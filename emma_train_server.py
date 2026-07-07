#!/usr/bin/env python3
"""
EMMA-BC 服务器端训练脚本
═══════════════════════════════════════════════
功能:
  1. 接收上传的音频数据集 (wav 文件 + labels.json)
  2. 用 wav2vec2-base 提取音频特征 (GPU 加速)
  3. 训练 sklearn 集成模型 (LR + SVC + GB + Ridge)
  4. 输出 .pkl 模型文件 + metrics.json
  5. 全程写日志到 train_log.txt

用法:
  python emma_train_server.py --data_dir /path/to/dataset --modality A

数据集目录结构:
  dataset_dir/
  ├── audio/
  │   ├── 001.wav
  │   ├── 002.wav
  │   └── ...
  └── labels.json   [{"file": "001.wav", "phq8": 5}, ...]

输出:
  output_dir/
  ├── model_<modality>_<timestamp>.pkl
  ├── metrics.json
  └── train_log.txt
"""
import os
import sys
import json
import time
import argparse
import warnings
from datetime import datetime

import numpy as np
import torch
import soundfile as sf
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import RidgeCV, LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import GradientBoostingClassifier, VotingClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score,
    recall_score, precision_score, mean_absolute_error, r2_score,
    confusion_matrix,
)

warnings.filterwarnings("ignore")

PHQ_THRESHOLD = 10.0
WAV2VEC2_MODEL = "facebook/wav2vec2-base"
LOG_FILE = "train_log.txt"


def log(msg):
    """写日志 (同时输出到控制台和文件)"""
    ts = datetime.now().strftime("%H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line, flush=True)
    try:
        with open(LOG_FILE, "a", encoding="utf-8") as f:
            f.write(line + "\n")
    except Exception:
        pass


def detect_device():
    """检测 GPU"""
    if torch.cuda.is_available():
        n = torch.cuda.device_count()
        names = [torch.cuda.get_device_name(i) for i in range(n)]
        device = torch.device("cuda:0")
        log(f"使用 GPU: {n} 张 -> {names[0]}")
        return device
    log("[警告] 无可用 GPU，使用 CPU (会很慢)")
    return torch.device("cpu")


def load_audio(path, target_sr=16000, max_sec=180):
    """加载音频，统一 16kHz 单声道"""
    data, sr = sf.read(path, dtype="float32")
    if data.ndim > 1:
        data = data.mean(axis=1)
    if sr != target_sr:
        import librosa
        data = librosa.resample(data, orig_sr=sr, target_sr=target_sr)
    max_samples = int(target_sr * max_sec)
    if len(data) > max_samples:
        data = data[:max_samples]
    return data.astype(np.float32)


def extract_features(audio_paths, device):
    """用 wav2vec2 提取音频特征 (768d mean pooling)"""
    from transformers import Wav2Vec2Model, Wav2Vec2Processor

    log("加载 wav2vec2-base 模型...")
    processor = Wav2Vec2Processor.from_pretrained(WAV2VEC2_MODEL)
    model = Wav2Vec2Model.from_pretrained(WAV2VEC2_MODEL).to(device)
    model.eval()

    features = []
    total = len(audio_paths)
    for i, path in enumerate(audio_paths):
        waveform = load_audio(path)
        inputs = processor(waveform, sampling_rate=16000, return_tensors="pt", padding=True)
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model(**inputs)
            hidden = outputs.last_hidden_state.squeeze(0)

            if "attention_mask" in inputs:
                import torch.nn.functional as F
                mask = inputs["attention_mask"].float().unsqueeze(0).unsqueeze(0)
                mask_ds = F.max_pool1d(mask, kernel_size=320, stride=320).squeeze().to(device)
                if mask_ds.shape[0] > hidden.shape[0]:
                    mask_ds = mask_ds[:hidden.shape[0]]
                elif mask_ds.shape[0] < hidden.shape[0]:
                    mask_ds = F.pad(mask_ds, (0, hidden.shape[0] - mask_ds.shape[0]))
                pooled = (hidden * mask_ds.unsqueeze(-1)).sum(0) / mask_ds.sum().clamp(min=1)
            else:
                pooled = hidden.mean(0)

        features.append(pooled.cpu().numpy().astype(np.float32))

        if (i + 1) % 5 == 0 or i == total - 1:
            log(f"特征提取进度: {i+1}/{total} ({(i+1)/total*100:.0f}%)")

    return np.stack(features, axis=0)


def train_model(features, labels, modality="A"):
    """训练 sklearn 集成模型"""
    labels = np.array(labels, dtype=np.float32)
    y_binary = (labels >= PHQ_THRESHOLD).astype(int)
    n_pos = int(y_binary.sum())
    n_neg = len(y_binary) - n_pos

    log(f"样本分布: 总数={len(labels)}, 抑郁={n_pos}, 非抑郁={n_neg}")

    if n_pos < 3 or n_neg < 3:
        return {"success": False, "error": f"类别不均衡: 抑郁 {n_pos} / 非抑郁 {n_neg}，每类至少需要 3 个"}

    # 标准化
    scaler = StandardScaler()
    X = scaler.fit_transform(features)

    # 集成分类器
    log("训练集成分类器 (LR + SVC + GB)...")
    clf = VotingClassifier(estimators=[
        ("lr", LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42)),
        ("svc", SVC(kernel="rbf", probability=True, class_weight="balanced", random_state=42)),
        ("gb", GradientBoostingClassifier(n_estimators=100, max_depth=4, random_state=42)),
    ], voting="soft", weights=[2, 1, 1])
    clf.fit(X, y_binary)

    # 回归器
    log("训练回归器 (Ridge)...")
    reg = RidgeCV(alphas=[0.01, 0.1, 1.0, 10.0, 100.0])
    reg.fit(X, labels)

    # 交叉验证
    log("5 折交叉验证...")
    cv = StratifiedKFold(n_splits=min(5, n_pos, n_neg), shuffle=True, random_state=42)
    if min(n_pos, n_neg) >= 5:
        y_pred = cross_val_predict(clf, X, y_binary, cv=cv, method="predict")
        y_proba = cross_val_predict(clf, X, y_binary, cv=cv, method="predict_proba")[:, 1]
    else:
        y_pred = clf.predict(X)
        y_proba = clf.predict_proba(X)[:, 1]

    cm = confusion_matrix(y_binary, y_pred)
    tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)
    reg_pred = reg.predict(X)

    metrics = {
        "n_samples": len(labels),
        "n_depressed": n_pos,
        "modality": modality,
        "feature_dim": int(features.shape[1]),
        "classification": {
            "accuracy": round(float(accuracy_score(y_binary, y_pred)), 4),
            "f1_score": round(float(f1_score(y_binary, y_pred, zero_division=0)), 4),
            "auc_roc": round(float(roc_auc_score(y_binary, y_proba)), 4) if len(set(y_binary)) > 1 else None,
            "sensitivity": round(float(tp / (tp + fn + 1e-10)), 4),
            "specificity": round(float(tn / (tn + fp + 1e-10)), 4),
            "precision": round(float(precision_score(y_binary, y_pred, zero_division=0)), 4),
            "confusion_matrix": [[int(tn), int(fp)], [int(fn), int(tp)]],
        },
        "regression": {
            "mae_phq8": round(float(mean_absolute_error(labels, reg_pred)), 2),
            "r2_phq8": round(float(r2_score(labels, reg_pred)), 4),
        },
    }

    log(f"训练完成: Acc={metrics['classification']['accuracy']}, "
        f"F1={metrics['classification']['f1_score']}, "
        f"AUC={metrics['classification']['auc_roc']}")

    # 保存模型
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_name = f"emma_bc_{modality}_{timestamp}"
    model_data = {
        "scaler": scaler,
        "classifier": clf,
        "regressor": reg,
        "metrics": metrics,
        "modality": modality,
        "feature_dim": int(features.shape[1]),
        "phq_threshold": PHQ_THRESHOLD,
        "version": "2.0.0",
    }

    return {"success": True, "model_name": model_name, "model_data": model_data, "metrics": metrics}


def main():
    parser = argparse.ArgumentParser(description="EMMA-BC 服务器训练")
    parser.add_argument("--data_dir", required=True, help="数据集目录 (含 audio/ 和 labels.json)")
    parser.add_argument("--output_dir", required=True, help="输出目录")
    parser.add_argument("--modality", default="A", choices=["A", "AV"], help="模态")
    args = parser.parse_args()

    # 初始化日志
    log_path = os.path.join(args.output_dir, LOG_FILE)
    global LOG_FILE
    LOG_FILE = log_path
    os.makedirs(args.output_dir, exist_ok=True)

    # 清空旧日志
    with open(log_path, "w", encoding="utf-8") as f:
        f.write("")

    log("=" * 60)
    log("EMMA-BC 服务器训练任务启动")
    log(f"数据目录: {args.data_dir}")
    log(f"输出目录: {args.output_dir}")
    log(f"模态: {args.modality}")
    log("=" * 60)

    # 1. 加载标签
    labels_path = os.path.join(args.data_dir, "labels.json")
    if not os.path.exists(labels_path):
        log(f"[错误] 标签文件不存在: {labels_path}")
        sys.exit(1)

    with open(labels_path, "r", encoding="utf-8") as f:
        labels_data = json.load(f)

    log(f"加载标签: {len(labels_data)} 条")

    # 2. 加载音频
    audio_dir = os.path.join(args.data_dir, "audio")
    audio_paths = []
    labels = []
    for item in labels_data:
        ap = os.path.join(audio_dir, item["file"])
        if os.path.exists(ap):
            audio_paths.append(ap)
            labels.append(float(item["phq8"]))
        else:
            log(f"[跳过] 音频文件不存在: {item['file']}")

    if len(audio_paths) < 20:
        log(f"[错误] 有效样本不足: {len(audio_paths)} < 20")
        sys.exit(1)

    log(f"有效音频: {len(audio_paths)} 个")

    # 3. 提取特征
    device = detect_device()
    log("开始特征提取...")
    features = extract_features(audio_paths, device)
    log(f"特征矩阵: {features.shape}")

    # 4. 训练
    log("开始模型训练...")
    result = train_model(features, labels, args.modality)

    if not result["success"]:
        log(f"[错误] 训练失败: {result['error']}")
        sys.exit(1)

    # 5. 保存模型
    model_path = os.path.join(args.output_dir, f"{result['model_name']}.pkl")
    joblib.dump(result["model_data"], model_path)
    log(f"模型已保存: {model_path}")

    metrics_path = os.path.join(args.output_dir, "metrics.json")
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(result["metrics"], f, ensure_ascii=False, indent=2)

    # 6. 完成
    log("=" * 60)
    log("训练任务完成!")
    log(f"模型: {model_path}")
    log(f"指标: {metrics_path}")
    log("=" * 60)

    # 写完成标记
    with open(os.path.join(args.output_dir, "DONE"), "w") as f:
        f.write(result["model_name"])


if __name__ == "__main__":
    main()
