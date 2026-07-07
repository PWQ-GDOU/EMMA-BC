#!/usr/bin/env python3
"""
EMMA-BC OpenFace 特征提取脚本 (服务器端)
═══════════════════════════════════════════════
功能: 接收视频文件，用 OpenFace 提取面部特征 CSV

用法:
  python emma_openface_extract.py --video /path/to/video.webm --output /path/to/output/

输出:
  output/<video_name>.csv  (OpenFace 特征文件)
  output/DONE              (完成标记)
"""
import os
import sys
import argparse
import subprocess
from pathlib import Path
from datetime import datetime

OPENFACE_BIN = "/data/disk1/OpenFace/build/bin/FeatureExtraction"

# 日志文件路径 (在 main() 中设置)
_LOG_FILE = "openface_log.txt"


def log(msg):
    ts = datetime.now().strftime("%H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line, flush=True)
    try:
        with open(_LOG_FILE, "a", encoding="utf-8") as f:
            f.write(line + "\n")
    except Exception:
        pass


def main():
    global _LOG_FILE
    parser = argparse.ArgumentParser(description="EMMA-BC OpenFace 特征提取")
    parser.add_argument("--video", required=True, help="视频文件路径")
    parser.add_argument("--output", required=True, help="输出目录")
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)
    _LOG_FILE = os.path.join(args.output, "openface_log.txt")

    # 清空旧日志
    with open(_LOG_FILE, "w", encoding="utf-8") as f:
        f.write("")

    log("=" * 60)
    log("EMMA-BC OpenFace 特征提取任务启动")
    log(f"视频: {args.video}")
    log(f"输出: {args.output}")
    log("=" * 60)

    # 检查 OpenFace 是否安装
    if not os.path.exists(OPENFACE_BIN):
        log(f"[错误] OpenFace 未安装: {OPENFACE_BIN}")
        log("请等待 OpenFace 安装完成，或联系管理员安装")
        sys.exit(1)

    # 检查 HAAR 检测器文件 (OpenFace 自带的)
    haar_path = os.path.join(os.path.dirname(OPENFACE_BIN), "model", "haarcascade_frontalface_alt.xml")
    if not os.path.exists(haar_path):
        # 下载 OpenCV 自带的 HAAR 文件
        log("下载 HAAR 人脸检测器...")
        import urllib.request
        haar_url = "https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/haarcascade_frontalface_alt.xml"
        os.makedirs(os.path.dirname(haar_path), exist_ok=True)
        try:
            urllib.request.urlretrieve(haar_url, haar_path)
            log(f"HAAR 检测器已下载: {haar_path}")
        except Exception as e:
            log(f"[警告] HAAR 下载失败: {e} (OpenFace 可能用其他检测器)")

    # 检查视频文件
    if not os.path.exists(args.video):
        log(f"[错误] 视频文件不存在: {args.video}")
        sys.exit(1)

    # 如果是 webm 格式，先转为 mp4 (OpenFace 对 mp4/avi 支持更好)
    video_path = args.video
    if video_path.endswith(".webm"):
        log("检测到 webm 格式，转换为 mp4...")
        mp4_path = video_path.replace(".webm", ".mp4")
        try:
            result = subprocess.run(
                ["ffmpeg", "-y", "-i", video_path, "-c:v", "libx264", "-c:a", "aac", mp4_path],
                capture_output=True, text=True, timeout=120
            )
            if result.returncode == 0 and os.path.exists(mp4_path):
                video_path = mp4_path
                log(f"转换完成: {mp4_path}")
            else:
                log(f"[警告] 转换失败，尝试直接处理 webm: {result.stderr[:200]}")
        except FileNotFoundError:
            log("[警告] ffmpeg 未安装，尝试直接处理 webm")
        except Exception as e:
            log(f"[警告] 转换异常: {e}")

    # 运行 OpenFace FeatureExtraction
    # 默认模型已替换为 CLNF (SVR/CCNF patch experts, 无需 Google Drive 下载)
    log("启动 OpenFace 特征提取...")
    cmd = [
        OPENFACE_BIN,
        "-f", video_path,
        "-out_dir", args.output,
        "-aus",       # Action Units (核心特征)
        "-pose",      # 头部姿态
        "-gaze",      # 注视方向
        "-2Dfp",      # 2D 面部关键点
        "-3Dfp",      # 3D 面部关键点
        "-pdmparams", # PDM 参数
        "-q",         # 安静模式
    ]

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
        log(f"OpenFace 退出码: {result.returncode}")
        if result.stderr:
            log(f"OpenFace 输出: {result.stderr[:500]}")
    except subprocess.TimeoutExpired:
        log("[错误] OpenFace 超时 (10分钟)")
        sys.exit(1)
    except Exception as e:
        log(f"[错误] OpenFace 运行失败: {e}")
        sys.exit(1)

    # 查找生成的 CSV
    video_name = Path(video_path).stem
    csv_files = list(Path(args.output).glob(f"{video_name}*.csv"))

    if not csv_files:
        # 尝试查找任意 CSV
        csv_files = list(Path(args.output).glob("*.csv"))

    if csv_files:
        csv_path = str(csv_files[0])
        # 重命名为标准名
        final_csv = os.path.join(args.output, "features.csv")
        if csv_path != final_csv:
            os.rename(csv_path, final_csv)
        log(f"特征提取完成: {final_csv}")

        # 写完成标记
        with open(os.path.join(args.output, "DONE"), "w") as f:
            f.write(final_csv)

        log("=" * 60)
        log("OpenFace 特征提取任务完成!")
        log(f"CSV: {final_csv}")
        log("=" * 60)
    else:
        log("[错误] 未找到生成的 CSV 文件")
        sys.exit(1)


if __name__ == "__main__":
    main()
