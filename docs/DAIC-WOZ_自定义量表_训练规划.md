# DAIC-WOZ + 自定义量表 多模态训练规划

> **核心决策:**
> 1. DAIC-WOZ 视频模态忽略 → 只用音频+文本 → 与医院模态完全匹配
> 2. PHQ-8 替换为你的5个量表 → 预训练用 PHQ-8，微调用你的量表
> 3. 量表完美覆盖四维知识图谱全部维度

**版本:** v2.0 | **日期:** 2026-06-15

---

## 一、模态匹配方案

### 问题

| | DAIC-WOZ 有 | 医院有 | 匹配吗 |
|---|---|---|---|
| 音频 | ✅ | ✅ | ✅ |
| 文本 | ✅ 访谈转写 | ✅ 心情日志 | ⚠️ 文体不同 |
| 视频 | ✅ | ❌ | ❌ |
| 量表 | ✅ PHQ-8 | ✅ 5个量表 | ⚠️ 量表不同 |

### 方案：DAIC-WOZ 预训练时直接丢弃视频模态

```
DAIC-WOZ 预训练（现在就能做）:
  [🎤 音频] ──→ wav2vec2 ──┐
                           ├──→ MulT 双模态融合 → PHQ-8
  [📝 转写文本] ──→ BERT ──┘

医院微调（数据到了就做）:
  [🎤 音频] ──→ wav2vec2 (复用) ──┐
                                  ├──→ MulT 双模态融合 → [HADS, VAS, CFS, PROMIS, LSNS]
  [📝 心情日志] ──→ BERT (复用) ──┘

零模态不匹配 ✅
```

### 为什么这是最优解

| 方案 | 优点 | 缺点 |
|------|------|------|
| A: 预训练丢弃视频（本方案） | 模态完全匹配，零迁移损耗 | DAIC-WOZ 的视频信息没用到 |
| B: 预训练用三模态，医院用零填充 | 利用了更多信息 | 零向量与真实视频分布差异大，可能引入噪声 |
| C: 视频作为"特权信息"训练 | 编码器更强 | 需要额外设计蒸馏损失，复杂度高 |

**选择方案A的原因：** DAIC-WOZ 的视频是固定摄像头访谈——大多数时间是坐着的半身像。对抑郁检测的价值远不如音频大。丢弃它损失很小，换来模态完全匹配的确定性。

---

## 二、量表映射：你的5个量表 → 四维知识图谱

### 2.1 量表全景

```
configs/scales/
├── HADS.json          ← 🧠 心理维度（焦虑+抑郁）
├── VAS.json           ← 💪 生理维度（疼痛）
├── CFS.json           ← 💪 生理维度（癌因疲乏，3个子维度）
├── PROMIS_Physical_Function_6b.json  ← 🏃 身体功能维度
└── LSNS-6.json        ← 👥 社会健康维度
```

### 2.2 四维知识图谱映射

```
                    心理-生理-身体功能-社会健康 四维症状知识图谱
                    ════════════════════════════════════════

  🧠 心理维度                    💪 生理维度
  ┌────────────────┐            ┌────────────────┐
  │ HADS-A 焦虑    │            │ VAS 疼痛       │
  │ 分数: 0-21     │            │ 分数: 0-100    │
  │ ↑=越焦虑       │            │ ↑=越疼         │
  ├────────────────┤            ├────────────────┤
  │ HADS-D 抑郁    │            │ CFS 癌因疲乏    │
  │ 分数: 0-21     │            │ 总分: 15-75    │
  │ ↑=越抑郁       │            │ • 身体疲乏(4项)│
  └────────────────┘            │ • 认知疲乏(5项)│
                                │ • 情感疲乏(5项)│
                                │ ↑=越疲乏       │
  🏃 身体功能维度               └────────────────┘
  ┌────────────────┐
  │ PROMIS-PF-6b   │            👥 社会健康维度
  │ T分: μ=50 σ=10 │            ┌────────────────┐
  │ ↑=功能越好     │            │ LSNS-6 社会网络 │
  └────────────────┘            │ 总分: 0-30     │
                                │ ≤12=隔离风险   │
                                │ ↑=支持越强     │
                                └────────────────┘
```

### 2.3 模型输出配置

```yaml
# configs/hospital_scales.yaml
scales:
  prediction_targets:
    - name: "hads_anxiety"
      range: [0, 21]
      direction: "higher_worse"
      dimension: "psychological"
      
    - name: "hads_depression"
      range: [0, 21]
      direction: "higher_worse"
      dimension: "psychological"
      
    - name: "vas_pain"
      range: [0, 100]
      direction: "higher_worse"
      dimension: "physiological"
      
    - name: "cfs_total"
      range: [15, 75]
      direction: "higher_worse"
      dimension: "physiological"
      subscales: ["cfs_physical", "cfs_cognitive", "cfs_emotional"]
      
    - name: "promis_pf_t"
      range: [20, 80]        # T-score typical range
      direction: "higher_better"
      dimension: "physical_function"
      note: "需用PROMIS官方评分表将原始分转为T分"
      
    - name: "lsns6_total"
      range: [0, 30]
      direction: "higher_better"
      dimension: "social_health"
      threshold: {value: 12, interpretation: "social_isolation_risk"}
```

---

## 三、分阶段训练计划

### 阶段 A: DAIC-WOZ 双模态预训练

```
目标: 训练"从音频+文本中提取情绪表征"的共享编码器

数据: DAIC-WOZ (189人)
输入: 
  ├── 音频: 临床访谈录音 (wav2vec2 → 768维)
  └── 文本: 访谈转写 (BERT → 768维)
标签:
  └── PHQ-8 总分 (0-24)

模型: MulT_Bimodal(audio, text)
      d_model=256, n_layers=4, n_heads=8
      num_emotions=1

损失: MSE + CCC

产出:
  ├── checkpoints/daic_woz_pretrain.pt     (完整模型)
  ├── checkpoints/audio_encoder.pt         (音频编码器)
  └── checkpoints/text_encoder.pt          (文本编码器)

验证: 在 DAIC-WOZ 测试集上
  ├── RMSE ≤ 5.0 (PHQ-8范围0-24)
  └── CCC ≥ 0.6
```

### 阶段 B: 医院数据多任务微调

```
目标: 在中文乳腺癌患者数据上微调，输出5个量表分数

数据: 医院采集 (目标≥30例，含全部5个量表)
输入:
  ├── 音频: 标准化访谈录音 (wav2vec2 → 768维)
  ├── 文本: 心情日志 (中文BERT → 768维)
  └── 量表标签: [HADS-A, HADS-D, VAS, CFS, PROMIS_T, LSNS6]

模型: 加载阶段A编码器 + 新预测头
      num_emotions=6  (6个输出头)

训练策略:
  ├── 编码器: 解冻，lr=1e-5 (低学习率保护预训练知识)
  ├── 新预测头: lr=1e-4 (高学习率快速适应)
  └── 每个量表独立标准化到[0,1]再训练

损失: 多任务损失 = Σ (w_i * MSE_i)
      权重 w_i 根据各量表方差动态调整

产出:
  └── checkpoints/hospital_finetuned.pt
```

### 阶段 C（可选）: 时序知识图谱 + 风险预测 + DRL干预

```
在阶段B的多模态表征基础上，接入:
  ├── 时序知识图谱（CFS时序 + HADS时序 + VAS时序）
  ├── 阶段感知GNN（化疗周期特异性）
  └── DRL自适应干预
```

---

## 四、数据采集格式（医院用）

### 单次采集JSON格式

```json
{
  "patient_id": "P001",
  "time_point": "C1D3",
  "date": "2026-08-15",
  "treatment_phase": "chemo_cycle_1_day_3",
  
  "audio": {
    "file": "P001_C1D3_audio.wav",
    "duration_sec": 180,
    "sample_rate": 16000,
    "task": "standardized_interview"
  },
  
  "text": {
    "content": "今天化疗后第三天，感觉特别累……",
    "char_count": 85,
    "type": "mood_diary"
  },
  
  "scales": {
    "hads": {
      "anxiety_score": 12,
      "depression_score": 9,
      "items": [2,1,2,3,1,1,2, 2,1,1,2,0,1,2]
    },
    "vas_pain": {
      "score_mm": 45,
      "category": "moderate_pain"
    },
    "cfs": {
      "total_score": 48,
      "physical_fatigue": 12,
      "cognitive_fatigue": 18,
      "emotional_fatigue": 18
    },
    "promis_pf_6b": {
      "raw_score": 18,
      "t_score": 42.5,
      "interpretation": "below_average_physical_function"
    },
    "lsns6": {
      "total_score": 14,
      "family_support": 8,
      "friend_support": 6,
      "isolation_risk": false
    }
  }
}
```

---

## 五、MulT 模型代码适配

### 5.1 当前代码改动点

当前 `mul_t_model.py` 是 Audio+Video 版本，需要改为 Audio+Text：

```python
# 改动1: MulT_Bimodal 类修改
class MulT_Bimodal(nn.Module):
    def __init__(
        self,
        d_audio: int = 768,
        d_text: int = 768,        # 原来是 d_video=512，改为 d_text=768
        d_model: int = 256,
        ...
    ):
        # 改动2: video_proj → text_proj
        self.text_proj = nn.Sequential(...)
        self.text_tconv = TemporalConv1D(...)
        
        # 改动3: Crossmodal Transformers 改为 Audio↔Text
        self.audio_from_text = CrossmodalTransformer(...)
        self.text_from_audio = CrossmodalTransformer(...)
```

### 5.2 多任务预测头（新增）

```python
class MultiScaleHead(nn.Module):
    """多任务预测头：从共享表征预测多个临床量表分数"""
    
    def __init__(self, d_model=256, scales_config=None):
        super().__init__()
        self.heads = nn.ModuleDict({
            "hads_anxiety":    ScaleHead(d_model, range_max=21),
            "hads_depression": ScaleHead(d_model, range_max=21),
            "vas_pain":        ScaleHead(d_model, range_max=100),
            "cfs_total":       ScaleHead(d_model, range_max=75),
            "promis_pf_t":     ScaleHead(d_model, range_max=80),
            "lsns6_total":     ScaleHead(d_model, range_max=30),
        })
    
    def forward(self, fused_embedding):
        return {name: head(fused_embedding) for name, head in self.heads.items()}

class ScaleHead(nn.Module):
    def __init__(self, d_model, range_max):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_model//2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(d_model//2, 1),
            nn.Sigmoid(),         # 输出 [0, 1]
        )
        self.range_max = range_max
    
    def forward(self, x):
        return self.net(x) * self.range_max  # 映射到实际量表范围
```

---

## 六、立即行动清单

- [ ] **P0:** 修改 `mul_t_model.py` → Audio+Text 双模态版本
- [ ] **P0:** 新增 `MultiScaleHead` 多任务预测头
- [ ] **P0:** 申请 DAIC-WOZ 数据集（https://dcapswoz.ict.usc.edu/）
- [ ] **P1:** 准备中文BERT文本编码器（bert-base-chinese）
- [ ] **P1:** 编写 DAIC-WOZ 数据加载器
- [ ] **P2:** 编写医院数据格式模拟生成器
- [ ] **P2:** 完整的阶段A训练脚本

---

*文档版本: v2.0 | 最后更新: 2026-06-15*
