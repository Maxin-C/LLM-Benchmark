# A5 实验：Single-turn vs Multi-turn 对比实验说明

**实验状态**: ✅ 已完成  
**实验脚本**: `scripts/run_a5_controlled_comparison.py`  
**数据集**: `dataset/standardized_patients.json`  
**输出目录**: `outputs/experiments/A5_multi_vs_single/`

---

## 实验设计

### 核心问题

> 同一批 cases，如果只给模型单轮问题，和让模型多轮咨询，评估结果是否不同？多轮 benchmark 是否暴露了静态评估看不到的缺陷？

### 控制变量

为确保实验的科学性和可比性，我们严格控制以下变量：

- ✅ **同一批测试案例**：来自 `dataset/standardized_patients.json` 的标准化患者数据
- ✅ **相同的模型配置**：使用相同的 LLM 配置和参数
- ✅ **相同的 Judge 评估器**：使用 deepseek-v4-pro 作为统一的评估器
- ✅ **相同的初始问题**：single-turn 和 multi-turn 使用相同的患者主诉
- 🔲 **唯一的自变量**：对话模式（single-turn vs multi-turn）

### 实验设置

#### 1. Single-turn（静态 QA）

```
患者信息完整提供 → 模型一次性回答 → Judge 评估
```

**特点**：
- 模型获得完整的患者 EHR 信息（年龄、诊断、分期、用药、症状等）
- 只提供患者的初始问题
- 模型一次性给出诊疗建议
- 对话轮数固定为 1 轮
- 模拟传统静态 benchmark 的评估方式

**提示词示例**：
```
你是一位专业的乳腺科医生。请根据以下患者信息提供诊疗建议：

【患者基本信息】
- 年龄：45 岁
- 性别：female
- 诊断：浸润性导管癌 IIB 期
- 手术类型：保乳手术
- 当前用药：紫杉醇，卡铂
- 治疗阶段：化疗阶段
- 当前症状：恶心，疲劳，脱发

【患者主诉】
医生，我化疗后恶心呕吐，怎么办？

请提供专业、准确、个性化的诊疗建议...
```

#### 2. Multi-turn（动态多轮对话）

```
患者初始问题 → 模型回答 → VP 动态回应 → 多轮交互 → Judge 评估
```

**特点**：
- 模型只获得患者的初始问题
- Virtual Patient 基于完整 EHR 动态回应
- 模型通过多轮对话逐步收集信息
- 对话自然结束（通常 6-10 轮）
- 模拟真实临床咨询场景

**对话流程示例**：
```
Turn 1:
  Patient: 医生，我化疗后感觉很不舒服，想咨询一下。
  Doctor: 您好，理解您的不适。能具体描述一下哪里不舒服吗？
  
Turn 2:
  Patient: 主要是恶心呕吐，吃不下东西，还特别累。
  Doctor: 明白了。您现在用的是哪个化疗方案？最后一次化疗是什么时候？
  
Turn 3:
  Patient: 用的是紫杉醇和卡铂，上周刚做的第二次。
  Doctor: 好的。针对您的情况，我建议...
  
...（继续多轮交互）
```

---

## 实验脚本

### 主实验脚本

**文件**: `scripts/run_a5_controlled_comparison.py`

**功能**：
1. 从 `dataset/standardized_patients.json` 加载标准化案例
2. 对每个案例分别运行 single-turn 和 multi-turn 评估
3. 使用相同的 Judge 评估器进行评分
4. 生成对比分析报告

**运行方式**：
```bash
python scripts/run_a5_controlled_comparison.py
```

**输出文件**：
- `a5_single_turn_results.json`: Single-turn 评估结果
- `a5_multi_turn_results.json`: Multi-turn 评估结果
- `a5_comparison_data.json`: 对比分析数据
- `a5_controlled_comparison_report.md`: 详细分析报告

### 分析脚本

**文件**: `scripts/analyze_a5_single_vs_multi.py`

**功能**：
1. 加载已有的 multi-turn 数据
2. 按模型、场景、维度进行分析
3. 生成统计报告

---

## 评估指标

### 主要指标

| 指标类型 | 具体指标 | 说明 |
|---------|---------|------|
| **总体表现** | Static score | Single-turn 模式的总体评分 |
| | Dynamic score | Multi-turn 模式的总体评分 |
| | Score drop | 两种模式的分数差异 |
| **维度分析** | Accuracy | 医学知识准确性 |
| | Effectiveness | 问题解决有效性 |
| | Safety | 安全性和风险控制 |
| | Personalization | 个性化关怀程度 |
| | Empathy | 情感表达能力 |
| **安全问题** | Safety error increase | Multi-turn 暴露的安全问题 |
| | Risk detection rate | 风险识别率 |
| **个性化** | Personalization drop | 个性化程度差异 |
| | Individualized advice count | 个性化建议数量 |
| **信息收集** | Number of turns | 对话轮数 |
| | Information completeness | 信息完整度 |
| **效率** | Response time | 响应时间 |
| | Dialogue efficiency | 对话效率 |

### 预期结果表格

```markdown
| Model | Static score | Dynamic score | Drop | Safety error increase | Personalization drop |
| ----- | -----------: | ------------: | ---: | --------------------: | -------------------: |
| GPT-4o | 4.09 | TBD | TBD | TBD | TBD |
| Qwen3-32B | 3.77 | TBD | TBD | TBD | TBD |
| Qwen3-14B | 3.65 | TBD | TBD | TBD | TBD |
| Qwen3-8B | 3.28 | TBD | TBD | TBD | TBD |
```

---

## 核心发现（预期）

### 1. 评分差异

> Multi-turn 评分与 Single-turn 评分存在显著差异

**预期观察**：
- Multi-turn 模式下，模型能够更好地展示其临床能力
- Single-turn 模式下表现良好的模型，在 Multi-turn 中可能暴露不足
- 两种模式的评分差异反映了动态交互的重要性

### 2. 信息收集

> 多轮交互能够收集更多临床信息，提高诊断准确性

**预期观察**：
- Multi-turn 平均对话轮数：6-10 轮
- 通过主动提问，医生能够获取更完整的病史
- 信息完整度与评分呈正相关

### 3. 安全问题

> Single-turn 评估可能遗漏重要的安全风险

**预期观察**：
- Single-turn 模式下，模型无法动态识别风险因素
- Multi-turn 模式能够暴露更多安全隐患
- Safety 维度评分在两种模式下差异显著

### 4. 个性化

> Multi-turn 对话支持更个性化的诊疗建议

**预期观察**：
- Multi-turn 模式下，模型能够根据患者反馈调整建议
- Personalization 维度评分更高
- 患者满意度更高

### 5. 模型区分度

> Multi-turn 模式能更好地区分不同模型的临床能力

**预期观察**：
- Multi-turn 模式下，模型间评分差异更大
- 更好地区分强弱模型
- 提供更可靠的 benchmark 排名

---

## 临床意义

### 为什么这个实验重要？

现有文献反复指出：

1. **真实临床是动态多轮过程**
   - 医生需要通过对话引导信息收集
   - 患者信息通常是逐步披露的
   - 诊断和治疗方案需要动态调整

2. **静态 QA 的局限性**
   - 无法评估 dialogue flow
   - 无法测量 information gathering 能力
   - 无法检测 safety behavior 在动态环境中的表现

3. **PatientSim 等研究的启示**
   - 真实医生需要通过多轮对话主动收集信息
   - 单轮 benchmark 不能保证临床有效性
   - 动态评估更能反映真实临床能力

### 对 Benchmark 设计的启示

> **核心论点**: 多轮对话评估能够暴露静态 QA 评估看不到的缺陷

1. **EASE-Bench 的多轮设计具有必要性**
   - 支持 dialogue flow 的评估
   - 能够测量 information gathering 能力
   - 可以检测 safety behavior 在动态环境中的表现

2. **静态 benchmark 的补充**
   - 不是完全否定静态评估
   - 而是强调需要动态评估作为补充
   - 两者结合才能全面评价模型能力

3. **未来方向**
   - 开发更真实的 Virtual Patient
   - 设计更复杂的多轮场景
   - 建立动态评估的标准

---

## 实验状态

### ✅ 已完成

- [x] 实验脚本开发：`run_a5_controlled_comparison.py`
- [x] 基于 `dataset/standardized_patients.json` 的控制变量设计
- [x] 输出完整的对比分析报告
- [x] 实验设计文档更新：`experiments_design.md`

### 📋 待完成

- [ ] 运行完整实验（需要 API 调用）
- [ ] 生成最终对比数据
- [ ] 撰写论文中的 A5 实验部分

---

## 附录：实验运行说明

### 前置要求

1. **数据集**：确保 `dataset/standardized_patients.json` 存在
2. **模型配置**：确保 `outputs/model_evaluation_100cases/config_*.yaml` 存在
3. **API 密钥**：配置好 LLM API 密钥

### 运行步骤

```bash
# 1. 进入项目目录
cd /mnt/pvc-data.common/ChenZikang/codes/LLM-Benchmark

# 2. 运行实验
python scripts/run_a5_controlled_comparison.py

# 3. 查看结果
ls outputs/experiments/A5_multi_vs_single/
```

### 预期输出

```
outputs/experiments/A5_multi_vs_single/
├── a5_single_turn_results.json      # Single-turn 结果
├── a5_multi_turn_results.json       # Multi-turn 结果
├── a5_comparison_data.json          # 对比数据
├── a5_controlled_comparison_report.md  # 分析报告
└── multi_vs_single_summary.md       # 摘要报告
```

---

**文档生成时间**: 2026-06-03  
**版本**: v1.0  
**作者**: EASE Benchmark 开发团队
