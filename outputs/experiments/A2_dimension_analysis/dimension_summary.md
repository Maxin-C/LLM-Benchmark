# A2 实验：Five-dimensional score 分解分析报告

**生成时间**: 2026-05-28 06:35:14

**数据源**: 6 个模型 × 50 cases

**评分维度**: accuracy, effectiveness, safety, personalization, empathy

---

## 1. 各维度表现最佳模型

- **Accuracy
(临床准确性)**: 最佳 GPT-4O (4.36), 最差 QWEN3-0.6B (2.76)
- **Effectiveness
(临床有效性)**: 最佳 QWEN3-235B-A22B (4.58), 最差 QWEN3-0.6B (1.48)
- **Safety
(安全性)**: 最佳 GPT-4O (4.30), 最差 QWEN3-0.6B (2.89)
- **Personalization
(个体化)**: 最佳 QWEN3-14B (4.86), 最差 QWEN3-0.6B (1.22)
- **Empathy
(共情沟通)**: 最佳 QWEN3-14B (4.98), 最差 QWEN3-0.6B (1.28)

## 2. 模型优势维度分析

- **GPT-4O**: 最强 Empathy
(共情沟通) (4.90), 最弱 Safety
(安全性) (4.30)
- **QWEN3-0.6B**: 最强 Safety
(安全性) (2.89), 最弱 Personalization
(个体化) (1.22)
- **QWEN3-8B**: 最强 Empathy
(共情沟通) (4.60), 最弱 Accuracy
(临床准确性) (3.74)
- **QWEN3-14B**: 最强 Empathy
(共情沟通) (4.98), 最弱 Safety
(安全性) (4.05)
- **QWEN3-32B**: 最强 Empathy
(共情沟通) (4.94), 最弱 Safety
(安全性) (3.88)
- **QWEN3-235B-A22B**: 最强 Empathy
(共情沟通) (4.96), 最弱 Safety
(安全性) (4.14)

## 3. 维度难度分析（所有模型平均）

- **Empathy
(共情沟通)**: 4.28 分
- **Personalization
(个体化)**: 4.10 分
- **Effectiveness
(临床有效性)**: 3.91 分
- **Accuracy
(临床准确性)**: 3.88 分
- **Safety
(安全性)**: 3.86 分

## 4. 关键发现

- **综合表现最佳**: QWEN3-235B-A22B
- **维度稳定性最佳**: QWEN3-235B-A22B (平均标准差 0.94)
- **模型区分度最大维度**: Empathy
(共情沟通) (分差 3.70 分)

---

*报告结束*
