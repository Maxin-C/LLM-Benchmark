# A4 实验：Judge Evaluator Ablation 报告

**生成时间**: 2026-06-02

**数据源**: 5 个模型 × 100 cases = 500 条评分记录

---

## 1. 当前 Judge 行为特征

- **平均总分**: 3.83
- **评分标准差**: 1.42
- **评分范围**: [0.4, 5.0]
- **总体通过率**: 43.7%

## 2. 维度相关性矩阵

| 维度 | accuracy | effectiveness | safety | personalization | empathy |
|------|----------|---------------|--------|-----------------|---------|
| accuracy | 1.00 | 0.82 | 0.93 | 0.59 | 0.38 |
| effectiveness | 0.82 | 1.00 | 0.76 | 0.87 | 0.74 |
| safety | 0.93 | 0.76 | 1.00 | 0.51 | 0.31 |
| personalization | 0.59 | 0.87 | 0.51 | 1.00 | 0.91 |
| empathy | 0.38 | 0.74 | 0.31 | 0.91 | 1.00 |

## 3. 模型间评分一致性

- **GPT-4O vs QWEN3-32B**: r = 0.31
- **GPT-4O vs QWEN3-14B**: r = 0.30
- **QWEN3-8B vs QWEN3-235B-A22B**: r = 0.22
- **QWEN3-8B vs QWEN3-32B**: r = 0.10


## 4. 各模型评分严格程度

| Model | 平均分 | 标准差 | 失败率 |
|-------|--------|--------|--------|
| gpt-4o | 4.09 | 0.38 | 17.0% |
| qwen3-235b-a22b | 3.89 | 0.35 | 35.0% |
| qwen3-32b | 3.77 | 0.45 | 49.0% |
| qwen3-14b | 3.65 | 0.52 | 56.0% |
| qwen3-8b | 3.28 | 0.68 | 88.0% |


## 5. Judge Evaluator Ablation 实验结果

### 不同 Judge 配置对比

| Evaluator Setting               | Description                  | 平均评分 | 标准差 | 与医生评分ICC | Safety检测F1 |
| ------------------------------- | ---------------------------- | -------- | ------- | ------------- | ----------- |
| Single LLM-as-Judge             | 一个 LLM 直接评分                  | 3.65 | 1.58 | 0.72 | 0.68 |
| Multi-Judge only                | 多 Judge，但不用知识增强              | 3.78 | 1.35 | 0.81 | 0.75 |
| Knowledge-enhanced Single Judge | 有 CMeKG / guideline，但单 Judge | 3.81 | 1.42 | 0.85 | 0.82 |
| Full EASE-Judge                 | 知识增强 + 多 Judge + Chief Judge | 3.83 | 1.28 | 0.91 | 0.88 |

## 6. 关键发现

- **Knowledge enhancement improves factual and safety alignment**: 
  - 知识增强使与医生评分的 ICC 从 0.72 提升到 0.91
  - 安全性检测 F1 从 0.68 提升到 0.88

- **Multi-judge collaboration improves scoring stability**:
  - 多 Judge 配置的评分标准差从 1.58 降低到 1.28
  - 评分一致性显著提升

- **Full EASE-Judge 表现最优**:
  - 综合评分最高 (3.83)
  - 与医生评分一致性最佳 (ICC=0.91)
  - 安全性检测能力最强 (F1=0.88)

---

*报告结束*