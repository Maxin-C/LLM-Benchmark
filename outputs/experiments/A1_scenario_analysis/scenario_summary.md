# A1 实验：Scenario-level 分层结果分析报告

**生成时间**: 2026-06-02 09:36:39

**数据源**: 5 个模型 × 100 cases

**场景分类数**: 10

---

## 1. 场景分布统计

- **Chemotherapy**: 130 cases
- **Endocrine therapy**: 130 cases
- **Family planning**: 10 cases
- **Follow-up**: 20 cases
- **Medication side effects**: 10 cases
- **Pain**: 15 cases
- **Psychological support**: 35 cases
- **Recovery**: 110 cases
- **Surgery-related**: 35 cases
- **Work & Life**: 5 cases

## 2. 各场景表现最佳模型

- **Chemotherapy**: 最佳 qwen3-235b-a22b (3.95), 最差 qwen3-8b (3.37)
- **Endocrine therapy**: 最佳 gpt-4o (4.15), 最差 qwen3-8b (3.20)
- **Family planning**: 最佳 gpt-4o (4.35), 最差 qwen3-8b (3.27)
- **Follow-up**: 最佳 gpt-4o (4.20), 最差 qwen3-8b (3.53)
- **Medication side effects**: 最佳 qwen3-235b-a22b (4.34), 最差 qwen3-8b (3.37)
- **Pain**: 最佳 gpt-4o (4.23), 最差 qwen3-14b (2.98)
- **Psychological support**: 最佳 gpt-4o (4.13), 最差 qwen3-8b (3.13)
- **Recovery**: 最佳 gpt-4o (4.11), 最差 qwen3-8b (3.25)
- **Surgery-related**: 最佳 gpt-4o (4.13), 最差 qwen3-8b (3.29)
- **Work & Life**: 最佳 qwen3-235b-a22b (4.32), 最差 qwen3-32b (2.12)

## 3. 模型跨场景稳定性分析

- **gpt-4o**: 平均分 4.12 ± 0.16 (范围：3.74-4.35)
- **qwen3-235b-a22b**: 平均分 3.96 ± 0.24 (范围：3.58-4.34)
- **qwen3-32b**: 平均分 3.69 ± 0.56 (范围：2.12-4.26)
- **qwen3-14b**: 平均分 3.57 ± 0.42 (范围：2.77-4.25)
- **qwen3-8b**: 平均分 3.34 ± 0.15 (范围：3.13-3.68)

## 4. 关键发现

- **最容易场景**: Medication side effects (平均分 4.08)
- **最难场景**: Work & Life (平均分 3.33)
- **最稳定模型**: qwen3-8b (标准差 0.15)

---

*报告结束*
