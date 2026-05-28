# A1 实验：Scenario-level 分层结果分析报告

**生成时间**: 2026-05-28 06:32:19

**数据源**: 6 个模型 × 50 cases

**场景分类数**: 10

---

## 1. 场景分布统计

- **Chemotherapy**: 26 cases
- **Endocrine therapy**: 15 cases
- **Follow-up**: 4 cases
- **Medication side effects**: 103 cases
- **Nutrition**: 1 cases
- **Other**: 3 cases
- **Pain**: 25 cases
- **Psychological support**: 6 cases
- **Recovery**: 79 cases
- **Surgery-related**: 38 cases

## 2. 各场景表现最佳模型

- **Chemotherapy**: 最佳 qwen3-14b (5.00), 最差 qwen3-0.6b (1.53)
- **Endocrine therapy**: 最佳 qwen3-235b-a22b (4.40), 最差 qwen3-0.6b (2.47)
- **Follow-up**: 最佳 gpt-4o (5.00), 最差 None (5.00)
- **Medication side effects**: 最佳 qwen3-235b-a22b (4.54), 最差 qwen3-0.6b (1.73)
- **Nutrition**: 最佳 qwen3-8b (5.00), 最差 None (5.00)
- **Other**: 最佳 qwen3-14b (4.93), 最差 qwen3-14b (4.93)
- **Pain**: 最佳 qwen3-32b (4.85), 最差 qwen3-0.6b (2.00)
- **Psychological support**: 最佳 gpt-4o (5.00), 最差 qwen3-0.6b (1.00)
- **Recovery**: 最佳 qwen3-32b (4.65), 最差 qwen3-0.6b (2.05)
- **Surgery-related**: 最佳 qwen3-235b-a22b (4.97), 最差 qwen3-0.6b (2.14)

## 3. 模型跨场景稳定性分析

- **qwen3-14b**: 平均分 4.71 ± 0.27 (范围：4.27-5.00)
- **gpt-4o**: 平均分 4.66 ± 0.29 (范围：4.33-5.00)
- **qwen3-235b-a22b**: 平均分 4.55 ± 0.34 (范围：3.93-5.00)
- **qwen3-32b**: 平均分 4.21 ± 0.82 (范围：2.20-5.00)
- **qwen3-8b**: 平均分 4.19 ± 0.51 (范围：3.40-5.00)
- **qwen3-0.6b**: 平均分 1.85 ± 0.44 (范围：1.00-2.47)

## 4. 关键发现

- **最容易场景**: Nutrition (平均分 5.00)
- **最难场景**: Psychological support (平均分 2.73)
- **最稳定模型**: qwen3-14b (标准差 0.27)

---

*报告结束*
