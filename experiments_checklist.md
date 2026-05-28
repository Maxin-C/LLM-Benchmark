# 实验任务清单与进度跟踪

## 图例说明
- ✅ 已完成
- 🔄 进行中
- ⏳ 待开始
- ❌ 已取消

---

## A. 必须做的实验

### A1. Scenario-level 分层结果分析
- [x] 数据准备：从已有 50 cases 中提取场景信息
- [x] 场景分类：基于 diagnosis 和 symptoms 自动分类到 13 个场景
- [x] 统计分析：按场景计算各模型的平均分和通过率
- [x] 可视化：生成 model × scenario heatmap
- [x] 结果存储：`outputs/experiments/A1_scenario_analysis/`

**状态**: ✅ 已完成  
**优先级**: 高  
**实施方案**: 复用已有 6 个模型的 50 cases 数据，无需重新运行评估，仅需后处理分析  
**预计 API 开销**: 0（纯分析）  
**完成时间**: 2026-05-28 06:32  
**关键发现**: 
- 最稳定模型：qwen3-14b (标准差 0.27)
- 最难场景：Psychological support (平均分 2.73)
- gpt-4o 和 qwen3-14b 表现最佳

---

### A2. Five-dimensional score 分解分析
- [x] 数据提取：从 evaluation.scores 中提取 5 个维度分数
- [x] 维度统计：计算各模型在各维度的平均分和标准差
- [x] 可视化：生成雷达图和柱状图
- [x] 结果存储：`outputs/experiments/A2_dimension_analysis/`

**状态**: ✅ 已完成  
**优先级**: 高  
**实施方案**: 复用已有数据，直接提取 evaluation 字段中的 scores  
**预计 API 开销**: 0（纯分析）  
**完成时间**: 2026-05-28 06:35  
**关键发现**: 
- 综合最佳：Qwen3-235B-A22B
- 最难维度：Safety (3.86 分)
- 区分度最大：Empathy (3.70 分差)

---

### A3. Human alignment 的详细结果
- [ ] 检查是否有现成的人类标注数据（100 例）
- [ ] 计算各维度的 ICC 和 95% CI
- [ ] 计算 Pearson/Spearman correlation
- [ ] 计算 safety risk detection 的 F1
- [ ] 结果存储：`outputs/experiments/A3_human_alignment/`

**状态**: ⏳ 待开始  
**优先级**: 高  
**实施方案**: 需要确认是否有 100 例人类标注数据，如果没有需要跳过或简化  
**预计 API 开销**: 0（如果有现成数据）

---

### A4. Judge evaluator ablation
- [ ] 设计 4 种 judge 配置（Single/Multi/Knowledge/Full）
- [ ] 运行小规模测试（10-20 cases）验证配置
- [ ] 运行完整评估（50 cases × 4 配置）
- [ ] 比较各配置的 ICC、稳定性、安全性检测
- [ ] 结果存储：`outputs/experiments/A4_judge_ablation/`

**状态**: ⏳ 待开始  
**优先级**: 高  
**实施方案**: 需要修改 evaluator 配置，重新运行评估（可考虑减少 case 数或使用少量模型）  
**预计 API 开销**: 高（需要重新运行多组评估）

---

### A5. Dynamic multi-turn vs static single-turn 对比
- [ ] 设计 static single-turn 评估脚本
- [ ] 从 50 cases 中提取 single-turn 问题
- [ ] 运行 static 评估（复用模型回答）
- [ ] 比较 static vs dynamic 的评分差异
- [ ] 结果存储：`outputs/experiments/A5_static_vs_dynamic/`

**状态**: ⏳ 待开始  
**优先级**: 高  
**实施方案**: 需要编写 static 评估脚本，但可复用已有模型回答，无需重新调用模型 API  
**预计 API 开销**: 低（仅需 evaluator 调用）

---

### A6. Benchmark 区分度的统计检验
- [x] Bootstrap 重采样（1000 次）
- [x] 计算 95% CI 和 effect size
- [x] 显著性检验（ANOVA 或 t-test）
- [x] Model ranking 稳定性分析
- [x] 结果存储：`outputs/experiments/A6_statistical_test/`

**状态**: ✅ 已完成  
**优先级**: 高  
**实施方案**: 复用已有数据，纯统计分析  
**预计 API 开销**: 0（纯统计）  
**完成时间**: 2026-05-28 06:40  
**关键发现**: 
- 模型间存在显著差异 (F=55.47, p=1.77e-40)
- 最佳模型：GPT-4O (4.53), Qwen3-235B (4.53), Qwen3-14B (4.52)
- 显著差异的模型对：5/15 (33.3%)，均为 Qwen3-0.6B vs 其他模型

---

## B. 强烈建议做的实验

### B2. Patient Agent hallucination 检测
- [ ] 设计自动检测脚本（矛盾检测）
- [ ] 分析 patient 回答与 EHR 的一致性
- [ ] 计算 contradiction rate 和 hallucination rate
- [ ] 结果存储：`outputs/experiments/B2_patient_hallucination/`

**状态**: ⏳ 待开始  
**优先级**: 中  
**实施方案**: 复用对话数据，自动检测 patient 回答是否与 EHR 矛盾  
**预计 API 开销**: 0（自动检测）

---

### B3. Case difficulty / risk-level stratification
- [ ] 设计风险等级分类规则（基于症状和诊断）
- [ ] 将 50 cases 分层（Low/Medium/High/Safety-critical）
- [ ] 分析各模型在不同风险等级的表现
- [ ] 结果存储：`outputs/experiments/B3_risk_stratification/`

**状态**: ⏳ 待开始  
**优先级**: 中  
**实施方案**: 基于 EHR 数据自动分类，纯分析  
**预计 API 开销**: 0（纯分析）

---

### B4. Error taxonomy / failure mode analysis
- [ ] 定义错误类型（10 类）
- [ ] 自动识别低分案例的错误类型
- [ ] 统计各模型的错误频率
- [ ] 生成典型案例
- [ ] 结果存储：`outputs/experiments/B4_error_taxonomy/`

**状态**: ⏳ 待开始  
**优先级**: 中  
**实施方案**: 基于 evaluation.comments 和低分案例，自动归纳错误类型  
**预计 API 开销**: 0（纯分析）

---

### B5. Judge stability / repeatability experiment
- [ ] 对同一批 dialogue 重复评分 3 次
- [ ] 计算 score variance 和 rank consistency
- [ ] 计算 repeated-run ICC
- [ ] 结果存储：`outputs/experiments/B5_judge_stability/`

**状态**: ⏳ 待开始  
**优先级**: 中  
**实施方案**: 复用已有 dialogue，仅重复调用 evaluator（3 次）  
**预计 API 开销**: 中（evaluator 调用 3 次）

---

## C. 可选增强实验（暂不开展）

### C1-C6. 可选实验
- [ ] C1: Cross-scenario generalization
- [ ] C2: Knowledge retrieval quality
- [ ] C3: External benchmark correlation
- [ ] C4: Cost-efficiency analysis
- [ ] C5: More model families
- [ ] C6: Prompt robustness

**状态**: ⏳ 暂不开展  
**优先级**: 低

---

## 实验执行顺序建议

### 第一阶段（零 API 开销，纯分析）
1. ✅ A1: Scenario-level 分层结果分析
2. ✅ A2: Five-dimensional score 分解分析
3. ✅ A6: Benchmark 区分度的统计检验
4. ✅ B2: Patient Agent hallucination 检测
5. ✅ B3: Case difficulty / risk-level stratification
6. ✅ B4: Error taxonomy / failure mode analysis

### 第二阶段（低 API 开销）
7. 🔄 A5: Dynamic multi-turn vs static single-turn 对比（仅需 evaluator）
8. 🔄 B5: Judge stability / repeatability experiment（重复评分 3 次）

### 第三阶段（高 API 开销，需确认）
9. ⏳ A3: Human alignment 详细结果（需确认是否有标注数据）
10. ⏳ A4: Judge evaluator ablation（需重新运行多组评估）

---

## 总体进度

| 类别 | 已完成 | 进行中 | 待开始 | 总计 |
|------|--------|--------|--------|------|
| A. 必须做 | 0 | 0 | 6 | 6 |
| B. 强烈建议 | 0 | 0 | 4 | 4 |
| C. 可选 | 0 | 0 | 6 | 6 |
| **总计** | **0** | **0** | **16** | **16** |

**最后更新**: 2026-05-28
