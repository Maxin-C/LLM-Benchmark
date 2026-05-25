"""
分析ICC评估报告
计算judger与真实医生评价之间的差异
"""

import json
import numpy as np

# 加载评估报告
with open('outputs/icc_evaluation/icc_evaluation_report.json', 'r', encoding='utf-8') as f:
    report = json.load(f)

case_results = report['case_results']

# 维度名称
dimensions = ['accuracy', 'effectiveness', 'safety', 'personalization', 'empathy']

# 收集所有评分
judger_scores_list = []
doctor_scores_list = []

for case in case_results:
    judge_scores = case['judger_scores_array']
    doctor_scores = case['doctor_avg_scores']
    
    judger_scores_list.append(judge_scores)
    doctor_scores_list.append(doctor_scores)

# 转换为numpy数组
judger_arr = np.array(judger_scores_list)
doctor_arr = np.array(doctor_scores_list)

# 计算统计信息
print("="*70)
print("              Judge与真实医生评价差异分析报告")
print("="*70)

print(f"\n样本数量: {len(case_results)} 个案例")
print(f"Judge模型: deepseek-v4-pro")

print("\n" + "-"*70)
print("【各维度评分对比】")
print("-"*70)
print(f"{'维度':<20} {'医生平均':<12} {'Judge平均':<12} {'差异':<10} {'Judge偏高?':<10}")
print("-"*70)

total_diff = []
for i, dim in enumerate(dimensions):
    doc_avg = np.mean(doctor_arr[:, i])
    judge_avg = np.mean(judger_arr[:, i])
    diff = judge_avg - doc_avg
    total_diff.append(diff)
    bias = "偏高 ⬆" if diff > 0 else "偏低 ⬇" if diff < 0 else "一致"
    print(f"{dim:<20} {doc_avg:>10.2f}   {judge_avg:>10.2f}   {diff:>+8.2f}   {bias}")

print("-"*70)

# 计算相关系数
print("\n" + "-"*70)
print("【评分相关性分析】")
print("-"*70)

# 每个维度的相关系数
correlations = []
for i, dim in enumerate(dimensions):
    corr = np.corrcoef(judger_arr[:, i], doctor_arr[:, i])[0, 1]
    correlations.append(corr)
    print(f"{dim:<20} r = {corr:>6.3f}")

overall_corr = np.corrcoef(judger_arr.flatten(), doctor_arr.flatten())[0, 1]
print("-"*70)
print(f"{'总体':<20} r = {overall_corr:>6.3f}")

# 计算ICC（简化版）
print("\n" + "-"*70)
print("【ICC一致性分析】")
print("-"*70)

# 使用简化版ICC计算
def compute_icc(scores1, scores2):
    """
    简化版ICC计算
    scores1: Judge评分 (n_samples, n_dims)
    scores2: 医生评分 (n_samples, n_dims)
    """
    n = len(scores1)
    k = scores1.shape[1]
    
    icc_values = []
    for i in range(k):
        s1 = scores1[:, i]
        s2 = scores2[:, i]
        
        # 计算ICC(2,1)简化公式
        mean_diff_sq = np.mean((s1 - s2) ** 2)
        var_total = np.var(np.concatenate([s1, s2]))
        
        icc = 1 - mean_diff_sq / (2 * var_total) if var_total > 0 else 0
        icc_values.append(max(-1, min(1, icc)))  # 限制在[-1, 1]范围内
    
    return icc_values

icc_values = compute_icc(judger_arr, doctor_arr)

print(f"{'维度':<20} {'ICC值':<12} {'一致性等级':<15}")
print("-"*70)

def get_icc_level(icc):
    if icc >= 0.80:
        return "优秀"
    elif icc >= 0.60:
        return "良好"
    elif icc >= 0.40:
        return "中等"
    else:
        return "差"

for i, dim in enumerate(dimensions):
    level = get_icc_level(icc_values[i])
    print(f"{dim:<20} {icc_values[i]:>8.4f}   {level:<15}")

overall_icc = np.mean(icc_values)
print("-"*70)
print(f"{'总体ICC':<20} {overall_icc:>8.4f}   {get_icc_level(overall_icc):<15}")

# 偏差分析
print("\n" + "-"*70)
print("【偏差分析】")
print("-"*70)

print("Judge评分偏差模式:")
if np.mean(total_diff) > 0.3:
    print("  ⚠️ Judge整体评分偏高（过于宽容）")
elif np.mean(total_diff) < -0.3:
    print("  ⚠️ Judge整体评分偏低（过于严格）")
else:
    print("  ✅ Judge整体评分较为客观")

# 找出偏差最大的维度
max_diff_idx = np.argmax(np.abs(total_diff))
print(f"\n偏差最大的维度: {dimensions[max_diff_idx]} (差异: {total_diff[max_diff_idx]:+.2f})")

# 相关性最差的维度
min_corr_idx = np.argmin(correlations)
print(f"相关性最差的维度: {dimensions[min_corr_idx]} (r = {correlations[min_corr_idx]:.3f})")

print("\n" + "="*70)
print("                        结论与建议")
print("="*70)

if overall_icc >= 0.60:
    print("\n✅ 一致性评估: 良好")
    print("   Judge与真实医生的评估具有较好的一致性，可以作为评估工具使用。")
else:
    print("\n⚠️ 一致性评估: 需改进")
    print("   Judge与真实医生的评估存在一定差异，建议进行优化。")

print("\n主要问题:")
for i, dim in enumerate(dimensions):
    issues = []
    if abs(total_diff[i]) > 0.5:
        if total_diff[i] > 0:
            issues.append(f"{dim}评分偏高")
        else:
            issues.append(f"{dim}评分偏低")
    if correlations[i] < 0.3:
        issues.append(f"{dim}相关性弱")
    
    if issues:
        print(f"  - {', '.join(issues)}")

print("\n优化建议:")
if correlations[0] < 0.3:  # accuracy
    print("  1. 准确性(accuracy): 加强医学知识准确性评估标准")
if correlations[1] < 0.3:  # effectiveness
    print("  2. 有效性(effectiveness): 优化治疗建议有效性评估标准")
if correlations[2] < 0.3:  # safety
    print("  3. 安全性(safety): 加强对医疗风险识别的培训")
if correlations[3] < 0.3:  # personalization
    print("  4. 个性化(personalization): 强化患者个体差异评估标准")
if correlations[4] < 0.3:  # empathy
    print("  5. 共情(empathy): 提升情感关怀评估的准确性")

print("\n" + "="*70)
