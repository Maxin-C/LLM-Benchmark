#!/usr/bin/env python3
"""运行ICC一致性验证，比较judger与真实医生评估"""
import json
import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src', 'evaluation'))

from icc_validator import ICCValidator, load_human_evaluations

def extract_model_ratings(result_file: str) -> dict:
    """从模型评估结果中提取评分"""
    with open(result_file, 'r', encoding='utf-8') as f:
        results = json.load(f)
    
    ratings = {}
    for result in results:
        case_id = result.get('case_id')
        patient_id = result.get('patient_id')
        
        # 获取严格评估分数或原始评估分数
        if 'strict_evaluation' in result:
            score = result['strict_evaluation'].get('overall_score', 0)
        elif 'evaluation' in result:
            score = result['evaluation'].get('overall_score', 0)
        elif 'overall_score' in result:
            score = result['overall_score']
        else:
            score = 0
        
        # 使用case_id作为主键
        if case_id:
            ratings[str(case_id)] = score
        elif patient_id:
            ratings[patient_id] = score
    
    return ratings

def extract_human_ratings(human_data: list) -> dict:
    """从人工评估数据中提取评分"""
    ratings = {}
    
    for item in human_data:
        case_id = str(item.get('id'))
        
        # 获取人工评估分数（取所有评估的平均值）
        evaluations = item.get('evaluations', [])
        if evaluations:
            scores = []
            for eval_item in evaluations:
                num_scores = eval_item.get('num', [])
                if num_scores:
                    # 计算平均值（排除最后一个"是否认同模型回答存在误导性风险建议"）
                    # 取前4个评分的平均值，第5个是反向题
                    valid_scores = num_scores[:4]
                    if valid_scores:
                        avg_score = sum(valid_scores) / len(valid_scores)
                        scores.append(avg_score)
            
            if scores:
                # 取所有评估者的平均值
                final_score = sum(scores) / len(scores)
                # 转换为5分制（原始评分是1-4分）
                final_score_5point = (final_score / 4) * 5
                ratings[case_id] = final_score_5point
    
    return ratings

def main():
    # 路径配置
    human_eval_file = 'dataset/real_eval/human_evaluations.json'
    model_result_file = 'outputs/model_evaluation_50cases/re_evaluated/qwen3-14b_re_evaluated.json'
    
    print("="*70)
    print("ICC一致性验证 - 比较Judger与真实医生评估")
    print("="*70)
    
    # 加载人工评估数据
    print("\n[1/4] 加载人工评估数据...")
    with open(human_eval_file, 'r', encoding='utf-8') as f:
        human_data = json.load(f)
    human_ratings_dict = extract_human_ratings(human_data)
    print(f"  加载到 {len(human_ratings_dict)} 个人工评估案例")
    
    # 加载模型评估数据
    print("\n[2/4] 加载模型评估数据...")
    model_ratings_dict = extract_model_ratings(model_result_file)
    print(f"  加载到 {len(model_ratings_dict)} 个模型评估案例")
    
    # 匹配案例
    print("\n[3/4] 匹配评估案例...")
    matched_case_ids = set(human_ratings_dict.keys()) & set(model_ratings_dict.keys())
    print(f"  成功匹配 {len(matched_case_ids)} 个案例")
    
    if len(matched_case_ids) < 5:
        print(f"  ⚠️  警告：匹配案例数不足（{len(matched_case_ids)}个），结果可能不可靠")
    
    # 提取匹配的评分
    model_ratings = [model_ratings_dict[case_id] for case_id in matched_case_ids]
    human_ratings = [human_ratings_dict[case_id] for case_id in matched_case_ids]
    
    # 运行ICC验证
    print("\n[4/4] 执行ICC一致性验证...")
    validator = ICCValidator()
    results = validator.validate_consistency(model_ratings, human_ratings)
    
    # 输出结果
    print("\n" + "="*70)
    print("一致性验证结果")
    print("="*70)
    
    print("\n📊 ICC指标（组内相关系数）：")
    print(f"  ICC(1,1) (单评分者随机效应): {results.get('ICC(1,1)', 'N/A')}")
    print(f"    → {validator.interpret_icc(results.get('ICC(1,1)', 0))}")
    print(f"  ICC(2,1) (双评分者随机效应): {results.get('ICC(2,1)', 'N/A')}")
    print(f"    → {validator.interpret_icc(results.get('ICC(2,1)', 0))}")
    print(f"  ICC(3,1) (固定效应模型): {results.get('ICC(3,1)', 'N/A')}")
    print(f"    → {validator.interpret_icc(results.get('ICC(3,1)', 0))}")
    
    print("\n📈 相关分析：")
    print(f"  Pearson相关系数: {results.get('pearson_correlation', 'N/A')}")
    print(f"  Spearman秩相关: {results.get('spearman_correlation', 'N/A')}")
    
    print("\n📉 误差分析：")
    print(f"  平均绝对误差 (MAE): {results.get('mae', 'N/A')}")
    print(f"  均方根误差 (RMSE): {results.get('rmse', 'N/A')}")
    
    print("\n⚖️ Bland-Altman分析：")
    print(f"  平均差值: {results.get('mean_difference', 'N/A')}")
    print(f"  差值标准差: {results.get('std_difference', 'N/A')}")
    print(f"  95%一致性界限: [{results.get('loa_lower', 'N/A')}, {results.get('loa_upper', 'N/A')}]")
    
    print("\n📋 描述统计：")
    print(f"  案例数 (n): {results.get('n', 'N/A')}")
    print(f"  Judger平均分: {results.get('model_mean', 'N/A')}")
    print(f"  医生平均分: {results.get('human_mean', 'N/A')}")
    print(f"  Judger标准差: {results.get('model_std', 'N/A')}")
    print(f"  医生标准差: {results.get('human_std', 'N/A')}")
    
    print("\n✅ 一致性判定：")
    is_consistent = results.get('is_consistent', False)
    print(f"  综合判定: {'通过' if is_consistent else '未通过'}")
    
    print("\n" + "="*70)
    
    # 输出详细数据点
    if len(matched_case_ids) > 0:
        print("\n📝 匹配案例详情：")
        print(f"{'案例ID':<10} {'Judger评分':<12} {'医生评分':<12} {'差值':<10}")
        print("-" * 50)
        for case_id in sorted(matched_case_ids)[:10]:  # 只显示前10个
            m_score = model_ratings_dict[case_id]
            h_score = human_ratings_dict[case_id]
            diff = m_score - h_score
            print(f"{case_id:<10} {m_score:<12.2f} {h_score:<12.2f} {diff:<10.2f}")
        
        if len(matched_case_ids) > 10:
            print(f"  ... 还有 {len(matched_case_ids) - 10} 个案例未显示")
    
    print("\n" + "="*70)
    
    # 保存结果（处理numpy类型）
    output_file = 'outputs/model_evaluation_50cases/icc_validation_results.json'
    results_serializable = {}
    for k, v in results.items():
        if hasattr(v, 'item'):
            results_serializable[k] = v.item()
        else:
            results_serializable[k] = v
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results_serializable, f, ensure_ascii=False, indent=2)
    print(f"\n📁 验证结果已保存到: {output_file}")

if __name__ == '__main__':
    main()