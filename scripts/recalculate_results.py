#!/usr/bin/env python3
"""
重新计算评估结果脚本

基于现有的benchmark_results_xxx.json文件重新计算平均分数和通过率
不需要重新运行评估
"""

import json
import os
import glob

def recalculate_model_results(model_name, output_dir):
    """重新计算单个模型的评估结果"""
    result_path = os.path.join(output_dir, f'benchmark_results_{model_name}.json')
    
    if not os.path.exists(result_path):
        print(f"⚠️ 结果文件不存在: {result_path}")
        return None
    
    with open(result_path, 'r', encoding='utf-8') as f:
        benchmark_data = json.load(f)
    
    if not isinstance(benchmark_data, list):
        print(f"⚠️ 结果格式错误（不是列表）: {model_name}")
        return None
    
    # 计算平均分数和通过率
    total_score = 0
    passed_count = 0
    case_results = []
    
    for i, case in enumerate(benchmark_data):
        evaluation = case.get('evaluation', {})
        overall_score = evaluation.get('overall_score', 0)
        is_passed = evaluation.get('is_passed', False)
        
        total_score += overall_score
        if is_passed:
            passed_count += 1
        
        case_results.append({
            "case_id": i + 1,
            "overall_score": overall_score,
            "is_passed": is_passed,
            "duration": case.get('duration', 0)
        })
    
    avg_score = total_score / len(benchmark_data) if benchmark_data else 0
    pass_rate = (passed_count / len(benchmark_data)) * 100 if benchmark_data else 0
    total_duration = sum(case.get('duration', 0) for case in benchmark_data)
    
    return {
        "model_name": model_name,
        "model_type": "unknown",
        "num_cases": len(benchmark_data),
        "average_score": avg_score,
        "pass_rate": pass_rate,
        "duration": total_duration,
        "completed": True,
        "timestamp": "2026-05-25 00:00:00",
        "case_results": case_results
    }

def main():
    output_dir = 'outputs/model_evaluation'
    
    # 找到所有基准测试结果文件
    result_files = glob.glob(os.path.join(output_dir, 'benchmark_results_*.json'))
    
    if not result_files:
        print("❌ 未找到任何基准测试结果文件")
        return
    
    print("🚀 开始重新计算评估结果...")
    print("-" * 60)
    
    results = {}
    completed_models = []
    
    for result_file in result_files:
        # 从文件名中提取模型名称
        model_name = os.path.basename(result_file).replace('benchmark_results_', '').replace('.json', '')
        
        print(f"处理模型: {model_name}")
        
        result = recalculate_model_results(model_name, output_dir)
        if result:
            results[model_name] = result
            completed_models.append(model_name)
            print(f"   ✓ 平均分数: {result['average_score']:.2f}/5")
            print(f"   ✓ 通过率: {result['pass_rate']:.1f}%")
    
    # 更新checkpoint.json
    checkpoint = {
        "completed_models": completed_models,
        "results": results,
        "timestamp": "2026-05-25 00:00:00"
    }
    checkpoint_path = os.path.join(output_dir, 'checkpoint.json')
    with open(checkpoint_path, 'w', encoding='utf-8') as f:
        json.dump(checkpoint, f, ensure_ascii=False, indent=2)
    
    # 更新benchmark_summary.json
    summary_path = os.path.join(output_dir, 'benchmark_summary.json')
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    print("-" * 60)
    print("📊 重新计算完成！")
    print()
    print("模型区分度评估汇总报告")
    print("-" * 60)
    print(f"{'模型名称':<20} {'平均分数':<10} {'通过率':<10}")
    print("-" * 60)
    
    for model_name, result in results.items():
        print(f"{model_name:<20} {result['average_score']:<10.2f} {result['pass_rate']:<10.1f}%")
    
    print("-" * 60)

if __name__ == '__main__':
    main()
