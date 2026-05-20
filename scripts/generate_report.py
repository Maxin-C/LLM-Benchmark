#!/usr/bin/env python3
"""
报告生成器
"""

import argparse
import json
import os
from datetime import datetime

def main():
    parser = argparse.ArgumentParser(description='EASE Report Generator')
    parser.add_argument('--input', type=str, required=True,
                        help='Path to evaluation results directory')
    parser.add_argument('--output', type=str, default='outputs/reports',
                        help='Output directory for reports')
    args = parser.parse_args()
    
    # 创建输出目录
    os.makedirs(args.output, exist_ok=True)
    
    # 加载评估结果
    results = []
    for filename in os.listdir(args.input):
        if filename.endswith('.json'):
            filepath = os.path.join(args.input, filename)
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
                if isinstance(data, list):
                    results.extend(data)
                else:
                    results.append(data)
    
    if not results:
        print("未找到评估结果")
        return
    
    # 生成报告
    report = generate_evaluation_report(results)
    
    # 保存报告
    report_file = os.path.join(args.output, f'report_{datetime.now().strftime("%Y%m%d_%H%M%S")}.md')
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write(report)
    
    print(f"报告已保存到: {report_file}")

def generate_evaluation_report(results: list) -> str:
    """
    生成评估报告
    
    参数：
        results: 评估结果列表
    
    返回：
        Markdown格式的报告
    """
    report = f"""# EASE 评估报告

生成时间: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
评估总数: {len(results)}

---

## 一、总体统计

"""
    
    # 计算统计数据
    total_score = sum(r.get('overall_score', 0) for r in results)
    avg_score = total_score / len(results) if results else 0
    passed_count = sum(1 for r in results if r.get('is_passed', False))
    passed_rate = passed_count / len(results) * 100 if results else 0
    
    report += f"""| 指标 | 值 |
|------|-----|
| 评估总数 | {len(results)} |
| 平均综合评分 | {avg_score:.2f} |
| 通过数量 | {passed_count} |
| 通过率 | {passed_rate:.1f}% |

---

## 二、各维度评分分布

"""
    
    # 计算各维度统计
    dimensions = ['accuracy', 'effectiveness', 'safety', 'personalization', 'empathy']
    for dim in dimensions:
        scores = [r.get('scores', {}).get(dim, 0) for r in results]
        if scores:
            avg_dim_score = sum(scores) / len(scores)
            report += f"- **{dim}**: {avg_dim_score:.2f}\n"
    
    report += "\n---\n\n## 三、风险分析\n\n"
    
    # 风险分析
    critical_issues = []
    warnings = []
    
    for r in results:
        risk_report = r.get('risk_report', {})
        critical_issues.extend(risk_report.get('critical_issues', []))
        warnings.extend(risk_report.get('warnings', []))
    
    report += f"""### 3.1 严重问题 ({len(critical_issues)}个)

"""
    if critical_issues:
        for issue in critical_issues[:10]:  # 最多显示10个
            report += f"- {issue}\n"
        if len(critical_issues) > 10:
            report += f"- ... 还有 {len(critical_issues) - 10} 个\n"
    else:
        report += "无\n"
    
    report += f"""

### 3.2 警告 ({len(warnings)}个)

"""
    if warnings:
        warning_counts = {}
        for w in warnings:
            warning_counts[w] = warning_counts.get(w, 0) + 1
        
        for w, count in sorted(warning_counts.items(), key=lambda x: x[1], reverse=True)[:10]:
            report += f"- {w} ({count}次)\n"
    else:
        report += "无\n"
    
    report += "\n---\n\n## 四、场景分析\n\n"
    
    # 场景分析
    scenario_results = {}
    for r in results:
        scenario_name = r.get('scenario_name', '未知')
        if scenario_name not in scenario_results:
            scenario_results[scenario_name] = []
        scenario_results[scenario_name].append(r)
    
    for scenario_name, scenario_results_list in scenario_results.items():
        avg_score = sum(r.get('overall_score', 0) for r in scenario_results_list) / len(scenario_results_list)
        passed_count = sum(1 for r in scenario_results_list if r.get('is_passed', False))
        passed_rate = passed_count / len(scenario_results_list) * 100
        
        report += f"""### {scenario_name}

| 指标 | 值 |
|------|-----|
| 评估数 | {len(scenario_results_list)} |
| 平均评分 | {avg_score:.2f} |
| 通过率 | {passed_rate:.1f}% |

"""
    
    report += "---\n\n## 五、结论\n\n"
    
    if avg_score >= 4:
        conclusion = "评估结果优秀，模型表现良好。"
    elif avg_score >= 3:
        conclusion = "评估结果良好，模型基本满足要求。"
    elif avg_score >= 2:
        conclusion = "评估结果一般，建议进一步优化模型。"
    else:
        conclusion = "评估结果较差，需要大幅改进。"
    
    report += f"""{conclusion}

---

*报告结束*
"""
    
    return report

if __name__ == '__main__':
    main()
