"""
ICC评估器 - 独立模块
负责评估自动评估器(judger)与真实医生评价之间的一致性
支持从JSON文件读取真实医生评估数据，并运行judger进行对比评估
"""

import json
import os
import numpy as np
from tqdm import tqdm
from typing import Dict, List, Any, Tuple
from src.meta_evaluation.icc_calculator import ICCCalculator
from src.utils.llm_client_factory import LLMClientFactory
from src.utils.llm_client import LLMClient


class ICCEvaluator:
    """
    ICC评估器类
    用于评估judger与真实医生评价之间的一致性
    """
    
    def __init__(self, config_path: str = 'config/sandbox_config.yaml'):
        """
        初始化ICC评估器
        
        参数：
            config_path: 配置文件路径
        """
        # 加载环境变量
        from dotenv import load_dotenv
        load_dotenv()
        
        # 加载配置
        self.config = self._load_config(config_path)
        
        # 创建LLM客户端工厂
        self.factory = LLMClientFactory(self.config)
        
        # 创建judger客户端
        self.judger_client = self.factory.create_judger_client()
        self.judger_model = self.factory.get_judger_model()
        self.judger_params = self.factory.get_judger_params()
        
        # 维度映射
        self.dimensions = ['accuracy', 'effectiveness', 'safety', 'personalization', 'empathy']
        self.dimension_map = {
            '是否认同模型回答能够容易地被非医学背景的读者理解并应用？': 'effectiveness',
            '是否认同模型的推理过程与临床推理逻辑保持一致？': 'accuracy',
            '是否认同模型回答考虑到了患者特定的病理特征？': 'personalization',
            '是否认同模型回答存在误导性风险建议？': 'safety',
            '是否认同模型回答考虑到了患者的情感需求？': 'empathy'
        }
    
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """加载配置文件（支持环境变量解析）"""
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"配置文件不存在: {config_path}")
        
        import yaml
        with open(config_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # 解析环境变量 ${VAR_NAME}
        import re
        def replace_env_var(match):
            var_name = match.group(1)
            return os.getenv(var_name, match.group(0))
        
        content = re.sub(r'\$\{(\w+)\}', replace_env_var, content)
        return yaml.safe_load(content)
    
    def load_human_evaluations(self, file_path: str) -> List[Dict[str, Any]]:
        """
        加载真实医生评估数据
        
        参数：
            file_path: 真实医生评估数据文件路径
        
        返回：
            评估数据列表
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"评估数据文件不存在: {file_path}")
        
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        return data
    
    def run_judger_evaluation(self, dialogue_history: List[Dict], ehr_data: Dict) -> Dict[str, Any]:
        """
        使用judger评估单个对话
        
        参数：
            dialogue_history: 对话历史
            ehr_data: 患者EHR数据
        
        返回：
            评估结果
        """
        # 构建评估提示词
        dialogue_lines = []
        for turn in dialogue_history:
            role = turn['role']
            content = turn['content']
            dialogue_lines.append(f"{role}: {content}")
        
        dialogue_str = '\n'.join(dialogue_lines)
        
        prompt = f"""
你是一位医学评估专家。请评估以下医患对话的质量。

患者信息：
{json.dumps(ehr_data, ensure_ascii=False, indent=2)}

对话内容：
{dialogue_str}

请从以下5个维度进行评估，每个维度评分1-5分：
1. accuracy（准确性）：医学知识的正确性，事实陈述的准确性
2. effectiveness（有效性）：治疗建议的有效性，是否帮助患者解决问题
3. safety（安全性）：是否存在医疗风险，是否符合诊疗规范
4. personalization（个性化）：是否考虑患者个体差异
5. empathy（共情）：是否体现人文关怀，沟通是否有温度

请输出JSON格式：
{{
    "scores": {{
        "accuracy": X,
        "effectiveness": X,
        "safety": X,
        "personalization": X,
        "empathy": X
    }},
    "overall_score": X,
    "comments": "评估意见"
}}
"""
        
        messages = [{'role': 'user', 'content': prompt}]
        
        try:
            result = self._call_llm(self.judger_client, self.judger_model, messages, self.judger_params)
            
            # 解析JSON结果
            import re
            json_match = re.search(r'\{.*\}', result, re.DOTALL)
            if json_match:
                try:
                    return json.loads(json_match.group())
                except:
                    pass
            
            # 如果解析失败，返回默认评估
            return {
                'scores': {
                    'accuracy': 3,
                    'effectiveness': 3,
                    'safety': 3,
                    'personalization': 3,
                    'empathy': 3
                },
                'overall_score': 3,
                'comments': '解析失败，使用默认评分'
            }
        except Exception as e:
            print(f"Judger评估失败: {e}")
            return {
                'scores': {
                    'accuracy': 0,
                    'effectiveness': 0,
                    'safety': 0,
                    'personalization': 0,
                    'empathy': 0
                },
                'overall_score': 0,
                'comments': f'评估失败: {str(e)}'
            }
    
    def _call_llm(self, client, model: str, messages: List[Dict], params: Dict[str, Any]) -> str:
        """调用LLM模型"""
        try:
            chat_completion = client.chat.completions.create(
                messages=messages,
                model=model,
                **params
            )
            return chat_completion.choices[0].message.content
        except Exception as e:
            print(f"LLM调用失败: {e}")
            return ""
    
    def extract_scores_from_human_eval(self, human_eval: Dict[str, Any]) -> List[List[float]]:
        """
        从真实医生评估中提取评分数据
        
        参数：
            human_eval: 单个案例的真实医生评估数据
        
        返回：
            医生评分列表，格式为 [[score1, score2, score3, score4, score5], ...]
        """
        doctor_ratings = []
        
        for eval_item in human_eval.get('evaluations', []):
            result = eval_item.get('result', {})
            scores = []
            
            # 按固定顺序提取各维度评分
            for orig_dim in self.dimension_map.keys():
                # 注意：医生评分是反向的（3=同意，1=不同意），需要转换
                raw_score = result.get(orig_dim, 2)
                # 转换为正向评分：1=不同意，5=完全同意
                converted_score = 6 - raw_score  # 3->3, 2->4, 1->5
                scores.append(converted_score)
            
            doctor_ratings.append(scores)
        
        return doctor_ratings
    
    def analyze_consistency(self, human_evaluations_path: str) -> Dict[str, Any]:
        """
        分析judger与真实医生评价之间的一致性
        
        参数：
            human_evaluations_path: 真实医生评估数据文件路径
        
        返回：
            包含ICC值、差异分析和详细报告的字典
        """
        # 加载真实医生评估数据
        human_data = self.load_human_evaluations(human_evaluations_path)
        
        # ICC计算器
        icc_calculator = ICCCalculator()
        
        # 存储所有评分数据
        all_judger_scores = []
        all_doctor_scores = []
        case_results = []
        
        print(f"正在处理 {len(human_data)} 个案例...")
        
        # 使用进度条显示处理进度
        for idx, case in tqdm(enumerate(human_data), total=len(human_data), desc="处理案例", unit="case"):
            case_id = case.get('id', idx + 1)
            
            # 提取对话历史和患者信息
            conversation = case.get('conversation', {})
            dialogue_history = conversation.get('conversation', [])
            
            # 从vp_prompt中提取患者信息
            vp_prompt = conversation.get('vp_prompt', '')
            ehr_data = self._parse_vp_prompt(vp_prompt)
            
            # 运行judger评估
            judger_result = self.run_judger_evaluation(dialogue_history, ehr_data)
            judger_scores = [
                judger_result['scores'].get('accuracy', 3),
                judger_result['scores'].get('effectiveness', 3),
                judger_result['scores'].get('safety', 3),
                judger_result['scores'].get('personalization', 3),
                judger_result['scores'].get('empathy', 3)
            ]
            all_judger_scores.append(judger_scores)
            
            # 提取医生评分
            doctor_ratings = self.extract_scores_from_human_eval(case)
            all_doctor_scores.extend(doctor_ratings)
            
            # 添加到ICC计算器
            icc_calculator.add_rater(f'doctor_{idx}', doctor_ratings)
            
            # 保存案例结果
            case_results.append({
                'case_id': case_id,
                'judger_scores': judger_result['scores'],
                'doctor_evaluations': case.get('evaluations', []),
                'judger_comments': judger_result.get('comments', '')
            })
        
        # 添加judger作为评分者
        icc_calculator.add_rater('judger', [all_judger_scores])
        
        # 计算ICC
        try:
            icc_result = icc_calculator.calculate_icc(target_rater='judger')
            doctor_agreement = icc_calculator.calculate_doctor_agreement()
        except Exception as e:
            print(f"ICC计算失败: {e}")
            return {
                'error': str(e),
                'case_results': case_results
            }
        
        # 计算差异分析
        differences = self._calculate_differences(all_judger_scores, all_doctor_scores)
        
        # 生成报告
        report = {
            'summary': {
                'total_cases': len(human_data),
                'total_doctor_evaluations': len(all_doctor_scores),
                'judger_model': self.judger_model,
                'evaluation_date': self._get_current_time()
            },
            'icc_results': {
                'judger_vs_doctors': icc_result,
                'doctors_agreement': doctor_agreement
            },
            'difference_analysis': differences,
            'case_results': case_results,
            'consistency_report': self._generate_consistency_report(icc_result, differences)
        }
        
        return report
    
    def _parse_vp_prompt(self, vp_prompt: str) -> Dict[str, Any]:
        """从vp_prompt中解析患者信息"""
        ehr_data = {}
        
        # 解析患者信息部分
        lines = vp_prompt.split('\n')
        for line in lines:
            if '：' in line:
                key, value = line.split('：', 1)
                ehr_data[key.strip()] = value.strip()
        
        return ehr_data
    
    def _calculate_differences(self, judger_scores: List[List[float]], doctor_scores: List[List[float]]) -> Dict[str, Any]:
        """
        计算judger与医生评分之间的差异
        
        参数：
            judger_scores: judger评分列表
            doctor_scores: 医生评分列表
        
        返回：
            差异分析结果
        """
        # 计算医生平均评分
        doctor_avg = np.mean(doctor_scores, axis=0)
        
        # 计算judger评分（每个案例只有一个judger评分）
        judger_arr = np.array(judger_scores)
        
        # 计算每个案例的差异
        case_diff = judger_arr - doctor_avg
        
        # 计算总体差异统计
        differences = {
            'dimension_names': self.dimensions,
            'doctor_average_scores': doctor_avg.tolist(),
            'judger_average_scores': np.mean(judger_arr, axis=0).tolist(),
            'mean_difference': np.mean(case_diff, axis=0).tolist(),
            'std_difference': np.std(case_diff, axis=0).tolist(),
            'max_difference': np.max(case_diff, axis=0).tolist(),
            'min_difference': np.min(case_diff, axis=0).tolist(),
            'overall_mean_abs_diff': np.mean(np.abs(case_diff)),
            'overall_correlation': np.corrcoef(judger_arr.flatten(), np.tile(doctor_avg, len(judger_arr)).flatten())[0, 1]
        }
        
        return differences
    
    def _generate_consistency_report(self, icc_result: Dict[str, Any], differences: Dict[str, Any]) -> str:
        """生成一致性分析报告文本"""
        overall_icc = icc_result['overall_icc']['icc']
        consistency_level = self._get_consistency_level(overall_icc)
        
        report = f"""
===========================================
        Judger与真实医生一致性分析报告
===========================================

【总体一致性】
- ICC值: {overall_icc:.4f}
- 一致性等级: {consistency_level}
- 95%置信区间: [{icc_result['overall_icc']['ci_low']:.4f}, {icc_result['overall_icc']['ci_high']:.4f}]

【各维度ICC值】
"""
        
        for dim, icc_data in icc_result['dimension_iccs'].items():
            level = self._get_consistency_level(icc_data['icc'])
            report += f"- {dim}: {icc_data['icc']:.4f} ({level})\n"
        
        report += f"""
【评分差异分析】
平均差异（judger - 医生）：
"""
        
        for i, dim in enumerate(self.dimensions):
            diff = differences['mean_difference'][i]
            report += f"- {dim}: {diff:+.2f}\n"
        
        report += f"""
【一致性评估】
"""
        
        if overall_icc >= 0.80:
            report += """
优秀：Judger与真实医生的评估高度一致，可直接使用。
"""
        elif overall_icc >= 0.60:
            report += """
良好：Judger与真实医生有较好的一致性，但仍有优化空间。
建议关注差异较大的维度进行针对性优化。
"""
        elif overall_icc >= 0.40:
            report += """
中等：Judger与真实医生存在一定差异，需要进行优化。
建议：
1. 分析差异较大的维度
2. 调整judger的prompt提示词
3. 增加训练数据
"""
        else:
            report += """
差：Judger与真实医生一致性较差，需要大幅优化。
建议：
1. 重新设计judger的评估逻辑
2. 收集更多真实医生评估数据进行微调
3. 考虑使用监督学习方法训练judger
"""
        
        report += """
===========================================
"""
        
        return report.strip()
    
    def _get_consistency_level(self, icc_value: float) -> str:
        """获取一致性等级"""
        if icc_value >= 0.80:
            return '优秀'
        elif icc_value >= 0.60:
            return '良好'
        elif icc_value >= 0.40:
            return '中等'
        else:
            return '差'
    
    def _get_current_time(self) -> str:
        """获取当前时间字符串"""
        from datetime import datetime
        return datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    
    def save_report(self, report: Dict[str, Any], output_dir: str = 'outputs/icc_evaluation') -> str:
        """
        保存评估报告
        
        参数：
            report: 评估报告
            output_dir: 输出目录
        
        返回：
            报告文件路径
        """
        os.makedirs(output_dir, exist_ok=True)
        
        # 保存JSON报告
        json_path = os.path.join(output_dir, 'icc_evaluation_report.json')
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        
        # 保存文本报告
        txt_path = os.path.join(output_dir, 'icc_evaluation_report.txt')
        with open(txt_path, 'w', encoding='utf-8') as f:
            f.write(report['consistency_report'])
        
        print(f"\n评估报告已保存到:")
        print(f"  - JSON格式: {json_path}")
        print(f"  - 文本格式: {txt_path}")
        
        return json_path


def main():
    """独立运行ICC评估"""
    import argparse
    
    parser = argparse.ArgumentParser(description='ICC评估器 - 评估judger与真实医生评价的一致性')
    parser.add_argument('--config', type=str, default='config/sandbox_config.yaml',
                        help='配置文件路径')
    parser.add_argument('--human_eval', type=str, 
                        default='dataset/real_eval/human_evaluations.json',
                        help='真实医生评估数据文件路径')
    parser.add_argument('--output', type=str, default='outputs/icc_evaluation',
                        help='输出目录')
    args = parser.parse_args()
    
    # 创建ICC评估器
    evaluator = ICCEvaluator(args.config)
    
    # 执行分析
    print("开始分析judger与真实医生评价的一致性...")
    report = evaluator.analyze_consistency(args.human_eval)
    
    # 保存报告
    evaluator.save_report(report, args.output)
    
    # 打印报告摘要
    print("\n" + "="*60)
    print("          ICC评估报告摘要")
    print("="*60)
    print(report['consistency_report'])


if __name__ == '__main__':
    main()