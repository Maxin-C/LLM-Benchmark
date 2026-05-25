"""
ICC评估器 - 独立模块
负责评估自动评估器(judger)与真实医生评价之间的一致性
支持从JSON文件读取真实医生评估数据，并运行judger进行对比评估
支持缓存机制，避免重复调用API
"""

import json
import os
import hashlib
import numpy as np
from tqdm import tqdm
from typing import Dict, List, Any, Tuple, Optional
from src.meta_evaluation.icc_calculator import ICCCalculator
from src.utils.llm_client_factory import LLMClientFactory
from src.utils.llm_client import LLMClient


class CacheManager:
    """
    缓存管理器
    用于缓存judger的评估结果，避免重复调用API
    """
    
    def __init__(self, cache_dir: str = 'outputs/judger_cache'):
        """
        初始化缓存管理器
        
        参数：
            cache_dir: 缓存目录
        """
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)
    
    def _generate_cache_key(self, judger_model: str, judger_params: Dict, human_eval_path: str, 
                           sample_indices: List[int]) -> str:
        """
        生成缓存文件的关键字
        
        参数：
            judger_model: judger模型名称
            judger_params: judger参数
            human_eval_path: 真实医生评估数据文件路径
            sample_indices: 抽样索引列表
        
        返回：
            缓存关键字（MD5哈希）
        """
        # 组合所有关键信息
        key_parts = [
            judger_model,
            str(sorted(judger_params.items())),
            human_eval_path,
            str(sorted(sample_indices))
        ]
        key_string = '|'.join(key_parts)
        
        # 生成MD5哈希
        return hashlib.md5(key_string.encode()).hexdigest()
    
    def get_cache_path(self, cache_key: str) -> str:
        """获取缓存文件路径"""
        return os.path.join(self.cache_dir, f'judger_cache_{cache_key}.json')
    
    def load_cache(self, judger_model: str, judger_params: Dict, human_eval_path: str,
                   sample_indices: List[int]) -> Optional[Dict[str, Any]]:
        """
        加载缓存
        
        参数：
            judger_model: judger模型名称
            judger_params: judger参数
            human_eval_path: 真实医生评估数据文件路径
            sample_indices: 抽样索引列表
        
        返回：
            缓存数据，如果缓存不存在或无效则返回None
        """
        cache_key = self._generate_cache_key(judger_model, judger_params, human_eval_path, sample_indices)
        cache_path = self.get_cache_path(cache_key)
        
        if os.path.exists(cache_path):
            try:
                with open(cache_path, 'r', encoding='utf-8') as f:
                    cache_data = json.load(f)
                print(f"✅ 找到缓存文件: {cache_path}")
                print(f"   缓存包含 {len(cache_data.get('case_results', []))} 个案例的评估结果")
                return cache_data
            except Exception as e:
                print(f"⚠️ 缓存加载失败: {e}")
                return None
        else:
            print(f"ℹ️ 未找到缓存文件，将进行API评估...")
            return None
    
    def save_cache(self, cache_data: Dict[str, Any], judger_model: str, judger_params: Dict,
                  human_eval_path: str, sample_indices: List[int]) -> str:
        """
        保存缓存
        
        参数：
            cache_data: 缓存数据
            judger_model: judger模型名称
            judger_params: judger参数
            human_eval_path: 真实医生评估数据文件路径
            sample_indices: 抽样索引列表
        
        返回：
            缓存文件路径
        """
        cache_key = self._generate_cache_key(judger_model, judger_params, human_eval_path, sample_indices)
        cache_path = self.get_cache_path(cache_key)
        
        # 添加缓存元信息
        cache_data['_cache_info'] = {
            'judger_model': judger_model,
            'judger_params': judger_params,
            'human_eval_path': human_eval_path,
            'sample_indices': sample_indices
        }
        
        with open(cache_path, 'w', encoding='utf-8') as f:
            json.dump(cache_data, f, ensure_ascii=False, indent=2)
        
        print(f"💾 缓存已保存: {cache_path}")
        return cache_path


class ICCEvaluator:
    """
    ICC评估器类
    用于评估judger与真实医生评价之间的一致性
    """
    
    def __init__(self, config_path: str = 'config/sandbox_config.yaml', cache_dir: str = 'outputs/judger_cache'):
        """
        初始化ICC评估器
        
        参数：
            config_path: 配置文件路径
            cache_dir: 缓存目录
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
        
        # 创建缓存管理器
        self.cache_manager = CacheManager(cache_dir)
        
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
        import re
        
        with open(config_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        def replace_env_var(match):
            var_name = match.group(1)
            return os.getenv(var_name, match.group(0))
        
        content = re.sub(r'\$\{(\w+)\}', replace_env_var, content)
        return yaml.safe_load(content)
    
    def load_human_evaluations(self, file_path: str) -> List[Dict[str, Any]]:
        """加载真实医生评估数据"""
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"评估数据文件不存在: {file_path}")
        
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        return data
    
    def run_judger_evaluation(self, dialogue_history: List[Dict], ehr_data: Dict) -> Dict[str, Any]:
        """使用judger评估单个对话"""
        # 构建评估提示词
        dialogue_lines = []
        for turn in dialogue_history:
            role = turn['role']
            content = turn['content']
            dialogue_lines.append(f"{role}: {content}")
        
        dialogue_str = '\n'.join(dialogue_lines)
        
        prompt = f"""
作为医学评估专家，评估以下医患对话，输出JSON格式评分。

评分标准（1-5分，严格按标准评分）：
- accuracy: 医学知识正确性（5=完全准确，1=严重错误）
- effectiveness: 建议有效性（5=完全解决患者核心问题，1=未解决或误导）
- safety: 医疗安全性（5=完全安全，1=存在风险）
- personalization: 个性化程度（5=充分考虑患者年龄/性别/病史/检查结果，1=建议通用化）
- empathy: 情感关怀（5=适度关怀+专业态度，1=过度情绪化或冷漠）

注意：
1. empathy评分时，4分=有同理心且保持专业，5分需同时满足：非常有同理心+专业建议+情感支持到位。避免给5分除非对话确实出色。
2. personalization评分时，只有明确提及患者具体信息（如检查结果、用药史、症状特点）才能给高分，泛泛建议给低分。
3. effectiveness评分时，必须患者明确提出的核心问题得到回答才能给高分。

患者信息：{json.dumps(ehr_data, ensure_ascii=False)}

对话：{dialogue_str}

仅输出JSON：
{{"scores":{{"accuracy":X,"effectiveness":X,"safety":X,"personalization":X,"empathy":X}}}}
"""
        
        messages = [{'role': 'user', 'content': prompt}]
        
        try:
            result = self._call_llm(self.judger_client, self.judger_model, messages, self.judger_params)
            
            # 解析JSON结果
            import re
            json_match = re.search(r'\{.*\}', result, re.DOTALL)
            if json_match:
                try:
                    parsed = json.loads(json_match.group())
                    # 确保返回格式正确
                    scores = parsed.get('scores', {})
                    overall_score = round(sum([scores.get(k, 3) for k in ['accuracy', 'effectiveness', 'safety', 'personalization', 'empathy']]) / 5)
                    return {
                        'scores': {
                            'accuracy': scores.get('accuracy', 3),
                            'effectiveness': scores.get('effectiveness', 3),
                            'safety': scores.get('safety', 3),
                            'personalization': scores.get('personalization', 3),
                            'empathy': scores.get('empathy', 3)
                        },
                        'overall_score': overall_score,
                        'comments': ''
                    }
                except Exception as e:
                    print(f"JSON解析失败: {e}")
            
            return {
                'scores': {
                    'accuracy': 3, 'effectiveness': 3, 'safety': 3,
                    'personalization': 3, 'empathy': 3
                },
                'overall_score': 3,
                'comments': '解析失败，使用默认评分'
            }
        except Exception as e:
            print(f"Judger评估失败: {e}")
            return {
                'scores': {
                    'accuracy': 0, 'effectiveness': 0, 'safety': 0,
                    'personalization': 0, 'empathy': 0
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
        """从真实医生评估中提取评分数据"""
        doctor_ratings = []
        
        for eval_item in human_eval.get('evaluations', []):
            # 直接使用已计算好的 num 字段
            num_scores = eval_item.get('num', [])
            if num_scores and len(num_scores) == 5:
                doctor_ratings.append(num_scores)
        
        return doctor_ratings
    
    def analyze_consistency(self, human_evaluations_path: str, sample_size: int = None, 
                          random_seed: int = 42, use_cache: bool = True) -> Dict[str, Any]:
        """
        分析judger与真实医生评价之间的一致性
        
        参数：
            human_evaluations_path: 真实医生评估数据文件路径
            sample_size: 抽样数量（None表示使用全部数据）
            random_seed: 随机种子（用于可重复抽样）
            use_cache: 是否使用缓存（默认True）
        
        返回：
            包含ICC值、差异分析和详细报告的字典
        """
        # 加载真实医生评估数据
        human_data = self.load_human_evaluations(human_evaluations_path)
        total_data_size = len(human_data)
        
        # 生成样本索引
        sample_indices = list(range(len(human_data)))
        if sample_size is not None and sample_size < len(human_data):
            import random
            random.seed(random_seed)
            sample_indices = sorted(random.sample(range(len(human_data)), sample_size))
            human_data = [human_data[i] for i in sample_indices]
            print(f"随机抽取了 {sample_size} 个样本（总样本数: {total_data_size}）")
        
        # 尝试加载缓存
        if use_cache:
            cached_result = self.cache_manager.load_cache(
                self.judger_model, 
                {k: v for k, v in self.judger_params.items() if k != 'extra_body'},
                human_evaluations_path,
                sample_indices
            )
            
            if cached_result and len(cached_result.get('case_results', [])) == len(human_data):
                print("🎯 使用缓存结果进行分析...")
                
                # 从缓存中恢复数据
                case_results = cached_result['case_results']
                all_judger_scores = [c['judger_scores_array'] for c in case_results]
                
                # 准备ICC计算
                icc_calculator = ICCCalculator()
                
                # 添加Judge作为评分者
                icc_calculator.add_rater('judger', all_judger_scores)
                
                # 从缓存中提取医生原始评分并添加每位医生作为独立评分者
                all_doctor_ratings = [c.get('doctor_raw_ratings', []) for c in case_results]
                num_doctors = max(len(ratings) for ratings in all_doctor_ratings) if all_doctor_ratings else 0
                print(f"检测到 {num_doctors} 位医生的评分数据")
                
                for doctor_idx in range(num_doctors):
                    doctor_scores = []
                    for case_ratings in all_doctor_ratings:
                        if doctor_idx < len(case_ratings):
                            doctor_scores.append(case_ratings[doctor_idx])
                        else:
                            doctor_scores.append([3, 3, 3, 3, 3])
                    icc_calculator.add_rater(f'doctor_{doctor_idx + 1}', doctor_scores)
                
                try:
                    icc_result = icc_calculator.calculate_icc(target_rater='judger')
                    doctor_agreement = icc_calculator.calculate_doctor_agreement()
                except Exception as e:
                    print(f"ICC计算失败: {e}")
                    return cached_result
                
                # 计算差异时使用医生平均评分
                all_doctor_scores = [c['doctor_avg_scores'] for c in case_results]
                differences = self._calculate_differences(all_judger_scores, all_doctor_scores)
                
                report = {
                    'summary': {
                        'total_cases': len(human_data),
                        'total_doctor_evaluations': sum(len(c.get('doctor_evaluations', [])) for c in cached_result['case_results']),
                        'judger_model': self.judger_model,
                        'evaluation_date': self._get_current_time(),
                        'from_cache': True
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
        
        # 无缓存，执行API评估
        print(f"正在处理 {len(human_data)} 个案例...")
        print(f"使用judger模型: {self.judger_model}")
        
        icc_calculator = ICCCalculator()
        all_judger_scores = []
        all_doctor_ratings = []  # 存储所有医生的评分（每位医生单独存储）
        case_results = []
        
        for idx, case in tqdm(enumerate(human_data), total=len(human_data), desc="处理案例", unit="case"):
            case_id = case.get('id', idx + 1)
            
            conversation = case.get('conversation', {})
            dialogue_history = conversation.get('conversation', [])
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
            
            # 提取医生评分（保留每位医生的独立评分）
            doctor_ratings = self.extract_scores_from_human_eval(case)
            if doctor_ratings:
                avg_doctor_scores = np.mean(doctor_ratings, axis=0).tolist()
            else:
                avg_doctor_scores = [3, 3, 3, 3, 3]
            
            all_doctor_ratings.append(doctor_ratings)  # 保留每位医生的评分
            
            case_results.append({
                'case_id': case_id,
                'judger_scores': judger_result['scores'],
                'judger_scores_array': judger_scores,  # 用于缓存
                'doctor_evaluations': case.get('evaluations', []),
                'doctor_avg_scores': avg_doctor_scores,
                'doctor_raw_ratings': doctor_ratings,  # 保存原始医生评分用于ICC计算
                'judger_comments': judger_result.get('comments', '')
            })
        
        # 准备ICC计算 - 需要按医生维度组织数据
        num_doctors = max(len(ratings) for ratings in all_doctor_ratings) if all_doctor_ratings else 0
        print(f"检测到 {num_doctors} 位医生的评分数据")
        
        # 添加Judge作为评分者
        icc_calculator.add_rater('judger', all_judger_scores)
        
        # 添加每位医生作为独立评分者
        for doctor_idx in range(num_doctors):
            doctor_scores = []
            for case_ratings in all_doctor_ratings:
                if doctor_idx < len(case_ratings):
                    doctor_scores.append(case_ratings[doctor_idx])
                else:
                    # 如果某个案例缺少该医生的评分，使用平均值填充
                    doctor_scores.append([3, 3, 3, 3, 3])
            icc_calculator.add_rater(f'doctor_{doctor_idx + 1}', doctor_scores)
        
        # 保存缓存
        if use_cache:
            cache_data = {'case_results': case_results}
            self.cache_manager.save_cache(
                cache_data,
                self.judger_model,
                {k: v for k, v in self.judger_params.items() if k != 'extra_body'},
                human_evaluations_path,
                sample_indices
            )
        
        # 计算差异时使用医生平均评分
        all_doctor_scores = [c['doctor_avg_scores'] for c in case_results]
        
        # 计算ICC
        icc_calculator.add_rater('judger', all_judger_scores)
        icc_calculator.add_rater('doctor_avg', all_doctor_scores)
        
        try:
            icc_result = icc_calculator.calculate_icc(target_rater='judger')
            doctor_agreement = icc_calculator.calculate_doctor_agreement()
        except Exception as e:
            print(f"ICC计算失败: {e}")
            return {'error': str(e), 'case_results': case_results}
        
        differences = self._calculate_differences(all_judger_scores, all_doctor_scores)
        
        report = {
            'summary': {
                'total_cases': len(human_data),
                'total_doctor_evaluations': sum(len(c.get('doctor_evaluations', [])) for c in case_results),
                'judger_model': self.judger_model,
                'evaluation_date': self._get_current_time(),
                'from_cache': False
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
        lines = vp_prompt.split('\n')
        for line in lines:
            if '：' in line:
                key, value = line.split('：', 1)
                ehr_data[key.strip()] = value.strip()
        return ehr_data
    
    def _calculate_differences(self, judger_scores: List[List[float]], doctor_scores: List[List[float]]) -> Dict[str, Any]:
        """计算judger与医生评分之间的差异"""
        doctor_avg = np.mean(doctor_scores, axis=0)
        judger_arr = np.array(judger_scores)
        case_diff = judger_arr - doctor_avg
        
        return {
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
平均评分（医生 vs Judge）：
"""
        
        for i, dim in enumerate(self.dimensions):
            doc_score = differences['doctor_average_scores'][i]
            judge_score = differences['judger_average_scores'][i]
            diff = differences['mean_difference'][i]
            report += f"- {dim}: 医生={doc_score:.2f}, Judge={judge_score:.2f}, 差异={diff:+.2f}\n"
        
        report += f"""
【一致性评估】
"""
        
        if overall_icc >= 0.80:
            report += "\n优秀：Judge与真实医生的评估高度一致，可直接使用。\n"
        elif overall_icc >= 0.60:
            report += "\n良好：Judge与真实医生有较好的一致性，但仍有优化空间。\n"
        elif overall_icc >= 0.40:
            report += "\n中等：Judge与真实医生存在一定差异，需要进行优化。\n"
        else:
            report += "\n差：Judge与真实医生一致性较差，需要大幅优化。\n"
        
        report += "\n===========================================\n"
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
        """保存评估报告"""
        os.makedirs(output_dir, exist_ok=True)
        
        # 保存JSON报告
        json_path = os.path.join(output_dir, 'icc_evaluation_report.json')
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        
        # 保存文本报告
        txt_path = os.path.join(output_dir, 'icc_evaluation_report.txt')
        with open(txt_path, 'w', encoding='utf-8') as f:
            if 'consistency_report' in report:
                f.write(report['consistency_report'])
        
        print(f"\n评估报告已保存到:")
        print(f"  - JSON格式: {json_path}")
        print(f"  - 文本格式: {txt_path}")
        
        return json_path
    
    def clear_cache(self, human_eval_path: str = None) -> int:
        """
        清除缓存
        
        参数：
            human_eval_path: 如果指定，只清除与该文件相关的缓存；否则清除所有缓存
        
        返回：
            清除的缓存文件数量
        """
        count = 0
        if not os.path.exists(self.cache_manager.cache_dir):
            return 0
        
        for filename in os.listdir(self.cache_manager.cache_dir):
            if filename.startswith('judger_cache_') and filename.endswith('.json'):
                filepath = os.path.join(self.cache_manager.cache_dir, filename)
                
                if human_eval_path is None:
                    os.remove(filepath)
                    count += 1
                else:
                    # 检查缓存是否与指定文件相关
                    try:
                        with open(filepath, 'r', encoding='utf-8') as f:
                            cache_data = json.load(f)
                        cache_info = cache_data.get('_cache_info', {})
                        if cache_info.get('human_eval_path') == human_eval_path:
                            os.remove(filepath)
                            count += 1
                    except:
                        pass
        
        print(f"已清除 {count} 个缓存文件")
        return count


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
    parser.add_argument('--sample_size', type=int, default=None,
                        help='随机抽样数量（默认使用全部数据）')
    parser.add_argument('--random_seed', type=int, default=42,
                        help='随机种子（用于可重复抽样）')
    parser.add_argument('--no_cache', action='store_true',
                        help='禁用缓存，强制重新评估')
    parser.add_argument('--clear_cache', action='store_true',
                        help='清除所有缓存后退出')
    args = parser.parse_args()
    
    # 创建ICC评估器
    evaluator = ICCEvaluator(args.config)
    
    # 清除缓存模式
    if args.clear_cache:
        evaluator.clear_cache(args.human_eval)
        return
    
    # 执行分析
    print("开始分析judger与真实医生评价的一致性...")
    report = evaluator.analyze_consistency(
        args.human_eval, 
        args.sample_size, 
        args.random_seed,
        use_cache=not args.no_cache
    )
    
    # 保存报告
    evaluator.save_report(report, args.output)
    
    # 打印报告摘要
    if 'consistency_report' in report:
        print("\n" + "="*60)
        print("          ICC评估报告摘要")
        print("="*60)
        print(report['consistency_report'])


if __name__ == '__main__':
    main()