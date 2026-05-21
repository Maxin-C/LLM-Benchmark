#!/usr/bin/env python3
"""
EASE Benchmark主入口
专家锚定的自适应仿真评估框架
"""

import argparse
import logging
import os
from dotenv import load_dotenv

from src.utils.logging_utils import LoggingUtils
from src.utils.llm_client import LLMClient

def main():
    # 加载环境变量
    load_dotenv()
    
    # 设置日志
    logger = LoggingUtils.setup_logger('ease_benchmark')
    
    parser = argparse.ArgumentParser(description='EASE Benchmark - Expert-Anchored Adaptive Simulation Evaluation Framework')
    parser.add_argument('--mode', type=str, required=True, 
                        choices=['benchmark', 'evaluate', 'meta', 'report', 'test', 'kg_test'],
                        help='运行模式')
    parser.add_argument('--config', type=str, default='config/sandbox_config.yaml',
                        help='配置文件路径')
    parser.add_argument('--output', type=str, default='outputs',
                        help='输出目录')
    
    args = parser.parse_args()
    
    logger.info(f"启动EASE Benchmark，模式: {args.mode}")
    
    try:
        if args.mode == 'benchmark':
            run_benchmark(args)
        elif args.mode == 'evaluate':
            run_evaluation(args)
        elif args.mode == 'meta':
            run_meta_evaluation(args)
        elif args.mode == 'report':
            run_report(args)
        elif args.mode == 'test':
            run_tests(args)
        elif args.mode == 'kg_test':
            run_kg_test(args)
        
        logger.info("任务完成")
    except Exception as e:
        logger.error(f"执行失败: {str(e)}", exc_info=True)
        raise

def _get_llm_client():
    """获取LLM客户端实例"""
    api_key = os.getenv('EASE_LLM_API_KEY', 'sk-KaZVAPnsPr2oVbLq17511e02E979454bBd43E0B07b18344f')
    base_url = os.getenv('EASE_LLM_BASE_URL', 'https://api.pumpkinaigc.online/v1')
    model = os.getenv('EASE_LLM_MODEL', 'gpt-4o')
    
    return LLMClient(api_key=api_key, base_url=base_url, model=model)

def _get_graph_reasoner():
    """获取图推理引擎实例"""
    try:
        from src.kg.graph_reasoner import GraphReasoner
        
        # 使用本项目中的模型和数据
        model_path = 'dataset/kg_data/gnn_models/best_kg_model.pth'
        kg_path = 'dataset/kg_data/CMeKG.pkl'
        embedding_cache_path = 'dataset/kg_data/gnn_cache/bge_text_cache.npy'
        
        # 检查文件是否存在
        if os.path.exists(model_path) and os.path.exists(kg_path) and os.path.exists(embedding_cache_path):
            return GraphReasoner(model_path, kg_path, embedding_cache_path)
        else:
            print(f"警告：图推理引擎文件不存在")
            print(f"  模型路径: {model_path}")
            print(f"  图谱路径: {kg_path}")
            print(f"  嵌入路径: {embedding_cache_path}")
            return None
    except Exception as e:
        print(f"图推理引擎初始化失败: {str(e)}")
        return None

def run_benchmark(args):
    """运行基准测试"""
    from scripts.run_benchmark import main as benchmark_main
    
    benchmark_main()

def run_evaluation(args):
    """运行单次评估"""
    from src.evaluation.chief_judge import ChiefJudge
    from src.evaluation.evidence_checker import EvidenceChecker
    from src.evaluation.empathy_evaluator import EmpathyEvaluator
    
    # 获取LLM客户端
    llm_client = _get_llm_client()
    
    # 获取图推理引擎
    graph_reasoner = _get_graph_reasoner()
    
    # 初始化评估组件
    evidence_checker = EvidenceChecker(llm_client, graph_reasoner)
    empathy_evaluator = EmpathyEvaluator(llm_client)
    judge = ChiefJudge(llm_client, evidence_checker, empathy_evaluator)
    
    # 示例对话
    dialogue_history = [
        {'role': 'doctor', 'content': '你好，我是你的主治医生。根据你的病历，你被诊断为乳腺癌IIB期。'},
        {'role': 'patient', 'content': '医生，我很担心，这个病能治好吗？'},
        {'role': 'doctor', 'content': '请放心，你的情况属于中期，通过规范治疗，治愈率还是很高的。我们会为你制定个性化的治疗方案。'},
        {'role': 'patient', 'content': '那我需要做哪些治疗呢？'},
        {'role': 'doctor', 'content': '根据你的情况，我们建议先进行手术治疗，然后进行化疗和内分泌治疗。'}
    ]
    
    patient_state = {
        'demographics': {'age': 45, 'gender': 'female', 'occupation': '教师'},
        'medical_info': {
            'pathology_type': '浸润性导管癌',
            'stage': 'IIB期',
            'surgery_type': '乳房切除术',
            'treatment_stage': '术前评估',
            'medications': ['他莫昔芬']
        },
        'current_mood': 'worried',
        'interaction_history': [],
        'symptoms': ['乳房疼痛', '焦虑']
    }
    
    context = {
        'disease_name': '乳腺癌',
        'patient_info': patient_state['medical_info']
    }
    
    result = judge.evaluate(dialogue_history, patient_state, context)
    
    print("\n评估结果:")
    print(f"综合评分: {result['overall_score']}/5")
    print(f"是否通过: {'是' if result['is_passed'] else '否'}")
    print("\n各维度评分:")
    for dim, score in result['scores'].items():
        print(f"  {dim}: {score}/5")
    
    if result.get('deduction_reasons'):
        print("\n扣分理由:")
        for reason in result['deduction_reasons']:
            print(f"  - {reason}")

def run_meta_evaluation(args):
    """运行元评估"""
    from src.meta_evaluation.icc_calculator import ICCCalculator
    
    icc_calculator = ICCCalculator()
    
    # 模拟100个病例的评分数据
    import numpy as np
    np.random.seed(42)
    
    # 4位医生的评分
    doctor_1 = np.random.randint(1, 6, size=(100, 5)).tolist()
    doctor_2 = np.random.randint(1, 6, size=(100, 5)).tolist()
    doctor_3 = np.random.randint(1, 6, size=(100, 5)).tolist()
    doctor_4 = np.random.randint(1, 6, size=(100, 5)).tolist()
    
    # EASE-Judge的评分（与医生评分有一定相关性）
    ease_judge = []
    for i in range(100):
        base = np.array([doctor_1[i], doctor_2[i], doctor_3[i], doctor_4[i]]).mean(axis=0)
        noise = np.random.normal(0, 0.3, size=5)
        judge_scores = np.clip(base + noise, 1, 5).round().astype(int).tolist()
        ease_judge.append(judge_scores)
    
    # 添加评分者
    icc_calculator.add_rater('doctor_1', doctor_1)
    icc_calculator.add_rater('doctor_2', doctor_2)
    icc_calculator.add_rater('doctor_3', doctor_3)
    icc_calculator.add_rater('doctor_4', doctor_4)
    icc_calculator.add_rater('ease_judge', ease_judge)
    
    # 计算ICC
    result = icc_calculator.calculate_icc()
    
    print("\nICC计算结果:")
    print(f"总体ICC: {result['overall_icc']['icc']:.4f}")
    print(f"95%置信区间: [{result['overall_icc']['confidence_interval'][0]:.4f}, {result['overall_icc']['confidence_interval'][1]:.4f}]")
    print(f"一致性等级: {result['overall_icc']['consistency_level']}")
    
    print("\n各维度ICC:")
    for dim, data in result['dimension_iccs'].items():
        print(f"  {dim}: {data['icc']:.4f}")
    
    # 计算医生间一致性
    doctor_agreement = icc_calculator.calculate_doctor_agreement()
    print(f"\n医生间一致性ICC: {doctor_agreement['icc']:.4f}")

def run_report(args):
    """生成报告"""
    from scripts.generate_report import main as report_main
    
    # 创建模拟结果
    import json
    import os
    
    output_dir = 'outputs/evaluations'
    os.makedirs(output_dir, exist_ok=True)
    
    mock_results = [
        {
            'scenario_id': 'scenario_001',
            'scenario_name': '内分泌治疗咨询',
            'scores': {'accuracy': 4, 'effectiveness': 4, 'safety': 5, 'personalization': 3, 'empathy': 4},
            'overall_score': 4,
            'is_passed': True,
            'deduction_reasons': [],
            'risk_report': {'risk_level': 'low', 'critical_issues': [], 'warnings': []}
        },
        {
            'scenario_id': 'scenario_002',
            'scenario_name': '化疗副作用处理',
            'scores': {'accuracy': 3, 'effectiveness': 3, 'safety': 4, 'personalization': 4, 'empathy': 5},
            'overall_score': 4,
            'is_passed': True,
            'deduction_reasons': ['治疗方案说明不够详细'],
            'risk_report': {'risk_level': 'low', 'critical_issues': [], 'warnings': ['建议增加随访提醒']}
        }
    ]
    
    with open(os.path.join(output_dir, 'mock_results.json'), 'w', encoding='utf-8') as f:
        json.dump(mock_results, f, ensure_ascii=False)
    
    report_main()

def run_tests(args):
    """运行测试"""
    import subprocess
    result = subprocess.run(['python', '-m', 'pytest', 'tests/', '-v'], 
                          capture_output=True, text=True)
    print(result.stdout)
    if result.stderr:
        print("错误输出:", result.stderr)

def run_kg_test(args):
    """测试图推理引擎"""
    graph_reasoner = _get_graph_reasoner()
    
    if not graph_reasoner:
        print("图推理引擎初始化失败，无法进行测试")
        return
    
    print("=== 图推理引擎测试 ===")
    
    # 测试1: 查询疾病治疗方案
    print("\n1. 查询疾病治疗方案:")
    treatments = graph_reasoner.query_disease_treatment('乳腺癌', top_k=3)
    for treatment, score in treatments:
        print(f"   - {treatment} (相关性: {score:.2f})")
    
    # 测试2: 查询药物相互作用
    print("\n2. 查询药物相互作用:")
    interactions = graph_reasoner.query_drug_interaction('他莫昔芬', top_k=3)
    for drug, score in interactions:
        print(f"   - {drug} (强度: {score:.2f})")
    
    # 测试3: 查找相似节点
    print("\n3. 查找相似疾病:")
    similar = graph_reasoner.get_similar_nodes('乳腺癌', top_k=3)
    for node, score in similar:
        print(f"   - {node} (相似度: {score:.2f})")
    
    # 测试4: 预测边概率
    print("\n4. 预测药物相互作用概率:")
    prob = graph_reasoner.predict_edge('他莫昔芬', '来曲唑')
    print(f"   他莫昔芬和来曲唑之间存在相互作用的概率: {prob:.4f}")
    
    # 测试5: 查找推理路径
    print("\n5. 查找推理路径:")
    paths = graph_reasoner.reasoning_path('乳腺癌', '化疗', max_hops=2)
    if paths:
        for i, path in enumerate(paths[:3]):
            print(f"   路径{i+1}: {' -> '.join(path)}")
    else:
        print("   未找到路径")

if __name__ == '__main__':
    main()
