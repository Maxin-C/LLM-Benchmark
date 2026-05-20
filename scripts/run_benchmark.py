#!/usr/bin/env python3
"""
基准测试运行器
"""

import argparse
import json
import os
from typing import Dict, List, Any

def main():
    parser = argparse.ArgumentParser(description='EASE Benchmark Runner')
    parser.add_argument('--config', type=str, default='config/sandbox_config.yaml',
                        help='Path to configuration file')
    parser.add_argument('--output', type=str, default='outputs/evaluations',
                        help='Output directory for evaluation results')
    parser.add_argument('--scenario', type=str, default=None,
                        help='Specific scenario to run (optional)')
    args = parser.parse_args()
    
    # 创建输出目录
    os.makedirs(args.output, exist_ok=True)
    
    # 加载配置
    config = load_config(args.config)
    
    # 初始化模块
    from src.sandbox import VirtualPatientAgent, ScenarioManager, MonitorAgent
    from src.evaluation import ChiefJudge, EvidenceChecker, EmpathyEvaluator, KillSwitch
    from src.data_processing import EHRParser, KnowledgeGraphLoader, ExpertDataProcessor
    
    # 加载数据
    ehr_parser = EHRParser()
    kg_loader = KnowledgeGraphLoader()
    expert_processor = ExpertDataProcessor()
    
    # 加载场景管理器
    scenario_manager = ScenarioManager()
    
    if args.scenario:
        scenario = scenario_manager.get_scenario(args.scenario)
        if not scenario:
            print(f"场景 {args.scenario} 不存在")
            return
        scenarios = [scenario]
    else:
        scenarios = scenario_manager.get_all_scenarios()
    
    # 运行每个场景
    results = []
    for scenario in scenarios:
        print(f"运行场景: {scenario.name}")
        
        # 创建虚拟患者
        ehr_data = {
            'patient_id': 'test_patient',
            'age': 45,
            'gender': 'female',
            'occupation': '教师',
            'pathology_type': '浸润性导管癌',
            'stage': 'IIB期',
            'surgery_type': '乳房切除术',
            'medications': ['他莫昔芬'],
            'treatment_stage': '内分泌治疗'
        }
        
        dialogue_data = [
            {'sender': 'patient', 'content': '医生，我最近感觉乳房有点痛', 'message_index': 0},
            {'sender': 'patient', 'content': '还有点肿胀，不知道怎么办', 'message_index': 1}
        ]
        
        persona_graph = {
            'compliance': 'medium',
            'cognitive_biases': [],
            'social_network': {},
            'history_symptoms': ['恶心', '疲劳']
        }
        
        vp_agent = VirtualPatientAgent(ehr_data, dialogue_data, persona_graph)
        
        # 创建监控Agent
        monitor_agent = MonitorAgent()
        
        # 创建评估器
        evidence_checker = EvidenceChecker()
        empathy_evaluator = EmpathyEvaluator()
        kill_switch = KillSwitch()
        chief_judge = ChiefJudge(evidence_checker, empathy_evaluator, kill_switch)
        
        # 模拟对话
        dialogue_history = []
        current_round = 0
        max_rounds = 5
        
        while current_round < max_rounds:
            # 医生响应（模拟）
            doctor_response = generate_doctor_response(vp_agent.get_state())
            dialogue_history.append({'role': 'doctor', 'content': doctor_response})
            
            # 监控检查
            should_terminate, reason = monitor_agent.should_terminate(current_round, vp_agent.get_state(), doctor_response)
            if should_terminate:
                print(f"对话终止: {reason}")
                break
            
            # 更新患者状态
            from src.sandbox.virtual_patient import DoctorAction
            action = DoctorAction('treatment_plan', doctor_response)
            vp_agent.state_transition(action)
            
            # 患者响应
            patient_response = vp_agent.generate_response(doctor_response)
            dialogue_history.append({'role': 'patient', 'content': patient_response})
            
            current_round += 1
        
        # 评估对话
        context = {
            'disease_name': '乳腺癌',
            'patient_info': ehr_data
        }
        
        evaluation_result = chief_judge.evaluate(dialogue_history, vp_agent.get_state(), context)
        evaluation_result['scenario_id'] = scenario.scenario_id
        evaluation_result['scenario_name'] = scenario.name
        evaluation_result['dialogue_history'] = dialogue_history
        
        results.append(evaluation_result)
        
        # 打印结果
        print(f"综合评分: {evaluation_result['overall_score']}")
        print(f"是否通过: {'是' if evaluation_result['is_passed'] else '否'}")
        print()
    
    # 保存结果
    output_file = os.path.join(args.output, 'benchmark_results.json')
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    print(f"结果已保存到: {output_file}")

def load_config(config_path: str) -> Dict[str, Any]:
    """
    加载配置文件
    
    参数：
        config_path: 配置文件路径
    
    返回：
        配置字典
    """
    if not os.path.exists(config_path):
        print(f"配置文件不存在: {config_path}")
        return {}
    
    import yaml
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)

def generate_doctor_response(patient_state: Dict[str, Any]) -> str:
    """
    生成模拟医生响应
    
    参数：
        patient_state: 患者状态
    
    返回：
        医生响应
    """
    symptoms = patient_state.get('symptoms', [])
    treatment_stage = patient_state.get('medical_info', {}).get('treatment_stage', '')
    
    responses = [
        f"你目前的症状是{', '.join(symptoms)}，这在{treatment_stage}阶段是比较常见的。请继续按时服药，定期复查。",
        f"关于你的症状，建议你注意休息，保持良好的心态。如果疼痛加剧，请及时联系我。",
        f"根据你的情况，目前的治疗方案是合适的。请不要过于担心，有问题随时联系我。"
    ]
    
    import random
    return random.choice(responses)

if __name__ == '__main__':
    main()
