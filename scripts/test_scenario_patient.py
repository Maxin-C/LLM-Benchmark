#!/usr/bin/env python3
"""
测试虚拟患者场景动态加载功能
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.sandbox.virtual_patient import VirtualPatientAgent
from src.utils.scenario_manager import get_scenario_manager

def test_scenario_inference():
    """测试场景推断功能"""
    print("测试场景推断功能")
    print("=" * 60)
    
    # 创建模拟患者数据
    test_cases = [
        {
            'name': '化疗患者',
            'ehr_data': {
                'age': 45,
                'gender': '女性',
                'pathology_type': '乳腺癌',
                'stage': 'II期',
                'surgery_type': '乳房根治术',
                'treatment_stage': '化疗中',
                'medications': ['紫杉醇', '表阿霉素']
            },
            'dialogue_data': [{'content': '化疗副作用好大，怎么办？'}],
            'expected_scenario': '化疗相关'
        },
        {
            'name': '术后患者',
            'ehr_data': {
                'age': 50,
                'gender': '女性',
                'pathology_type': '乳腺癌',
                'stage': 'I期',
                'surgery_type': '保乳手术',
                'treatment_stage': '术后康复',
                'medications': []
            },
            'dialogue_data': [{'content': '手术后多久可以恢复性生活？'}],
            'expected_scenario': '性生活与康复'
        },
        {
            'name': '疼痛患者',
            'ehr_data': {
                'age': 38,
                'gender': '女性',
                'pathology_type': '乳腺增生',
                'stage': 'NA',
                'surgery_type': '无',
                'treatment_stage': '观察中',
                'medications': ['乳癖散结丸']
            },
            'dialogue_data': [{'content': '最近乳房胀痛得厉害，有什么办法缓解吗？'}],
            'expected_scenario': '疼痛管理'
        },
        {
            'name': '复查患者',
            'ehr_data': {
                'age': 55,
                'gender': '女性',
                'pathology_type': '乳腺癌',
                'stage': 'III期',
                'surgery_type': '乳房根治术',
                'treatment_stage': '定期复查',
                'medications': ['他莫昔芬']
            },
            'dialogue_data': [{'content': '明天要去复查了，需要准备什么吗？'}],
            'expected_scenario': '复查随访'
        }
    ]
    
    manager = get_scenario_manager()
    
    for test_case in test_cases:
        # 构建患者状态（简化版）
        state = {
            'demographics': {
                'age': test_case['ehr_data']['age'],
                'gender': test_case['ehr_data']['gender'],
                'occupation': '职员'
            },
            'medical_info': {
                'pathology_type': test_case['ehr_data']['pathology_type'],
                'stage': test_case['ehr_data']['stage'],
                'surgery_type': test_case['ehr_data']['surgery_type'],
                'medications': test_case['ehr_data']['medications'],
                'treatment_stage': test_case['ehr_data']['treatment_stage']
            },
            'symptoms': [],
            'concerns': [msg['content'] for msg in test_case['dialogue_data']],
            'communication_style': {'formality': 'informal'},
            'emotion': 'neutral',
            'compliance': 'medium',
            'cognitive_biases': [],
            'social_network': {'support_level': '中等'},
            'history_symptoms': [],
            'current_mood': 'neutral',
            'disease_progression': 0,
            'adverse_reactions': [],
            'interaction_history': []
        }
        
        # 手动测试场景推断
        treatment_stage = state['medical_info']['treatment_stage']
        concerns = state['concerns']
        
        inferred_scenario = '复查随访'  # 默认
        
        if '化疗' in treatment_stage or any('化疗' in c for c in concerns):
            inferred_scenario = '化疗相关'
        elif '手术' in treatment_stage or any('手术' in c for c in concerns):
            inferred_scenario = '手术相关'
        elif '放疗' in treatment_stage or any('放疗' in c for c in concerns):
            inferred_scenario = '放疗相关'
        elif '内分泌' in treatment_stage or any('内分泌' in c for c in concerns):
            inferred_scenario = '内分泌治疗'
        
        concern_keywords = {
            '性生活': '性生活与康复', '性': '性生活与康复',
            '复发': '复发转移', '转移': '复发转移',
            '复查': '复查随访', '检查': '复查随访',
            '怀孕': '生育哺乳', '生育': '生育哺乳', '哺乳': '生育哺乳',
            '疼痛': '疼痛管理', '痛': '疼痛管理', '胀': '疼痛管理',
            '饮食': '饮食营养', '吃': '饮食营养',
            '心理': '心理支持', '担心': '心理支持', '焦虑': '心理支持'
        }
        
        for concern in concerns:
            for keyword, scenario in concern_keywords.items():
                if keyword in concern:
                    inferred_scenario = scenario
                    break
        
        status = "✅" if inferred_scenario == test_case['expected_scenario'] else "❌"
        print(f"{status} {test_case['name']}:")
        print(f"   推断场景: {inferred_scenario}")
        print(f"   期望场景: {test_case['expected_scenario']}")
        print()

def test_prompt_generation():
    """测试提示生成功能"""
    print("\n测试场景提示生成")
    print("=" * 60)
    
    manager = get_scenario_manager()
    
    # 测试几个典型场景
    test_scenarios = ['性生活与康复', '化疗相关', '疼痛管理']
    
    for scenario_type in test_scenarios:
        config = manager.get_scenario_config(scenario_type)
        if config:
            prompt = manager.generate_prompt(scenario_type, include_examples=True)
            print(f"【{scenario_type}】")
            print(f"描述: {config.description}")
            print(f"关键词: {', '.join(config.keywords)}")
            print(f"示例数: {len(config.examples)}")
            print(f"提示预览:\n{prompt[:300]}...")
            print()

def test_scenario_manager():
    """测试场景管理器"""
    print("\n测试场景管理器")
    print("=" * 60)
    
    manager = get_scenario_manager()
    
    # 获取所有场景
    scenarios = manager.get_scenarios()
    print(f"可用场景数: {len(scenarios)}")
    print(f"场景列表: {', '.join(scenarios)}")
    
    # 关键词搜索
    print("\n关键词搜索测试:")
    search_results = manager.get_scenarios_by_keyword('性')
    print(f"  '性' -> {search_results}")
    
    search_results = manager.get_scenarios_by_keyword('化疗')
    print(f"  '化疗' -> {search_results}")
    
    search_results = manager.get_scenarios_by_keyword('疼痛')
    print(f"  '疼痛' -> {search_results}")
    
    # 获取随机案例
    print("\n随机案例测试:")
    case = manager.get_random_case('化疗相关')
    if case:
        print(f"  场景: {case.scenario_type}")
        print(f"  问题预览: {case.input_text[:100]}...")

if __name__ == "__main__":
    test_scenario_inference()
    test_prompt_generation()
    test_scenario_manager()
    
    print("\n🎉 所有测试完成！")