"""
仿真沙盒测试
"""

import pytest
from src.sandbox import VirtualPatientAgent, ScenarioManager, MonitorAgent

class TestVirtualPatientAgent:
    """测试虚拟患者Agent"""
    
    def test_initialization(self):
        """测试虚拟患者初始化"""
        ehr_data = {
            'patient_id': 'VP001',
            'age': 45,
            'gender': 'female',
            'pathology_type': '浸润性导管癌',
            'stage': 'IIB期',
            'surgery_type': '乳房切除术',
            'medications': ['他莫昔芬'],
            'treatment_stage': '内分泌治疗'
        }
        
        dialogue_data = [
            {'sender': 'patient', 'content': '医生，我最近感觉乳房有点痛'}
        ]
        
        persona_graph = {
            'compliance': 'medium',
            'cognitive_biases': [],
            'social_network': {},
            'history_symptoms': ['恶心']
        }
        
        vp_agent = VirtualPatientAgent(ehr_data, dialogue_data, persona_graph)
        
        assert vp_agent is not None
        assert vp_agent.state['demographics']['age'] == 45
        assert vp_agent.state['medical_info']['stage'] == 'IIB期'
        assert vp_agent.state['compliance'] == 'medium'
    
    def test_state_transition(self):
        """测试状态转移"""
        ehr_data = {'patient_id': 'VP001', 'age': 45, 'gender': 'female'}
        dialogue_data = []
        persona_graph = {'compliance': 'medium'}
        
        vp_agent = VirtualPatientAgent(ehr_data, dialogue_data, persona_graph)
        
        from src.sandbox.virtual_patient import DoctorAction
        
        # 测试药物错误
        action = DoctorAction('treatment_plan', '推荐错误药物')
        action.add_error('wrong_medication')
        vp_agent.state_transition(action)
        
        assert '药物不良反应' in vp_agent.state['adverse_reactions']
        assert vp_agent.state['current_mood'] == 'anxious'
    
    def test_generate_response(self):
        """测试生成响应"""
        ehr_data = {'patient_id': 'VP001', 'age': 45}
        dialogue_data = [{'sender': 'patient', 'content': '医生，我有点担心'}]
        persona_graph = {'compliance': 'medium'}
        
        vp_agent = VirtualPatientAgent(ehr_data, dialogue_data, persona_graph)
        response = vp_agent.generate_response('你好，有什么可以帮你的？')
        
        assert isinstance(response, str)
        assert len(response) > 0

class TestScenarioManager:
    """测试场景管理器"""
    
    def test_scenario_loading(self):
        """测试场景加载"""
        scenario_manager = ScenarioManager()
        
        scenario_data = {
            'scenario_id': 'test_scenario',
            'name': '测试场景',
            'category': 'test',
            'description': '测试场景描述',
            'setup': {},
            'evaluation_criteria': []
        }
        
        scenario_manager.load_scenario(scenario_data)
        scenario = scenario_manager.get_scenario('test_scenario')
        
        assert scenario is not None
        assert scenario.name == '测试场景'
    
    def test_scenario_activation(self):
        """测试场景激活"""
        scenario_manager = ScenarioManager()
        
        scenario_data = {
            'scenario_id': 'test_scenario',
            'name': '测试场景',
            'category': 'test',
            'description': '测试场景描述',
            'setup': {},
            'evaluation_criteria': []
        }
        
        scenario_manager.load_scenario(scenario_data)
        result = scenario_manager.activate_scenario('test_scenario')
        
        assert result is True
        assert scenario_manager.get_active_scenario().scenario_id == 'test_scenario'

class TestMonitorAgent:
    """测试监控Agent"""
    
    def test_red_line_check(self):
        """测试红线检查"""
        monitor = MonitorAgent()
        
        # 默认情况下不应该触发红线
        result = monitor.check_red_line('正常的医疗建议', {})
        assert result is False
    
    def test_termination_check(self):
        """测试终止检查"""
        monitor = MonitorAgent()
        monitor.set_max_rounds(5)
        
        # 未达到最大轮数时不应终止
        should_terminate, reason = monitor.should_terminate(3, {})
        assert should_terminate is False
        
        # 达到最大轮数时应终止
        should_terminate, reason = monitor.should_terminate(5, {})
        assert should_terminate is True
        assert reason == '达到最大对话轮数'

if __name__ == '__main__':
    pytest.main([__file__, '-v'])
