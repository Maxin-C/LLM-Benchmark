"""
场景管理器
管理仿真场景的定义、加载和执行
"""

from typing import Dict, List, Any, Callable
import json

class Scenario:
    """
    场景类
    表示一个特定的临床场景
    """
    
    def __init__(self, scenario_id: str, name: str, category: str, description: str,
                 setup: Dict[str, Any], evaluation_criteria: List[Dict[str, Any]]):
        self.scenario_id = scenario_id
        self.name = name
        self.category = category
        self.description = description
        self.setup = setup
        self.evaluation_criteria = evaluation_criteria
        self.is_active = False
    
    def activate(self) -> None:
        """激活场景"""
        self.is_active = True
    
    def deactivate(self) -> None:
        """停用场景"""
        self.is_active = False

class ScenarioManager:
    """
    场景管理器类
    负责场景的加载、管理和执行
    """
    
    def __init__(self):
        self.scenarios = {}
        self.active_scenario = None
    
    def load_scenario(self, scenario_data: Dict[str, Any]) -> None:
        """
        加载场景
        
        参数：
            scenario_data: 场景数据
        """
        scenario = Scenario(
            scenario_id=scenario_data['scenario_id'],
            name=scenario_data['name'],
            category=scenario_data['category'],
            description=scenario_data['description'],
            setup=scenario_data.get('setup', {}),
            evaluation_criteria=scenario_data.get('evaluation_criteria', [])
        )
        
        self.scenarios[scenario.scenario_id] = scenario
    
    def load_from_json(self, file_path: str) -> None:
        """
        从JSON文件加载场景
        
        参数：
            file_path: JSON文件路径
        """
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        if isinstance(data, list):
            for scenario_data in data:
                self.load_scenario(scenario_data)
        elif isinstance(data, dict) and 'scenarios' in data:
            for scenario_data in data['scenarios']:
                self.load_scenario(scenario_data)
    
    def get_scenario(self, scenario_id: str) -> Scenario:
        """
        获取场景
        
        参数：
            scenario_id: 场景ID
        
        返回：
            场景对象
        """
        return self.scenarios.get(scenario_id)
    
    def activate_scenario(self, scenario_id: str) -> bool:
        """
        激活场景
        
        参数：
            scenario_id: 场景ID
        
        返回：
            是否激活成功
        """
        if scenario_id in self.scenarios:
            # 先停用当前激活的场景
            if self.active_scenario:
                self.active_scenario.deactivate()
            
            self.active_scenario = self.scenarios[scenario_id]
            self.active_scenario.activate()
            return True
        
        return False
    
    def get_active_scenario(self) -> Scenario:
        """
        获取当前激活的场景
        
        返回：
            当前激活的场景
        """
        return self.active_scenario
    
    def get_scenarios_by_category(self, category: str) -> List[Scenario]:
        """
        按类别获取场景
        
        参数：
            category: 场景类别
        
        返回：
            场景列表
        """
        return [s for s in self.scenarios.values() if s.category == category]
    
    def get_all_scenarios(self) -> List[Scenario]:
        """
        获取所有场景
        
        返回：
            场景列表
        """
        return list(self.scenarios.values())
    
    def get_scenario_categories(self) -> List[str]:
        """
        获取所有场景类别
        
        返回：
            类别列表
        """
        categories = set()
        for scenario in self.scenarios.values():
            categories.add(scenario.category)
        return list(categories)
    
    def create_scenario(self, scenario_id: str, name: str, category: str, description: str,
                        setup: Dict[str, Any] = None, evaluation_criteria: List[Dict[str, Any]] = None) -> Scenario:
        """
        创建新场景
        
        参数：
            scenario_id: 场景ID
            name: 场景名称
            category: 场景类别
            description: 场景描述
            setup: 场景设置
            evaluation_criteria: 评估标准
        
        返回：
            创建的场景
        """
        scenario = Scenario(
            scenario_id=scenario_id,
            name=name,
            category=category,
            description=description,
            setup=setup or {},
            evaluation_criteria=evaluation_criteria or []
        )
        
        self.scenarios[scenario_id] = scenario
        return scenario
    
    def delete_scenario(self, scenario_id: str) -> bool:
        """
        删除场景
        
        参数：
            scenario_id: 场景ID
        
        返回：
            是否删除成功
        """
        if scenario_id in self.scenarios:
            # 如果删除的是当前激活的场景，取消激活
            if self.active_scenario and self.active_scenario.scenario_id == scenario_id:
                self.active_scenario.deactivate()
                self.active_scenario = None
            
            del self.scenarios[scenario_id]
            return True
        
        return False

class BreastCancerScenarioManager(ScenarioManager):
    """
    乳腺癌专用场景管理器
    包含预设的乳腺癌相关场景
    """
    
    def __init__(self):
        super().__init__()
        self._initialize_default_scenarios()
    
    def _initialize_default_scenarios(self) -> None:
        """
        初始化默认场景
        """
        scenarios = [
            {
                'scenario_id': 'targeted_therapy_followup',
                'name': '靶向治疗随访',
                'category': 'targeted_therapy',
                'description': '患者正在接受靶向治疗，需要进行定期随访评估',
                'setup': {
                    'patient_type': 'breast_cancer',
                    'treatment_stage': 'targeted_therapy',
                    'expected_duration': 12
                },
                'evaluation_criteria': [
                    {'name': '疗效评估', 'weight': 0.3},
                    {'name': '剂量调整', 'weight': 0.2},
                    {'name': '耐药监测', 'weight': 0.3},
                    {'name': '不良反应管理', 'weight': 0.2}
                ]
            },
            {
                'scenario_id': 'endocrine_switch',
                'name': '内分泌药物转换',
                'category': 'endocrine_therapy',
                'description': '患者需要从一种内分泌药物转换到另一种',
                'setup': {
                    'patient_type': 'breast_cancer',
                    'treatment_stage': 'endocrine_therapy',
                    'current_drug': 'tamoxifen',
                    'target_drug': 'letrozole'
                },
                'evaluation_criteria': [
                    {'name': '药物选择合理性', 'weight': 0.4},
                    {'name': '副作用管理', 'weight': 0.3},
                    {'name': '转换时机', 'weight': 0.3}
                ]
            },
            {
                'scenario_id': 'lymphedema_recognition',
                'name': '淋巴水肿识别',
                'category': 'complication',
                'description': '患者出现上肢肿胀，需要识别是否为淋巴水肿',
                'setup': {
                    'patient_type': 'breast_cancer',
                    'post_surgery_months': 6,
                    'symptoms': ['arm_swelling', 'heaviness']
                },
                'evaluation_criteria': [
                    {'name': '症状识别', 'weight': 0.4},
                    {'name': '干预时机', 'weight': 0.3},
                    {'name': '处理规范', 'weight': 0.3}
                ]
            },
            {
                'scenario_id': 'psychological_crisis',
                'name': '心理危机干预',
                'category': 'psychological',
                'description': '患者出现抑郁倾向，需要进行心理支持',
                'setup': {
                    'patient_type': 'breast_cancer',
                    'mood_state': 'depressed',
                    'risk_level': 'moderate'
                },
                'evaluation_criteria': [
                    {'name': '情绪识别', 'weight': 0.3},
                    {'name': '支持性沟通', 'weight': 0.4},
                    {'name': '转诊建议', 'weight': 0.3}
                ]
            },
            {
                'scenario_id': 'complication_management',
                'name': '并发症管理',
                'category': 'complication',
                'description': '患者出现治疗相关并发症',
                'setup': {
                    'patient_type': 'breast_cancer',
                    'complication_type': 'infection',
                    'severity': 'mild'
                },
                'evaluation_criteria': [
                    {'name': '鉴别诊断', 'weight': 0.4},
                    {'name': '处理及时性', 'weight': 0.3},
                    {'name': '预防措施', 'weight': 0.3}
                ]
            }
        ]
        
        for scenario_data in scenarios:
            self.load_scenario(scenario_data)
