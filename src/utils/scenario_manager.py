#!/usr/bin/env python3
"""
场景管理器 - 用于管理乳腺癌领域的对话场景和经典案例
支持动态加载场景，改善虚拟患者的真实性和针对性
"""

import json
import os
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass

@dataclass
class ScenarioCase:
    """场景案例数据结构"""
    input_text: str
    output_text: str
    scenario_type: str
    keywords: List[str]

@dataclass
class ScenarioConfig:
    """场景配置"""
    scenario_type: str
    description: str
    keywords: List[str]
    examples: List[ScenarioCase]
    prompt_template: str

class ScenarioManager:
    """场景管理器"""
    
    def __init__(self, data_path: str = "dataset/dialogue_data/med_dialogue_qa.json"):
        self.data_path = data_path
        self.scenarios: Dict[str, ScenarioConfig] = {}
        self.all_cases: List[ScenarioCase] = []
        self._load_scenarios()
    
    def _load_scenarios(self):
        """加载所有场景"""
        # 定义场景配置
        scenario_definitions = {
            "性生活与康复": {
                "description": "乳腺癌患者术后性生活相关问题，包括对病情的影响、时机选择等",
                "keywords": ["性生活", "性", "亲密关系"]
            },
            "化疗相关": {
                "description": "化疗方案、副作用处理、疗程安排等问题",
                "keywords": ["化疗", "紫杉醇", "表阿霉素", "疗程"]
            },
            "手术相关": {
                "description": "手术方式选择、术后恢复、并发症处理等",
                "keywords": ["手术", "切除", "根治", "保乳"]
            },
            "复发转移": {
                "description": "复发风险评估、转移症状识别、后续治疗方案",
                "keywords": ["复发", "转移", "风险"]
            },
            "乳腺纤维瘤": {
                "description": "乳腺纤维瘤的诊断、治疗和随访",
                "keywords": ["乳腺纤维瘤", "纤维瘤"]
            },
            "乳腺增生": {
                "description": "乳腺增生的症状、治疗和日常护理",
                "keywords": ["乳腺增生", "小叶增生"]
            },
            "疼痛管理": {
                "description": "乳房疼痛的原因分析和缓解方法",
                "keywords": ["胀痛", "疼痛", "刺痛"]
            },
            "复查随访": {
                "description": "定期复查计划、检查项目解读、随访频率",
                "keywords": ["复查", "超声", "MRI", "检查", "随访"]
            },
            "饮食营养": {
                "description": "饮食建议、营养补充、忌口事项",
                "keywords": ["饮食", "营养", "忌口", "咖啡", "蜂胶"]
            },
            "生育哺乳": {
                "description": "乳腺癌患者的生育计划、哺乳指导",
                "keywords": ["怀孕", "生育", "哺乳", "溢乳"]
            },
            "心理支持": {
                "description": "患者心理状态、焦虑情绪管理",
                "keywords": ["心理", "焦虑", "抑郁", "担心"]
            },
            "内分泌治疗": {
                "description": "内分泌治疗方案、药物副作用、雌激素管理",
                "keywords": ["内分泌", "雌激素", "他莫昔芬", "诺雷德"]
            },
            "放疗相关": {
                "description": "放疗计划、副作用处理、效果评估",
                "keywords": ["放疗", "放射治疗"]
            }
        }
        
        # 加载对话数据
        if os.path.exists(self.data_path):
            with open(self.data_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # 构建案例
            for item in data:
                input_text = item.get('input', '')
                output_text = item.get('output', '')
                
                # 判断场景类型
                for scenario_type, config in scenario_definitions.items():
                    if any(kw in input_text for kw in config['keywords']):
                        case = ScenarioCase(
                            input_text=input_text,
                            output_text=output_text,
                            scenario_type=scenario_type,
                            keywords=config['keywords']
                        )
                        self.all_cases.append(case)
                        break
        
        # 构建场景配置
        for scenario_type, config in scenario_definitions.items():
            examples = [c for c in self.all_cases if c.scenario_type == scenario_type][:10]
            self.scenarios[scenario_type] = ScenarioConfig(
                scenario_type=scenario_type,
                description=config['description'],
                keywords=config['keywords'],
                examples=examples,
                prompt_template=self._build_prompt_template(scenario_type)
            )
    
    def _build_prompt_template(self, scenario_type: str) -> str:
        """构建场景提示模板"""
        templates = {
            "性生活与康复": """
你是一位乳腺癌患者，刚刚完成治疗，身体正在恢复中。
你对术后性生活有很多疑问和担忧，担心会影响病情。
请用自然、真诚的方式表达你的困惑和担忧。
""",
            "化疗相关": """
你是一位正在接受化疗的乳腺癌患者。
你对化疗方案、副作用、疗程安排有很多问题。
请用自然、真诚的方式表达你的疑问和感受。
""",
            "手术相关": """
你是一位即将接受或刚刚完成手术的乳腺癌患者。
你对手术方式、恢复过程、并发症等有很多疑问。
请用自然、真诚的方式表达你的担忧和问题。
""",
            "复发转移": """
你是一位担心癌症复发或已经发现转移迹象的患者。
你非常焦虑，想了解复发的风险和应对措施。
请用自然、真诚的方式表达你的担忧。
""",
            "乳腺纤维瘤": """
你被诊断出乳腺纤维瘤，对这个疾病不太了解。
你想知道是否需要手术，以及日常需要注意什么。
请用自然、真诚的方式表达你的疑问。
""",
            "乳腺增生": """
你长期受乳腺增生困扰，经常感到乳房胀痛。
你想知道如何缓解症状，以及是否有恶变风险。
请用自然、真诚的方式表达你的困扰。
""",
            "疼痛管理": """
你经常感到乳房疼痛，影响了日常生活。
你想知道疼痛的原因和缓解方法。
请用自然、真诚的方式表达你的痛苦。
""",
            "复查随访": """
你需要进行定期复查，对检查项目和结果有很多疑问。
你想知道复查的频率和注意事项。
请用自然、真诚的方式表达你的疑问。
""",
            "饮食营养": """
你想了解乳腺癌患者的饮食建议和忌口事项。
你对某些食物是否适合食用有疑问。
请用自然、真诚的方式表达你的疑问。
""",
            "生育哺乳": """
你是一位年轻的乳腺癌患者，有生育计划。
你想知道治疗期间和治疗后是否可以怀孕哺乳。
请用自然、真诚的方式表达你的担忧。
""",
            "心理支持": """
你在治疗过程中感到非常焦虑和恐惧。
你需要心理支持和安慰。
请用自然、真诚的方式表达你的情绪。
""",
            "内分泌治疗": """
你正在接受内分泌治疗，对药物副作用和效果有疑问。
你想知道治疗需要持续多久，以及如何应对副作用。
请用自然、真诚的方式表达你的疑问。
""",
            "放疗相关": """
你正在接受或即将接受放疗。
你对放疗的过程、副作用和效果有很多疑问。
请用自然、真诚的方式表达你的担忧。
"""
        }
        return templates.get(scenario_type, "")
    
    def get_scenarios(self) -> List[str]:
        """获取所有场景类型"""
        return list(self.scenarios.keys())
    
    def get_scenario_config(self, scenario_type: str) -> Optional[ScenarioConfig]:
        """获取指定场景的配置"""
        return self.scenarios.get(scenario_type)
    
    def get_scenarios_by_keyword(self, keyword: str) -> List[str]:
        """根据关键词搜索场景"""
        matched = []
        for scenario_type, config in self.scenarios.items():
            if any(keyword.lower() in kw.lower() for kw in config.keywords):
                matched.append(scenario_type)
        return matched
    
    def get_random_case(self, scenario_type: Optional[str] = None) -> Optional[ScenarioCase]:
        """获取随机案例"""
        import random
        if scenario_type:
            cases = [c for c in self.all_cases if c.scenario_type == scenario_type]
        else:
            cases = self.all_cases
        
        if cases:
            return random.choice(cases)
        return None
    
    def get_top_cases(self, scenario_type: str, limit: int = 5) -> List[ScenarioCase]:
        """获取场景的典型案例"""
        config = self.scenarios.get(scenario_type)
        if config:
            return config.examples[:limit]
        return []
    
    def generate_prompt(self, scenario_type: str, include_examples: bool = True) -> str:
        """生成场景提示"""
        config = self.scenarios.get(scenario_type)
        if not config:
            return ""
        
        prompt = config.prompt_template
        
        if include_examples and config.examples:
            prompt += "\n\n参考案例：\n"
            for i, example in enumerate(config.examples[:3], 1):
                prompt += f"案例{i}：{example.input_text[:100]}...\n"
        
        return prompt
    
    def save_scenario_config(self, output_path: str):
        """保存场景配置到文件"""
        config_data = {}
        for scenario_type, config in self.scenarios.items():
            config_data[scenario_type] = {
                "description": config.description,
                "keywords": config.keywords,
                "example_count": len(config.examples),
                "prompt_template": config.prompt_template
            }
        
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(config_data, f, ensure_ascii=False, indent=2)
    
    def get_scenario_statistics(self) -> Dict[str, int]:
        """获取各场景案例数量统计"""
        stats = {}
        for scenario_type in self.scenarios:
            stats[scenario_type] = len([c for c in self.all_cases if c.scenario_type == scenario_type])
        return stats

# 全局实例
_scenario_manager = None

def get_scenario_manager() -> ScenarioManager:
    """获取场景管理器单例"""
    global _scenario_manager
    if _scenario_manager is None:
        _scenario_manager = ScenarioManager()
    return _scenario_manager

if __name__ == "__main__":
    # 测试场景管理器
    manager = ScenarioManager()
    
    print("场景管理器测试")
    print("=" * 60)
    
    # 输出场景列表
    scenarios = manager.get_scenarios()
    print(f"识别到 {len(scenarios)} 种场景类型：")
    for scenario in scenarios:
        print(f"  - {scenario}")
    
    print()
    
    # 输出场景统计
    stats = manager.get_scenario_statistics()
    print("场景案例统计：")
    for scenario, count in sorted(stats.items(), key=lambda x: x[1], reverse=True):
        print(f"  {scenario}: {count} 例")
    
    print()
    
    # 测试生成提示
    example_scenario = "性生活与康复"
    prompt = manager.generate_prompt(example_scenario, include_examples=True)
    print(f"【{example_scenario}】场景提示：")
    print(prompt[:500], "...")
    
    # 保存配置
    manager.save_scenario_config("outputs/scenario_config.json")
    print("\n✅ 场景配置已保存到 outputs/scenario_config.json")