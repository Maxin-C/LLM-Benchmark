"""
测试配置是否正确
验证各角色的模型配置
"""

import os
import sys
from dotenv import load_dotenv

# 加载环境变量
load_dotenv()

# 加载配置
def load_config(config_path: str):
    import yaml
    import re
    
    with open(config_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    def replace_env_var(match):
        var_name = match.group(1)
        return os.getenv(var_name, match.group(0))
    
    content = re.sub(r'\$\{(\w+)\}', replace_env_var, content)
    return yaml.safe_load(content)

config = load_config('config/sandbox_config.yaml')

print("="*60)
print("          配置验证报告")
print("="*60)

# 虚拟患者配置
vp = config['virtual_patient']
print("\n【虚拟患者】（固定模型）")
print(f"  模型: {vp['model']}")
print(f"  API: {vp['base_url']}")
print(f"  Thinking模式: {'启用' if vp.get('thinking_enabled') else '禁用'}")
print(f"  Reasoning Effort: {vp.get('reasoning_effort', 'N/A')}")

# 虚拟医生配置
doctor = config['virtual_doctor']
print("\n【虚拟医生】（待测试模型）")
print(f"  模型: {doctor['model']}")
print(f"  API: {doctor['base_url']}")
print(f"  Thinking模式: {'启用' if doctor.get('thinking_enabled') else '禁用'}")

# 评估器配置
judger = config['judger']
print("\n【评估器】")
print(f"  模型: {judger['model']}")
print(f"  API: {judger['base_url']}")

# 监控器配置
monitor = config['dialogue_monitor']
print("\n【监控器】")
print(f"  模型: {monitor['model']}")
print(f"  API: {monitor['base_url']}")

print("\n" + "="*60)
print("配置验证完成！")
print("="*60)