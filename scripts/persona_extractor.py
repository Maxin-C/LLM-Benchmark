#!/usr/bin/env python3
"""
患者Persona提取工具

从格式化的聊天记录中提取患者特征，构建虚拟患者Persona。
"""

import json
import os
import sys
import glob

# 添加项目根目录到Python路径
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from tqdm import tqdm
from src.persona.builder import PersonaBuilder


def main():
    print("=" * 60)
    print("患者Persona提取工具")
    print("=" * 60)
    
    # 获取项目根目录
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    # 配置路径
    input_dir = os.path.join(project_root, "dataset", "formated_data")
    output_dir = os.path.join(project_root, "dataset", "persona_data")
    
    print(f"输入目录: {input_dir}")
    print(f"输出目录: {output_dir}")
    print()
    
    # 查找所有JSON文件
    json_files = sorted(glob.glob(os.path.join(input_dir, "*.json")))
    
    if not json_files:
        print("错误: 未找到任何JSON文件")
        return
    
    print(f"找到 {len(json_files)} 个JSON文件:")
    for f in json_files:
        print(f"  - {os.path.basename(f)}")
    print()
    
    # 构建Persona
    builder = PersonaBuilder(output_dir)
    
    print("开始提取Persona...")
    result = builder.build_from_files(json_files)
    
    print(f"\n提取完成！")
    print(f"总患者数: {result['total_patients']}")
    print(f"总消息数: {result['total_messages']}")
    
    # 保存Persona数据
    print("\n保存Persona数据...")
    persona_path = builder.save_personas(result)
    print(f"  - Persona数据已保存到: {persona_path}")
    
    # 生成并保存知识图谱
    print("\n生成知识图谱...")
    graph = builder.generate_knowledge_graph(result['personas'])
    graph_path = builder.save_knowledge_graph(graph)
    print(f"  - 知识图谱已保存到: {graph_path}")
    
    # 输出统计信息
    print("\n" + "=" * 60)
    print("分析报告")
    print("=" * 60)
    
    # 常见症状
    print("\n[常见症状]")
    for symptom, count in result['analysis']['common_symptoms'][:10]:
        print(f"  {symptom}: {count} 次提及")
    
    # 常见关注点
    print("\n[常见关注点]")
    for concern, count in result['analysis']['common_concerns'][:10]:
        print(f"  {concern}: {count} 次提及")
    
    # 问题意图分布
    print("\n[问题意图分布]")
    for intent, count in result['analysis']['patient_question_frequency']:
        print(f"  {intent}: {count} 次")
    
    # 响应率
    print(f"\n[医护响应率] {result['analysis']['staff_response_rate']:.2%}")
    
    print("\n" + "=" * 60)
    print("所有处理完成！")
    print("=" * 60)


if __name__ == "__main__":
    main()
