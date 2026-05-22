#!/usr/bin/env python3
"""
使用 LLM 智能优化指南 JSON 文件

这个脚本会：
1. 读取已生成的指南 JSON 文件
2. 使用 LLM 理解内容并重新结构化
3. 生成清晰、准确的章节划分
"""

import os
import json
import sys
from typing import Dict, List, Any

# 加载环境变量
from dotenv import load_dotenv
load_dotenv()

def load_json_file(file_path: str) -> Dict[str, Any]:
    """加载 JSON 文件"""
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def save_json_file(file_path: str, data: Dict[str, Any]):
    """保存 JSON 文件"""
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

def create_optimization_prompt(guideline: Dict[str, Any]) -> str:
    """
    创建 LLM 优化提示词
    
    参数：
        guideline: 原始指南数据
    
    返回：
        LLM 提示词
    """
    title = guideline.get('title', '')
    sections = guideline.get('sections', [])
    
    # 准备内容预览（限制总长度）
    content_preview = ""
    total_chars = 0
    max_chars = 8000  # 限制总字符数
    
    for i, section in enumerate(sections[:50]):  # 最多取前 50 个章节
        section_text = f"{section.get('section_name', '')}: {section.get('content', '')}\n\n"
        if total_chars + len(section_text) > max_chars:
            break
        content_preview += section_text
        total_chars += len(section_text)
    
    prompt = f"""你是一个专业的医学指南结构化处理专家。请对以下指南内容进行智能优化和重新结构化。

## 指南信息
标题：{title}

## 当前问题
目前的章节划分存在以下问题：
1. 章节标题包含页眉、页脚等无关信息
2. 章节划分过细，很多相关内容被拆分
3. 部分章节标题不能准确反映内容

## 原始内容预览
{content_preview}

## 任务要求
请重新组织这份指南，要求：

1. **提取核心章节**：识别指南的主要结构，提取 5-15 个核心章节
2. **清理章节标题**：确保标题简洁明了（不超过 30 个字），准确反映内容
3. **合并相关内容**：将相关的短章节合并为完整的章节
4. **去除无关信息**：删除页眉、页脚、页码、参考文献列表等
5. **保持医学准确性**：确保医学术语准确，内容完整

## 输出格式
请严格按照以下 JSON 格式输出：

```json
{{
    "guideline_id": "",
    "title": "完整标题",
    "version": "版本号（如 2015 版、2022 版）",
    "source": "发布机构或来源",
    "summary": "指南内容简介（100-200 字）",
    "sections": [
        {{"section_name": "清晰的章节标题 1", "content": "完整的章节内容 1，合并了相关内容，去除了无关信息"}},
        {{"section_name": "清晰的章节标题 2", "content": "完整的章节内容 2"}}
    ]
}}
```

## 注意事项
- 章节数量控制在 5-15 个之间
- 每个章节的内容应该完整、连贯
- 章节标题应该能够准确概括内容
- 保持医学术语的专业性和准确性
- 使用中文输出

请输出优化后的 JSON 结构："""
    
    return prompt

def optimize_guideline_with_llm(guideline: Dict[str, Any], llm_client, model: str) -> Dict[str, Any]:
    """
    使用 LLM 优化指南
    
    参数：
        guideline: 原始指南数据
        llm_client: OpenAI 客户端
        model: 模型名称
    
    返回：
        优化后的指南数据
    """
    prompt = create_optimization_prompt(guideline)
    
    try:
        print("  正在调用 LLM 进行优化...")
        
        response = llm_client.chat.completions.create(
            messages=[
                {"role": "system", "content": "你是一个专业的医学指南结构化处理专家，擅长将非结构化文本转换为结构化格式。"},
                {"role": "user", "content": prompt}
            ],
            model=model,
            temperature=0.3,
            max_tokens=4000
        )
        
        response_text = response.choices[0].message.content
        print(f"  LLM 响应长度：{len(response_text)} 字符")
        
        # 提取 JSON 部分
        start_idx = response_text.find('{')
        end_idx = response_text.rfind('}') + 1
        
        if start_idx >= 0 and end_idx > start_idx:
            json_str = response_text[start_idx:end_idx]
            optimized = json.loads(json_str)
            return optimized
        else:
            print("  警告：无法从响应中提取 JSON，使用规则优化")
            return simple_optimize(guideline)
            
    except Exception as e:
        print(f"  LLM 优化失败：{e}")
        return simple_optimize(guideline)

def process_all_guidelines(input_dir: str, output_dir: str):
    """
    处理所有指南文件
    
    参数：
        input_dir: 输入目录
        output_dir: 输出目录
    """
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)
    
    # 获取所有 JSON 文件
    json_files = [f for f in os.listdir(input_dir) if f.endswith('.json')]
    
    print(f"找到 {len(json_files)} 个指南文件")
    
    # 从环境变量获取配置
    api_key = os.getenv('EASE_LLM_API_KEY')
    base_url = os.getenv('EASE_LLM_BASE_URL', 'https://api.pumpkinaigc.online/v1')
    model = os.getenv('EASE_LLM_MODEL', 'gpt-4o')
    
    print(f"API Key: {'已配置' if api_key else '未配置'}")
    print(f"Base URL: {base_url}")
    print(f"Model: {model}")
    
    # 初始化 OpenAI 客户端
    llm_client = None
    if api_key:
        try:
            from openai import OpenAI
            llm_client = OpenAI(api_key=api_key, base_url=base_url)
            print("✓ LLM 客户端初始化成功\n")
        except Exception as e:
            print(f"✗ LLM 客户端初始化失败：{e}")
            print("将使用规则优化方法\n")
    else:
        print("✗ 未配置 API Key")
        print("将使用规则优化方法\n")
    
    # 处理每个文件
    for json_file in json_files:
        input_path = os.path.join(input_dir, json_file)
        output_path = os.path.join(output_dir, json_file)
        
        print(f"\n{'='*60}")
        print(f"处理文件：{json_file}")
        print(f"{'='*60}")
        
        # 加载原始数据
        guideline = load_json_file(input_path)
        original_sections = len(guideline.get('sections', []))
        print(f"原始章节数：{original_sections}")
        
        # 优化
        if llm_client:
            optimized = optimize_guideline_with_llm(guideline, llm_client, model)
        else:
            # 使用规则优化（备用方案）
            optimized = simple_optimize(guideline)
        
        optimized_sections = len(optimized.get('sections', []))
        print(f"优化后章节数：{optimized_sections}")
        
        # 保存
        save_json_file(output_path, optimized)
        print(f"✓ 已保存到：{output_path}")

def simple_optimize(guideline: Dict[str, Any]) -> Dict[str, Any]:
    """
    简单的规则优化（备用方案）
    
    参数：
        guideline: 原始指南数据
    
    返回：
        优化后的指南数据
    """
    optimized = guideline.copy()
    sections = guideline.get('sections', [])
    
    # 合并内容过短的章节
    merged = []
    current_section = None
    
    for section in sections:
        content = section.get('content', '')
        section_name = section.get('section_name', '')
        
        # 如果章节标题很短或内容很短，尝试合并
        if len(section_name) < 20 or len(content) < 100:
            if current_section:
                current_section['content'] += '\n' + content
            else:
                current_section = section.copy()
        else:
            if current_section:
                merged.append(current_section)
            current_section = section.copy()
    
    if current_section:
        merged.append(current_section)
    
    # 过滤掉内容过短的章节
    filtered = [s for s in merged if len(s.get('content', '')) > 200]
    
    optimized['sections'] = filtered
    
    # 尝试从标题中提取版本信息
    title = optimized.get('title', '')
    if '2022' in title:
        optimized['version'] = '2022 版'
    elif '2015' in title:
        optimized['version'] = '2015 版'
    
    # 尝试从标题中提取来源
    if '专家共识' in title:
        optimized['source'] = '中国乳腺癌领域专家'
    elif '指南' in title:
        optimized['source'] = '国家卫生健康委员会'
    
    return optimized

def main():
    # 输入和输出目录
    input_dir = "dataset/guidlines/formated_data"
    output_dir = "dataset/guidlines/formated_data_llm_optimized"
    
    print("="*60)
    print("使用 LLM 智能优化指南 JSON 文件")
    print("="*60)
    print()
    
    # 删除之前处理失败的旧文件
    if os.path.exists(output_dir):
        import shutil
        shutil.rmtree(output_dir)
        print(f"已删除旧的优化目录：{output_dir}")
    
    process_all_guidelines(input_dir, output_dir)
    
    print("\n" + "="*60)
    print("优化完成！")
    print("="*60)

if __name__ == "__main__":
    main()
