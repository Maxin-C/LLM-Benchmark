#!/usr/bin/env python3
"""
使用LLM优化指南PDF处理脚本
将PDF格式的临床指南转换为结构化的JSON格式
"""

import os
import json
import pdfplumber
from typing import Dict, List, Any

def extract_text_from_pdf(pdf_path: str) -> str:
    """
    从PDF文件中提取文本
    
    参数：
        pdf_path: PDF文件路径
    
    返回：
        提取的文本内容
    """
    text = ""
    try:
        with pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
    except Exception as e:
        print(f"读取PDF文件失败：{e}")
        return ""
    
    return text

def create_llm_prompt(pdf_title: str, raw_text: str) -> str:
    """
    创建LLM处理提示词
    
    参数：
        pdf_title: PDF标题
        raw_text: 原始文本
    
    返回：
        LLM提示词
    """
    prompt = f"""
你是一个专业的医学指南结构化处理助手。请将以下指南内容转换为结构化的JSON格式。

指南标题：{pdf_title}

要求：
1. 提取指南的完整标题、版本号、发布机构等信息
2. 将内容按照逻辑结构划分为多个章节，每个章节包含：
   - section_name: 章节标题（简洁明了）
   - content: 章节内容（完整、准确）
3. 确保章节划分合理，内容完整
4. 去除页眉、页脚、页码等无关信息
5. 保持医学术语的准确性
6. 使用中文输出

请按照以下JSON格式输出：
{{
    "guideline_id": "指南ID（可以留空）",
    "title": "完整标题",
    "version": "版本号",
    "source": "发布机构/来源",
    "sections": [
        {{
            "section_name": "章节标题1",
            "content": "章节内容1"
        }},
        {{
            "section_name": "章节标题2", 
            "content": "章节内容2"
        }}
    ]
}}

原始文本内容：
{raw_text[:5000]}  # 限制长度，避免超出token限制

请输出JSON格式的结果：
"""
    return prompt

def process_guideline_with_llm(pdf_path: str, output_path: str, llm_client=None):
    """
    使用LLM处理指南PDF文件
    
    参数：
        pdf_path: PDF文件路径
        output_path: 输出JSON文件路径
        llm_client: LLM客户端（可选）
    """
    print(f"处理文件：{pdf_path}")
    
    # 提取文件名作为标题
    filename = os.path.basename(pdf_path)
    title = filename.replace('.pdf', '')
    
    # 提取文本
    raw_text = extract_text_from_pdf(pdf_path)
    
    if not raw_text:
        print(f"警告：无法从 {pdf_path} 提取文本")
        return
    
    # 如果没有LLM客户端，使用简单的规则处理
    if not llm_client:
        print("使用简单规则处理（建议使用LLM优化）...")
        guideline = simple_process(raw_text, title)
    else:
        # 使用LLM处理
        print("使用LLM处理...")
        prompt = create_llm_prompt(title, raw_text)
        response = llm_client.chat_completion(
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3
        )
        
        try:
            # 解析LLM返回的JSON
            guideline_text = response.choices[0].message.content
            # 提取JSON部分
            start_idx = guideline_text.find('{')
            end_idx = guideline_text.rfind('}') + 1
            if start_idx >= 0 and end_idx > start_idx:
                json_str = guideline_text[start_idx:end_idx]
                guideline = json.loads(json_str)
            else:
                guideline = simple_process(raw_text, title)
        except Exception as e:
            print(f"LLM处理失败，使用简单规则：{e}")
            guideline = simple_process(raw_text, title)
    
    # 保存为JSON
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(guideline, f, ensure_ascii=False, indent=2)
    
    print(f"保存到：{output_path}")
    print(f"提取了 {len(guideline.get('sections', []))} 个章节")

def simple_process(raw_text: str, title: str) -> Dict[str, Any]:
    """
    简单的规则处理（备用方案）
    
    参数：
        raw_text: 原始文本
        title: 标题
    
    返回：
        结构化的指南数据
    """
    guideline = {
        "guideline_id": "",
        "title": title,
        "version": "",
        "source": "",
        "sections": []
    }
    
    # 简单的文本分割逻辑
    lines = raw_text.split('\n')
    current_section = None
    current_content = []
    
    for line in lines:
        line = line.strip()
        if not line:
            continue
        
        # 检测章节标题（简单的启发式规则）
        if (line.endswith(('：', '。', '.')) or 
            any(keyword in line for keyword in ['诊断', '治疗', '分期', '随访', '原则', '方案', '适应症', '禁忌症', '共识', '策略', '推荐'])):
            # 保存上一个章节
            if current_section and current_content:
                guideline["sections"].append({
                    "section_name": current_section,
                    "content": " ".join(current_content)
                })
            
            current_section = line
            current_content = []
        else:
            if current_section:
                current_content.append(line)
    
    # 保存最后一个章节
    if current_section and current_content:
        guideline["sections"].append({
            "section_name": current_section,
            "content": " ".join(current_content)
        })
    
    # 如果没有检测到章节，将整个文本作为一个章节
    if not guideline["sections"]:
        guideline["sections"].append({
            "section_name": "概述",
            "content": raw_text
        })
    
    return guideline

def main():
    # 输入和输出目录
    input_dir = "dataset/guidlines/original_pdf"
    output_dir = "dataset/guidlines/formated_data"
    
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)
    
    # 获取所有PDF文件
    pdf_files = [f for f in os.listdir(input_dir) if f.endswith('.pdf')]
    
    print(f"找到 {len(pdf_files)} 个PDF文件")
    
    # 尝试初始化 LLM 客户端（可选）
    llm_client = None
    try:
        import sys
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
        from src.utils.llm_client import LLMClient
        llm_client = LLMClient()
        print("LLM 客户端初始化成功")
    except Exception as e:
        print(f"LLM 客户端初始化失败，将使用简单规则处理：{e}")
    
    # 处理每个PDF文件
    for pdf_file in pdf_files:
        pdf_path = os.path.join(input_dir, pdf_file)
        output_file = pdf_file.replace('.pdf', '.json')
        output_path = os.path.join(output_dir, output_file)
        
        process_guideline_with_llm(pdf_path, output_path, llm_client)
    
    print("\n所有文件处理完成！")

if __name__ == "__main__":
    main()
