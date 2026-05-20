#!/usr/bin/env python3
"""
聊天记录格式化转换脚本

将 dataset/raw_data/ 中的原始聊天记录 JSON 文件
处理并输出到 dataset/formated_data/ 目录。

输入格式: [{"role": "sender_name", "content": "message_content"}, ...]
输出格式: 每条消息作为一个独立的 JSON 对象，包含群组信息、发送者、内容等字段。
"""

import json
import os
import re
import sys
from datetime import datetime

# 路径配置
RAW_DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "dataset", "raw_data")
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "dataset", "formated_data")


def get_group_name(folder_name: str, json_filename: str) -> str:
    """
    从文件夹名和 JSON 文件名推断群组名称。
    
    优先使用 JSON 文件名（去掉扩展名），因为文件名通常包含更有意义的群组名称。
    如果 JSON 文件名是数字 ID 或无意义名称，则使用文件夹名。
    """
    # 去掉 .json 扩展名
    name = os.path.splitext(json_filename)[0]
    
    # 如果文件名是纯数字或太短，使用文件夹名
    if name.isdigit() or len(name) < 3:
        return folder_name
    
    return name


def clean_role_name(role: str) -> str:
    """
    清理发送者名称，去除多余的空格和特殊字符。
    """
    if not role:
        return "unknown"
    
    # 去除首尾空白
    role = role.strip()
    
    # 如果为空，返回 unknown
    if not role:
        return "unknown"
    
    return role


def restore_url(content: str) -> str:
    """
    修复微信导出时被截断的 URL 链接。
    
    微信导出工具会移除 URL 中的 :// 和 . 等字符，导致链接变成乱码。
    例如: "httpmpweixinqqcom..." -> "https://mp.weixin.qq.com/..."
    
    支持的 URL 模式:
    - http(s)开头但缺少 :// 的链接
    - mp.weixin.qq.com 链接
    - 其他常见域名
    """
    if not content:
        return content
    
    result = content
    
    # 1. 修复 httpsmpweixinqqcom... -> https://mp.weixin.qq.com/...
    # 先处理带 s 的，避免被后面的规则误匹配
    result = re.sub(
        r'(?<![a-zA-Z0-9])https(mpweixinqqcom)',
        r'https://mp.weixin.qq.com',
        result
    )
    
    # 2. 修复 httpmpweixinqqcom... -> https://mp.weixin.qq.com/...
    result = re.sub(
        r'(?<![a-zA-Z0-9])http(mpweixinqqcom)',
        r'https://mp.weixin.qq.com',
        result
    )
    
    # 3. 修复其他 http(s)后直接跟域名的链接（域名中至少包含一个点）
    # 例如: httpkgqqcom... -> https://kg.qq.com/...
    # 注意: 域名中的 . 也被移除了，所以我们需要匹配连续的字母数字
    result = re.sub(
        r'(?<![a-zA-Z0-9])https?([a-zA-Z][a-zA-Z0-9]{1,60})(?:com|cn|net|org|edu|gov|io|tv|cc|me|top|xin|club|wang|vip)',
        lambda m: f'https://{m.group(1)}{m.group(0)[-len(m.group(0))+len(m.group(1))+5:]}' if m.group(0)[:5] == 'https' else f'http://{m.group(1)}{m.group(0)[-len(m.group(0))+len(m.group(1))+4:]}',
        result
    )
    
    return result


def clean_content(content: str) -> str:
    """
    清理消息内容，去除多余空白，并修复被截断的 URL。
    """
    if not content:
        return ""
    
    # 去除首尾空白
    content = content.strip()
    
    # 修复被截断的 URL
    content = restore_url(content)
    
    return content


def process_json_file(filepath: str, group_name: str) -> list:
    """
    处理单个 JSON 文件，返回格式化后的消息列表。
    """
    with open(filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    if not isinstance(data, list):
        print(f"  [警告] 文件 {filepath} 不是列表格式，跳过")
        return []
    
    formatted_messages = []
    
    for idx, item in enumerate(data):
        if not isinstance(item, dict):
            continue
        
        role = item.get("role", "")
        content = item.get("content", "")
        
        # 跳过空消息
        if not content and not role:
            continue
        
        # 清理数据
        role_clean = clean_role_name(role)
        content_clean = clean_content(content)
        
        # 构建格式化消息
        formatted_msg = {
            "group": group_name,
            "sender": role_clean,
            "content": content_clean,
            "message_index": idx,
        }
        
        formatted_messages.append(formatted_msg)
    
    return formatted_messages


def process_txt_file(filepath: str, group_name: str) -> list:
    """
    处理 TXT/MD 格式的聊天记录文件。
    
    格式示例:
    2026-05-17 16:25:24 谢探（邵院_护士长）
    【视频号】:
        浙大邵逸夫医院骨科徐文斌：...
    
    2026-05-17 16:26:19 谢探（邵院_护士长）
    本周推送---经常这里那里疼的可以看下...
    """
    formatted_messages = []
    
    with open(filepath, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    msg_index = 0
    i = 0
    while i < len(lines):
        line = lines[i].rstrip('\n')
        
        # 匹配时间戳模式: YYYY-MM-DD HH:MM:SS sender_name
        timestamp_match = re.match(r'^(\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}:\d{2})\s+(.+)$', line)
        
        if timestamp_match:
            timestamp = timestamp_match.group(1)
            sender = timestamp_match.group(2).strip()
            
            # 收集属于此消息的所有内容，直到遇到下一个时间戳或文件结束
            message_content_lines = []
            i += 1
            
            # 继续读取行，直到遇到下一个时间戳或文件结束
            while i < len(lines):
                next_line = lines[i].rstrip('\n')
                
                # 检查下一行是否是新的时间戳
                if re.match(r'^\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}:\d{2}', next_line):
                    break
                
                # 添加当前行到消息内容
                message_content_lines.append(next_line)
                i += 1
            
            # 将内容行合并为单个字符串，保留原有的换行
            message_content = '\n'.join(message_content_lines).strip()
            
            if message_content or sender:
                formatted_msg = {
                    "group": group_name,
                    "sender": clean_role_name(sender),
                    "content": clean_content(message_content),
                    "timestamp": timestamp,
                    "message_index": msg_index,
                }
                
                formatted_messages.append(formatted_msg)
                msg_index += 1
        else:
            # 如果第一行不是时间戳，跳过
            i += 1
    
    return formatted_messages


def save_formatted_data(all_messages: list, output_filename: str):
    """
    保存格式化后的数据到输出目录。
    """
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    output_path = os.path.join(OUTPUT_DIR, output_filename)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(all_messages, f, ensure_ascii=False, indent=2)
    
    print(f"  [输出] 已保存 {len(all_messages)} 条消息到 {output_path}")
    return output_path


def main():
    print("=" * 60)
    print("聊天记录格式化转换工具")
    print(f"输入目录: {RAW_DATA_DIR}")
    print(f"输出目录: {OUTPUT_DIR}")
    print("=" * 60)
    
    # 确保输出目录存在
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # 收集所有需要处理的文件
    all_formatted_messages = []
    group_stats = {}
    
    # 遍历 raw_data 目录下的所有文件夹
    for folder_name in sorted(os.listdir(RAW_DATA_DIR)):
        folder_path = os.path.join(RAW_DATA_DIR, folder_name)
        
        if not os.path.isdir(folder_path):
            continue
        
        print(f"\n[处理] 文件夹: {folder_name}")
        
        # 查找 JSON 文件（排除 _dev.json 和 _train.json）
        json_files = sorted([
            f for f in os.listdir(folder_path)
            if f.endswith('.json')
            and not f.endswith('_dev.json')
            and not f.endswith('_train.json')
        ])
        
        # 查找 TXT/MD 文件
        text_files = sorted([
            f for f in os.listdir(folder_path)
            if f.endswith('.txt') or f.endswith('.md')
        ])
        
        folder_messages = []
        
        # 处理 JSON 文件
        for json_file in json_files:
            filepath = os.path.join(folder_path, json_file)
            group_name = get_group_name(folder_name, json_file)
            
            print(f"  [JSON] 处理文件: {json_file}")
            print(f"  [JSON] 群组名称: {group_name}")
            
            messages = process_json_file(filepath, group_name)
            folder_messages.extend(messages)
            print(f"  [JSON] 提取 {len(messages)} 条消息")
        
        # 处理 TXT/MD 文件 - 仅当该文件夹没有 JSON 文件时才处理
        # 因为 JSON 文件已经包含了完整的聊天记录，TXT/MD 文件可能包含
        # 混合格式（原始格式 + Markdown 格式），直接解析会导致大量错误
        if not json_files:
            for text_file in text_files:
                filepath = os.path.join(folder_path, text_file)
                group_name = get_group_name(folder_name, text_file)
                
                print(f"  [TXT] 处理文件: {text_file}")
                print(f"  [TXT] 群组名称: {group_name}")
                
                messages = process_txt_file(filepath, group_name)
                folder_messages.extend(messages)
                print(f"  [TXT] 提取 {len(messages)} 条消息")
        else:
            if text_files:
                print(f"  [跳过] 已存在 JSON 文件，跳过 TXT/MD 文件: {', '.join(text_files)}")
        
        if folder_messages:
            # 去重：同一群组内按 (sender, content) 去重
            seen_in_group = set()
            unique_messages = []
            for msg in folder_messages:
                key = (msg["sender"], msg["content"])
                if key not in seen_in_group:
                    seen_in_group.add(key)
                    unique_messages.append(msg)
            
            dup_count = len(folder_messages) - len(unique_messages)
            group_stats[folder_name] = len(unique_messages)
            all_formatted_messages.extend(unique_messages)
            print(f"  [完成] 共提取 {len(folder_messages)} 条消息（去重后 {len(unique_messages)} 条，去除 {dup_count} 条重复）")
        else:
            print(f"  [跳过] 未找到可处理的数据文件")
    
    # 保存汇总数据
    print("\n" + "=" * 60)
    print("处理完成！")
    print(f"总消息数: {len(all_formatted_messages)}")
    print("\n各群组统计:")
    for group, count in sorted(group_stats.items(), key=lambda x: x[1], reverse=True):
        print(f"  {group}: {count} 条消息")
    
    # 保存所有数据到一个文件
    if all_formatted_messages:
        save_formatted_data(all_formatted_messages, "all_chat_records.json")
    
    # 按群组分别保存
    print("\n按群组分别保存:")
    group_messages = {}
    for msg in all_formatted_messages:
        group_key = msg["group"]
        if group_key not in group_messages:
            group_messages[group_key] = []
        group_messages[group_key].append(msg)
    
    for group_name, messages in sorted(group_messages.items()):
        # 生成安全的文件名
        safe_filename = re.sub(r'[\\/*?:"<>|]', '_', group_name) + ".json"
        save_formatted_data(messages, safe_filename)
    
    print("\n" + "=" * 60)
    print("所有处理完成！")
    print(f"输出目录: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
