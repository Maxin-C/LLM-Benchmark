import re
import json
from typing import List, Dict


def clean_content(content: str) -> str:
    """
    清理消息内容，去除噪声
    """
    if not content:
        return ""
    
    # 去除 @提及
    content = re.sub(r'@[\u4e00-\u9fa5a-zA-Z0-9_-]+\s*', '', content)
    
    # 去除表情符号（简单处理）
    content = re.sub(r'[^\u4e00-\u9fa5a-zA-Z0-9，。！？、；：""''（）《》【】\s]', '', content)
    
    # 去除多余空格和换行
    content = ' '.join(content.split())
    
    return content.strip()


def is_medical_staff(sender: str) -> bool:
    """
    判断发送者是否为医护人员
    """
    staff_keywords = ['护士长', '护士', '医生', '主任', '护师', '医院']
    return any(keyword in sender for keyword in staff_keywords)


def filter_patient_messages(messages: List[Dict]) -> List[Dict]:
    """
    过滤出患者的消息
    """
    return [msg for msg in messages if not is_medical_staff(msg['sender'])]


def load_json_file(filepath: str) -> List[Dict]:
    """
    加载JSON文件
    """
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)
