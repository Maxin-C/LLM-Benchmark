"""LLM客户端模块 - 统一封装LLM调用"""

from openai import OpenAI
import json
import logging
from typing import Dict, List, Any, Optional

logger = logging.getLogger(__name__)

class LLMClient:
    """LLM客户端封装类"""
    
    def __init__(self, api_key: str, base_url: str = "https://api.pumpkinaigc.online/v1", 
                 model: str = "gpt-4o", temperature: float = 0.7, max_tokens: int = 512):
        """
        初始化LLM客户端
        
        Args:
            api_key: API密钥
            base_url: API基础URL
            model: 使用的模型名称
            temperature: 温度参数，控制输出随机性
            max_tokens: 最大输出token数
        """
        self.client = OpenAI(
            api_key=api_key,
            base_url=base_url
        )
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
    
    def chat(self, system_prompt: str, user_prompt: str, 
             temperature: Optional[float] = None, 
             max_tokens: Optional[int] = None) -> str:
        """
        调用LLM进行对话
        
        Args:
            system_prompt: 系统提示词，定义角色和规则
            user_prompt: 用户输入
            temperature: 温度参数（可选，覆盖默认值）
            max_tokens: 最大token数（可选，覆盖默认值）
        
        Returns:
            LLM的回复内容
        """
        try:
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ]
            
            response = self.client.chat.completions.create(
                messages=messages,
                model=self.model,
                temperature=temperature if temperature is not None else self.temperature,
                max_tokens=max_tokens if max_tokens is not None else self.max_tokens
            )
            
            return response.choices[0].message.content.strip()
        except Exception as e:
            logger.error(f"LLM调用失败: {str(e)}")
            return ""
    
    def chat_json(self, system_prompt: str, user_prompt: str,
                  temperature: float = 0.1) -> Dict[str, Any]:
        """
        调用LLM并返回JSON格式结果
        
        Args:
            system_prompt: 系统提示词
            user_prompt: 用户输入
            temperature: 温度参数（默认0.1，保证输出稳定性）
        
        Returns:
            解析后的JSON字典
        """
        try:
            response = self.chat(system_prompt, user_prompt, temperature=temperature)
            return json.loads(response)
        except json.JSONDecodeError:
            logger.error(f"LLM返回非JSON格式: {response}")
            return {}
        except Exception as e:
            logger.error(f"LLM调用失败: {str(e)}")
            return {}
    
    def batch_chat(self, system_prompt: str, prompts: List[str]) -> List[str]:
        """
        批量调用LLM
        
        Args:
            system_prompt: 系统提示词
            prompts: 用户输入列表
        
        Returns:
            回复列表
        """
        results = []
        for prompt in prompts:
            result = self.chat(system_prompt, prompt)
            results.append(result)
        return results

class LLMClientSingleton:
    """LLM客户端单例类"""
    
    _instance = None
    
    @classmethod
    def get_instance(cls, api_key: Optional[str] = None, **kwargs) -> LLMClient:
        """
        获取或创建LLM客户端实例
        
        Args:
            api_key: API密钥
            **kwargs: 其他参数
            
        Returns:
            LLMClient实例
        """
        if cls._instance is None:
            if api_key is None:
                raise ValueError("必须提供API密钥")
            cls._instance = LLMClient(api_key, **kwargs)
        return cls._instance
    
    @classmethod
    def reset_instance(cls):
        """重置单例实例"""
        cls._instance = None
