#!/usr/bin/env python3
"""
LLM客户端工厂 - 根据配置创建不同角色的LLM客户端
支持：虚拟患者(待评估模型)、虚拟医生(deepseek-v4-pro)、评估器(deepseek-v4-pro)、监控器(deepseek-v4-pro)
支持reasoning模型的thinking输出记录（DeepSeek官方API支持thinking模式）
支持本地模型部署（如Qwen3-0.6B）
"""

import os
import torch
from typing import Dict, Any, Optional, Tuple
from openai import OpenAI

class _LocalChatCompletions:
    """
    模拟OpenAI的chat.completions接口
    """
    
    def __init__(self, client):
        self._client = client
    
    def create(self, messages: list, model: str = None, **kwargs) -> Any:
        """
        模拟OpenAI的chat.completions.create接口
        
        Args:
            messages: 消息列表
            model: 模型名称
            **kwargs: 其他参数
        
        Returns:
            模拟的OpenAI响应对象
        """
        response_text = self._client.generate(messages, **kwargs)
        
        # 创建模拟的响应对象
        class Choice:
            class Message:
                content: str = response_text
            
            message = Message()
        
        class ChatCompletion:
            choices = [Choice()]
        
        return ChatCompletion()


class _LocalChat:
    """
    模拟OpenAI的chat对象
    """
    
    def __init__(self, client):
        self.completions = _LocalChatCompletions(client)


class LocalModelClient:
    """
    本地模型客户端类
    用于加载和调用本地部署的LLM模型
    兼容OpenAI API接口: client.chat.completions.create()
    """
    
    def __init__(self, model_path: str, device: str = 'cuda:0'):
        """
        初始化本地模型客户端
        
        Args:
            model_path: 本地模型路径
            device: 运行设备（默认cuda:0）
        """
        self.model_path = model_path
        self.device = device
        
        # 延迟加载模型
        self._model = None
        self._tokenizer = None
        
        # 创建兼容的chat对象（包含completions）
        self.chat = _LocalChat(self)
    
    def _load_model(self):
        """延迟加载模型和tokenizer"""
        if self._model is None:
            from transformers import AutoModelForCausalLM, AutoTokenizer
            
            print(f"正在加载本地模型: {self.model_path}")
            self._tokenizer = AutoTokenizer.from_pretrained(self.model_path, trust_remote_code=True)
            self._model = AutoModelForCausalLM.from_pretrained(
                self.model_path,
                torch_dtype=torch.float16,
                trust_remote_code=True
            ).to(self.device)
            print("本地模型加载完成")
    
    def generate(self, messages: list, **kwargs) -> str:
        """
        调用本地模型进行对话
        
        Args:
            messages: 消息列表，格式为 [{"role": "user", "content": "..."}]
            **kwargs: 其他参数（temperature, max_tokens等）
        
        Returns:
            模型回复内容
        """
        self._load_model()
        
        temperature = kwargs.get('temperature', 0.7)
        max_tokens = kwargs.get('max_tokens', 512)
        
        # 格式化输入
        text = self._tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        
        # 编码输入
        model_inputs = self._tokenizer([text], return_tensors='pt').to(self.device)
        
        # 生成回复
        with torch.no_grad():
            generated_ids = self._model.generate(
                **model_inputs,
                max_new_tokens=max_tokens,
                temperature=temperature
            )
        
        # 解码输出（只解码新生成的部分）
        generated_ids = generated_ids[0][len(model_inputs.input_ids[0]):]
        output = self._tokenizer.decode(generated_ids, skip_special_tokens=True)
        
        # 清理输出中的特殊标签
        output = output.replace('<think>', '').replace('</think>', '').strip()
        
        return output


class LLMClientFactory:
    """
    LLM客户端工厂类
    根据不同角色创建对应的LLM客户端
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
    
    def create_virtual_patient_client(self) -> OpenAI:
        """
        创建虚拟患者客户端（待评估模型）
        
        返回：
            OpenAI客户端实例
        """
        vp_config = self.config.get('virtual_patient', {})
        return OpenAI(
            api_key=vp_config.get('api', {}).get('api_key') or os.getenv('EASE_VP_API_KEY'),
            base_url=vp_config.get('api', {}).get('base_url') or os.getenv('EASE_VP_BASE_URL')
        )
    
    def create_virtual_doctor_client(self) -> OpenAI:
        """
        创建虚拟医生客户端（待测试模型）
        
        返回：
            OpenAI客户端实例或本地模型客户端
        """
        doctor_config = self.config.get('virtual_doctor', {})
        
        # 检查是否为本地模型配置
        if 'model_type' in doctor_config and doctor_config['model_type'] == 'local':
            local_config = doctor_config.get('local', {})
            model_path = local_config.get('model_path') or os.getenv('EASE_DOCTOR_LOCAL_PATH')
            device = local_config.get('device', 'cuda:0')
            return LocalModelClient(model_path, device)
        
        # 默认使用API模式
        return OpenAI(
            api_key=doctor_config.get('api', {}).get('api_key') or os.getenv('EASE_DOCTOR_API_KEY'),
            base_url=doctor_config.get('api', {}).get('base_url') or os.getenv('EASE_DOCTOR_BASE_URL')
        )
    
    def create_judger_client(self) -> OpenAI:
        """
        创建评估器客户端（使用deepseek-v4-pro）
        
        返回：
            OpenAI客户端实例
        """
        judger_config = self.config.get('judger', {})
        return OpenAI(
            api_key=judger_config.get('api', {}).get('api_key') or os.getenv('EASE_JUDGER_API_KEY'),
            base_url=judger_config.get('api', {}).get('base_url') or os.getenv('EASE_JUDGER_BASE_URL')
        )
    
    def create_monitor_client(self) -> OpenAI:
        """
        创建对话监控器客户端（使用deepseek-v4-pro）
        
        返回：
            OpenAI客户端实例
        """
        monitor_config = self.config.get('dialogue_monitor', {})
        return OpenAI(
            api_key=monitor_config.get('api', {}).get('api_key') or os.getenv('EASE_MONITOR_API_KEY'),
            base_url=monitor_config.get('api', {}).get('base_url') or os.getenv('EASE_MONITOR_BASE_URL')
        )
    
    def get_virtual_patient_model(self) -> str:
        """获取虚拟患者使用的模型名称"""
        vp_config = self.config.get('virtual_patient', {})
        return vp_config.get('model') or os.getenv('EASE_VP_MODEL', 'gpt-4o')
    
    def is_virtual_patient_reasoning(self) -> bool:
        """判断虚拟患者是否为reasoning模型"""
        vp_config = self.config.get('virtual_patient', {})
        is_reasoning = vp_config.get('is_reasoning') or os.getenv('EASE_VP_IS_REASONING', 'false')
        return str(is_reasoning).lower() == 'true'
    
    def get_virtual_doctor_model(self) -> str:
        """获取虚拟医生使用的模型名称"""
        doctor_config = self.config.get('virtual_doctor', {})
        
        # 检查是否为本地模型配置
        if 'model_type' in doctor_config and doctor_config['model_type'] == 'local':
            local_config = doctor_config.get('local', {})
            return local_config.get('model_name', 'local-model')
        
        # 默认使用API模式
        return doctor_config.get('model', 'deepseek-v4-pro')
    
    def get_judger_model(self) -> str:
        """获取评估器使用的模型名称（deepseek-v4-pro）"""
        judger_config = self.config.get('judger', {})
        return judger_config.get('model', 'deepseek-v4-pro')
    
    def get_monitor_model(self) -> str:
        """获取监控器使用的模型名称（deepseek-v4-pro）"""
        monitor_config = self.config.get('dialogue_monitor', {})
        return monitor_config.get('model', 'deepseek-v4-pro')
    
    def get_doctor_params(self) -> Dict[str, Any]:
        """获取医生模型的参数配置（支持thinking模式）"""
        doctor_config = self.config.get('virtual_doctor', {})
        params = {
            'temperature': doctor_config.get('temperature', 0.8),
            'presence_penalty': doctor_config.get('presence_penalty', 0),
            'frequency_penalty': doctor_config.get('frequency_penalty', 0),
            'top_p': doctor_config.get('top_p', 0.9)
        }
        
        # 如果启用了thinking模式，添加相关参数
        if doctor_config.get('thinking_enabled', False):
            params['reasoning_effort'] = doctor_config.get('reasoning_effort', 'high')
            params['extra_body'] = {'thinking': {'type': 'enabled'}}
        
        return params
    
    def get_judger_params(self) -> Dict[str, Any]:
        """获取评估器模型的参数配置"""
        judger_config = self.config.get('judger', {})
        return {
            'temperature': judger_config.get('temperature', 0.1),
            'presence_penalty': judger_config.get('presence_penalty', 0),
            'frequency_penalty': judger_config.get('frequency_penalty', 0),
            'top_p': judger_config.get('top_p', 1.0)
        }
    
    def get_monitor_params(self) -> Dict[str, Any]:
        """获取监控器模型的参数配置"""
        monitor_config = self.config.get('dialogue_monitor', {})
        return {
            'max_tokens': monitor_config.get('max_tokens', 10),
            'temperature': monitor_config.get('temperature', 0.1),
            'presence_penalty': monitor_config.get('presence_penalty', 0),
            'frequency_penalty': monitor_config.get('frequency_penalty', 0),
            'top_p': monitor_config.get('top_p', 1.0)
        }
    
    def get_virtual_patient_params(self) -> Dict[str, Any]:
        """获取虚拟患者模型的参数配置（支持reasoning模型和额外参数）"""
        params = {
            'temperature': 0.7,
            'top_p': 0.9
        }
        
        # 获取额外参数配置
        vp_config = self.config.get('virtual_patient', {})
        extra_params = vp_config.get('extra_params', {})
        
        # 添加额外参数
        params.update(extra_params)
        
        if self.is_virtual_patient_reasoning():
            # DeepSeek官方API的thinking模式参数
            params['reasoning_effort'] = 'high'
            params['extra_body'] = {'thinking': {'type': 'enabled'}}
        
        return params
    
    def get_virtual_patient_extra_params(self) -> Dict[str, Any]:
        """获取虚拟患者模型的额外参数（用于reasoning模式）"""
        if self.is_virtual_patient_reasoning():
            return {
                'reasoning_effort': 'high',
                'extra_body': {'thinking': {'type': 'enabled'}}
            }
        return {}