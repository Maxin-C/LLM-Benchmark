#!/usr/bin/env python3
"""
LLM客户端工厂 - 根据配置创建不同角色的LLM客户端
支持：虚拟患者(待评估模型)、虚拟医生(deepseek-v4-pro)、评估器(deepseek-v4-pro)、监控器(deepseek-v4-pro)
支持reasoning模型的thinking输出记录（DeepSeek官方API支持thinking模式）
"""

import os
from typing import Dict, Any, Optional, Tuple
from openai import OpenAI

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
        model_type = vp_config.get('model_type', 'api')
        
        if model_type == 'api':
            api_config = vp_config.get('api', {})
            return OpenAI(
                api_key=api_config.get('api_key') or os.getenv('EASE_VP_API_KEY'),
                base_url=api_config.get('base_url') or os.getenv('EASE_VP_BASE_URL')
            )
        elif model_type == 'local':
            # 本地模型支持（预留）
            local_config = vp_config.get('local', {})
            model_path = local_config.get('model_path') or os.getenv('EASE_VP_LOCAL_PATH')
            raise NotImplementedError(f"本地模型支持尚未实现: {model_path}")
        else:
            raise ValueError(f"未知的模型类型: {model_type}")
    
    def create_virtual_doctor_client(self) -> OpenAI:
        """
        创建虚拟医生客户端（使用deepseek-v4-pro）
        
        返回：
            OpenAI客户端实例
        """
        doctor_config = self.config.get('virtual_doctor', {})
        return OpenAI(
            api_key=doctor_config.get('api_key') or os.getenv('EASE_DOCTOR_API_KEY'),
            base_url=doctor_config.get('base_url') or os.getenv('EASE_DOCTOR_BASE_URL')
        )
    
    def create_judger_client(self) -> OpenAI:
        """
        创建评估器客户端（使用deepseek-v4-pro）
        
        返回：
            OpenAI客户端实例
        """
        judger_config = self.config.get('judger', {})
        return OpenAI(
            api_key=judger_config.get('api_key') or os.getenv('EASE_JUDGER_API_KEY'),
            base_url=judger_config.get('base_url') or os.getenv('EASE_JUDGER_BASE_URL')
        )
    
    def create_monitor_client(self) -> OpenAI:
        """
        创建对话监控器客户端（使用deepseek-v4-pro）
        
        返回：
            OpenAI客户端实例
        """
        monitor_config = self.config.get('dialogue_monitor', {})
        return OpenAI(
            api_key=monitor_config.get('api_key') or os.getenv('EASE_MONITOR_API_KEY'),
            base_url=monitor_config.get('base_url') or os.getenv('EASE_MONITOR_BASE_URL')
        )
    
    def get_virtual_patient_model(self) -> str:
        """获取虚拟患者使用的模型名称"""
        vp_config = self.config.get('virtual_patient', {}).get('api', {})
        return vp_config.get('model') or os.getenv('EASE_VP_MODEL', 'gpt-4o')
    
    def is_virtual_patient_reasoning(self) -> bool:
        """判断虚拟患者是否为reasoning模型"""
        vp_config = self.config.get('virtual_patient', {})
        is_reasoning = vp_config.get('is_reasoning') or os.getenv('EASE_VP_IS_REASONING', 'false')
        return str(is_reasoning).lower() == 'true'
    
    def get_virtual_doctor_model(self) -> str:
        """获取虚拟医生使用的模型名称（deepseek-v4-pro）"""
        doctor_config = self.config.get('virtual_doctor', {})
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
        """获取虚拟患者模型的参数配置（支持reasoning模型）"""
        params = {
            'temperature': 0.7,
            'top_p': 0.9
        }
        
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
