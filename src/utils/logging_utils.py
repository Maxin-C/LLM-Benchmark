"""
日志工具
"""

import logging
import os
from datetime import datetime
from typing import Dict, Any

class LoggingUtils:
    """
    日志工具类
    """
    
    @staticmethod
    def setup_logger(name: str, log_dir: str = 'logs', level: int = logging.INFO) -> logging.Logger:
        """
        设置日志记录器
        
        参数：
            name: 日志记录器名称
            log_dir: 日志目录
            level: 日志级别
        
        返回：
            日志记录器
        """
        # 创建日志目录
        os.makedirs(log_dir, exist_ok=True)
        
        # 创建日志记录器
        logger = logging.getLogger(name)
        logger.setLevel(level)
        
        # 避免重复添加处理器
        if logger.handlers:
            return logger
        
        # 创建格式器
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        
        # 创建文件处理器
        log_file = os.path.join(log_dir, f'{name}_{datetime.now().strftime("%Y%m%d")}.log')
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        
        # 创建控制台处理器
        console_handler = logging.StreamHandler()
        console_handler.setLevel(level)
        console_handler.setFormatter(formatter)
        
        # 添加处理器
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)
        
        return logger
    
    @staticmethod
    def log_evaluation(result: Dict[str, Any], logger: logging.Logger = None) -> None:
        """
        记录评估结果
        
        参数：
            result: 评估结果
            logger: 日志记录器
        """
        if not logger:
            logger = LoggingUtils.setup_logger('evaluation')
        
        logger.info("=" * 50)
        logger.info("评估结果记录")
        logger.info("=" * 50)
        
        if 'scores' in result:
            logger.info("各维度评分：")
            for dimension, score in result['scores'].items():
                logger.info(f"  {dimension}: {score}")
        
        if 'overall_score' in result:
            logger.info(f"综合评分：{result['overall_score']}")
        
        if 'is_passed' in result:
            logger.info(f"是否通过：{'是' if result['is_passed'] else '否'}")
        
        if 'deduction_reasons' in result and result['deduction_reasons']:
            logger.info("扣分理由：")
            for reason in result['deduction_reasons']:
                logger.info(f"  - {reason}")
        
        if 'risk_report' in result:
            risk_level = result['risk_report'].get('risk_level', 'unknown')
            logger.info(f"风险等级：{risk_level}")
            
            critical_issues = result['risk_report'].get('critical_issues', [])
            if critical_issues:
                logger.warning("严重问题：")
                for issue in critical_issues:
                    logger.warning(f"  - {issue}")
        
        logger.info("=" * 50)
    
    @staticmethod
    def log_scenario(scenario: Dict[str, Any], logger: logging.Logger = None) -> None:
        """
        记录场景信息
        
        参数：
            scenario: 场景信息
            logger: 日志记录器
        """
        if not logger:
            logger = LoggingUtils.setup_logger('scenario')
        
        logger.info(f"场景ID: {scenario.get('scenario_id')}")
        logger.info(f"场景名称: {scenario.get('name')}")
        logger.info(f"场景类别: {scenario.get('category')}")
        logger.info(f"场景描述: {scenario.get('description')}")
    
    @staticmethod
    def log_conversation(dialogue_history: list, logger: logging.Logger = None) -> None:
        """
        记录对话历史
        
        参数：
            dialogue_history: 对话历史
            logger: 日志记录器
        """
        if not logger:
            logger = LoggingUtils.setup_logger('conversation')
        
        logger.info("对话历史：")
        for i, turn in enumerate(dialogue_history):
            role = turn.get('role', 'unknown')
            content = turn.get('content', '')
            logger.info(f"{i+1}. [{role}] {content}")
    
    @staticmethod
    def log_error(message: str, exception: Exception = None, logger: logging.Logger = None) -> None:
        """
        记录错误信息
        
        参数：
            message: 错误消息
            exception: 异常对象
            logger: 日志记录器
        """
        if not logger:
            logger = LoggingUtils.setup_logger('error')
        
        if exception:
            logger.error(f"{message}: {str(exception)}", exc_info=True)
        else:
            logger.error(message)
    
    @staticmethod
    def log_info(message: str, logger: logging.Logger = None) -> None:
        """
        记录一般信息
        
        参数：
            message: 消息
            logger: 日志记录器
        """
        if not logger:
            logger = LoggingUtils.setup_logger('info')
        
        logger.info(message)
