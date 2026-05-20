"""
EASE-Judge评估引擎模块
包含循证审查官、人文关怀员、主审法官和Kill-Switch机制
"""

from .evidence_checker import EvidenceChecker
from .empathy_evaluator import EmpathyEvaluator
from .chief_judge import ChiefJudge
from .kill_switch import KillSwitch

__all__ = [
    'EvidenceChecker',
    'EmpathyEvaluator',
    'ChiefJudge',
    'KillSwitch'
]
