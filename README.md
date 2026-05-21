# EASE Benchmark

**EASE**: **E**xpert-**A**nchored **S**imulation **E**valuation Framework

专家锚定的自适应仿真评估框架，用于评估医疗对话系统的性能。

## 项目概述

EASE Benchmark是一个基于大型语言模型（LLM）和图神经网络（GNN）的医疗对话评估框架。该框架能够：

- 构建基于真实对话数据和EHR数据的虚拟患者
- 使用LLM驱动的多智能体系统进行对话仿真
- 通过循证审查和人文关怀评估医生表现
- 计算组内相关系数（ICC）进行元评估和效度验证

## 核心功能

### 🧠 多智能体系统
- **虚拟患者Agent**: 基于真实对话数据和EHR数据构建，能够模拟真实患者行为
- **循证审查官**: 使用LLM和知识图谱进行事实核查与红线拦截
- **人文关怀评估员**: 评估医生的共情能力和沟通质量
- **主审法官**: 综合各评估意见，输出李克特5维评分
- **监控Agent**: 监控对话过程，注入干扰或触发熔断

### 🔍 图推理引擎
- 基于GATv2模型的知识图谱推理
- 支持疾病治疗方案查询、药物相互作用检测
- 基于图嵌入的相似性推理

### 📊 元评估体系
- ICC一致性检验（支持输入100份对话和4位医生评估）
- 指南覆盖度分析
- 模型梯度敏感度测试

## 技术架构

```
┌─────────────────────────────────────────────────────────────────┐
│                      EASE Benchmark                            │
├─────────────────────────────────────────────────────────────────┤
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐         │
│  │  虚拟患者    │  │   评估引擎   │  │   监控Agent  │         │
│  │ Virtual      │  │ Evaluation   │  │ Monitor      │         │
│  │ Patient      │  │ Engine       │  │ Agent        │         │
│  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘         │
│         │                 │                  │                  │
│         ▼                 ▼                  ▼                  │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │                    LLM Client                           │    │
│  │              (GPT-4o / OpenAI API)                     │    │
│  └─────────────────────────────────────────────────────────┘    │
│                              │                                 │
│                              ▼                                 │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │                    图推理引擎                            │    │
│  │              (GATv2 + CMeKG Knowledge Graph)            │    │
│  └─────────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────────┘
```

## 快速开始

### 环境要求
- Python 3.10+
- PyTorch 2.1+
- OpenAI API Key

### 安装依赖

```bash
# 克隆项目
git clone <repository-url>
cd LLM-Benchmark

# 创建虚拟环境
conda create -n ease-benchmark python=3.10
conda activate ease-benchmark

# 安装依赖
pip install -r requirements.txt
```

### 配置环境变量

复制并编辑 `.env` 文件：

```bash
# 复制示例配置
cp .env.example .env

# 编辑配置
vim .env
```

配置内容：
```bash
# LLM API配置
EASE_LLM_API_KEY=your-api-key-here
EASE_LLM_BASE_URL=https://api.pumpkinaigc.online/v1
EASE_LLM_MODEL=gpt-4o

# 输出目录
EASE_OUTPUT_DIR=outputs

# 日志级别
EASE_LOG_LEVEL=INFO
```

### 运行评估

```bash
# 运行单次评估
python main.py --mode evaluate

# 运行元评估（ICC计算）
python main.py --mode meta

# 测试图推理引擎
python main.py --mode kg_test

# 运行完整基准测试
python main.py --mode benchmark
```

## 项目结构

```
LLM-Benchmark/
├── src/                          # 源代码目录
│   ├── utils/                    # 工具模块
│   │   ├── llm_client.py        # LLM客户端封装
│   │   └── logging_utils.py     # 日志工具
│   ├── data_processing/          # 数据处理模块
│   │   └── kg_loader.py         # 知识图谱加载器
│   ├── kg/                       # 图推理模块
│   │   └── graph_reasoner.py    # 图推理引擎
│   ├── sandbox/                  # 仿真沙盒
│   │   ├── virtual_patient.py   # 虚拟患者Agent
│   │   └── monitor_agent.py     # 监控Agent
│   ├── evaluation/               # 评估引擎
│   │   ├── evidence_checker.py  # 循证审查官
│   │   ├── empathy_evaluator.py # 人文关怀评估员
│   │   └── chief_judge.py       # 主审法官
│   └── meta_evaluation/         # 元评估模块
│       └── icc_calculator.py    # ICC计算器
├── dataset/                      # 数据集目录
│   ├── ehr_data/                 # EHR数据
│   ├── kg_data/                  # 知识图谱数据
│   │   ├── CMeKG.pkl            # 知识图谱
│   │   ├── gnn_models/          # GNN模型
│   │   ├── gnn_cache/           # 嵌入缓存
│   │   └── guidelines/          # 临床指南
│   └── persona_data/            # 患者画像数据
├── config/                       # 配置文件
├── scripts/                      # 脚本
├── outputs/                      # 输出目录
├── tests/                        # 测试用例
├── main.py                       # 主入口
├── requirements.txt              # 依赖清单
├── .env                          # 环境变量配置
└── .gitignore                   # Git忽略配置
```

## 核心模块说明

### LLM客户端 (`src/utils/llm_client.py`)
统一管理LLM调用，支持：
- 标准对话接口
- JSON格式输出解析
- 批量调用支持

### 图推理引擎 (`src/kg/graph_reasoner.py`)
基于预训练GATv2模型的知识图谱推理引擎：
- 节点嵌入查询
- 相似节点查找
- 链接预测
- 多跳推理路径查找

### 评估引擎 (`src/evaluation/`)
- **EvidenceChecker**: 事实核查与红线拦截
- **EmpathyEvaluator**: 共情能力评估
- **ChiefJudge**: 综合五维评分（准确性、有效性、安全性、个性化、共情）

### 元评估 (`src/meta_evaluation/icc_calculator.py`)
支持输入100份对话和4位医生评估结果，自动计算：
- 总体ICC
- 各维度ICC
- 95%置信区间
- 一致性等级

## 使用示例

### 评估医生对话

```python
from src.evaluation.chief_judge import ChiefJudge
from src.evaluation.evidence_checker import EvidenceChecker
from src.evaluation.empathy_evaluator import EmpathyEvaluator
from src.utils.llm_client import LLMClient

# 初始化LLM客户端
llm_client = LLMClient(api_key="your-key", base_url="https://api.pumpkinaigc.online/v1")

# 初始化评估组件
evidence_checker = EvidenceChecker(llm_client)
empathy_evaluator = EmpathyEvaluator(llm_client)
judge = ChiefJudge(llm_client, evidence_checker, empathy_evaluator)

# 对话历史
dialogue_history = [
    {"role": "doctor", "content": "你好，我是你的主治医生..."},
    {"role": "patient", "content": "医生，我很担心我的病情..."},
    {"role": "doctor", "content": "请放心，我们会为你制定个性化治疗方案..."}
]

# 患者状态
patient_state = {
    "demographics": {"age": 45, "gender": "female"},
    "medical_info": {"pathology_type": "乳腺癌", "stage": "IIB期"}
}

# 评估
result = judge.evaluate(dialogue_history, patient_state, {})
print(f"综合评分: {result['overall_score']}/5")
```

### 计算ICC

```python
from src.meta_evaluation.icc_calculator import ICCCalculator

icc_calculator = ICCCalculator()

# 添加评分者（4位医生 + EASE-Judge）
icc_calculator.add_rater("doctor_1", doctor_1_scores)
icc_calculator.add_rater("doctor_2", doctor_2_scores)
icc_calculator.add_rater("doctor_3", doctor_3_scores)
icc_calculator.add_rater("doctor_4", doctor_4_scores)
icc_calculator.add_rater("ease_judge", ease_judge_scores)

# 计算ICC
result = icc_calculator.calculate_icc()
print(f"总体ICC: {result['overall_icc']['icc']:.4f}")
```

## 数据说明

### 知识图谱数据
- **格式**: Pickle
- **来源**: CMeKG医疗知识图谱
- **包含**: 疾病、药物、治疗方案等实体及关系

### EHR数据
- **格式**: Excel
- **包含**: 患者人口统计学信息、诊断信息、治疗记录

### 对话数据
- **格式**: JSON
- **来源**: 真实医患对话记录

## 许可证

本项目仅供研究使用。

## 贡献

欢迎提交Issue和Pull Request！

## 联系方式

如有问题或建议，请联系项目维护者。
