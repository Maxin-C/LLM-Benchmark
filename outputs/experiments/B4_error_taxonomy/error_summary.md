# B4 实验：Error Taxonomy / Failure Mode Analysis 报告

**生成时间**: 2026-06-02 10:01:32

**数据源**: 6 个模型的低分案例

---

## 1. 低分案例统计

- **总低分案例数**: 52


- **QWEN3-8B**: 25 个低分案例, 平均每案例 5.36 个错误
- **QWEN3-14B**: 14 个低分案例, 平均每案例 5.71 个错误
- **QWEN3-32B**: 9 个低分案例, 平均每案例 5.33 个错误
- **QWEN3-235B-A22B**: 4 个低分案例, 平均每案例 5.75 个错误

## 2. 错误类型分布

- **Unsafe Recommendation**: 52 次
- **Treatment Error**: 52 次
- **Underconfidence**: 46 次
- **Overconfidence**: 44 次
- **Irrelevant Response**: 34 次
- **Factual Error**: 32 次
- **Missing Information**: 24 次
- **Inconsistency**: 1 次

## 3. 各模型主要错误类型

- **QWEN3-8B**:
  - Unsafe Recommendation: 25 次
  - Treatment Error: 25 次
  - Underconfidence: 23 次
- **QWEN3-14B**:
  - Unsafe Recommendation: 14 次
  - Overconfidence: 14 次
  - Treatment Error: 14 次
- **QWEN3-32B**:
  - Unsafe Recommendation: 9 次
  - Treatment Error: 9 次
  - Factual Error: 8 次
- **QWEN3-235B-A22B**:
  - Unsafe Recommendation: 4 次
  - Overconfidence: 4 次
  - Underconfidence: 4 次

## 4. 错误类型定义

### Factual Error
- **描述**: Provided incorrect medical facts or information
- **关键词**: 错误, 不正确, 不是, 没有, 错误地, 误

### Missing Information
- **描述**: Omitted critical information or failed to answer questions
- **关键词**: 不知道, 不清楚, 无法回答, 未提及, 缺少, 遗漏

### Unsafe Recommendation
- **描述**: Provided potentially harmful advice
- **关键词**: 建议, 应该, 可以, 不要, 禁止, 避免

### Inconsistency
- **描述**: Internal contradictions or inconsistencies in the answer
- **关键词**: 但是, 然而, 矛盾, 不一致, 相反, 却

### Overconfidence
- **描述**: Gave definitive conclusions when uncertain
- **关键词**: 肯定, 一定, 绝对, 毫无疑问, 显然, 必然

### Underconfidence
- **描述**: Overly cautious, failed to provide sufficient information
- **关键词**: 可能, 也许, 或许, 不确定, 建议咨询

### Communication Issue
- **描述**: Unclear, rigid language or lack of empathy
- **关键词**: 抱歉, 不好意思, 简单来说, 直白地说, 老实说

### Irrelevant Response
- **描述**: Answer unrelated to the question or off-topic
- **关键词**: 另外, 顺便说, 补充一下, 关于

### Knowledge Gap
- **描述**: Lack of necessary medical knowledge
- **关键词**: 根据我的知识, 据我所知, 研究表明, 医学上

### Treatment Error
- **描述**: Provided incorrect treatment recommendations
- **关键词**: 治疗, 药物, 手术, 化疗, 放疗, 内分泌

## 5. 关键发现

- **最常见错误类型**: Unsafe Recommendation (52 次, 18.2%)
- **错误最多的模型**: QWEN3-8B (134 个错误)
- **错误最少的模型**: QWEN3-235B-A22B (23 个错误)

---

*报告结束*
