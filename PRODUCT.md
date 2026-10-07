# PRODUCT.md — Knowledge System V0

> 状态：FROZEN。Worker 不得为了方便实现而改变产品边界。

## 1. 产品问题

企业知识通常以文件、聊天、网页、表格等形式存在，但员工实际工作以“当前任务/问题”为中心。文件结构并不等于员工任务结构。员工经常知道问题是什么意思，却不知道正式规则叫什么、在哪个文件、哪个章节，也可能需要跨文档核验。

产品目标：**将企业知识从“以文件为中心的存储”转变为“以员工任务为中心的获取”。**

系统不是替员工知道答案，而是让正式知识在需要时，以容易理解、容易核验、适合当前任务的形式出现。

## 2. 产品职责

系统可以：
- 定位正式知识；
- 按范围/文档/类别浏览；
- 在员工记不起正式术语时进行语义发现；
- 组织规则的条件、阈值、例外和来源；
- 将当前 Case 已知事实与规则条件比较；
- 展示匹配、缺失、冲突、歧义与 Knowledge Gap；
- 始终保留回到原始资料的路径。

系统不可以：
- 退化成无关问题也回答的通用聊天机器人；
- 替员工做最终业务审批/审核决定；
- 用模型推测填补企业规则缺失；
- 发明企业术语、规则、来源、页码或事实；
- 让 LLM 覆盖用户明确事实、权限或权威元数据。

## 3. Knowledge Request

V0 支持五类请求：

1. `Locate`：已知名称/范围，精确或近精确定位。
2. `Browse`：浏览某范围内的一组知识；结果集合本身可以是正确答案。
3. `Discover`：用户知道含义但不知道正式名称，用语义能力跨越自然表达与正式知识表达的鸿沟。
4. `Apply / Compare`：检索规则，并把 Case 事实与规则条件进行辅助比较；最终业务决定仍由人承担。
5. `Out of Scope`：明确告知当前企业知识库不覆盖，不回退为通用问答。

同时支持 Source Navigation：文档 → 全部知识；文档+页面/Sheet/Section → 对应知识。

## 4. 核心原则

**确定性优先，语义能力兜底；LLM 只解释不确定的部分，不重新解释已经确定的部分。**

能力顺序：`ID / Metadata → Keyword → Browse → Semantic Retrieval → LLM → Human`

**缺信息 ≠ 必须先拦住用户提问。** 能形成有用检索时先给证据，再展示缺失条件；只有信息不足到无法形成有用检索时才阻塞式澄清。

## 5. Evidence Map

最终输出不是自由生成答案，而是 Evidence Map，至少能表达：
- 当前请求/主题；
- 相关正式规则；
- 条件/阈值/例外（若可可靠派生）；
- Case 已匹配事实；
- 缺失/未知条件；
- 相关知识；
- Knowledge Gap / ambiguity；
- Source Reference；
- 可用于继续缩小范围的提示。

## 6. Knowledge Boundary

必须区分：
- `Case Information Missing`：规则清楚，但 Case 事实不足；
- `Knowledge Missing`：当前知识库没有检索到所需知识；
- `Knowledge Ambiguous`：正式资料自身冲突、不完整或歧义。

系统只能说“当前知识库未检索到相关内容”，不能把 Not Found 说成“企业没有这种规定”。

`Retrieved ≠ Answerable`：检索到了标题或片段，不代表存在足够规则支持结论。

## 7. Source Traceability

每条关键 Evidence 必须尽可能让员工重新找到原始资料：
1. 稳定 URL 可用：完整原始名称 + URL + 精确位置；
2. 无 URL：完整文件名/页面名 + Page/Sheet/Section 等可靠位置；
3. 无稳定位置：至少完整原始名称；
4. 无法确定来源身份：不得作为高置信 Evidence。

不得伪造页码。内部服务器路径默认不作为员工主要来源线索。

## 8. Grounding Failure

Evidence 本身不足时，不继续生成“看起来合理”的答案，而进入 `Verification Mode`：展示当前原文、问题/不确定性、为什么不能依赖、来源位置，以及员工应去哪里核验。

员工核验后的反馈进入 Feedback Log；**Employee Feedback ≠ Formal Knowledge**，V0 不自动修改正式知识。

## 9. V0 成功标准

- 能从真实企业规则资料形成可追溯 Knowledge Units；
- Locate/Browse/Discover/Apply/Out-of-Scope 有明确行为；
- Hybrid Retrieval 能用评测证明相对基线的价值；
- Case 比较不越过正式规则边界；
- Evidence 异常能恢复、诊断或进入 Verification Mode；
- LLM 不能直接改写最终事实 Evidence；
- Trace 足以解释一次请求如何得到当前结果；
- 人仍承担最终业务责任。

## 10. 一句话定义

**一个以企业正式知识为 Source of Truth，把员工任务映射到可核验 Evidence，并在知识边界处明确停下来的企业知识系统。**
