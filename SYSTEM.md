# SYSTEM.md — Knowledge System V0

> 状态：FROZEN。结构性修改必须退出当前 Runtime，由 Human 修改契约后开启新一轮。

## 1. 总体架构

```text
Enterprise Documents
→ Loader
→ Parser
→ Structure Recognition
→ Knowledge Units
  + Semantic Content
  + Metadata
  + Source Reference
  + Knowledge Lineage
  + Optional Rule Structure
→ Keyword Index + Vector Index

Employee
→ Raw Query
→ Context Resolution
→ Knowledge Request
→ Hybrid Retrieval
→ Candidate Evidence
→ Evidence Quality Check
  → problem: Recovery → Diagnose
→ Candidate Rules
→ Rule-driven Fact Extraction
→ Case ↔ Rule Comparison
→ Grounded Knowledge Organizer
→ EvidenceMapPlan
→ Schema + Reference Validation
→ Deterministic Hydration
→ Evidence Map
  → reliable: Employee
  → unreliable: Verification Mode → Employee → Feedback Log
```

## 2. Knowledge Authority

权威层级：
1. `Source Evidence`：原始正式资料，最高事实权威；
2. `Knowledge Unit`：检索单位；
3. `Derived Rule Structure`：由原文派生的条件/逻辑/阈值/例外等，权威低于原文。

无法可靠结构化时允许保留原文，不强行生成字段。歧义必须成为数据，不能被模型消除。

## 3. Knowledge Unit

至少包含：
- `unit_id`
- `content/original_content`
- `semantic_content`
- `metadata`
- `source_reference`
- `lineage`
- optional `rule_structure`

Chunk 是知识检索单位，不是固定字符窗口。层级文档按结构；FAQ 按 Q+A；表格保留表头与单元格关系；CSV 按行/结构单元；普通文本再使用动态窗口。

V0 本轮知识导入仅支持 HTML、Markdown 和 CSV。

- HTML：按标题层级和结构块切分。
- Markdown：按标题层级和结构单元切分，并保留代码块、列表、引用和表格语义。
- CSV：表头作为字段定义，每一数据行生成一个结构化知识单元，并保留原始行号。
- 所有格式必须生成稳定的 source_id、unit_id、SourceReference 和 lineage。
- 导入结果区分 SUCCESS、PARTIAL 和 FAILED。

Text、DOCX、XLS/XLSX、PDF、图片及 OCR 不属于本轮范围。

## 4. Semantic Content 与 Metadata

Semantic Content 描述“这条知识说什么”；Metadata 描述“它是谁、属于哪、对谁有效、现在能不能用、从哪来”。

Metadata 可包含 scope/status/version/effective time/source/permission 等。Embedding 不负责判断权限、适用性或权威性。

## 5. Retrieval

分离：Document Routing / Query Understanding / Retrieval。

- Keyword/exact：身份、专名、文件名、规则编号、缩写等；
- Vector：语义变体与用户自然表达；
- Filter：只提前排除确定不可能/不允许的候选（权限、失效、明确错误 scope）；
- Rerank：改善正确 Evidence 的排序，不替代 Recall。

Recall 与 Ranking 独立评测。Browse 不强求 Top-1。

## 6. QueryContext 与 CaseContext

`QueryContext`：当前要找什么知识、scope、request type。  
`CaseContext`：当前现实 Case 的原始描述和按需提取的事实。

不建立覆盖开放世界的全局 Case Schema。CaseContext V0 可保持轻量：
- `raw_descriptions[]`
- `extracted_facts[]`

## 7. Rule-driven Fact Extraction

先 Retrieval 得到 Candidate Rules，再由规则条件产生 Information Requirements，然后只从用户描述中提取当前规则真正需要的事实。

```text
Query/Case
→ Retrieval
→ Candidate Rules
→ Rule Conditions
→ Information Requirements
→ Fact Extraction
→ Compare
```

规则是“问题生成器”。不要在 Retrieval 前把 Query 与全语料规则逐条比较。

## 8. Rule Structure 与 Comparison
 
Rule Structure 可包含：
- conditions
- logical_relations (AND/OR)
- thresholds
- exceptions
- consequence
- ambiguity
- information_requirements

V0 确定性比较由普通程序完成，例如数值阈值。无法机械判断的条件保持 UNKNOWN/HUMAN_REQUIRED，不在 V0 建完整 Semantic Rule Engine/DSL/AST。

## 9. Evidence Quality / Recovery / Diagnosis

LLM 可发出异常信号，但不拥有最终故障归因权。

问题分三层：
- Retrieval Problem：候选错误/上下文不足；
- Processing Problem：parse/chunk/structure 损失；
- Source Knowledge Problem：原始资料本身不完整/歧义/冲突/缺失。

只有第三类是真正 Knowledge Gap。

Recovery 最多进行有限次数（V0 1–2 次）：Neighbor Expansion → Parent Unit → Section Expansion → Re-retrieval → Original Source Context。

Knowledge Lineage 至少支持从 Retrieved Evidence 回溯到原始来源与处理版本，可记录 document_id/source_position/parent_section_id/parser_version/chunker_version/processing_version。

## 10. Grounded Organizer

Organizer 只有组织权，没有知识写入权。它可以选择、分组、排序、引用 Evidence ID，生成 `EvidenceMapPlan`；不能创造企业事实、阈值、条件、来源或修改 ComparisonResult。

Organizer 可以看原始 Retrieved Evidence 与派生结构，以避免 ingestion 信息损失被掩盖。

所有 Organizer 输出的 `evidence_id` 必须属于当前 Candidate Evidence Set；未知 ID → `ORGANIZATION_ERROR`。

**LLM 写目录，程序装正文。** 最终事实内容由程序按 Evidence ID 确定性 Hydration。

## 11. Verification Mode / Feedback

Evidence 不足或来源自身存在问题时进入 Verification Mode，而不是继续猜。

Feedback 记录为观测/维护信号，关联 evidence/trace，但不自动升级为 Formal Knowledge。

## 12. Session / State / Runtime

产品 V0 支持 Task-level multi-turn 与 Session follow-up，不做跨会话长期个人记忆。保存结构化工作上下文，不依赖完整聊天历史。

Context relation：CONTINUE / UPDATE / NEW / AMBIGUOUS；继承不安全时澄清。

产品 AgentState 可包含：session_id/raw_query/query_context/case_context/knowledge_request/context_resolution/clarification/candidate_evidence/evidence_quality/recovery_state/comparison_result/evidence_map_plan/evidence_map/error。

权限、knowledge_commit、retrieval_service、vector store、LLM/config 属于 Runtime dependencies，不塞进普通 State。

## 13. Graph Boundary

Node 表达有意义的状态转换/控制决策，不把每个 Python 函数都变成节点。`retrieve` 内部仍可为普通 Python pipeline。

概念图：START → resolve_context → understand_request → clarify? → retrieve → check_evidence → recover/diagnose? → rule analysis → compare → organize → validate → hydrate → verification? → trace → END。

## 14. Persistence / Version

- Git：Knowledge source history；
- JSON：轻量 system state；
- JSONL：trace/evaluation/experiment/feedback；
- Keyword/Vector index：派生数据，可重建。

Knowledge source commit 与 indexed commit 分离。启动时只检查版本，不默认全量重建。只有完整成功后才推进 indexed_commit。

## 15. Evaluation / Trace

Golden Dataset 约 20–30 个真实需求 × 多种表达，覆盖 Locate/Browse/Discover/Apply/Out-of-Scope、边界、澄清。Golden 描述可接受行为与 Evidence 边界，不固定最终措辞。

Trace 至少记录 raw/retrieval query、intent/scope、recall/rerank、final evidence IDs、knowledge version、quality signal、recovery、diagnosis、comparison、organizer references。

## 16. V0 不变量

1. Original Query 不丢失。
2. Explicit Facts 不被 LLM 静默修改。
3. Source 可追溯。
4. Derived Structure 权威低于 Source。
5. Permission 是硬边界。
6. Not Found ≠ Not Exist。
7. Missing Case Info ≠ Knowledge Gap。
8. Retrieved ≠ Answerable。
9. Retrieval Failure ≠ Processing Failure ≠ Source Problem。
10. LLM 可读 Evidence，不可重写 Evidence。
11. Organizer 只有组织权。
12. Rule 驱动 Fact Extraction。
13. Evidence failure → Verification，不 hallucinate。
14. Employee Feedback 是 observation，不是 Formal Knowledge。
15. 最终业务责任属于 Human。

## 17. 一句话定义

**这是一个以企业正式知识为 Source of Truth、以 Hybrid Retrieval 为核心、以 Evidence 为输出、以 LLM 处理语义不确定性、以确定性程序控制事实与引用、并在 Grounding 失败时主动进入 Verification Mode 的企业 RAG 系统。**
