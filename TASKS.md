# TASKS.md — Knowledge System V0 Construction Plan

> 目标：给 Runtime 使用，不是给 Human 做逐步验收清单。每个 Task 有机器可判断的 Acceptance Criteria。除结构性阻塞外，Runtime 自主推进。

## 0. 总体施工策略

优先做**薄的端到端 Vertical Slice**，尽早得到一个能从真实资料检索并返回原始 Evidence 的可运行系统，再逐层增加语义、规则比较、Recovery、Organizer 与 Verification。禁止先把所有基础设施造完再集成。

测试遵循 `BUILD.md`：Task 只做最小/定向测试；Phase Gate 做集成；V0 完成做 Full Regression + Golden Dataset。

---

## Phase 0 — Skeleton & Contracts

### T-000 Project skeleton
**目标**：建立 `src/ tests/ data/ scripts/` 与配置入口，保留当前冻结文档。

**Acceptance**：
- 包可导入；
- CLI/最小入口可启动；
- 测试框架可运行一个 smoke test；
- 不引入非必要 DB/Redis/Kafka 等基础设施。

### T-001 Core data contracts
**目标**：定义 KnowledgeUnit、SourceReference、KnowledgeLineage、ParseStatus、Evidence、Trace 基础模型。

**Acceptance**：
- schema 可序列化/反序列化；
- SourceReference 支持 name/url/section/page/sheet 等可选定位；
- Lineage 能记录 parser/chunker/processing version；
- 派生结构与原始 Source 字段明确分离。

**Phase Gate P0**：smoke + contracts tests。

---

## Phase 1 — Real Knowledge Ingestion Vertical Slice

### T-100 HTML loader/parser
以真实 `微信支付分级审核规则-结构化速查手册.html` 为首个 fixture/source；若 Worker 环境未包含该文件，使用等价本地 fixture 建立接口，不虚构其业务内容。

**Acceptance**：
- 原始文本可保留；
- 标题/层级/表格等结构尽可能保留；
- ParseStatus 正确；
- 不可靠 page 不生成。

### T-101 Structure-aware Knowledge Units
**Acceptance**：
- 不是固定字符粗切；
- unit_id 稳定、可复现；
- 每个 unit 可回溯 SourceReference + Lineage；
- 原文与 semantic_content 分离。

### T-102 Source navigation baseline
**Acceptance**：可按 document/section 查询 Knowledge Units；返回完整 source identity。

**Phase Gate P1**：真实 HTML → Knowledge Units → source navigation 端到端通过。

---

## Phase 2 — Deterministic Retrieval Baseline

### T-200 Keyword/exact index
**Acceptance**：规则名、关键术语、文档名可定位；输出 Evidence ID + source。

### T-201 Permission/status filter contract
**Acceptance**：无权限/明确失效内容不能进入候选；Filter 与 Ranking 分离。

### T-202 Locate/Browse baseline
**Acceptance**：
- Locate 可返回精确/近精确 Evidence；
- Browse 可返回 scope 内集合，不强制 Top-1；
- Out-of-Scope/Not Found 不表述成“企业没有”。

### T-203 Baseline trace
**Acceptance**：记录 raw query、retrieval query、candidate/final IDs、filter、source/version。

**Phase Gate P2**：不依赖 Vector/LLM，也能完成可追溯 Locate/Browse。

---

## Phase 3 — Semantic / Hybrid Retrieval

### T-300 Vector index
**Acceptance**：index 可重建；index version 与 knowledge version 分离。

### T-301 Hybrid recall/fusion
**Acceptance**：Keyword 与 Vector 结果可合并；各路径 rank/score 可 Trace。

### T-302 Optional rerank
只有在基线 Evaluation 证明排序问题时启用；否则保留接口但不强制复杂模型。

### T-303 Discover request
**Acceptance**：模糊自然表达可发现正式 Knowledge Unit；正式名称必须来自 KB Evidence。

**Phase Gate P3**：一组 Discover cases 相对 keyword baseline 有可观测改进，且精确专名不因语义层退化。

---

## Phase 4 — Query & Multi-turn Context

### T-400 Request classification
支持 Locate/Browse/Discover/Apply/Out-of-Scope。

### T-401 QueryContext / CaseContext
**Acceptance**：
- scope/intent 与 Case raw descriptions 分离；
- 后续“现在52个赞”等更新 Case，而不是重写 QueryContext；
- original query 保留。

### T-402 Clarification policy
**Acceptance**：只有无法形成有用 retrieval 才阻塞；否则先返回规则/缺失信息。

**Phase Gate P4**：多轮上下文 cases 可重复测试。

---

## Phase 5 — Derived Rule Structure

### T-500 Rule structure extraction
从 Source 派生 conditions/logical_relations/thresholds/exceptions/consequence/ambiguity；无法可靠结构化时允许为空/partial。

### T-501 Information requirements
Rule condition 可产生当前 Case 需要的事实类型/问题。

### T-502 Authority checks
**Acceptance**：Derived Structure 不覆盖/修改 original source；任何结构字段可回指 evidence/source。

**Phase Gate P5**：选定真实规则可形成结构；歧义规则保持 ambiguity，不被强行确定。

---

## Phase 6 — Rule-driven Fact Extraction & Compare

### T-600 On-demand fact extraction
只根据 Candidate Rules 的 Information Requirements 从 Case 描述提取事实。

### T-601 Deterministic comparison
数值/布尔/枚举等明确条件由程序比较；语义不确定条件为 UNKNOWN/HUMAN_REQUIRED。

### T-602 Apply/Compare behavior
**Acceptance**：输出 matched/missing/unknown，不输出最终责任性业务决定。

**Phase Gate P6**：至少覆盖阈值变化、duplicate、缺失事实、OR/AND 条件案例。

---

## Phase 7 — Evidence Quality, Recovery & Diagnosis

### T-700 Evidence quality signal
检测 POSSIBLY_INCOMPLETE/BROKEN_CONTEXT/MISSING_CONTEXT/POSSIBLY_AMBIGUOUS 等信号。

### T-701 Limited recovery
最多 1–2 轮 Neighbor/Parent/Section/Re-retrieval/Source expansion。

### T-702 Failure diagnosis
区分 Retrieval Problem / Processing Problem / Source Knowledge Problem。

**Acceptance**：Source 本身没问题时，不把系统 retrieval/chunk 问题误报为 Knowledge Gap。

**Phase Gate P7**：人为构造断句/错误候选/原文歧义三类 fixture，诊断路径可 Trace。

---

## Phase 8 — Grounded Organizer & Evidence Map

### T-800 EvidenceMapPlan schema
只允许 section/group/order/evidence_ids 等组织信息。

### T-801 Grounded Organizer
LLM 可读原始 Evidence + derived structure + compare result，但只输出 Plan。

### T-802 Reference Guard
未知 evidence_id → ORGANIZATION_ERROR；不得静默接受。

### T-803 Deterministic Hydration
按 ID 从 Evidence Store 回填事实原文/结构/source，生成最终 Evidence Map。

**Phase Gate P8**：证明 Organizer 无法凭输出新 ID/新事实进入最终 Evidence。

---

## Phase 9 — Verification Mode & Feedback

### T-900 Verification Mode
Evidence 不可靠时展示：原文、问题、为何不可确定、Source、核验路径。

### T-901 Feedback Log
记录 employee feedback + target evidence/trace；不得自动修改 formal knowledge。

**Phase Gate P9**：Knowledge Missing / Source Ambiguous / Recovery Failed 均有可理解行为。

---

## Phase 10 — Product Orchestration / Session

### T-1000 Product AgentState
实现 SYSTEM 定义的轻量 State。

### T-1001 Graph/control flow
Node 只承载有意义状态转换；retrieval 等内部仍是普通 pipeline。

### T-1002 Session snapshot/resume
一轮 CLI 为 Session；关闭后保留结构化 snapshot；显式 resume 恢复，不做跨会话个人记忆。

**Phase Gate P10**：完整多轮任务可中断/恢复，Trace 与 State 分离。

---

## Phase 11 — Evaluation & Regression

### T-1100 Golden Dataset
构建约 20–30 个真实需求 × 多表达，覆盖五类 Request、边界、澄清、Knowledge Gap。

### T-1101 Metrics/report
至少区分 Recall 与 Ranking；Browse 使用集合型可接受标准。

### T-1102 Regression runner
支持小范围 case/tag 运行与 full suite；避免每个 Task 全量执行。

**Phase Gate P11**：形成 baseline vs current 的可复现报告。

---

## Phase 12 — Knowledge Version / Index State

### T-1200 System state
维护 source_commit/indexed_commit/index_status/last_check_at。

### T-1201 Incremental update contract
启动只检查版本；OUT_OF_SYNC 才更新；完整成功后推进 indexed_commit。

**Phase Gate P12**：可模拟 source A→B、失败不推进 indexed_commit、成功后同步。

---

## Phase 13 — Final Product Surface & V0 Acceptance

### T-1300 CLI/service surface
提供可真实体验的入口；核心逻辑不绑定 UI。

### T-1301 Demo scenarios
至少包含：精确 Locate、Browse、模糊 Discover、Case threshold update、duplicate、Knowledge Missing、Source Ambiguity→Verification、Out-of-Scope。

### T-1302 Final regression
按 BUILD 运行 Full Regression + Golden Dataset + final review。

### T-1303 V0 completion report
输出：完成能力、已知限制、评测结果、关键 Trace、未解决 Knowledge Gap、下一轮候选改进；不在本轮擅自增加机制。

**PROJECT COMPLETE**：所有冻结产品不变量通过；无未解释结构性失败；最终成品可由 Human 独立运行体验。
## Phase 14 — HTML / Markdown / CSV 统一导入

### T-1400 — 统一导入接口与格式路由

- 建立统一 ingestion result 和 loader registry。
- 根据扩展名确定性识别 HTML、Markdown、CSV。
- 统一错误、警告、来源引用、稳定 ID 和解析状态。
- 将现有 HTML loader 接入统一入口，不重写已有解析逻辑。
- 本任务不运行测试。

### T-1401 — Markdown 导入

- 支持 .md 和 .markdown。
- 按标题层级和结构单元生成知识单元。
- 保留标题路径、代码块、列表、引用和表格。
- 生成稳定来源引用与 lineage。
- 本任务不运行测试。

### T-1402 — CSV 导入

- 支持 .csv 和 UTF-8 / UTF-8-SIG。
- 表头作为字段定义，每一行生成结构化知识单元。
- 保留文件、表头、原始行号和字段值来源。
- 对空表头、列数异常和无效编码返回 PARTIAL 或 FAILED。
- 本任务不运行测试。

### T-1403 — 批量导入 CLI 与结果合并

- 增加 ingest 命令，支持单文件和目录。
- 目录中仅处理 HTML、Markdown、CSV。
- 多文件结果按确定性顺序合并。
- 支持重复来源和知识单元去重。
- 原子写入 KnowledgeUnit corpus JSON。
- 本任务完成后视为功能合并完成。
- 本任务不运行测试。

### P14-GATE — 合并后统一验证

- 仅在 T-1400 至 T-1403 全部完成后执行一次。
- Reviewer 执行新格式的聚焦测试和必要的集成测试。
- 验证 HTML、Markdown、CSV 混合目录导入、稳定 ID、来源追踪、重复导入、异常输入，以及导入结果可被 query 使用。
- 不执行全项目测试。
- Gate 失败后仅运行与失败项直接相关的测试。
