# Knowledge System V0 Completion Report

## 1. 状态与边界

Knowledge System V0 已完成实现，并于 T-1302 Final Review 获得 `PASS`，允许建立最终验证 checkpoint。本报告依据 [PRODUCT.md](PRODUCT.md)、[SYSTEM.md](SYSTEM.md)、[DECISIONS.md](DECISIONS.md)、[BUILD.md](BUILD.md)、[TASKS.md](TASKS.md) 以及最终审查证据 [.runtime/review_result.json](.runtime/review_result.json) 汇总，不替代这些冻结契约或审查记录。

仓库内的 Golden Dataset 和 Demo 均为 **SYNTHETIC / DEMO_ONLY**。它们验证产品行为、边界与可复现性，不证明真实企业政策的正确性、完整性、授权有效性或生产质量，不得用于真实业务判断。

本版本仍坚持：系统提供知识定位、证据、辅助比较和核验路径；**最终业务判断与责任始终由 Human 承担**。

## 2. 已完成能力矩阵

| 能力域 | V0 已完成能力 | 主要实现/证据 |
|---|---|---|
| Ingestion / Knowledge Units | HTML 解析、结构感知 Knowledge Unit、原文与 semantic content 分离、Source Reference、Lineage、解析状态、Source Navigation | [ingestion/html.py](src/knowledge_system/ingestion/html.py)、[knowledge_units.py](src/knowledge_system/knowledge_units.py)、[source_navigation.py](src/knowledge_system/source_navigation.py) |
| Filter / Retrieval | 权限、状态、scope 硬过滤；Keyword/exact baseline；本地 Vector 与 Hybrid recall/fusion；可选 rerank 接口 | [candidate_filter.py](src/knowledge_system/candidate_filter.py)、[baseline_retrieval.py](src/knowledge_system/baseline_retrieval.py)、[hybrid_retrieval.py](src/knowledge_system/hybrid_retrieval.py) |
| 五类请求 | Locate、Browse、Discover、Apply/Compare、Out-of-Scope；Browse 使用集合语义，不要求 Top-1 | [request_classification.py](src/knowledge_system/request_classification.py)、[service.py](src/knowledge_system/service.py) |
| Context / Case | QueryContext 与 CaseContext 分离；original query 保留；显式 CONTINUE/UPDATE/NEW/AMBIGUOUS 关系 | [context.py](src/knowledge_system/context.py)、[agent_state.py](src/knowledge_system/agent_state.py) |
| Rule / Fact / Compare | 派生 Rule Structure、Information Requirements、按需事实提取、确定性条件比较、matched/missing/unknown 与 duplicate 处理 | [rule_structure.py](src/knowledge_system/rule_structure.py)、[information_requirements.py](src/knowledge_system/information_requirements.py)、[fact_extraction.py](src/knowledge_system/fact_extraction.py)、[deterministic_comparison.py](src/knowledge_system/deterministic_comparison.py) |
| Quality / Recovery / Diagnosis | Evidence 质量信号、有限 1–2 轮恢复、Retrieval/Processing/Source Knowledge 问题分离 | [evidence_quality.py](src/knowledge_system/evidence_quality.py)、[limited_recovery.py](src/knowledge_system/limited_recovery.py)、[failure_diagnosis.py](src/knowledge_system/failure_diagnosis.py) |
| Grounded Organizer | Organizer 仅输出引用已有 Evidence ID 的计划；Reference Guard 拒绝未知 ID；程序确定性 hydration | [grounded_organizer.py](src/knowledge_system/grounded_organizer.py)、[reference_guard.py](src/knowledge_system/reference_guard.py)、[evidence_map.py](src/knowledge_system/evidence_map.py) |
| Verification / Feedback | Evidence 不可靠时进入 Verification Mode；Feedback 只作为 observation，不自动成为正式知识 | [verification_mode.py](src/knowledge_system/verification_mode.py)、[feedback_log.py](src/knowledge_system/feedback_log.py) |
| Graph / Session | 有意义的状态转换图；结构化 snapshot/save/resume；不保存完整聊天历史，不做跨 session 个人记忆 | [product_graph.py](src/knowledge_system/product_graph.py)、[session_snapshot.py](src/knowledge_system/session_snapshot.py) |
| Evaluation | 24-case/72-expression Golden、Recall/Ranking/Browse/Behavior/Safety 分项指标、case/tag/full selection 与 baseline/current 可复现报告 | [golden_dataset.py](src/knowledge_system/golden_dataset.py)、[evaluation_metrics.py](src/knowledge_system/evaluation_metrics.py)、[regression_runner.py](src/knowledge_system/regression_runner.py)、[golden_dataset_v1.json](data/golden_dataset_v1.json) |
| Version / Index | source commit 与 indexed commit 分离；OUT_OF_SYNC 才更新；完整成功后才推进 indexed commit | [system_state.py](src/knowledge_system/system_state.py)、[incremental_update.py](src/knowledge_system/incremental_update.py) |
| Product surface | UI 无关 `KnowledgeService`、CLI JSON/text、稳定退出码、8 个合成 Demo | [service.py](src/knowledge_system/service.py)、[cli.py](src/knowledge_system/cli.py)、[demo_runner.py](src/knowledge_system/demo_runner.py)、[data/demo](data/demo) |

## 3. Human Quickstart

以下命令从仓库根目录执行。需要 Python 3.10+；在未安装 package 的源码工作区中，PowerShell 先设置本次终端的 `PYTHONPATH`：

```powershell
$env:PYTHONPATH = "src"
python -m knowledge_system --help
python -m knowledge_system --version
```

查看并运行 **SYNTHETIC / DEMO_ONLY** 场景：

```powershell
python -m knowledge_system demo list --format text
python -m knowledge_system demo all --format text
python -m knowledge_system demo run exact-locate --format json
```

`demo all` 依次覆盖 8 个场景：`exact-locate`、`browse-scope`、`fuzzy-discover`、`case-threshold-update`、`duplicate-compare`、`knowledge-missing`、`source-ambiguity`、`out-of-scope`。清单和输入见 [scenarios.v1.json](data/demo/scenarios.v1.json)，合成 Knowledge Units 见 [corpus.v1.json](data/demo/corpus.v1.json)。

直接查询同一合成 corpus，并分别查看 JSON/text 输出：

```powershell
python -m knowledge_system query --corpus data/demo/corpus.v1.json --query "定位演示积分复核门槛规则" --session-id human-v0-json --scope demo-payments --knowledge-version demo-v1 --index-version demo-index-demo-v1 --format json
python -m knowledge_system query --corpus data/demo/corpus.v1.json --query "浏览全部演示支付规则" --session-id human-v0-text --scope demo-payments --format text
```

保存并显式恢复结构化 session snapshot：

```powershell
python -m knowledge_system query --corpus data/demo/corpus.v1.json --query "定位演示积分复核门槛规则" --session-id human-v0-resume --scope demo-payments --format json --save-snapshot .\human-v0.snapshot.json
python -m knowledge_system resume --snapshot .\human-v0.snapshot.json --session-id human-v0-resume --format json
```

CLI 的稳定退出码为：`0` 成功、`2` 输入/本地数据错误、`3` 内部错误。错误写入 stderr；正常结果写入 stdout。Snapshot 仅恢复显式结构化工作上下文，不包含完整聊天历史或跨 session 个人记忆。上述 Demo 数据不是正式企业资料。

## 4. T-1302 Final Review 结果

以下数字来自最终审查记录 [.runtime/review_result.json](.runtime/review_result.json)。各指标独立报告，**没有复合总分**。

- Full regression：43 个测试模块，346/346 expanded executions 通过，0 failed、0 skipped、0 omitted。
- Full Golden：24 cases / 72 expressions；同一 Selection Plan 运行两次，pair object、JSON bytes 与 strict round-trip 均一致。
- Baseline：Recall `0.764705882353`；MRR `0.764705882353`；NDCG `0.764705882353`；Browse coverage/precision/pass rate 均为 `1.0`；Behavior `0.569444444444`；Safety `1.0`。
- Current：Recall `0.862745098039`；MRR `0.830882352941`；NDCG `0.838159635978`；Browse coverage/precision/pass rate 均为 `1.0`；Behavior `0.597222222222`；Safety `1.0`。
- Hybrid/Discover：Discover Recall 从 `0.50` 提升至 `0.75`；Discover MRR 从 `0.50` 提升至 `0.614583333333`。
- Locate：Recall 和 MRR 均为 `0.888888888889 → 0.888888888889`，无退化。
- Browse 按集合指标评估，不按 Top-1；permission/inactive/unknown Evidence violations 为 `0`；regressed expression IDs 为空。
- Demo：8 个场景运行两轮，16/16 通过，输出 byte-stable。
- 系统不变量：15/15 通过。
- Adversarial matrix：10/10 通过。

Golden 的 plan fingerprint 为 `52cfb0d79024b06a9e9f5264bca5cc9ad5f685792c49f2a21ab6198fc0c5ff9c`。这些结果证明当前合成评测范围内的行为与可复现性，不外推到真实企业政策或生产流量。

## 5. Trace 与演示证据映射

实际 `QueryResult.trace` 可序列化字段定义在 [contracts.py](src/knowledge_system/contracts.py)：

- 身份与输入：`trace_id`、`raw_query`、`retrieval_query`；
- 选择过程：`candidate_evidence_ids`、`final_evidence_ids`、`applied_filters`、`filter_summary`；
- 版本与来源：`knowledge_version`、`index_version`、`source_references`、`processing_versions`；
- 检索与排序：`retrieval_paths`、`fusion_scores`、`rerank_summary`、`discover_summary`。

下列信息是可核验的审计伴随信息，但并非全部塞入同一个 `Trace` 对象：

- intent 由结果的 `request_type` / State 的 `knowledge_request` 承载；scope 来自显式 query 输入和 `QueryContext`；
- Source/Lineage 与原文位于每条 `evidence`；
- quality 位于 `quality_signals`，并决定 `verification`；
- recovery、diagnosis、comparison、organizer references 分别由对应模块、AgentState/控制流或 Demo step 输出承载。当前薄 CLI surface 不会伪造这些字段，也不声称每次普通 query 都执行了所有阶段。

| 演示 | 可直接核验的信息 |
|---|---|
| `exact-locate` | raw/retrieval query、request type、candidate/final Evidence IDs、filters、版本、Evidence 原文、Source 与 Lineage |
| `browse-scope` | 显式 scope 内的 Evidence 集合；验证 Browse 不是 Top-1 |
| `fuzzy-discover` | `retrieval_paths`、fusion/discover/rerank 摘要与正式 Evidence 名称，验证 hybrid semantic 路径 |
| `case-threshold-update` | original query 保留、Case raw descriptions 追加、comparison 的 matched/missing/unknown/not_satisfied |
| `duplicate-compare` | comparison、duplicate group count 与逻辑结果，验证重复输入确定性处理 |
| `knowledge-missing` | 空 Evidence、Knowledge Missing 边界与核验提示；不表达为“企业没有” |
| `source-ambiguity` | quality signals、原始 Evidence/Source、Verification Mode 及核验路径；不猜测结论 |
| `out-of-scope` | intent/boundary、空 Evidence，且不回退通用问答 |

更深层的 recovery/diagnosis/organizer/reference-guard 行为由相应实现和 Final Review 的 346/346 tests、15/15 invariants、10/10 adversarial evidence 覆盖；Demo 并未宣称把每个内部阶段都暴露为 CLI 字段。

## 6. 已知限制与未解决 Gaps

1. **数据真实性边界**：Golden 与 Demo 全部是 synthetic；没有验证真实企业政策的质量、完整性、适用性或授权配置。
2. **Golden Behavior gap**：Current Behavior pass rate 为 `0.597222222222`，不是满分。
3. **OOS fixture/metric 矛盾**：15 个 Out-of-Scope expressions 同时要求零 Evidence 和 per-Evidence source/lineage coverage，形成 provenance fixture/metric contradiction；此矛盾尚未在本轮改写 fixture 或 metric。
4. **非 OOS 行为/provenance gaps**：仍有 14 个 expressions 存在行为或 provenance gap。
5. **Recall-only Locate gap**：另有 1 个 Locate expression 仅 Recall 失败；Locate 聚合指标未退化不代表每个表达均命中。
6. **本地语义实现**：只验证本地确定性 vector index；外部 embedding/LLM 的质量、可用性、延迟、成本与供应商变化未评估。
7. **真实授权与安全集成**：未连接真实企业 authorization provider；OS ACL 等应用过滤之外的纵深控制未做生产验证。
8. **格式覆盖**：没有生产级验证真实世界格式广度；OCR/图片型 PDF 不属于 V0 必需能力，文本型 PDF 也仅有限支持。
9. **规模与运行质量**：未做 load/soak、并发、性能、容量或 crash durability release benchmark。
10. **外部环境**：未验证生产凭据、外部网络服务、真实保密 corpus 或运维部署流程。
11. **历史审计限制**：最终 `review_result.json` 是 latest-result storage，精确 P11 历史数值 artifact 已被后续审查覆盖；当前确定性 rerun 与 events 只能证明无已观察到的回归。包含仓库当时无 commits 且项目显示 untracked，无法提供受保护文件的历史归属证明。

这些 gap 是透明的 V0 限制，不应被解释为已经解决，也不应以 Final Review PASS 掩盖。

## 7. 下一轮候选改进（尚未实现）

下表仅给出由现有证据触发的候选方向；**均未在 V0/T-1303 中实现**，也没有新增阈值、数据机制或修改冻结契约。是否进入下一轮必须由 Human 体验成品后决定，并先更新契约。

| 已有证据 | 下一轮候选（未实现） |
|---|---|
| Behavior `0.597222222222` 与 14 个非 OOS gaps | 逐 expression 归因 behavior/provenance 失败，再决定是 adapter、fixture 还是产品行为需调整 |
| 15 个 OOS provenance 矛盾 | 明确零 Evidence 场景的 provenance 语义，并修订互相冲突的 fixture/metric 契约 |
| 1 个 recall-only Locate gap | 对该表达做 tokenizer/alias/keyword/hybrid Trace 分析，在不损害精确专名的前提下评估最小召回改进 |
| Synthetic-only 证据 | 由 Human 提供经授权、脱敏的真实企业 corpus 和验收边界，建立独立真实资料评测集 |
| Local deterministic vector only | 在证据证明需要后，对候选 embedding/LLM 做质量、延迟、成本、故障与数据合规评估 |
| 真实授权 provider 未接入 | 设计真实身份/权限集成与 application filter、OS/平台控制的纵深验证方案 |
| OCR/格式广度未覆盖 | 按真实失败样本决定 DOCX/CSV/PDF/OCR parser 的优先级，不预先扩张格式机制 |
| 无性能/并发/耐久 benchmark | 先定义真实 workload、SLO 与故障模型，再增加 load/soak/concurrency/crash-recovery 验证 |
| latest-result 与无 Git 历史限制 | 评估不可覆盖的审查 artifact 归档和正式版本化/发布审计策略 |

## 8. 完成责任与停止条件

T-1302 Final Review 已确认 V0 可以由 Human 独立运行体验。T-1303 只生成本报告，不修改产品实现、测试、数据、指标或冻结契约。

当 Orchestrator 验收本报告并将 Runtime 推进至 `PROJECT_COMPLETE` 后，依照 [BUILD.md](BUILD.md) 必须停止自动新增机制。后续变化由 Human 基于完整成品体验，重新修改契约并启动新一轮 Runtime。任何 Evidence Map、Comparison 或 Verification 输出都仅用于辅助核验；最终业务判断和责任不转移给系统。
