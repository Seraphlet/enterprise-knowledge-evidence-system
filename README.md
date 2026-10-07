# Enterprise Knowledge Evidence System

一个以正式知识为事实来源、以可追溯 Evidence 为输出，并在知识边界处明确停下来的企业知识系统 V0。

> **重要声明：** 仓库中的 Golden Dataset 与 Demo 全部为 **SYNTHETIC / DEMO_ONLY**。它们验证软件行为、边界和可复现性，不证明任何真实企业政策的正确性、完整性、适用性、授权有效性或生产质量。系统只辅助定位、比较与核验；最终业务判断和责任始终由人承担。

## 中文

### 项目定位

传统企业知识库通常按文件组织，而员工是从当前任务或问题出发。本项目把自然语言请求映射到可核验的正式 Evidence，并保留原文、来源、处理版本和选择过程。

V0 支持五类 Knowledge Request：

- `Locate`：按规则名、术语、文档或近精确表达定位知识；
- `Browse`：浏览某个 scope 内的知识集合，不把正确性简化成 Top-1；
- `Discover`：在不知道正式名称时，通过 hybrid retrieval 发现正式 Knowledge Unit；
- `Apply / Compare`：以规则驱动事实提取，确定性比较已知、缺失与未知条件；
- `Out of Scope`：明确停止，不退化成通用问答。

系统不会替用户做最终审批，不会用模型补写缺失政策，也不会把“未检索到”表述为“企业不存在该规定”。

### 核心能力与架构

```text
Enterprise Documents
  -> structure-aware parsing -> Knowledge Units + Source/Lineage
  -> keyword index + local vector index

Employee Request
  -> context/request classification
  -> permission/status/scope filter
  -> exact/keyword + vector retrieval -> fusion/optional rerank
  -> evidence quality -> limited recovery/diagnosis
  -> rule-driven fact extraction -> deterministic comparison
  -> grounded plan -> reference guard -> deterministic hydration
  -> Evidence Map or Verification Mode
```

主要设计边界：

- 原始 Source Evidence 的权威高于 Knowledge Unit 和派生 Rule Structure；
- 权限、明确失效状态和错误 scope 是硬过滤边界，不交给 embedding 或 LLM 判断；
- Organizer 只能排列并引用候选 Evidence ID，不能创造新事实或新来源；
- 数值、布尔、枚举等明确条件由普通程序比较；无法可靠判断的条件保持 `UNKNOWN` / `HUMAN_REQUIRED`；
- Evidence 不可靠时进入 Verification Mode，并给出原文、问题、来源和人工核验路径；
- Session snapshot 只保存显式结构化上下文，不保存完整聊天历史，也不提供跨会话个人记忆。

### V0 状态与 Final Review

V0 已完成并通过 T-1302 Final Review。以下结果来自仓库最终审查，指标分别报告，不使用复合总分：

| 检查项 | 结果 |
|---|---:|
| Full regression | 346 / 346 |
| Full Golden | 24 cases / 72 expressions；同一计划双跑一致 |
| Current Recall | `0.862745098039` |
| Current MRR | `0.830882352941` |
| Current NDCG | `0.838159635978` |
| Discover Recall | `0.50 -> 0.75` |
| Locate | Recall/MRR 均无退化 |
| Browse / Safety | 各项 `1.0`；安全违规 `0` |
| Demo | 8 个场景 x 2 轮，16 / 16 |
| 系统不变量 | 15 / 15 |
| Adversarial matrix | 10 / 10 |

这些数字只证明当前合成评测范围内的行为和可复现性，不能外推到真实政策或生产流量。完整证据、精确口径和限制见 [V0 完成报告](V0_COMPLETION_REPORT.md)。

### 安装与源码运行

前提：Python 3.10+。项目当前没有声明第三方运行时依赖。

在仓库根目录使用 PowerShell：

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -e .
knowledge-system --help
knowledge-system --version
```

也可以不安装 package，直接从源码运行：

```powershell
$env:PYTHONPATH = "src"
python -m knowledge_system --help
```

### Quickstart

查看并运行内置合成场景：

```powershell
knowledge-system demo list --format text
knowledge-system demo run exact-locate --format json
knowledge-system demo all --format text
```

8 个场景覆盖 exact locate、scope browse、fuzzy discover、case threshold update、duplicate compare、knowledge missing、source ambiguity 和 out-of-scope。

查询本地 KnowledgeUnit JSON corpus：

```powershell
knowledge-system query `
  --corpus data/demo/corpus.v1.json `
  --query "定位演示积分复核门槛规则" `
  --session-id demo-query-001 `
  --scope demo-payments `
  --knowledge-version demo-v1 `
  --index-version demo-index-v1 `
  --format json
```

保存并恢复显式 session snapshot：

```powershell
knowledge-system query `
  --corpus data/demo/corpus.v1.json `
  --query "浏览全部演示支付规则" `
  --session-id demo-resume-001 `
  --scope demo-payments `
  --format text `
  --save-snapshot .\demo.snapshot.json

knowledge-system resume `
  --snapshot .\demo.snapshot.json `
  --session-id demo-resume-001 `
  --format json
```

CLI 正常结果写入 stdout，错误写入 stderr；退出码为 `0`（成功）、`2`（输入或本地数据错误）、`3`（内部错误）。`KnowledgeService` 是 UI 无关的 Python 接口，本仓库没有声称提供网络服务端。

### 项目结构

```text
src/knowledge_system/   核心模型、摄取、检索、比较、Evidence 与 CLI
tests/                  单元、集成、边界与对抗测试
data/demo/              SYNTHETIC / DEMO_ONLY corpus 与场景
data/golden_dataset_v1.json
                        合成 Golden Dataset
data/fixtures/          测试 fixture
PRODUCT.md              产品目标与边界
SYSTEM.md               系统架构与不变量
DECISIONS.md            已冻结的设计决策
TASKS.md                V0 施工与验收计划
BUILD.md                多角色构建与审查契约
V0_COMPLETION_REPORT.md 最终能力、指标、Trace 与限制
```

### Evidence、来源与安全边界

每条关键 Evidence 尽可能保留完整来源名称、URL 或 section/page/sheet 等可靠位置，以及 parser/chunker/processing version。系统区分：

- Case Information Missing：规则存在，但当前 Case 信息不足；
- Knowledge Missing：当前知识库没有检索到所需内容；
- Knowledge Ambiguous：正式资料自身冲突、不完整或含糊；
- Retrieval / Processing / Source Knowledge Problem：分别诊断检索、处理和源资料问题。

`Retrieved != Answerable`。检索到标题或片段不代表足以支持结论。Employee Feedback 只作为 observation 记录，不会自动升级成 Formal Knowledge。真实身份系统、企业授权 provider、生产访问控制和真实保密语料尚未接入或验证。

### 已知限制

- Golden 与 Demo 仅为合成数据；真实企业政策、权限配置和生产质量未验证；
- Current Behavior pass rate 为 `0.597222222222`，不是满分；
- Out-of-Scope fixture 对“零 Evidence”与 per-Evidence provenance 同时提出要求，存在尚未解决的指标矛盾；
- 仍有 14 个非 OOS expressions 存在 behavior 或 provenance gap，另有 1 个 Locate expression 为 recall-only gap；
- 语义检索只验证了本地确定性 vector index，未评估外部 embedding/LLM 的质量、延迟、成本和合规；
- 尚未完成真实授权集成、生产级格式广度、OCR/图片型 PDF、负载/并发/容量/崩溃耐久或部署验证；文本型 PDF 仅有限支持。

### 文档

- [产品契约](PRODUCT.md)
- [系统契约](SYSTEM.md)
- [设计决策](DECISIONS.md)
- [任务与验收计划](TASKS.md)
- [构建与评审契约](BUILD.md)
- [V0 完成报告](V0_COMPLETION_REPORT.md)

---

## English

### Positioning

Enterprise knowledge is usually organized around files, while employees work from a task or question. This project maps natural-language requests to verifiable formal Evidence while preserving source text, provenance, processing versions, and the selection trail.

V0 supports five Knowledge Request types:

- `Locate`: find knowledge by rule name, term, document, or near-exact wording;
- `Browse`: return a relevant set within a scope rather than reducing correctness to Top-1;
- `Discover`: use hybrid retrieval to find a formal Knowledge Unit when the official name is unknown;
- `Apply / Compare`: extract facts from rule requirements and deterministically report matched, missing, and unknown conditions;
- `Out of Scope`: stop explicitly instead of falling back to general question answering.

The system does not make final approvals, use a model to fill missing policy, or turn “not retrieved” into “the enterprise has no such rule.”

### Core capabilities and architecture

```text
Enterprise Documents
  -> structure-aware parsing -> Knowledge Units + Source/Lineage
  -> keyword index + local vector index

Employee Request
  -> context/request classification
  -> permission/status/scope filter
  -> exact/keyword + vector retrieval -> fusion/optional rerank
  -> evidence quality -> limited recovery/diagnosis
  -> rule-driven fact extraction -> deterministic comparison
  -> grounded plan -> reference guard -> deterministic hydration
  -> Evidence Map or Verification Mode
```

Key boundaries:

- original Source Evidence has higher authority than Knowledge Units and derived Rule Structures;
- permission, inactive status, and clearly wrong scope are hard filters, not embedding or LLM decisions;
- the Organizer may only order and reference candidate Evidence IDs; it cannot create facts or sources;
- explicit numeric, Boolean, and enum conditions are compared by deterministic code; uncertain conditions remain `UNKNOWN` / `HUMAN_REQUIRED`;
- unreliable Evidence triggers Verification Mode with source text, the problem, provenance, and a human verification path;
- session snapshots contain explicit structured context only, not full chat history or cross-session personal memory.

### V0 status and Final Review

V0 is complete and passed the T-1302 Final Review. Metrics are reported independently; there is no composite score.

| Check | Result |
|---|---:|
| Full regression | 346 / 346 |
| Full Golden | 24 cases / 72 expressions; identical-plan double run was reproducible |
| Current Recall | `0.862745098039` |
| Current MRR | `0.830882352941` |
| Current NDCG | `0.838159635978` |
| Discover Recall | `0.50 -> 0.75` |
| Locate | no Recall or MRR regression |
| Browse / Safety | each `1.0`; `0` safety violations |
| Demo | 8 scenarios x 2 runs, 16 / 16 |
| System invariants | 15 / 15 |
| Adversarial matrix | 10 / 10 |

These results establish behavior and reproducibility only within the current synthetic evaluation scope. They do not validate real enterprise policies or production traffic. See the [V0 Completion Report](V0_COMPLETION_REPORT.md) for exact definitions, evidence, and limitations.

### Installation and source execution

Prerequisite: Python 3.10+. The project currently declares no third-party runtime dependencies.

From the repository root in PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -e .
knowledge-system --help
knowledge-system --version
```

To run directly from source without installing the package:

```powershell
$env:PYTHONPATH = "src"
python -m knowledge_system --help
```

### Quickstart

List and run the bundled synthetic scenarios:

```powershell
knowledge-system demo list --format text
knowledge-system demo run exact-locate --format json
knowledge-system demo all --format text
```

The eight scenarios cover exact locate, scoped browse, fuzzy discover, case threshold update, duplicate compare, knowledge missing, source ambiguity, and out-of-scope behavior.

Query a local KnowledgeUnit JSON corpus:

```powershell
knowledge-system query `
  --corpus data/demo/corpus.v1.json `
  --query "定位演示积分复核门槛规则" `
  --session-id demo-query-001 `
  --scope demo-payments `
  --knowledge-version demo-v1 `
  --index-version demo-index-v1 `
  --format json
```

Save and explicitly resume a session snapshot:

```powershell
knowledge-system query `
  --corpus data/demo/corpus.v1.json `
  --query "浏览全部演示支付规则" `
  --session-id demo-resume-001 `
  --scope demo-payments `
  --format text `
  --save-snapshot .\demo.snapshot.json

knowledge-system resume `
  --snapshot .\demo.snapshot.json `
  --session-id demo-resume-001 `
  --format json
```

Normal output goes to stdout and errors go to stderr. Exit codes are `0` for success, `2` for input/local-data errors, and `3` for internal errors. `KnowledgeService` is a UI-independent Python interface; this repository does not claim to provide a network server.

### Repository layout

```text
src/knowledge_system/   Core models, ingestion, retrieval, comparison, Evidence, and CLI
tests/                  Unit, integration, boundary, and adversarial tests
data/demo/              SYNTHETIC / DEMO_ONLY corpus and scenarios
data/golden_dataset_v1.json
                        Synthetic Golden Dataset
data/fixtures/          Test fixtures
PRODUCT.md              Product goals and boundaries
SYSTEM.md               Architecture and invariants
DECISIONS.md            Frozen design decisions
TASKS.md                V0 implementation and acceptance plan
BUILD.md                Multi-role build and review contract
V0_COMPLETION_REPORT.md Final capabilities, metrics, Trace, and limitations
```

### Evidence, provenance, and safety boundaries

Each material Evidence item preserves as much reliable provenance as available: the full source name, a URL or section/page/sheet location, and parser/chunker/processing versions. The system distinguishes:

- Case Information Missing: the rule exists, but case facts are incomplete;
- Knowledge Missing: required knowledge was not retrieved from the current knowledge base;
- Knowledge Ambiguous: the formal source is conflicting, incomplete, or ambiguous;
- Retrieval / Processing / Source Knowledge Problem: separate diagnoses for retrieval, transformation, and source-material failures.

`Retrieved != Answerable`. Finding a title or fragment does not establish enough support for a conclusion. Employee Feedback remains an observation and is not promoted automatically to Formal Knowledge. Real identity systems, enterprise authorization providers, production access controls, and confidential corpora have not been integrated or validated.

### Known limitations

- Golden and Demo data are synthetic; real enterprise policy, authorization configuration, and production quality are unvalidated;
- the Current Behavior pass rate is `0.597222222222`, not a perfect score;
- the Out-of-Scope fixture has an unresolved metric contradiction between zero Evidence and per-Evidence provenance requirements;
- 14 non-OOS expressions still have behavior or provenance gaps, and one additional Locate expression has a recall-only gap;
- semantic retrieval was validated only with the local deterministic vector index; external embedding/LLM quality, latency, cost, and compliance were not evaluated;
- real authorization integration, production format breadth, OCR/image PDF support, load/concurrency/capacity/crash-durability testing, and deployment validation remain out of scope; text PDF support is limited.

### Documentation

- [Product contract](PRODUCT.md)
- [System contract](SYSTEM.md)
- [Design decisions](DECISIONS.md)
- [Task and acceptance plan](TASKS.md)
- [Build and review contract](BUILD.md)
- [V0 Completion Report](V0_COMPLETION_REPORT.md)
