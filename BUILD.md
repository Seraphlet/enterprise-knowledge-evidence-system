# BUILD.md — Autonomous Multi-Agent Runtime Contract

> 本文件定义“谁来造、怎么造、什么时候测试、什么时候停”。Runtime 一旦 START，除 Human 生命周期状态外，不接受中途人工遥控式修改。

## 1. Runtime Philosophy

Human 定义 Runtime Contract；Agent 在 Contract 内自治。Human 不做逐步验收。

**Implementation Error → Runtime 自己闭环。**  
**Evidence-backed Structural Error → Graceful Pause → Human。**

只要没有系统性/结构性错误，就先把 V0 做成完整可运行成品，再由 Human 用成品产生下一轮设计判断。

## 2. Roles

- `Orchestrator`：调度/状态转换/故障升级；不写业务实现。
- `Builder`：实现；不能宣布最终 PASS。
-`Reviewer`：仅在 Phase Gate、大型跨阶段更新、关键共享组件或发布节点进行独立审查与测试；原则上不修改业务实现。
- `Watchdog`：只读监察权限/状态/日志/Git 异常；不修复。
- `Human`：冻结契约与 Runtime 生命周期最高权力。

## 3. Human Runtime State — Highest Priority

`.runtime/HUMAN_RUNTIME_STATE` 只允许：
- `RUN`
- `PAUSE_AFTER_CURRENT`

禁止把它扩展为“删掉重写/跳过某测试/换架构”等中途业务指令。

### Hard invariant

**NO DISPATCH WITHOUT HUMAN-STATE CHECK.**

Orchestrator 每次派发任何新工作前必须重新读取该文件。

- RUN：允许按 TASKS/state 派发。
- PAUSE_AFTER_CURRENT：已派出的当前工作正常完成、review/状态收口后，不派发下一工作；state → PAUSED。

需要立即中断时由 Human 直接终止 Agent 软件/进程，不在 V0 设计复杂 emergency kill 协议。

## 4. Runtime Files / Ownership

```text
.runtime/
├── HUMAN_RUNTIME_STATE       # Human write; Agents read-only
├── state.json                # Orchestrator owns
├── current_task.json         # Orchestrator owns
├── builder_result.json       # Builder owns
├── review_result.json        # Reviewer owns
├── watchdog_alert.json       # Watchdog owns
└── events.jsonl              # append-only audit/handoff log
```

原则：一个文件一个主要 writer，减少 Lost Update。所有结果携带 `actor/task_id/based_on_runtime_version`。

## 5. Optimistic Concurrency Guard

任何 Agent 执行写入前检查：
1. actor/role 是否匹配；
2. task_id 是否匹配；
3. `based_on_runtime_version == state.runtime_version`；
4. writable_scope 是否允许；
5. Human Runtime State 是否允许 Orchestrator 派发新工作。

不满足 → 不覆盖状态，返回 `RUNTIME_CONFLICT` / `STALE_RESULT` / `POLICY_VIOLATION`。

`state.json` 每次合法状态转换 `runtime_version += 1`。

## 6. Dispatch Cycle

Runtime 默认优先完成项目。普通任务采用 Builder-only 流程，Reviewer 仅用于大型更新、Phase Gate 和发布检查。

### 6.1 普通任务流程

普通任务指单个 `T-xxx` 实现任务，且不涉及 Phase Gate、跨阶段迁移、关键共享组件或发布状态。

```text
Orchestrator reads HUMAN_RUNTIME_STATE + state + TASKS
→ if RUN and work ready: write current_task
→ Builder implements + minimal self-check → builder_result
→ Orchestrator validates actor/task/runtime_version/writable_scope
→ READY_FOR_REVIEW is treated as BUILDER_ACCEPTED for ordinary tasks
→ Orchestrator advances to the next task without dispatching Reviewer

## 7. Testing Policy — Completion-first, Risk-based Escalation

测试目标是在保持关键证据的前提下优先完成项目。禁止为每个普通任务重复运行 Reviewer、宽回归或全量测试。

### 7.1 普通 Builder 任务

普通 `T-xxx` 任务只运行与本次修改直接相关的最小测试：

- static check、compile check 或 import smoke；
- 新增或修改模块的 focused unit tests；
- Reviewer 已提供失败证据时，只运行精确 regression tests；
- 只有直接依赖发生变化时，才增加少量 adjacent smoke。

普通任务默认禁止：

- 全量测试；
- Phase integration tests；
- 与修改无直接关系的影响回归；
- 重建全部 embedding/index；
- 为普通任务单独派发 Reviewer。

Builder 必须在 `builder_result.json` 中记录：

- 实际执行的测试命令；
- 测试数量和结果；
- 未执行的宽回归或全量测试；
- writable scope audit；
- 已知风险或被推迟到 Phase Gate 的验证范围。

### 7.2 Phase Gate 与大型更新

Phase Gate 或其他大型更新运行与该阶段能力对应的集成测试：

- 覆盖本阶段所有已累计的 Builder-only 任务；
- 覆盖 TASKS 中明确要求的 Phase Gate fixture；
- 覆盖关键跨模块数据流和 provenance；
- 只在共享组件影响明确时增加 impact regression；
- 不因 Phase Gate 自动运行整个项目的全量测试。

Reviewer 必须记录：

- 独立测试矩阵；
- 实际测试范围和数量；
- 未覆盖范围；
- PASS/FAIL 证据；
- 是否允许建立新的 verified checkpoint。

### 7.3 Failure Retry

Reviewer 或 Builder 发现局部失败后：

- Builder 只运行失败复现、修复回归和必要的最小相邻测试；
- Reviewer 只复验原失败范围和必要影响范围；
- 不因单点修复重新运行整个 Phase Gate；
- 同类失败重复出现时，执行 Root Cause Escalation；
- 只有根因可能影响整个阶段时，才重新运行完整 Phase integration tests。

### 7.4 Full Regression

全量测试只允许在以下节点运行：

- `V0_COMPLETE_PENDING_REGRESSION`；
- release candidate；
- Golden Dataset；
- Final Review；
- 有证据证明关键共享组件可能造成全项目回归。

除上述节点外，不得运行 full regression。

### 7.5 Test Frequency Principle

测试频率遵循以下优先级：

```text
ordinary task
→ minimal Builder checks only

local retry
→ exact regression only

Phase Gate / major phase transition
→ independent Reviewer + phase integration

release candidate
→ full regression + Golden Dataset + Final Review

## 8. Failure / Retry Policy

一次 FAIL 默认是实现问题，由 Builder 根据 Reviewer Evidence 修复。

重复同类失败不能无限循环。Orchestrator 必须进行 Root Cause Escalation，至少区分：
- Implementation bug
- Test/fixture bug
- Environment/dependency problem
- Requirement misunderstanding
- Architecture/contract conflict

局部可修复 → 自主修复继续。环境问题 → 在契约内诊断/修复；不得通过破坏冻结边界“让测试绿”。

## 9. Structural Block

只有以下情况允许主动暂停并找 Human：
- PRODUCT 要求互相冲突；
- SYSTEM 架构被实现证据证明不可成立；
- DECISIONS 发生不可调和冲突；
- 修复必须突破明确权限/安全边界；
- 连续失败经排除 implementation/test/environment 后指向架构；
- 外部约束使既定方案不可实现，且替代方案会改变冻结结构。

不得因为“实现困难/想重构/测试麻烦”宣称结构性错误。

Structural Block 必须输出：task、observed、attempts、ruled_out、conflicting_contract、evidence、why_local_fix_insufficient、possible_options（可选）、runtime state。然后 Graceful Pause。

## 10. Git / Safety

Git 是恢复与审计的硬保险：
- 保留可检查 diff；
- 禁止 destructive reset/history rewrite/删除保护资料等危险操作，除非 Human 明确在 Runtime 外批准；
- 冻结文档 `PRODUCT.md/SYSTEM.md/DECISIONS.md/BUILD.md` 默认 protected；Worker 不得自行改写；
- 删除文件、批量覆盖、secret/生产外部写入属于危险操作，应由实际工具权限/审批阻止，而不是只靠 Prompt。

若 OS/Agent 平台支持 ACL，应优先把 Prompt 权限逐步升级为程序/系统权限；V0 不要求为了 ACL 阻塞产品施工。

## 11. Watchdog

Watchdog 只读项目、state、events、git diff。优先用确定性 invariant 检查：
- Builder 修改 protected docs；
- Reviewer 修改 `src/**`；
- 非 Orchestrator 修改 state；
- actor != expected_actor；
- stale runtime version；
- protected file deletion；
- event/state 不一致。

正常时静默。异常写 `watchdog_alert.json` 并记录 event；严重越权可要求 Orchestrator 在当前工作收口后 BLOCK/PAUSE。

## 12. Events / Handoff Log

`events.jsonl` 是 append-only 的通信历史 + 审计历史 + 故障现场。每条建议包含：seq/time/runtime_version/actor/action/task/result/evidence refs。

`state.json` = 现在是什么状态。  
`events.jsonl` = 为什么变成这个状态。

## 13. Resume

第二天 Human 将 `HUMAN_RUNTIME_STATE` 从 `PAUSE_AFTER_CURRENT` 改为 `RUN`，启动 Orchestrator。

Orchestrator：
1. 读取 Human state；
2. 读取 state/events/current results；
3. 检查 Git 与 last verified checkpoint 一致性；
4. 恢复未完成的合法状态；
5. 从 next_task/next_actor 继续。

不得依赖昨天的聊天上下文恢复项目。

## 14. Completion

最后 Phase 完成后自动进入：
`V0_COMPLETE_PENDING_REGRESSION → Full Regression → Golden Dataset → Final Review → PROJECT_COMPLETE`。

PROJECT_COMPLETE 后停止自动新增机制，输出 completion report 给 Human。下一轮修改由 Human 基于完整成品体验重新修改契约后启动。

## Phase 14 测试策略

T-1400 至 T-1403 的 Builder 可以编写测试，但不得执行自动化测试、全量测试或回归测试。

T-1403 完成功能合并后，才进入 P14-GATE。P14-GATE 由 Reviewer 一次性执行：

1. HTML / Markdown / CSV 导入聚焦测试；
2. 混合格式批量导入集成测试；
3. 导入结果进入 query 的最小链路测试；
4. 必要的既有 HTML 行为回归测试。

不得执行项目全量测试。若 Gate 失败，只允许在修复完成后重新执行相关失败测试。