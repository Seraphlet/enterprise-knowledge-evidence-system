# Runtime Protocol Quick Reference

## Priority
Human Runtime State > Frozen Contract > TASKS > Orchestrator decision > Worker implementation preference.

## State transitions (suggested)
READY → BUILDING → BUILD_DONE → REVIEWING → PASS/RETRY → next task.
Phase end → PHASE_GATE → PASS → next phase.
Structural evidence → STRUCTURAL_BLOCKED → PAUSED.
Human `PAUSE_AFTER_CURRENT` → current dispatched work closes → PAUSED.
Final → V0_COMPLETE_PENDING_REGRESSION → FINAL_REVIEW → PROJECT_COMPLETE.

## Result guard
Every result: actor + task_id + based_on_runtime_version. Stale or wrong actor is rejected, never merged silently.

## Runtime maxim
TASKS = 路线图；state = 当前位置；HUMAN_RUNTIME_STATE = 方向盘；events = 行车记录仪。
