"""Bounded deterministic orchestration for Evidence recovery attempts."""

from __future__ import annotations

import json
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from enum import Enum

from .contracts import Evidence, KnowledgeLineage, SourceReference
from .evidence_quality import EvidenceQualitySignal, EvidenceQualitySignalType


class RecoveryAction(str, Enum):
    """Allowed recovery actions in their fixed execution priority."""

    NEIGHBOR_EXPANSION = "NEIGHBOR_EXPANSION"
    PARENT_UNIT = "PARENT_UNIT"
    SECTION_EXPANSION = "SECTION_EXPANSION"
    RE_RETRIEVAL = "RE_RETRIEVAL"
    ORIGINAL_SOURCE_CONTEXT = "ORIGINAL_SOURCE_CONTEXT"


class RecoveryStatus(str, Enum):
    """Completion state without assigning a failure diagnosis."""

    NOT_NEEDED = "NOT_NEEDED"
    RESOLVED = "RESOLVED"
    UNRESOLVED = "UNRESOLVED"


@dataclass(frozen=True)
class RecoveryTarget:
    """A target derived verbatim from existing source or lineage identity."""

    target_id: str
    source_reference: SourceReference
    lineage: KnowledgeLineage


@dataclass(frozen=True)
class RecoveryRequest:
    """Pure request supplied to an injected deterministic executor."""

    round_number: int
    action: RecoveryAction
    target: RecoveryTarget
    trigger_signals: tuple[EvidenceQualitySignal, ...]


@dataclass(frozen=True)
class RecoveryExecutionResult:
    """Executor output; remaining signals are explicit, never inferred."""

    recovered_evidence: tuple[Evidence, ...]
    remaining_signals: tuple[EvidenceQualitySignal, ...]
    failure_reason: str | None = None


RecoveryExecutor = Callable[[RecoveryRequest], RecoveryExecutionResult]


@dataclass(frozen=True)
class RecoveryRoundTrace:
    """Complete record of one bounded recovery attempt."""

    round_number: int
    action: RecoveryAction
    target: RecoveryTarget
    trigger_signals: tuple[EvidenceQualitySignal, ...]
    recovered_evidence: tuple[Evidence, ...]
    failure_reason: str | None
    remaining_signals: tuple[EvidenceQualitySignal, ...]


@dataclass(frozen=True)
class RecoveryResult:
    """Recovery evidence and trace without diagnosis or Knowledge Gap claims."""

    original_evidence: Evidence
    status: RecoveryStatus
    recovered_evidence: tuple[Evidence, ...]
    remaining_signals: tuple[EvidenceQualitySignal, ...]
    trace: tuple[RecoveryRoundTrace, ...]
    max_rounds: int
    verification_required: bool
    unresolved_reason: str | None = None
    attempted_actions: tuple[tuple[RecoveryAction, RecoveryTarget], ...] = field(
        default_factory=tuple
    )


_ACTION_ORDER = {
    action: index for index, action in enumerate(RecoveryAction)
}
_ROUTES = {
    EvidenceQualitySignalType.POSSIBLY_INCOMPLETE: (
        RecoveryAction.NEIGHBOR_EXPANSION,
        RecoveryAction.PARENT_UNIT,
        RecoveryAction.SECTION_EXPANSION,
        RecoveryAction.RE_RETRIEVAL,
        RecoveryAction.ORIGINAL_SOURCE_CONTEXT,
    ),
    EvidenceQualitySignalType.BROKEN_CONTEXT: (
        RecoveryAction.RE_RETRIEVAL,
        RecoveryAction.ORIGINAL_SOURCE_CONTEXT,
    ),
    EvidenceQualitySignalType.MISSING_CONTEXT: (
        RecoveryAction.PARENT_UNIT,
        RecoveryAction.SECTION_EXPANSION,
        RecoveryAction.RE_RETRIEVAL,
        RecoveryAction.ORIGINAL_SOURCE_CONTEXT,
    ),
    EvidenceQualitySignalType.POSSIBLY_AMBIGUOUS: (
        RecoveryAction.SECTION_EXPANSION,
        RecoveryAction.ORIGINAL_SOURCE_CONTEXT,
    ),
}


def _signal_key(signal: EvidenceQualitySignal) -> str:
    return json.dumps(signal.to_dict(), ensure_ascii=False, sort_keys=True)


def _unique_signals(
    signals: Iterable[EvidenceQualitySignal],
) -> tuple[EvidenceQualitySignal, ...]:
    unique: dict[str, EvidenceQualitySignal] = {}
    for signal in signals:
        if not isinstance(signal, EvidenceQualitySignal):
            raise TypeError("signals must contain EvidenceQualitySignal items")
        unique.setdefault(_signal_key(signal), signal)
    return tuple(unique[key] for key in sorted(unique))


def _target(
    action: RecoveryAction, signal: EvidenceQualitySignal
) -> RecoveryTarget | None:
    source = signal.source_reference
    lineage = signal.lineage
    target_id: str | None = None
    if action is RecoveryAction.NEIGHBOR_EXPANSION:
        if lineage.source_position:
            target_id = f"{lineage.document_id}:{lineage.source_position}"
    elif action is RecoveryAction.PARENT_UNIT:
        target_id = lineage.parent_section_id
    elif action is RecoveryAction.SECTION_EXPANSION:
        if source.section:
            target_id = f"{source.name}:{source.section}"
    elif action is RecoveryAction.RE_RETRIEVAL:
        target_id = lineage.document_id or None
    elif action is RecoveryAction.ORIGINAL_SOURCE_CONTEXT:
        target_id = source.url or (source.name if source.name.strip() else None)
    if target_id is None or not target_id.strip():
        return None
    return RecoveryTarget(target_id, source, lineage)


def _candidate_requests(
    remaining: tuple[EvidenceQualitySignal, ...],
    attempted: set[tuple[RecoveryAction, RecoveryTarget]],
) -> tuple[tuple[RecoveryAction, RecoveryTarget, tuple[EvidenceQualitySignal, ...]], ...]:
    grouped: dict[
        tuple[RecoveryAction, RecoveryTarget], list[EvidenceQualitySignal],
    ] = {}
    for signal in remaining:
        for action in _ROUTES[signal.signal_type]:
            target = _target(action, signal)
            if target is None or (action, target) in attempted:
                continue
            grouped.setdefault((action, target), []).append(signal)
    candidates = (
        (action, target, _unique_signals(triggers))
        for (action, target), triggers in grouped.items()
    )
    return tuple(
        sorted(
            candidates,
            key=lambda item: (
                _ACTION_ORDER[item[0]],
                item[1].target_id,
                item[1].source_reference.to_json(),
                item[1].lineage.to_json(),
            ),
        )
    )


def _evidence_key(evidence: Evidence) -> tuple[str, str, str, str]:
    return (
        evidence.evidence_id,
        evidence.unit_id,
        evidence.source_reference.to_json(),
        evidence.lineage.to_json(),
    )


def _merge_evidence(
    known: dict[tuple[str, str, str, str], Evidence],
    candidates: Iterable[Evidence],
) -> tuple[Evidence, ...]:
    added: list[Evidence] = []
    for evidence in candidates:
        if not isinstance(evidence, Evidence):
            raise TypeError("recovered_evidence must contain Evidence items")
        if not evidence.evidence_id.strip():
            raise ValueError("recovered Evidence must have an evidence_id")
        key = _evidence_key(evidence)
        existing = known.get(key)
        if existing is not None:
            if existing != evidence:
                raise ValueError(
                    "recovered Evidence identity has conflicting content"
                )
            continue
        known[key] = evidence
        added.append(evidence)
    return tuple(added)


def _validate_signal_provenance(
    signals: tuple[EvidenceQualitySignal, ...],
    known_evidence: Iterable[Evidence],
) -> None:
    identities = {
        (
            item.evidence_id,
            item.source_reference.to_json(),
            item.lineage.to_json(),
        )
        for item in known_evidence
    }
    for signal in signals:
        identity = (
            signal.evidence_id,
            signal.source_reference.to_json(),
            signal.lineage.to_json(),
        )
        if identity not in identities:
            raise ValueError("quality signal is not linked to known Evidence")


def run_limited_recovery(
    original_evidence: Evidence,
    signals: Iterable[EvidenceQualitySignal],
    executor: RecoveryExecutor | None = None,
    *,
    max_rounds: int = 2,
) -> RecoveryResult:
    """Execute at most one or two deterministic, source-grounded attempts."""

    if not isinstance(original_evidence, Evidence):
        raise TypeError("original_evidence must be an Evidence")
    if type(max_rounds) is not int or max_rounds not in (1, 2):
        raise ValueError("max_rounds must be 1 or 2")
    remaining = _unique_signals(signals)
    _validate_signal_provenance(remaining, (original_evidence,))
    if not remaining:
        return RecoveryResult(
            original_evidence=original_evidence,
            status=RecoveryStatus.NOT_NEEDED,
            recovered_evidence=(),
            remaining_signals=(),
            trace=(),
            max_rounds=max_rounds,
            verification_required=False,
        )
    if executor is None or not callable(executor):
        raise TypeError("executor must be callable when quality signals exist")

    known = {_evidence_key(original_evidence): original_evidence}
    recovered: list[Evidence] = []
    traces: list[RecoveryRoundTrace] = []
    attempted: set[tuple[RecoveryAction, RecoveryTarget]] = set()
    unresolved_reason: str | None = None

    for round_number in range(1, max_rounds + 1):
        candidates = _candidate_requests(remaining, attempted)
        if not candidates:
            unresolved_reason = "no_untried_recovery_target"
            break
        action, target, trigger_signals = candidates[0]
        attempted.add((action, target))
        request = RecoveryRequest(
            round_number, action, target, trigger_signals
        )
        execution = executor(request)
        if not isinstance(execution, RecoveryExecutionResult):
            raise TypeError("executor must return RecoveryExecutionResult")
        if execution.failure_reason is not None and not execution.failure_reason.strip():
            raise ValueError("failure_reason must be non-empty when present")
        added = _merge_evidence(known, execution.recovered_evidence)
        recovered.extend(added)
        next_remaining = _unique_signals(execution.remaining_signals)
        _validate_signal_provenance(next_remaining, known.values())
        if execution.failure_reason is not None and not next_remaining:
            raise ValueError(
                "failed recovery must retain at least one remaining signal"
            )
        traces.append(
            RecoveryRoundTrace(
                round_number=round_number,
                action=action,
                target=target,
                trigger_signals=trigger_signals,
                recovered_evidence=added,
                failure_reason=execution.failure_reason,
                remaining_signals=next_remaining,
            )
        )
        remaining = next_remaining
        if not remaining:
            return RecoveryResult(
                original_evidence=original_evidence,
                status=RecoveryStatus.RESOLVED,
                recovered_evidence=tuple(recovered),
                remaining_signals=(),
                trace=tuple(traces),
                max_rounds=max_rounds,
                verification_required=False,
                attempted_actions=tuple(
                    (item.action, item.target) for item in traces
                ),
            )

    if unresolved_reason is None:
        unresolved_reason = "round_budget_exhausted"
    return RecoveryResult(
        original_evidence=original_evidence,
        status=RecoveryStatus.UNRESOLVED,
        recovered_evidence=tuple(recovered),
        remaining_signals=remaining,
        trace=tuple(traces),
        max_rounds=max_rounds,
        verification_required=True,
        unresolved_reason=unresolved_reason,
        attempted_actions=tuple(
            (item.action, item.target) for item in traces
        ),
    )
