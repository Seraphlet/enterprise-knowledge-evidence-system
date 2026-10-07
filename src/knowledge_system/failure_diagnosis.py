"""Deterministic diagnosis over completed quality and recovery evidence."""

from __future__ import annotations

import json
from collections.abc import Iterable
from dataclasses import dataclass
from enum import Enum

from .contracts import Evidence, KnowledgeLineage, SourceReference
from .evidence_quality import EvidenceQualitySignal, EvidenceQualitySignalType
from .limited_recovery import (
    RecoveryAction,
    RecoveryResult,
    RecoveryRoundTrace,
    RecoveryStatus,
)
from .rule_structure import SourceFragment


class FailureDiagnosisType(str, Enum):
    RETRIEVAL_PROBLEM = "RETRIEVAL_PROBLEM"
    PROCESSING_PROBLEM = "PROCESSING_PROBLEM"
    SOURCE_KNOWLEDGE_PROBLEM = "SOURCE_KNOWLEDGE_PROBLEM"
    UNDETERMINED = "UNDETERMINED"


class DiagnosisBoundary(str, Enum):
    """Known non-Knowledge-Gap boundaries supplied by the caller."""

    CASE_INFORMATION_MISSING = "CASE_INFORMATION_MISSING"
    NOT_FOUND = "NOT_FOUND"
    RETRIEVAL_FAILURE = "RETRIEVAL_FAILURE"


class SourceFindingType(str, Enum):
    REQUIRED_CONTENT_PRESENT = "REQUIRED_CONTENT_PRESENT"
    REQUIRED_CONTENT_MISSING = "REQUIRED_CONTENT_MISSING"
    SOURCE_CONFLICT = "SOURCE_CONFLICT"
    SOURCE_AMBIGUITY = "SOURCE_AMBIGUITY"


@dataclass(frozen=True)
class SourceInspection:
    """Explicit observation from a successfully recovered original source."""

    finding_type: SourceFindingType
    evidence_id: str
    source_reference: SourceReference
    lineage: KnowledgeLineage
    reason: str
    complete_scope_checked: bool
    source_fragments: tuple[SourceFragment, ...] = ()


@dataclass(frozen=True)
class FailureDiagnosis:
    """Auditable diagnosis; only source problems permit a Knowledge Gap."""

    diagnosis_type: FailureDiagnosisType
    verification_required: bool
    knowledge_gap_allowed: bool
    reason: str
    signals: tuple[EvidenceQualitySignal, ...]
    recovery_trace: tuple[RecoveryRoundTrace, ...]
    evidence: tuple[Evidence, ...]
    source_references: tuple[SourceReference, ...]
    lineages: tuple[KnowledgeLineage, ...]
    source_inspections: tuple[SourceInspection, ...]
    boundary: DiagnosisBoundary | None = None


def _signal_key(signal: EvidenceQualitySignal) -> str:
    return json.dumps(signal.to_dict(), ensure_ascii=False, sort_keys=True)


def _signals(recovery: RecoveryResult) -> tuple[EvidenceQualitySignal, ...]:
    unique: dict[str, EvidenceQualitySignal] = {}
    for trace in recovery.trace:
        for signal in (*trace.trigger_signals, *trace.remaining_signals):
            unique.setdefault(_signal_key(signal), signal)
    for signal in recovery.remaining_signals:
        unique.setdefault(_signal_key(signal), signal)
    return tuple(unique[key] for key in sorted(unique))


def _evidence_key(evidence: Evidence) -> tuple[str, str, str, str]:
    return (
        evidence.evidence_id,
        evidence.unit_id,
        evidence.source_reference.to_json(),
        evidence.lineage.to_json(),
    )


def _all_evidence(recovery: RecoveryResult) -> tuple[Evidence, ...]:
    unique: dict[tuple[str, str, str, str], Evidence] = {}
    for evidence in (recovery.original_evidence, *recovery.recovered_evidence):
        unique.setdefault(_evidence_key(evidence), evidence)
    return tuple(unique.values())


def _trace_integrity_error(recovery: RecoveryResult) -> str | None:
    if len(recovery.trace) > recovery.max_rounds or recovery.max_rounds not in (1, 2):
        return "invalid_recovery_round_budget"
    expected_rounds = tuple(range(1, len(recovery.trace) + 1))
    if tuple(item.round_number for item in recovery.trace) != expected_rounds:
        return "recovery_round_sequence_incomplete"
    if recovery.attempted_actions != tuple(
        (item.action, item.target) for item in recovery.trace
    ):
        return "recovery_attempt_map_inconsistent"
    if recovery.status is RecoveryStatus.NOT_NEEDED:
        if recovery.trace or recovery.remaining_signals:
            return "not_needed_recovery_contains_attempts"
    elif recovery.status is RecoveryStatus.RESOLVED:
        if not recovery.trace or recovery.remaining_signals:
            return "resolved_recovery_trace_incomplete"
    elif recovery.status is RecoveryStatus.UNRESOLVED:
        if not recovery.verification_required or not recovery.remaining_signals:
            return "unresolved_recovery_state_incomplete"

    known = {_evidence_key(recovery.original_evidence): recovery.original_evidence}
    recovered_result = {
        _evidence_key(item): item for item in recovery.recovered_evidence
    }
    for item in recovery.recovered_evidence:
        key = _evidence_key(item)
        existing = known.get(key)
        if existing is not None and existing != item:
            return "recovered_evidence_identity_conflict"
        known[key] = item
    for trace in recovery.trace:
        if not trace.trigger_signals:
            return "recovery_trace_missing_trigger"
        if not any(
            signal.source_reference == trace.target.source_reference
            and signal.lineage == trace.target.lineage
            for signal in trace.trigger_signals
        ):
            return "recovery_target_provenance_mismatch"
        for evidence in trace.recovered_evidence:
            key = _evidence_key(evidence)
            if key not in recovered_result or recovered_result[key] != evidence:
                return "recovery_trace_evidence_mismatch"
        if trace.failure_reason is not None and not trace.remaining_signals:
            return "failed_recovery_lost_remaining_signal"
    if recovery.trace and recovery.trace[-1].remaining_signals != recovery.remaining_signals:
        return "recovery_remaining_signals_mismatch"
    return None


def _inspection_error(
    inspections: tuple[SourceInspection, ...], recovery: RecoveryResult
) -> str | None:
    original_source_evidence: dict[tuple[str, str, str], Evidence] = {}
    expanded_source_evidence: dict[tuple[str, str, str], Evidence] = {}
    for trace in recovery.trace:
        if trace.failure_reason is None and trace.action in {
            RecoveryAction.SECTION_EXPANSION,
            RecoveryAction.ORIGINAL_SOURCE_CONTEXT,
        }:
            for evidence in trace.recovered_evidence:
                key = (
                    evidence.evidence_id,
                    evidence.source_reference.to_json(),
                    evidence.lineage.to_json(),
                )
                expanded_source_evidence[key] = evidence
                if trace.action is RecoveryAction.ORIGINAL_SOURCE_CONTEXT:
                    original_source_evidence[key] = evidence

    for inspection in inspections:
        if not isinstance(inspection, SourceInspection):
            return "invalid_source_inspection_type"
        if not isinstance(inspection.finding_type, SourceFindingType):
            return "invalid_source_finding_type"
        if not isinstance(inspection.source_reference, SourceReference) or not isinstance(
            inspection.lineage, KnowledgeLineage
        ):
            return "invalid_source_inspection_provenance"
        if not inspection.reason.strip():
            return "source_inspection_reason_missing"
        key = (
            inspection.evidence_id,
            inspection.source_reference.to_json(),
            inspection.lineage.to_json(),
        )
        allowed_evidence = (
            expanded_source_evidence
            if inspection.finding_type
            is SourceFindingType.REQUIRED_CONTENT_PRESENT
            else original_source_evidence
        )
        evidence = allowed_evidence.get(key)
        if evidence is None:
            return (
                "source_inspection_not_linked_to_expanded_source_trace"
                if inspection.finding_type
                is SourceFindingType.REQUIRED_CONTENT_PRESENT
                else "source_inspection_not_linked_to_original_source_trace"
            )
        if not inspection.complete_scope_checked:
            return "original_source_scope_not_completely_checked"
        for fragment in inspection.source_fragments:
            if not (
                0 <= fragment.start < fragment.end <= len(evidence.original_content)
                and evidence.original_content[fragment.start:fragment.end]
                == fragment.text
            ):
                return "source_inspection_fragment_untraceable"
        if inspection.finding_type is SourceFindingType.REQUIRED_CONTENT_MISSING:
            if inspection.source_fragments:
                return "missing_source_finding_contains_fragments"
        elif inspection.finding_type is SourceFindingType.SOURCE_CONFLICT:
            if len(set(inspection.source_fragments)) < 2:
                return "source_conflict_requires_two_fragments"
        elif not inspection.source_fragments:
            return "source_finding_requires_fragment"
    return None


def _references(
    evidence: tuple[Evidence, ...],
) -> tuple[tuple[SourceReference, ...], tuple[KnowledgeLineage, ...]]:
    sources: dict[str, SourceReference] = {}
    lineages: dict[str, KnowledgeLineage] = {}
    for item in evidence:
        sources.setdefault(item.source_reference.to_json(), item.source_reference)
        lineages.setdefault(item.lineage.to_json(), item.lineage)
    return tuple(sources.values()), tuple(lineages.values())


def _result(
    diagnosis_type: FailureDiagnosisType,
    reason: str,
    recovery: RecoveryResult,
    signals: tuple[EvidenceQualitySignal, ...],
    inspections: tuple[SourceInspection, ...],
    boundary: DiagnosisBoundary | None,
) -> FailureDiagnosis:
    evidence = _all_evidence(recovery)
    sources, lineages = _references(evidence)
    source_problem = diagnosis_type is FailureDiagnosisType.SOURCE_KNOWLEDGE_PROBLEM
    return FailureDiagnosis(
        diagnosis_type=diagnosis_type,
        verification_required=diagnosis_type in {
            FailureDiagnosisType.SOURCE_KNOWLEDGE_PROBLEM,
            FailureDiagnosisType.UNDETERMINED,
        },
        knowledge_gap_allowed=source_problem,
        reason=reason,
        signals=signals,
        recovery_trace=recovery.trace,
        evidence=evidence,
        source_references=sources,
        lineages=lineages,
        source_inspections=inspections,
        boundary=boundary,
    )


def diagnose_failure(
    recovery: RecoveryResult,
    *,
    source_inspections: Iterable[SourceInspection] = (),
    boundary: DiagnosisBoundary | None = None,
) -> FailureDiagnosis:
    """Classify only when completed trace and source evidence prove the cause."""

    if not isinstance(recovery, RecoveryResult):
        raise TypeError("recovery must be a RecoveryResult")
    if boundary is not None and not isinstance(boundary, DiagnosisBoundary):
        raise TypeError("boundary must be a DiagnosisBoundary or None")
    inspections = tuple(source_inspections)
    signals = _signals(recovery)

    if boundary is not None:
        return _result(
            FailureDiagnosisType.UNDETERMINED,
            f"non_knowledge_gap_boundary:{boundary.value}",
            recovery,
            signals,
            inspections,
            boundary,
        )

    trace_error = _trace_integrity_error(recovery)
    if trace_error is not None:
        return _result(
            FailureDiagnosisType.UNDETERMINED,
            trace_error,
            recovery,
            signals,
            inspections,
            None,
        )
    if recovery.status is not RecoveryStatus.RESOLVED:
        return _result(
            FailureDiagnosisType.UNDETERMINED,
            "recovery_not_completed",
            recovery,
            signals,
            inspections,
            None,
        )

    inspection_error = _inspection_error(inspections, recovery)
    if inspection_error is not None:
        return _result(
            FailureDiagnosisType.UNDETERMINED,
            inspection_error,
            recovery,
            signals,
            inspections,
            None,
        )

    finding_types = {item.finding_type for item in inspections}
    if len(finding_types) > 1:
        return _result(
            FailureDiagnosisType.UNDETERMINED,
            "source_inspections_conflict",
            recovery,
            signals,
            inspections,
            None,
        )
    if finding_types & {
        SourceFindingType.REQUIRED_CONTENT_MISSING,
        SourceFindingType.SOURCE_CONFLICT,
        SourceFindingType.SOURCE_AMBIGUITY,
    }:
        return _result(
            FailureDiagnosisType.SOURCE_KNOWLEDGE_PROBLEM,
            "original_source_contains_confirmed_defect",
            recovery,
            signals,
            inspections,
            None,
        )

    signal_types = {item.signal_type for item in signals}
    if finding_types == {SourceFindingType.REQUIRED_CONTENT_PRESENT}:
        if signal_types and signal_types <= {
            EvidenceQualitySignalType.POSSIBLY_INCOMPLETE,
            EvidenceQualitySignalType.BROKEN_CONTEXT,
        }:
            return _result(
                FailureDiagnosisType.PROCESSING_PROBLEM,
                "original_source_complete_but_derived_evidence_broken",
                recovery,
                signals,
                inspections,
                None,
            )
        return _result(
            FailureDiagnosisType.UNDETERMINED,
            "source_complete_but_processing_defect_not_proven",
            recovery,
            signals,
            inspections,
            None,
        )

    successful_non_source = any(
        trace.action is not RecoveryAction.ORIGINAL_SOURCE_CONTEXT
        and trace.failure_reason is None
        and trace.recovered_evidence
        for trace in recovery.trace
    )
    if (
        successful_non_source
        and signal_types
        and signal_types
        <= {
            EvidenceQualitySignalType.MISSING_CONTEXT,
            EvidenceQualitySignalType.BROKEN_CONTEXT,
        }
    ):
        return _result(
            FailureDiagnosisType.RETRIEVAL_PROBLEM,
            "context_recovery_found_source_grounded_evidence",
            recovery,
            signals,
            inspections,
            None,
        )

    return _result(
        FailureDiagnosisType.UNDETERMINED,
        "available_evidence_does_not_prove_failure_class",
        recovery,
        signals,
        inspections,
        None,
    )
