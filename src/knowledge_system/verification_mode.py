"""Deterministic, source-grounded Verification Mode presentation."""

from __future__ import annotations

import json
import re
from collections.abc import Iterable
from dataclasses import dataclass
from enum import Enum
from urllib.parse import urlsplit

from .contracts import Evidence, KnowledgeLineage, SourceReference
from .evidence_quality import EvidenceQualitySignal, EvidenceQualitySignalType
from .failure_diagnosis import (
    FailureDiagnosis,
    FailureDiagnosisType,
    SourceFindingType,
)
from .limited_recovery import RecoveryResult, RecoveryStatus
from .rule_structure import SourceFragment


class VerificationReason(str, Enum):
    KNOWLEDGE_MISSING = "KNOWLEDGE_MISSING"
    SOURCE_AMBIGUOUS = "SOURCE_AMBIGUOUS"
    RECOVERY_FAILED = "RECOVERY_FAILED"
    RETRIEVAL_UNCERTAINTY = "RETRIEVAL_UNCERTAINTY"
    PROCESSING_UNCERTAINTY = "PROCESSING_UNCERTAINTY"


class LocatorLevel(str, Enum):
    STABLE_URL = "STABLE_URL"
    SOURCE_POSITION = "SOURCE_POSITION"
    SOURCE_NAME = "SOURCE_NAME"


@dataclass(frozen=True)
class VerificationLocator:
    """Employee-facing locator containing only existing source fields."""

    level: LocatorLevel
    source_name: str | None
    url: str | None
    section: str | None
    page: int | None
    sheet: str | None


@dataclass(frozen=True)
class VerificationRecord:
    """One evidence/reason pair with verbatim source and reason codes only."""

    reason: VerificationReason
    evidence_id: str
    original_content: str
    source_reference: SourceReference
    lineage: KnowledgeLineage
    source_fragments: tuple[SourceFragment, ...]
    problem_codes: tuple[str, ...]
    uncertainty_codes: tuple[str, ...]
    verification_locator: VerificationLocator | None


@dataclass(frozen=True)
class VerificationMode:
    """No-op or immutable deterministic Verification Mode output."""

    active: bool
    records: tuple[VerificationRecord, ...]


_REASON_ORDER = {
    reason: index for index, reason in enumerate(VerificationReason)
}
_INTERNAL_PATH_RE = re.compile(r"^(?:[A-Za-z]:[\\/]|\\\\|//)")


def _stable_url(value: str | None) -> str | None:
    if not isinstance(value, str) or not value or value != value.strip():
        return None
    parsed = urlsplit(value)
    if parsed.scheme.casefold() not in {"http", "https"} or not parsed.netloc:
        return None
    return value


def _source_name(value: str) -> str | None:
    if not isinstance(value, str) or not value.strip():
        return None
    if _INTERNAL_PATH_RE.match(value):
        return None
    return value


def _locator(source: SourceReference) -> VerificationLocator | None:
    name = _source_name(source.name)
    url = _stable_url(source.url)
    section = source.section if isinstance(source.section, str) and source.section else None
    sheet = source.sheet if isinstance(source.sheet, str) and source.sheet else None
    page = source.page if type(source.page) is int and source.page > 0 else None
    if url is not None:
        return VerificationLocator(
            LocatorLevel.STABLE_URL, name, url, section, page, sheet
        )
    if name is not None and any(item is not None for item in (section, page, sheet)):
        return VerificationLocator(
            LocatorLevel.SOURCE_POSITION, name, None, section, page, sheet
        )
    if name is not None:
        return VerificationLocator(
            LocatorLevel.SOURCE_NAME, name, None, None, None, None
        )
    return None


def _evidence_identity(evidence: Evidence) -> tuple[str, str, str]:
    return (
        evidence.evidence_id,
        evidence.source_reference.to_json(),
        evidence.lineage.to_json(),
    )


def _evidence_set(items: Iterable[Evidence]) -> tuple[Evidence, ...]:
    unique: dict[tuple[str, str, str], Evidence] = {}
    ids: dict[str, Evidence] = {}
    for evidence in items:
        if not isinstance(evidence, Evidence):
            raise TypeError("evidence must contain Evidence items")
        existing_id = ids.get(evidence.evidence_id)
        if existing_id is not None and existing_id != evidence:
            raise ValueError("Evidence ID has conflicting authoritative content")
        ids.setdefault(evidence.evidence_id, evidence)
        key = _evidence_identity(evidence)
        existing = unique.get(key)
        if existing is not None and existing != evidence:
            raise ValueError("Evidence identity has conflicting content")
        unique.setdefault(key, evidence)
    return tuple(
        sorted(
            unique.values(),
            key=lambda item: (
                item.evidence_id,
                item.source_reference.to_json(),
                item.lineage.to_json(),
            ),
        )
    )


def _signal_key(signal: EvidenceQualitySignal) -> str:
    return json.dumps(signal.to_dict(), ensure_ascii=False, sort_keys=True)


def _signals(
    quality_signals: Iterable[EvidenceQualitySignal],
    diagnosis: FailureDiagnosis | None,
    recovery: RecoveryResult | None,
) -> tuple[EvidenceQualitySignal, ...]:
    unique: dict[str, EvidenceQualitySignal] = {}
    sources = list(quality_signals)
    if diagnosis is not None:
        sources.extend(diagnosis.signals)
    if recovery is not None:
        sources.extend(recovery.remaining_signals)
        for trace in recovery.trace:
            sources.extend(trace.trigger_signals)
            sources.extend(trace.remaining_signals)
    for signal in sources:
        if not isinstance(signal, EvidenceQualitySignal):
            raise TypeError("quality_signals must contain EvidenceQualitySignal items")
        unique.setdefault(_signal_key(signal), signal)
    return tuple(unique[key] for key in sorted(unique))


def _validate_fragment(evidence: Evidence, fragment: SourceFragment) -> None:
    if not isinstance(fragment, SourceFragment) or not (
        0 <= fragment.start < fragment.end <= len(evidence.original_content)
        and evidence.original_content[fragment.start:fragment.end] == fragment.text
    ):
        raise ValueError("SourceFragment is not verbatim Evidence content")


def _signal_reason(signal_type: EvidenceQualitySignalType) -> VerificationReason:
    if signal_type is EvidenceQualitySignalType.POSSIBLY_AMBIGUOUS:
        return VerificationReason.SOURCE_AMBIGUOUS
    if signal_type is EvidenceQualitySignalType.MISSING_CONTEXT:
        return VerificationReason.RETRIEVAL_UNCERTAINTY
    return VerificationReason.PROCESSING_UNCERTAINTY


def _diagnosis_reason(diagnosis: FailureDiagnosis) -> VerificationReason:
    if diagnosis.diagnosis_type is FailureDiagnosisType.RETRIEVAL_PROBLEM:
        return VerificationReason.RETRIEVAL_UNCERTAINTY
    if diagnosis.diagnosis_type is FailureDiagnosisType.PROCESSING_PROBLEM:
        return VerificationReason.PROCESSING_UNCERTAINTY
    return VerificationReason.RETRIEVAL_UNCERTAINTY


def build_verification_mode(
    evidence: Iterable[Evidence],
    *,
    quality_signals: Iterable[EvidenceQualitySignal] = (),
    recovery: RecoveryResult | None = None,
    diagnosis: FailureDiagnosis | None = None,
) -> VerificationMode:
    """Build verification records without generating facts or explanations."""

    if recovery is not None and not isinstance(recovery, RecoveryResult):
        raise TypeError("recovery must be a RecoveryResult or None")
    if diagnosis is not None and not isinstance(diagnosis, FailureDiagnosis):
        raise TypeError("diagnosis must be a FailureDiagnosis or None")
    evidence_items = _evidence_set(evidence)
    signals = _signals(quality_signals, diagnosis, recovery)
    recovery_unresolved = recovery is not None and (
        recovery.status is RecoveryStatus.UNRESOLVED
        or recovery.verification_required
    )
    diagnosis_requires = diagnosis is not None and diagnosis.verification_required
    if not signals and not recovery_unresolved and not diagnosis_requires:
        return VerificationMode(False, ())

    by_identity = {_evidence_identity(item): item for item in evidence_items}
    by_id = {item.evidence_id: item for item in evidence_items}
    categories: dict[str, set[VerificationReason]] = {
        item.evidence_id: set() for item in evidence_items
    }
    problems: dict[str, set[str]] = {item.evidence_id: set() for item in evidence_items}
    uncertainty: dict[str, set[str]] = {
        item.evidence_id: set() for item in evidence_items
    }
    fragments: dict[str, dict[tuple[int, int, str], SourceFragment]] = {
        item.evidence_id: {} for item in evidence_items
    }

    for signal in signals:
        key = (
            signal.evidence_id,
            signal.source_reference.to_json(),
            signal.lineage.to_json(),
        )
        target = by_identity.get(key)
        if target is None:
            raise ValueError("quality signal is not linked to authoritative Evidence")
        categories[target.evidence_id].add(_signal_reason(signal.signal_type))
        problems[target.evidence_id].add(f"quality:{signal.signal_type.value}")
        uncertainty[target.evidence_id].add(f"signal_reason:{signal.reason}")
        if signal.source_fragment is not None:
            _validate_fragment(target, signal.source_fragment)
            fragment = signal.source_fragment
            fragments[target.evidence_id][
                (fragment.start, fragment.end, fragment.text)
            ] = fragment

    if recovery_unresolved and recovery is not None:
        recovery_ids = {
            recovery.original_evidence.evidence_id,
            *(item.evidence_id for item in recovery.recovered_evidence),
        }
        for evidence_id in recovery_ids:
            if evidence_id not in by_id:
                raise ValueError("recovery is not linked to authoritative Evidence")
            categories[evidence_id].add(VerificationReason.RECOVERY_FAILED)
            problems[evidence_id].add(f"recovery:{recovery.status.value}")
            if recovery.unresolved_reason is not None:
                uncertainty[evidence_id].add(
                    f"recovery_reason:{recovery.unresolved_reason}"
                )
            for trace in recovery.trace:
                if trace.failure_reason is not None:
                    uncertainty[evidence_id].add(
                        f"recovery_failure:{trace.action.value}:"
                        f"{trace.failure_reason}"
                    )

    if diagnosis_requires and diagnosis is not None:
        diagnosis_ids = {item.evidence_id for item in diagnosis.evidence}
        if not diagnosis_ids:
            diagnosis_ids = set(by_id)
        for evidence_id in diagnosis_ids:
            if evidence_id not in by_id:
                raise ValueError("diagnosis is not linked to authoritative Evidence")
            if diagnosis.diagnosis_type is not FailureDiagnosisType.SOURCE_KNOWLEDGE_PROBLEM:
                categories[evidence_id].add(_diagnosis_reason(diagnosis))
            problems[evidence_id].add(
                f"diagnosis:{diagnosis.diagnosis_type.value}"
            )
            uncertainty[evidence_id].add(f"diagnosis_reason:{diagnosis.reason}")
        for inspection in diagnosis.source_inspections:
            target = by_id.get(inspection.evidence_id)
            if target is None or (
                target.source_reference != inspection.source_reference
                or target.lineage != inspection.lineage
            ):
                raise ValueError("source inspection is not linked to Evidence")
            if inspection.finding_type is SourceFindingType.REQUIRED_CONTENT_MISSING:
                reason = VerificationReason.KNOWLEDGE_MISSING
            elif inspection.finding_type in {
                SourceFindingType.SOURCE_AMBIGUITY,
                SourceFindingType.SOURCE_CONFLICT,
            }:
                reason = VerificationReason.SOURCE_AMBIGUOUS
            else:
                reason = _diagnosis_reason(diagnosis)
            categories[target.evidence_id].add(reason)
            problems[target.evidence_id].add(
                f"source_finding:{inspection.finding_type.value}"
            )
            uncertainty[target.evidence_id].add(
                f"source_inspection_reason:{inspection.reason}"
            )
            for fragment in inspection.source_fragments:
                _validate_fragment(target, fragment)
                fragments[target.evidence_id][
                    (fragment.start, fragment.end, fragment.text)
                ] = fragment

    records: list[VerificationRecord] = []
    for item in evidence_items:
        item_fragments = tuple(
            fragments[item.evidence_id][key]
            for key in sorted(fragments[item.evidence_id])
        )
        for reason in sorted(categories[item.evidence_id], key=_REASON_ORDER.get):
            records.append(
                VerificationRecord(
                    reason=reason,
                    evidence_id=item.evidence_id,
                    original_content=item.original_content,
                    source_reference=item.source_reference,
                    lineage=item.lineage,
                    source_fragments=item_fragments,
                    problem_codes=tuple(sorted(problems[item.evidence_id])),
                    uncertainty_codes=tuple(sorted(uncertainty[item.evidence_id])),
                    verification_locator=_locator(item.source_reference),
                )
            )
    return VerificationMode(bool(records), tuple(records))
