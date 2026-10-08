"""Deterministic, non-diagnostic quality signals for retrieved Evidence."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from .contracts import (
    Evidence,
    JsonValue,
    KnowledgeLineage,
    KnowledgeUnit,
    ParseStatus,
    SourceReference,
)
from .rule_authority import AuthorityViolation, validate_rule_authority
from .rule_structure import RuleExtractionResult, SourceFragment


class EvidenceQualitySignalType(str, Enum):
    """Observable quality concerns; these are not failure diagnoses."""

    POSSIBLY_INCOMPLETE = "POSSIBLY_INCOMPLETE"
    BROKEN_CONTEXT = "BROKEN_CONTEXT"
    MISSING_CONTEXT = "MISSING_CONTEXT"
    POSSIBLY_AMBIGUOUS = "POSSIBLY_AMBIGUOUS"


@dataclass(frozen=True)
class EvidenceQualitySignal:
    """One traceable observation about Evidence quality."""

    signal_type: EvidenceQualitySignalType
    reason: str
    evidence_id: str
    source_reference: SourceReference
    lineage: KnowledgeLineage
    source_fragment: SourceFragment | None = None

    def to_dict(self) -> dict[str, JsonValue]:
        return {
            "signal_type": self.signal_type.value,
            "reason": self.reason,
            "evidence_id": self.evidence_id,
            "source_reference": self.source_reference.to_dict(),
            "lineage": self.lineage.to_dict(),
            "source_fragment": (
                self.source_fragment.to_dict()
                if self.source_fragment is not None
                else None
            ),
        }


_SIGNAL_ORDER = {
    signal_type: index
    for index, signal_type in enumerate(EvidenceQualitySignalType)
}


_CONTEXT_DEPENDENT_PARAGRAPHS = frozenset(
    {
        "然后问：",
        "而是：",
        "完全一致。",
        "初学时很容易理解成：",
    }
)


def _signal(
    evidence: Evidence,
    signal_type: EvidenceQualitySignalType,
    reason: str,
    source_fragment: SourceFragment | None = None,
) -> EvidenceQualitySignal:
    return EvidenceQualitySignal(
        signal_type=signal_type,
        reason=reason,
        evidence_id=evidence.evidence_id,
        source_reference=evidence.source_reference,
        lineage=evidence.lineage,
        source_fragment=source_fragment,
    )


def _identity(signal: EvidenceQualitySignal) -> tuple[object, ...]:
    fragment = signal.source_fragment
    return (
        signal.signal_type.value,
        signal.reason,
        signal.evidence_id,
        signal.source_reference.to_json(),
        signal.lineage.to_json(),
        (
            (fragment.text, fragment.start, fragment.end)
            if fragment is not None
            else None
        ),
    )


def _stable_unique(
    signals: list[EvidenceQualitySignal],
) -> tuple[EvidenceQualitySignal, ...]:
    unique: dict[tuple[object, ...], EvidenceQualitySignal] = {}
    for signal in signals:
        unique.setdefault(_identity(signal), signal)
    return tuple(
        sorted(
            unique.values(),
            key=lambda item: (
                _SIGNAL_ORDER[item.signal_type],
                (
                    item.source_fragment.start
                    if item.source_fragment is not None
                    else -1
                ),
                (
                    item.source_fragment.end
                    if item.source_fragment is not None
                    else -1
                ),
                item.reason,
            ),
        )
    )


def detect_evidence_quality(
    evidence: Evidence,
    *,
    knowledge_unit: KnowledgeUnit | None = None,
    rule_extraction: RuleExtractionResult | None = None,
) -> tuple[EvidenceQualitySignal, ...]:
    """Return source-grounded observations without recovery or diagnosis."""

    if not isinstance(evidence, Evidence):
        raise TypeError("evidence must be an Evidence")
    if not evidence.evidence_id.strip():
        raise ValueError("evidence.evidence_id must not be empty")
    if knowledge_unit is not None and not isinstance(
        knowledge_unit, KnowledgeUnit
    ):
        raise TypeError("knowledge_unit must be a KnowledgeUnit or None")
    if rule_extraction is not None and not isinstance(
        rule_extraction, RuleExtractionResult
    ):
        raise TypeError("rule_extraction must be a RuleExtractionResult or None")

    signals: list[EvidenceQualitySignal] = []

    if (
        knowledge_unit is not None
        and knowledge_unit.metadata.get("structure_kind") == "paragraph"
        and evidence.original_content.strip() in _CONTEXT_DEPENDENT_PARAGRAPHS
    ):
        signals.append(
            _signal(
                evidence,
                EvidenceQualitySignalType.POSSIBLY_INCOMPLETE,
                "context_dependent_fragment",
            )
        )

    if not evidence.original_content.strip():
        signals.append(
            _signal(
                evidence,
                EvidenceQualitySignalType.POSSIBLY_INCOMPLETE,
                "source_content_empty",
            )
        )
        signals.append(
            _signal(
                evidence,
                EvidenceQualitySignalType.MISSING_CONTEXT,
                "source_content_unavailable",
            )
        )

    if not evidence.source_reference.name.strip():
        signals.append(
            _signal(
                evidence,
                EvidenceQualitySignalType.MISSING_CONTEXT,
                "source_identity_missing",
            )
        )

    required_lineage = (
        evidence.lineage.document_id,
        evidence.lineage.parser_version,
        evidence.lineage.chunker_version,
        evidence.lineage.processing_version,
    )
    if any(not value.strip() for value in required_lineage):
        signals.append(
            _signal(
                evidence,
                EvidenceQualitySignalType.BROKEN_CONTEXT,
                "lineage_identity_incomplete",
            )
        )

    if knowledge_unit is not None:
        if knowledge_unit.parse_status is ParseStatus.PARTIAL:
            signals.append(
                _signal(
                    evidence,
                    EvidenceQualitySignalType.POSSIBLY_INCOMPLETE,
                    "parse_status_partial",
                )
            )
        elif knowledge_unit.parse_status is ParseStatus.FAILED:
            signals.append(
                _signal(
                    evidence,
                    EvidenceQualitySignalType.POSSIBLY_INCOMPLETE,
                    "parse_status_failed",
                )
            )

        unit_links = (
            (
                knowledge_unit.unit_id == evidence.unit_id,
                "evidence_unit_identity_mismatch",
            ),
            (
                knowledge_unit.original_content == evidence.original_content,
                "evidence_unit_content_mismatch",
            ),
            (
                knowledge_unit.source_reference == evidence.source_reference,
                "evidence_unit_source_mismatch",
            ),
            (
                knowledge_unit.lineage == evidence.lineage,
                "evidence_unit_lineage_mismatch",
            ),
        )
        for matches, reason in unit_links:
            if not matches:
                signals.append(
                    _signal(
                        evidence,
                        EvidenceQualitySignalType.BROKEN_CONTEXT,
                        reason,
                    )
                )

    if (
        evidence.lineage.parent_section_id
        and not (evidence.source_reference.section or "").strip()
    ):
        signals.append(
            _signal(
                evidence,
                EvidenceQualitySignalType.MISSING_CONTEXT,
                "parent_section_locator_missing",
            )
        )

    if evidence.derived_structure is not None:
        ambiguity = evidence.derived_structure.get("ambiguity")
        if ambiguity is not None and not isinstance(ambiguity, list):
            signals.append(
                _signal(
                    evidence,
                    EvidenceQualitySignalType.BROKEN_CONTEXT,
                    "derived_ambiguity_not_a_list",
                )
            )
        elif isinstance(ambiguity, list):
            for index, item in enumerate(ambiguity):
                fragment: SourceFragment | None = None
                if isinstance(item, dict):
                    text = item.get("text")
                    start = item.get("start")
                    end = item.get("end")
                    if (
                        isinstance(text, str)
                        and type(start) is int
                        and type(end) is int
                        and 0 <= start < end <= len(evidence.original_content)
                        and evidence.original_content[start:end] == text
                    ):
                        fragment = SourceFragment(text, start, end)
                if fragment is None:
                    signals.append(
                        _signal(
                            evidence,
                            EvidenceQualitySignalType.BROKEN_CONTEXT,
                            f"derived_ambiguity_untraceable:{index}",
                        )
                    )
                else:
                    signals.append(
                        _signal(
                            evidence,
                            EvidenceQualitySignalType.POSSIBLY_AMBIGUOUS,
                            "explicit_rule_ambiguity",
                            fragment,
                        )
                    )

    if rule_extraction is not None:
        if rule_extraction.source_evidence != evidence:
            signals.append(
                _signal(
                    evidence,
                    EvidenceQualitySignalType.BROKEN_CONTEXT,
                    "rule_source_evidence_mismatch",
                )
            )
        else:
            try:
                validate_rule_authority(rule_extraction)
            except AuthorityViolation as violation:
                signals.append(
                    _signal(
                        evidence,
                        EvidenceQualitySignalType.BROKEN_CONTEXT,
                        "rule_structure_untraceable:"
                        f"{violation.field_path}:{violation.reason}",
                    )
                )
            else:
                structure = rule_extraction.structure
                if structure is not None:
                    for fragment in structure.ambiguity:
                        signals.append(
                            _signal(
                                evidence,
                                EvidenceQualitySignalType.POSSIBLY_AMBIGUOUS,
                                "explicit_rule_ambiguity",
                                fragment,
                            )
                        )

    return _stable_unique(signals)
