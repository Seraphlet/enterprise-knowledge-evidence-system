"""Read-only authority and provenance checks for Derived Rule Structure."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from .contracts import KnowledgeLineage, SourceReference
from .rule_structure import (
    RuleExtractionResult,
    RuleExtractionStatus,
    SourceFragment,
    Threshold,
    _LOGIC_RE,
    _THRESHOLD_RE,
)


class RuleAuthority(str, Enum):
    """Fixed authority order from the frozen system contract."""

    SOURCE_EVIDENCE = "SOURCE_EVIDENCE"
    DERIVED_RULE_STRUCTURE = "DERIVED_RULE_STRUCTURE"


class AuthorityViolation(ValueError):
    """A derived field cannot be verified against its claimed source."""

    def __init__(self, field_path: str, reason: str) -> None:
        self.field_path = field_path
        self.reason = reason
        super().__init__(f"{field_path}: {reason}")


@dataclass(frozen=True)
class AuthorityCheckResult:
    """Successful audit result; source remains authoritative and unchanged."""

    evidence_id: str
    source_reference: SourceReference
    lineage: KnowledgeLineage
    checked_fields: tuple[str, ...]
    authority_order: tuple[RuleAuthority, RuleAuthority] = (
        RuleAuthority.SOURCE_EVIDENCE,
        RuleAuthority.DERIVED_RULE_STRUCTURE,
    )


def _require_identity(value: str, field_path: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise AuthorityViolation(field_path, "required source identity is empty")


def _check_fragment(
    source: str, fragment: SourceFragment, field_path: str
) -> None:
    if not isinstance(fragment, SourceFragment):
        raise AuthorityViolation(field_path, "field has no SourceFragment")
    if not (0 <= fragment.start < fragment.end <= len(source)):
        raise AuthorityViolation(field_path, "invalid source offsets")
    if source[fragment.start : fragment.end] != fragment.text:
        raise AuthorityViolation(field_path, "source text does not match offsets")


def _check_logical_relation(
    source: str, relation: str, fragment: SourceFragment, field_path: str
) -> None:
    _check_fragment(source, fragment, f"{field_path}.source")
    match = _LOGIC_RE.fullmatch(fragment.text)
    if match is None:
        raise AuthorityViolation(field_path, "source is not a logical marker")
    expected = "AND" if match.group("and") is not None else "OR"
    if relation != expected:
        raise AuthorityViolation(field_path, "relation contradicts source marker")


def _source_metric(value: str) -> str:
    metric = value.strip()
    for prefix in ("如果", "若", "当"):
        if metric.startswith(prefix):
            return metric[len(prefix) :].lstrip()
    return metric


def _check_threshold(
    source: str, threshold: Threshold, field_path: str
) -> None:
    _check_fragment(source, threshold.source, f"{field_path}.source")
    match = _THRESHOLD_RE.fullmatch(threshold.source.text)
    if match is None:
        raise AuthorityViolation(field_path, "source is not a literal threshold")
    if threshold.metric != _source_metric(match.group("metric")):
        raise AuthorityViolation(field_path, "metric contradicts threshold source")
    if threshold.operator != match.group("operator"):
        raise AuthorityViolation(field_path, "operator contradicts threshold source")
    if isinstance(threshold.value, bool) or float(threshold.value) != float(
        match.group("value")
    ):
        raise AuthorityViolation(field_path, "value contradicts threshold source")
    if threshold.unit != match.group("unit"):
        raise AuthorityViolation(field_path, "unit contradicts threshold source")


def validate_rule_authority(
    extraction: RuleExtractionResult,
) -> AuthorityCheckResult:
    """Reject any Derived Rule field that cannot be traced to Source Evidence."""

    if not isinstance(extraction, RuleExtractionResult):
        raise TypeError("extraction must be a RuleExtractionResult")

    evidence = extraction.source_evidence
    _require_identity(evidence.evidence_id, "evidence.evidence_id")
    _require_identity(evidence.unit_id, "evidence.unit_id")
    _require_identity(evidence.source_reference.name, "source_reference.name")
    for name in (
        "document_id",
        "parser_version",
        "chunker_version",
        "processing_version",
    ):
        _require_identity(getattr(evidence.lineage, name), f"lineage.{name}")

    structure = extraction.structure
    if extraction.status is RuleExtractionStatus.EMPTY and structure is not None:
        raise AuthorityViolation("structure", "EMPTY extraction contains structure")
    if extraction.status is not RuleExtractionStatus.EMPTY and structure is None:
        raise AuthorityViolation("structure", "non-EMPTY extraction has no structure")

    checked: list[str] = []
    if structure is not None:
        source = evidence.original_content
        fragment_groups = (
            ("conditions", structure.conditions),
            ("exceptions", structure.exceptions),
            ("consequences", structure.consequences),
            ("ambiguity", structure.ambiguity),
        )
        for group_name, fragments in fragment_groups:
            for index, fragment in enumerate(fragments):
                path = f"{group_name}[{index}]"
                _check_fragment(source, fragment, path)
                checked.append(path)

        for index, item in enumerate(structure.logical_relations):
            path = f"logical_relations[{index}]"
            _check_logical_relation(source, item.relation, item.source, path)
            checked.append(path)

        for index, item in enumerate(structure.thresholds):
            path = f"thresholds[{index}]"
            _check_threshold(source, item, path)
            checked.append(path)

    return AuthorityCheckResult(
        evidence_id=evidence.evidence_id,
        source_reference=evidence.source_reference,
        lineage=evidence.lineage,
        checked_fields=tuple(checked),
    )
