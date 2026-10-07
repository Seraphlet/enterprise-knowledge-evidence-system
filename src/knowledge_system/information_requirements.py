"""Information Requirements derived only from reliable Rule conditions."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from hashlib import sha256

from .contracts import KnowledgeLineage, SourceReference
from .rule_structure import RuleExtractionResult, SourceFragment, Threshold


class RequirementCertainty(str, Enum):
    """Whether the condition can be used without resolving source ambiguity."""

    RELIABLE = "RELIABLE"
    HUMAN_REQUIRED = "HUMAN_REQUIRED"


@dataclass(frozen=True)
class InformationRequirement:
    """A source-grounded description of Case information needed by one rule."""

    requirement_id: str
    fact_type: str
    question: str
    certainty: RequirementCertainty
    metric_hints: tuple[str, ...]
    condition: SourceFragment
    evidence_id: str
    source_reference: SourceReference
    lineage: KnowledgeLineage
    threshold: Threshold | None = None


def _requirement_id(evidence_id: str, condition: SourceFragment) -> str:
    identity = (
        f"{evidence_id}\0{condition.start}\0{condition.end}\0{condition.text}"
    )
    return f"requirement-{sha256(identity.encode('utf-8')).hexdigest()[:20]}"


def _validate_condition(source: str, condition: SourceFragment) -> None:
    if not (0 <= condition.start <= condition.end <= len(source)):
        raise ValueError("condition contains invalid source offsets")
    if source[condition.start : condition.end] != condition.text:
        raise ValueError("condition does not match Evidence.original_content")


def generate_information_requirements(
    extraction: RuleExtractionResult,
) -> tuple[InformationRequirement, ...]:
    """Generate requirements without extracting facts or comparing a Case."""

    if not isinstance(extraction, RuleExtractionResult):
        raise TypeError("extraction must be a RuleExtractionResult")
    if extraction.structure is None or not extraction.structure.conditions:
        return ()

    evidence = extraction.source_evidence
    structure = extraction.structure
    certainty = (
        RequirementCertainty.HUMAN_REQUIRED
        if structure.ambiguity
        else RequirementCertainty.RELIABLE
    )
    requirements: list[InformationRequirement] = []
    for condition in structure.conditions:
        _validate_condition(evidence.original_content, condition)
        condition_thresholds = tuple(
            threshold
            for threshold in structure.thresholds
            if threshold.source.start >= condition.start
            and threshold.source.end <= condition.end
        )
        atomic_sources = (
            tuple((threshold.source, threshold) for threshold in condition_thresholds)
            if len(condition_thresholds) > 1
            else ((condition, condition_thresholds[0] if condition_thresholds else None),)
        )
        for atomic_condition, threshold in atomic_sources:
            _validate_condition(evidence.original_content, atomic_condition)
            metric_hints = (
                (threshold.metric,)
                if threshold is not None and len(condition_thresholds) > 1
                else tuple(
                    dict.fromkeys(item.metric for item in condition_thresholds)
                )
            )
            requirements.append(
                InformationRequirement(
                    requirement_id=_requirement_id(
                        evidence.evidence_id, atomic_condition
                    ),
                    fact_type=atomic_condition.text,
                    question=(
                        "请提供当前 Case 中用于核对该条件的事实："
                        f"{atomic_condition.text}"
                    ),
                    certainty=certainty,
                    metric_hints=metric_hints,
                    condition=atomic_condition,
                    evidence_id=evidence.evidence_id,
                    source_reference=evidence.source_reference,
                    lineage=evidence.lineage,
                    threshold=threshold,
                )
            )
    return tuple(requirements)
