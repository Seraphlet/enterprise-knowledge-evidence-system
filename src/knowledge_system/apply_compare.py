"""Evidence-preserving Apply/Compare organization without business decisions."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Iterable

from .context import CaseContext
from .deterministic_comparison import (
    ComparisonOutcome,
    ConditionComparison,
    ExactCondition,
    compare_condition,
)
from .rule_authority import validate_rule_authority
from .rule_structure import RuleExtractionResult, SourceFragment, Threshold


class LogicalOutcome(str, Enum):
    """Conservative truth of compared rule conditions, not a business decision."""

    SATISFIED = "SATISFIED"
    NOT_SATISFIED = "NOT_SATISFIED"
    INDETERMINATE = "INDETERMINATE"


@dataclass(frozen=True)
class DuplicateComparisonGroup:
    """Repeated input for one condition identity, retained but counted once."""

    identity: tuple[str, str, int, int]
    canonical: ConditionComparison
    duplicates: tuple[ConditionComparison, ...]


@dataclass(frozen=True)
class ApplyCompareResult:
    """Evidence-backed condition classifications and conservative logic only."""

    extraction: RuleExtractionResult
    comparisons: tuple[ConditionComparison, ...]
    matched: tuple[ConditionComparison, ...]
    missing: tuple[ConditionComparison, ...]
    unknown: tuple[ConditionComparison, ...]
    not_satisfied: tuple[ConditionComparison, ...]
    duplicate_groups: tuple[DuplicateComparisonGroup, ...] = field(
        default_factory=tuple
    )
    unevaluated_conditions: tuple[SourceFragment, ...] = field(
        default_factory=tuple
    )
    logical_relation: str | None = None
    logical_outcome: LogicalOutcome = LogicalOutcome.INDETERMINATE
    logical_reason: str = "no_comparisons"


def _condition_source(comparison: ConditionComparison) -> SourceFragment:
    condition = comparison.condition
    if isinstance(condition, Threshold):
        return condition.source
    if isinstance(condition, ExactCondition):
        return condition.source
    return condition


def _identity(
    comparison: ConditionComparison,
) -> tuple[str, str, int, int]:
    source = _condition_source(comparison)
    return (
        comparison.requirement.evidence_id,
        comparison.requirement.requirement_id,
        source.start,
        source.end,
    )


def _fragment_signature(fragment: SourceFragment) -> tuple[str, int, int]:
    return (fragment.text, fragment.start, fragment.end)


def _value_signature(value: object) -> tuple[str, str]:
    return (f"{type(value).__module__}.{type(value).__qualname__}", repr(value))


def _threshold_signature(threshold: Threshold) -> tuple[object, ...]:
    return (
        "threshold",
        threshold.metric,
        threshold.operator,
        _value_signature(threshold.value),
        threshold.unit,
        _fragment_signature(threshold.source),
    )


def _condition_signature(comparison: ConditionComparison) -> tuple[object, ...]:
    condition = comparison.condition
    source = _condition_source(comparison)
    if isinstance(condition, Threshold):
        return _threshold_signature(condition)
    if isinstance(condition, ExactCondition):
        return (
            "exact",
            _value_signature(condition.expected_value),
            _fragment_signature(source),
        )
    return ("source", _fragment_signature(source))


def _comparison_signature(
    comparison: ConditionComparison,
) -> tuple[object, ...]:
    requirement = comparison.requirement
    fact = comparison.fact
    observations = tuple(
        (
            _value_signature(item.value),
            item.unit,
            item.source.description_index,
            item.source.start,
            item.source.end,
            item.source.text,
        )
        for item in fact.observations
    )
    return (
        _condition_signature(comparison),
        requirement.requirement_id,
        requirement.fact_type,
        requirement.question,
        requirement.certainty,
        requirement.metric_hints,
        _fragment_signature(requirement.condition),
        (
            _threshold_signature(requirement.threshold)
            if isinstance(requirement.threshold, Threshold)
            else None
        ),
        requirement.evidence_id,
        requirement.source_reference,
        requirement.lineage,
        fact.requirement_id,
        fact.evidence_id,
        fact.fact_type,
        fact.status,
        _value_signature(fact.value),
        observations,
        fact.reason,
        comparison.outcome,
        comparison.reason,
    )


def _validate_comparison(
    extraction: RuleExtractionResult,
    comparison: ConditionComparison,
    case_context: CaseContext,
) -> None:
    structure = extraction.structure
    assert structure is not None
    valid_requirement_sources = {
        _fragment_signature(item) for item in structure.conditions
    } | {
        _fragment_signature(item.source) for item in structure.thresholds
    }
    if (
        _fragment_signature(comparison.requirement.condition)
        not in valid_requirement_sources
    ):
        raise ValueError("comparison requirement condition is not in RuleStructure")
    condition = comparison.condition
    if isinstance(condition, Threshold):
        candidate = _threshold_signature(condition)
        canonical_thresholds = {
            _threshold_signature(item): item for item in structure.thresholds
        }
        canonical = canonical_thresholds.get(candidate)
        if canonical is None:
            raise ValueError("comparison threshold is not in RuleStructure")
        requirement_threshold = comparison.requirement.threshold
        if not isinstance(requirement_threshold, Threshold):
            raise ValueError("numeric requirement has no canonical threshold")
        if _threshold_signature(requirement_threshold) != candidate:
            raise ValueError(
                "requirement threshold does not match canonical threshold"
            )
        if comparison.requirement.metric_hints != (canonical.metric,):
            raise ValueError(
                "requirement metric_hints do not match canonical metric"
            )

    verified = compare_condition(
        condition,
        comparison.requirement,
        comparison.fact,
        extraction.source_evidence,
        case_context,
    )
    if (
        verified.outcome is not comparison.outcome
        or verified.reason != comparison.reason
    ):
        raise ValueError("comparison outcome does not match verified inputs")


def _deduplicate(
    comparisons: tuple[ConditionComparison, ...],
) -> tuple[
    tuple[ConditionComparison, ...], tuple[DuplicateComparisonGroup, ...]
]:
    canonical: dict[tuple[str, str, int, int], ConditionComparison] = {}
    duplicates: dict[
        tuple[str, str, int, int], list[ConditionComparison]
    ] = {}
    for comparison in comparisons:
        identity = _identity(comparison)
        existing = canonical.get(identity)
        if existing is None:
            canonical[identity] = comparison
            continue
        if _comparison_signature(comparison) != _comparison_signature(existing):
            raise ValueError("duplicate condition identity has conflicting inputs")
        duplicates.setdefault(identity, []).append(comparison)
    groups = tuple(
        DuplicateComparisonGroup(identity, canonical[identity], tuple(items))
        for identity, items in duplicates.items()
    )
    return tuple(canonical.values()), groups


def _logical_result(
    comparisons: tuple[ConditionComparison, ...],
    relation: str | None,
    has_unevaluated: bool,
) -> tuple[LogicalOutcome, str]:
    truths = tuple(
        True
        if item.outcome is ComparisonOutcome.SATISFIED
        else False
        if item.outcome is ComparisonOutcome.NOT_SATISFIED
        else None
        for item in comparisons
    )
    if not truths:
        return LogicalOutcome.INDETERMINATE, "no_comparisons"
    if len(truths) == 1 and not has_unevaluated:
        if truths[0] is True:
            return LogicalOutcome.SATISFIED, "single_condition_satisfied"
        if truths[0] is False:
            return LogicalOutcome.NOT_SATISFIED, "single_condition_not_satisfied"
        return LogicalOutcome.INDETERMINATE, "single_condition_uncertain"
    if relation is None:
        return LogicalOutcome.INDETERMINATE, "logical_relation_unavailable"
    if relation == "AND":
        if False in truths:
            return LogicalOutcome.NOT_SATISFIED, "and_has_not_satisfied"
        if not has_unevaluated and all(value is True for value in truths):
            return LogicalOutcome.SATISFIED, "and_all_satisfied"
        return LogicalOutcome.INDETERMINATE, "and_has_uncertainty"
    if True in truths:
        return LogicalOutcome.SATISFIED, "or_has_satisfied"
    if not has_unevaluated and all(value is False for value in truths):
        return LogicalOutcome.NOT_SATISFIED, "or_all_not_satisfied"
    return LogicalOutcome.INDETERMINATE, "or_has_uncertainty"


def _expected_condition_sources(
    extraction: RuleExtractionResult,
) -> tuple[SourceFragment, ...]:
    """List atomic threshold sources plus conditions with no threshold."""

    structure = extraction.structure
    assert structure is not None
    expected = [item.source for item in structure.thresholds]
    for condition in structure.conditions:
        has_threshold = any(
            condition.start <= item.source.start
            and item.source.end <= condition.end
            for item in structure.thresholds
        )
        if not has_threshold:
            expected.append(condition)
    return tuple(sorted(expected, key=lambda item: (item.start, item.end)))


def organize_apply_comparisons(
    extraction: RuleExtractionResult,
    comparisons: Iterable[ConditionComparison],
    case_context: CaseContext,
) -> ApplyCompareResult:
    """Classify existing comparisons and conservatively propagate AND/OR."""

    if not isinstance(extraction, RuleExtractionResult):
        raise TypeError("extraction must be a RuleExtractionResult")
    if extraction.structure is None:
        raise ValueError("RuleExtractionResult has no RuleStructure")
    if not isinstance(case_context, CaseContext):
        raise TypeError("case_context must be a CaseContext")
    selected = tuple(comparisons)
    if any(not isinstance(item, ConditionComparison) for item in selected):
        raise TypeError("comparisons must contain ConditionComparison items")

    validate_rule_authority(extraction)
    for comparison in selected:
        _validate_comparison(extraction, comparison, case_context)
    canonical, duplicate_groups = _deduplicate(selected)

    matched = tuple(
        item for item in canonical
        if item.outcome is ComparisonOutcome.SATISFIED
    )
    missing = tuple(
        item for item in canonical
        if item.outcome is ComparisonOutcome.MISSING
    )
    unknown = tuple(
        item for item in canonical
        if item.outcome in {
            ComparisonOutcome.UNKNOWN,
            ComparisonOutcome.HUMAN_REQUIRED,
        }
    )
    not_satisfied = tuple(
        item for item in canonical
        if item.outcome is ComparisonOutcome.NOT_SATISFIED
    )

    represented = {
        (_condition_source(item).start, _condition_source(item).end)
        for item in canonical
    }
    unevaluated = tuple(
        condition
        for condition in _expected_condition_sources(extraction)
        if (condition.start, condition.end) not in represented
    )
    relations = {
        item.relation for item in extraction.structure.logical_relations
    }
    relation = next(iter(relations)) if len(relations) == 1 else None
    logical_outcome, logical_reason = _logical_result(
        canonical, relation, bool(unevaluated)
    )
    if len(relations) > 1:
        logical_outcome = LogicalOutcome.INDETERMINATE
        logical_reason = "mixed_logical_relations_without_expression_tree"

    return ApplyCompareResult(
        extraction=extraction,
        comparisons=canonical,
        matched=matched,
        missing=missing,
        unknown=unknown,
        not_satisfied=not_satisfied,
        duplicate_groups=duplicate_groups,
        unevaluated_conditions=unevaluated,
        logical_relation=relation,
        logical_outcome=logical_outcome,
        logical_reason=logical_reason,
    )
