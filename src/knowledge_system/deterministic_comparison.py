"""Single-condition deterministic comparison without rule aggregation."""

from __future__ import annotations

import operator
import math
from dataclasses import dataclass
from decimal import Decimal
from enum import Enum
from typing import Callable

from .context import CaseContext, FactValue
from .contracts import Evidence
from .fact_extraction import (
    ExtractedRequirementFact,
    FactExtractionStatus,
)
from .information_requirements import (
    InformationRequirement,
    RequirementCertainty,
)
from .rule_structure import SourceFragment, Threshold


class ComparisonOutcome(str, Enum):
    """Outcome for one condition/fact pair, never a rule-level decision."""

    SATISFIED = "SATISFIED"
    NOT_SATISFIED = "NOT_SATISFIED"
    MISSING = "MISSING"
    UNKNOWN = "UNKNOWN"
    HUMAN_REQUIRED = "HUMAN_REQUIRED"


@dataclass(frozen=True)
class ExactCondition:
    """An explicit boolean or enum equality condition grounded in source."""

    source: SourceFragment
    expected_value: bool | str

    def __post_init__(self) -> None:
        if type(self.expected_value) not in {bool, str}:
            raise TypeError("expected_value must be exactly bool or str")
        literal = str(self.expected_value)
        if not literal or literal not in self.source.text:
            raise ValueError("expected_value is not explicit in condition source")


ComparableCondition = Threshold | ExactCondition | SourceFragment


@dataclass(frozen=True)
class ConditionComparison:
    """Auditable comparison retaining all existing provenance objects."""

    condition: ComparableCondition
    requirement: InformationRequirement
    fact: ExtractedRequirementFact
    outcome: ComparisonOutcome
    reason: str


_NUMERIC_OPERATORS: dict[str, Callable[[int | float, int | float], bool]] = {
    "不少于": operator.ge,
    "不低于": operator.ge,
    "至少": operator.ge,
    "大于等于": operator.ge,
    ">=": operator.ge,
    "≥": operator.ge,
    "小于等于": operator.le,
    "不超过": operator.le,
    "至多": operator.le,
    "<=": operator.le,
    "≤": operator.le,
    "超过": operator.gt,
    "大于": operator.gt,
    ">": operator.gt,
    "低于": operator.lt,
    "小于": operator.lt,
    "<": operator.lt,
    "=": operator.eq,
    "等于": operator.eq,
}


def _source(condition: ComparableCondition) -> SourceFragment:
    if isinstance(condition, Threshold):
        return condition.source
    if isinstance(condition, ExactCondition):
        return condition.source
    return condition


def _validate_links(
    condition: ComparableCondition,
    requirement: InformationRequirement,
    fact: ExtractedRequirementFact,
    evidence: Evidence,
    case_context: CaseContext,
) -> None:
    if requirement.evidence_id != evidence.evidence_id:
        raise ValueError("requirement evidence_id does not match Evidence")
    if requirement.source_reference != evidence.source_reference:
        raise ValueError("requirement source_reference does not match Evidence")
    if requirement.lineage != evidence.lineage:
        raise ValueError("requirement lineage does not match Evidence")
    if fact.requirement_id != requirement.requirement_id:
        raise ValueError("fact requirement_id does not match requirement")
    if fact.evidence_id != requirement.evidence_id:
        raise ValueError("fact evidence_id does not match requirement")
    if fact.fact_type != requirement.fact_type:
        raise ValueError("fact_type does not match requirement")

    parent = requirement.condition
    if not (0 <= parent.start < parent.end <= len(evidence.original_content)):
        raise ValueError("requirement condition has invalid Evidence offsets")
    if evidence.original_content[parent.start:parent.end] != parent.text:
        raise ValueError("requirement condition does not match Evidence source")

    source = _source(condition)
    if not (
        parent.start <= source.start < source.end <= parent.end
    ):
        raise ValueError("condition is outside requirement source")
    relative_start = source.start - parent.start
    relative_end = source.end - parent.start
    if parent.text[relative_start:relative_end] != source.text:
        raise ValueError("condition does not match requirement source")

    for observation in fact.observations:
        span = observation.source
        if type(span.description_index) is not int or not (
            0 <= span.description_index < len(case_context.raw_descriptions)
        ):
            raise ValueError("fact observation has invalid description_index")
        description = case_context.raw_descriptions[span.description_index]
        if (
            type(span.start) is not int
            or type(span.end) is not int
            or not (0 <= span.start < span.end <= len(description))
        ):
            raise ValueError("fact observation has invalid Case offsets")
        if description[span.start:span.end] != span.text:
            raise ValueError("fact observation does not match Case source")


def _result(
    condition: ComparableCondition,
    requirement: InformationRequirement,
    fact: ExtractedRequirementFact,
    outcome: ComparisonOutcome,
    reason: str,
) -> ConditionComparison:
    return ConditionComparison(condition, requirement, fact, outcome, reason)


def _fact_boundary(
    condition: ComparableCondition,
    requirement: InformationRequirement,
    fact: ExtractedRequirementFact,
) -> ConditionComparison | None:
    if fact.status is FactExtractionStatus.MISSING:
        return _result(
            condition, requirement, fact,
            ComparisonOutcome.MISSING, "fact_missing",
        )
    if fact.status is FactExtractionStatus.UNKNOWN:
        outcome = (
            ComparisonOutcome.HUMAN_REQUIRED
            if requirement.certainty is RequirementCertainty.HUMAN_REQUIRED
            else ComparisonOutcome.UNKNOWN
        )
        return _result(
            condition, requirement, fact, outcome,
            fact.reason or "fact_unknown",
        )
    if requirement.certainty is RequirementCertainty.HUMAN_REQUIRED:
        return _result(
            condition, requirement, fact,
            ComparisonOutcome.HUMAN_REQUIRED,
            "requirement_human_required",
        )
    if fact.value is None:
        return _result(
            condition, requirement, fact,
            ComparisonOutcome.UNKNOWN, "fact_value_unreliable",
        )
    if not fact.observations:
        return _result(
            condition, requirement, fact,
            ComparisonOutcome.UNKNOWN, "fact_provenance_missing",
        )
    return None


def _numeric_comparison(
    threshold: Threshold,
    requirement: InformationRequirement,
    fact: ExtractedRequirementFact,
) -> ConditionComparison:
    numeric_values = (
        threshold.value,
        fact.value,
        *(item.value for item in fact.observations),
    )
    if any(_is_non_finite_number(value) for value in numeric_values):
        return _result(
            threshold, requirement, fact,
            ComparisonOutcome.UNKNOWN, "numeric_value_not_finite_real",
        )
    if not all(type(value) in {int, float} for value in numeric_values):
        return _result(
            threshold, requirement, fact,
            ComparisonOutcome.UNKNOWN, "numeric_type_incompatible",
        )
    if any(item.value != fact.value for item in fact.observations):
        return _result(
            threshold, requirement, fact,
            ComparisonOutcome.UNKNOWN, "fact_observations_inconsistent",
        )
    observed_units = {item.unit for item in fact.observations}
    if observed_units != {threshold.unit}:
        return _result(
            threshold, requirement, fact,
            ComparisonOutcome.UNKNOWN, "numeric_unit_incompatible",
        )
    compare = _NUMERIC_OPERATORS.get(threshold.operator)
    if compare is None:
        return _result(
            threshold, requirement, fact,
            ComparisonOutcome.HUMAN_REQUIRED,
            "unsupported_numeric_operator",
        )
    satisfied = compare(fact.value, threshold.value)
    return _result(
        threshold,
        requirement,
        fact,
        (
            ComparisonOutcome.SATISFIED
            if satisfied
            else ComparisonOutcome.NOT_SATISFIED
        ),
        "deterministic_numeric_comparison",
    )


def _exact_comparison(
    condition: ExactCondition,
    requirement: InformationRequirement,
    fact: ExtractedRequirementFact,
) -> ConditionComparison:
    expected = condition.expected_value
    if type(fact.value) is not type(expected):
        return _result(
            condition, requirement, fact,
            ComparisonOutcome.UNKNOWN, "exact_type_incompatible",
        )
    if any(
        type(item.value) is not type(fact.value) or item.value != fact.value
        for item in fact.observations
    ):
        return _result(
            condition, requirement, fact,
            ComparisonOutcome.UNKNOWN, "fact_observations_inconsistent",
        )
    if any(item.unit is not None for item in fact.observations):
        return _result(
            condition, requirement, fact,
            ComparisonOutcome.UNKNOWN, "exact_value_has_unit",
        )
    return _result(
        condition,
        requirement,
        fact,
        (
            ComparisonOutcome.SATISFIED
            if fact.value == expected
            else ComparisonOutcome.NOT_SATISFIED
        ),
        "deterministic_exact_comparison",
    )


def _is_non_finite_number(value: object) -> bool:
    """Identify non-finite float/Decimal values before any operator runs."""

    if isinstance(value, Decimal):
        return not value.is_finite()
    if type(value) is float:
        return not math.isfinite(value)
    return False


def compare_condition(
    condition: ComparableCondition,
    requirement: InformationRequirement,
    fact: ExtractedRequirementFact,
    evidence: Evidence,
    case_context: CaseContext,
) -> ConditionComparison:
    """Compare one explicit condition and fact without rule aggregation."""

    if not isinstance(condition, (Threshold, ExactCondition, SourceFragment)):
        raise TypeError("condition must be Threshold, ExactCondition, or SourceFragment")
    if not isinstance(requirement, InformationRequirement):
        raise TypeError("requirement must be an InformationRequirement")
    if not isinstance(fact, ExtractedRequirementFact):
        raise TypeError("fact must be an ExtractedRequirementFact")
    if not isinstance(evidence, Evidence):
        raise TypeError("evidence must be an Evidence")
    if not isinstance(case_context, CaseContext):
        raise TypeError("case_context must be a CaseContext")
    _validate_links(condition, requirement, fact, evidence, case_context)

    boundary = _fact_boundary(condition, requirement, fact)
    if boundary is not None:
        return boundary
    if isinstance(condition, Threshold):
        return _numeric_comparison(condition, requirement, fact)
    if isinstance(condition, ExactCondition):
        return _exact_comparison(condition, requirement, fact)
    return _result(
        condition,
        requirement,
        fact,
        ComparisonOutcome.HUMAN_REQUIRED,
        "non_mechanical_condition",
    )
