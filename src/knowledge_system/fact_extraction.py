"""Deterministic, requirement-driven extraction from Case raw descriptions."""

from __future__ import annotations

import re
import unicodedata
from collections.abc import Iterable
from dataclasses import dataclass, field
from enum import Enum

from .context import CaseContext, FactValue
from .information_requirements import (
    InformationRequirement,
    RequirementCertainty,
)


class FactExtractionStatus(str, Enum):
    """Outcome for one existing Information Requirement."""

    FOUND = "FOUND"
    MISSING = "MISSING"
    UNKNOWN = "UNKNOWN"


@dataclass(frozen=True)
class CaseSourceSpan:
    """Verbatim span in one CaseContext.raw_descriptions item."""

    description_index: int
    start: int
    end: int
    text: str


@dataclass(frozen=True)
class FactObservation:
    """One literal value observed in Case text with exact provenance."""

    value: FactValue
    source: CaseSourceSpan
    unit: str | None = None


@dataclass(frozen=True)
class ExtractedRequirementFact:
    """Fact extraction outcome tied to its requirement and Candidate Evidence."""

    requirement_id: str
    evidence_id: str
    fact_type: str
    status: FactExtractionStatus
    value: FactValue = None
    observations: tuple[FactObservation, ...] = field(default_factory=tuple)
    reason: str | None = None


_NUMBER = r"-?\d+(?:\.\d+)?"
_SEPARATOR = r"\s*(?:为|是|[:：=])?\s*"
_UNIT_TOKENS = ("小时", "分钟", "个", "次", "条", "人", "元", "天", "分", "%", "％")
_UNIT = "(?:" + "|".join(re.escape(item) for item in _UNIT_TOKENS) + ")"
_VALUE_WITH_UNIT = rf"(?P<value>{_NUMBER})\s*(?P<unit>{_UNIT})?"
_TOKEN_END = r"(?![\dA-Za-z_\u4e00-\u9fff])"


def _numeric_value(text: str) -> int | float:
    return float(text) if "." in text else int(text)


def _has_numeric_continuation(
    description: str, match: re.Match[str]
) -> bool:
    """Reject a candidate that is only the prefix of a range or value list."""

    unit = match.group("unit")
    cursor = match.end("unit") if unit is not None else match.end("value")
    saw_separator = False
    while cursor < len(description):
        character = description[cursor]
        if character.isspace():
            cursor += 1
            continue
        if unicodedata.category(character)[0] in {"P", "S"}:
            saw_separator = True
            cursor += 1
            continue
        break
    return (
        saw_separator
        and cursor < len(description)
        and description[cursor].isdecimal()
    )


def _metric_patterns(metric: str) -> tuple[re.Pattern[str], ...]:
    """Build local patterns from one requirement metric, not a global schema."""

    escaped = re.escape(metric)
    patterns = [
        re.compile(rf"{escaped}{_SEPARATOR}{_VALUE_WITH_UNIT}{_TOKEN_END}"),
        re.compile(
            rf"{_VALUE_WITH_UNIT}{_SEPARATOR}{escaped}"
            rf"(?![A-Za-z0-9_\u4e00-\u9fff])"
        ),
    ]
    if metric.endswith("数") and len(metric) > 1:
        stem = metric[:-1]
        escaped_stem = re.escape(stem)
        patterns.extend(
            (
                re.compile(
                    rf"{escaped_stem}{_SEPARATOR}{_VALUE_WITH_UNIT}{_TOKEN_END}"
                ),
                re.compile(
                    rf"{_VALUE_WITH_UNIT}\s*{re.escape(stem[-1])}"
                    rf"(?![A-Za-z0-9_\u4e00-\u9fff])"
                ),
            )
        )
    return tuple(patterns)


def _supported_units(requirement: InformationRequirement) -> frozenset[str]:
    """Read literal units from this requirement's source condition only."""

    unit_pattern = re.compile(rf"{_NUMBER}\s*(?P<unit>{_UNIT})")
    return frozenset(
        match.group("unit") for match in unit_pattern.finditer(requirement.fact_type)
    )


def _observations(
    requirement: InformationRequirement,
    raw_descriptions: tuple[str, ...],
) -> tuple[FactObservation, ...]:
    metric = requirement.metric_hints[0]
    patterns = _metric_patterns(metric)
    supported_units = _supported_units(requirement)
    observations: dict[tuple[int, int, int, int | float], FactObservation] = {}
    for description_index, description in enumerate(raw_descriptions):
        for pattern in patterns:
            for match in pattern.finditer(description):
                unit = match.group("unit")
                if unit is not None and unit not in supported_units:
                    continue
                if _has_numeric_continuation(description, match):
                    continue
                value = _numeric_value(match.group("value"))
                key = (description_index, match.start(), match.end(), value)
                observations.setdefault(
                    key,
                    FactObservation(
                        value=value,
                        source=CaseSourceSpan(
                            description_index=description_index,
                            start=match.start(),
                            end=match.end(),
                            text=match.group(0),
                        ),
                        unit=unit,
                    ),
                )
    return tuple(
        observations[key]
        for key in sorted(
            observations,
            key=lambda item: (item[0], item[1], item[2], item[3]),
        )
    )


def extract_required_facts(
    requirements: Iterable[InformationRequirement],
    case_context: CaseContext,
) -> tuple[ExtractedRequirementFact, ...]:
    """Extract only facts requested by Candidate Rule requirements.

    The function does not inspect a rule corpus, mutate CaseContext, compare a
    value to a condition, or infer semantic facts.
    """

    if not isinstance(case_context, CaseContext):
        raise TypeError("case_context must be a CaseContext")
    selected = tuple(requirements)
    if any(not isinstance(item, InformationRequirement) for item in selected):
        raise TypeError("requirements must contain InformationRequirement items")

    results: list[ExtractedRequirementFact] = []
    for requirement in selected:
        common = {
            "requirement_id": requirement.requirement_id,
            "evidence_id": requirement.evidence_id,
            "fact_type": requirement.fact_type,
        }
        if requirement.certainty is RequirementCertainty.HUMAN_REQUIRED:
            results.append(
                ExtractedRequirementFact(
                    **common,
                    status=FactExtractionStatus.UNKNOWN,
                    reason="requirement_human_required",
                )
            )
            continue
        if len(requirement.metric_hints) != 1:
            results.append(
                ExtractedRequirementFact(
                    **common,
                    status=FactExtractionStatus.UNKNOWN,
                    reason="no_single_numeric_metric",
                )
            )
            continue

        observations = _observations(requirement, case_context.raw_descriptions)
        if not observations:
            results.append(
                ExtractedRequirementFact(
                    **common,
                    status=FactExtractionStatus.MISSING,
                    reason="required_fact_not_found",
                )
            )
            continue

        values = {item.value for item in observations}
        if len(values) != 1:
            results.append(
                ExtractedRequirementFact(
                    **common,
                    status=FactExtractionStatus.UNKNOWN,
                    observations=observations,
                    reason="conflicting_case_values",
                )
            )
            continue

        results.append(
            ExtractedRequirementFact(
                **common,
                status=FactExtractionStatus.FOUND,
                value=observations[0].value,
                observations=observations,
            )
        )
    return tuple(results)
