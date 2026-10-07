"""Conservative deterministic extraction of optional rule structure."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum

from .contracts import Evidence, JsonValue


class RuleExtractionStatus(str, Enum):
    """How much reliable structure was derived from the source text."""

    EMPTY = "EMPTY"
    PARTIAL = "PARTIAL"
    STRUCTURED = "STRUCTURED"


@dataclass(frozen=True)
class SourceFragment:
    """Verbatim source fragment and its offsets in Evidence.original_content."""

    text: str
    start: int
    end: int

    def to_dict(self) -> dict[str, JsonValue]:
        return {"text": self.text, "start": self.start, "end": self.end}


@dataclass(frozen=True)
class LogicalRelation:
    """A directly written AND/OR marker, not an executable expression tree."""

    relation: str
    source: SourceFragment

    def to_dict(self) -> dict[str, JsonValue]:
        return {"relation": self.relation, "source": self.source.to_dict()}


@dataclass(frozen=True)
class Threshold:
    """A literal numeric threshold; no comparison behavior is attached."""

    metric: str
    operator: str
    value: int | float
    unit: str | None
    source: SourceFragment

    def to_dict(self) -> dict[str, JsonValue]:
        return {
            "metric": self.metric,
            "operator": self.operator,
            "value": self.value,
            "unit": self.unit,
            "source": self.source.to_dict(),
        }


@dataclass(frozen=True)
class RuleStructure:
    """Optional fields derived from source, never a replacement for source."""

    conditions: tuple[SourceFragment, ...] = field(default_factory=tuple)
    logical_relations: tuple[LogicalRelation, ...] = field(default_factory=tuple)
    thresholds: tuple[Threshold, ...] = field(default_factory=tuple)
    exceptions: tuple[SourceFragment, ...] = field(default_factory=tuple)
    consequences: tuple[SourceFragment, ...] = field(default_factory=tuple)
    ambiguity: tuple[SourceFragment, ...] = field(default_factory=tuple)

    def to_dict(self) -> dict[str, JsonValue]:
        return {
            "conditions": [item.to_dict() for item in self.conditions],
            "logical_relations": [
                item.to_dict() for item in self.logical_relations
            ],
            "thresholds": [item.to_dict() for item in self.thresholds],
            "exceptions": [item.to_dict() for item in self.exceptions],
            "consequence": [item.to_dict() for item in self.consequences],
            "ambiguity": [item.to_dict() for item in self.ambiguity],
        }


@dataclass(frozen=True)
class RuleExtractionResult:
    """Source Evidence plus lower-authority optional derived structure."""

    source_evidence: Evidence
    status: RuleExtractionStatus
    structure: RuleStructure | None


_LABEL_RE = re.compile(
    r"(?im)^(?P<label>条件|condition|逻辑关系|逻辑|logic|阈值|threshold|"
    r"例外|exception|结果|后果|consequence|歧义|ambiguity)\s*[:：]\s*"
    r"(?P<value>.+?)\s*$"
)
_IF_RE = re.compile(
    r"(?:如果|若)(?P<condition>[^。；;\n]+?)(?:，?则)[，,]?"
    r"(?P<consequence>[^。；;\n]+)"
)
_WHEN_RE = re.compile(
    r"当(?P<condition>[^。；;\n]+?)时[，,]?"
    r"(?P<consequence>[^。；;\n]+)"
)
_LOGIC_RE = re.compile(
    r"(?P<and>并且|且|同时|\bAND\b)|(?P<or>或者|或|\bOR\b)", re.I
)
_THRESHOLD_RE = re.compile(
    r"(?P<metric>[\u4e00-\u9fffA-Za-z_]"
    r"[\u4e00-\u9fffA-Za-z0-9_ ]{0,20}?)\s*"
    r"(?P<operator>不少于|不低于|至少|大于等于|小于等于|不超过|至多|"
    r"超过|大于|低于|小于|>=|<=|>|<|≥|≤|=|等于)\s*"
    r"(?P<value>\d+(?:\.\d+)?)\s*"
    r"(?P<unit>个|次|元|%|％|天|小时|分钟|人|条|分)?"
)
_EXCEPTION_RE = re.compile(
    r"(?:但|但是)?(?P<exception>[^，,。；;\n]{1,40}除外)|"
    r"(?P<unless>除非[^，,。；;\n]+)"
)
_AMBIGUITY_RE = re.compile(
    r"可能|视情况|原则上|通常|必要时|酌情|未明确|不明确"
)
_METRIC_PREFIX_RE = re.compile(
    r"(?:(?:如果|若|当)\s*)?"
    r"(?:(?:并且|且|同时|或者|或|\bAND\b|\bOR\b)\s*)?",
    re.I,
)


def _fragment(match: re.Match[str], group: str) -> SourceFragment:
    return SourceFragment(match.group(group), match.start(group), match.end(group))


def _unique_fragments(items: list[SourceFragment]) -> tuple[SourceFragment, ...]:
    unique: dict[tuple[int, int], SourceFragment] = {}
    for item in items:
        unique.setdefault((item.start, item.end), item)
    return tuple(sorted(unique.values(), key=lambda item: (item.start, item.end)))


def _number(value: str) -> int | float:
    return float(value) if "." in value else int(value)


def _metric_source(match: re.Match[str]) -> tuple[str, SourceFragment]:
    """Exclude surrounding condition/relation syntax from one threshold atom."""

    raw_metric = match.group("metric")
    prefix = _METRIC_PREFIX_RE.match(raw_metric)
    prefix_end = prefix.end() if prefix is not None else 0
    metric = raw_metric[prefix_end:].strip()
    metric_start = match.start("metric") + prefix_end
    source = SourceFragment(
        match.string[metric_start : match.end()], metric_start, match.end()
    )
    return metric, source


def _logical_relations(
    contexts: tuple[SourceFragment, ...],
) -> tuple[LogicalRelation, ...]:
    """Extract relations only inside a condition or explicit logic field."""

    unique: dict[tuple[str, int, int], LogicalRelation] = {}
    for context in contexts:
        for match in _LOGIC_RE.finditer(context.text):
            start = context.start + match.start()
            end = context.start + match.end()
            relation = "AND" if match.group("and") is not None else "OR"
            unique.setdefault(
                (relation, start, end),
                LogicalRelation(
                    relation,
                    SourceFragment(match.group(0), start, end),
                ),
            )
    return tuple(
        sorted(
            unique.values(),
            key=lambda item: (item.source.start, item.source.end),
        )
    )


def extract_rule_structure(evidence: Evidence) -> RuleExtractionResult:
    """Derive only explicit, locally verifiable fields from source Evidence."""

    if not isinstance(evidence, Evidence):
        raise TypeError("evidence must be an Evidence")

    text = evidence.original_content
    conditions: list[SourceFragment] = []
    consequences: list[SourceFragment] = []
    exceptions: list[SourceFragment] = []
    ambiguity: list[SourceFragment] = []
    explicit_logic: list[SourceFragment] = []

    label_targets = {
        "条件": conditions,
        "condition": conditions,
        "例外": exceptions,
        "exception": exceptions,
        "结果": consequences,
        "后果": consequences,
        "consequence": consequences,
        "歧义": ambiguity,
        "ambiguity": ambiguity,
    }
    for match in _LABEL_RE.finditer(text):
        label = match.group("label").casefold()
        if label in {"逻辑关系", "逻辑", "logic"}:
            explicit_logic.append(_fragment(match, "value"))
            continue
        target = label_targets.get(label)
        if target is not None:
            target.append(_fragment(match, "value"))

    for pattern in (_IF_RE, _WHEN_RE):
        for match in pattern.finditer(text):
            conditions.append(_fragment(match, "condition"))
            consequences.append(_fragment(match, "consequence"))

    condition_fragments = _unique_fragments(conditions)
    logical_relations = _logical_relations(
        (*condition_fragments, *_unique_fragments(explicit_logic))
    )

    thresholds: list[Threshold] = []
    for match in _THRESHOLD_RE.finditer(text):
        metric, source = _metric_source(match)
        thresholds.append(
            Threshold(
                metric=metric,
                operator=match.group("operator"),
                value=_number(match.group("value")),
                unit=match.group("unit"),
                source=source,
            )
        )

    for match in _EXCEPTION_RE.finditer(text):
        group = "exception" if match.group("exception") is not None else "unless"
        exceptions.append(_fragment(match, group))

    for match in _AMBIGUITY_RE.finditer(text):
        ambiguity.append(SourceFragment(match.group(0), match.start(), match.end()))

    structure = RuleStructure(
        conditions=condition_fragments,
        logical_relations=logical_relations,
        thresholds=tuple(thresholds),
        exceptions=_unique_fragments(exceptions),
        consequences=_unique_fragments(consequences),
        ambiguity=_unique_fragments(ambiguity),
    )
    populated = any(structure.to_dict().values())
    if not populated:
        return RuleExtractionResult(evidence, RuleExtractionStatus.EMPTY, None)

    status = (
        RuleExtractionStatus.STRUCTURED
        if structure.conditions and structure.consequences
        else RuleExtractionStatus.PARTIAL
    )
    return RuleExtractionResult(evidence, status, structure)
