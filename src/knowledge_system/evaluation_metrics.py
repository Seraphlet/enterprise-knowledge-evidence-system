"""Pure deterministic metrics and reports for the synthetic Golden Dataset.

This module evaluates supplied predictions only.  It does not retrieve, run the
product graph, call a model, access the network, or select cases by tag.
"""

from __future__ import annotations

import json
import math
import re
from collections import defaultdict
from dataclasses import dataclass, fields, is_dataclass
from enum import Enum
from types import MappingProxyType
from typing import Any, Iterable, Mapping

from .golden_dataset import (
    BoundaryTruth,
    ClarificationTruth,
    ForbiddenBehavior,
    GoldenCase,
    GoldenDataset,
    OutputMode,
    VerificationTruth,
)
from .request_classification import KnowledgeRequestType


REPORT_SCHEMA = "knowledge-system.evaluation-report"
REPORT_SCHEMA_VERSION = 1
COMPARISON_SCHEMA = "knowledge-system.evaluation-comparison"
COMPARISON_SCHEMA_VERSION = 1
_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,127}$")


class EvaluationValidationError(ValueError):
    """Prediction, run identity, or serialized report is structurally invalid."""


@dataclass(frozen=True)
class Prediction:
    expression_id: str
    request_type: KnowledgeRequestType
    ranked_evidence_ids: tuple[str, ...]
    output_mode: OutputMode
    boundary: BoundaryTruth
    clarification: ClarificationTruth
    verification: VerificationTruth
    comparison_outcome: str | None = None
    missing_information: tuple[str, ...] = ()
    trace_fields: tuple[str, ...] = ()
    source_evidence_ids: tuple[str, ...] = ()
    lineage_evidence_ids: tuple[str, ...] = ()
    observed_forbidden_behaviors: tuple[ForbiddenBehavior, ...] = ()
    final_business_decision: bool = False

    def __post_init__(self) -> None:
        _identity(self.expression_id, "prediction.expression_id")
        _enum_instance(self.request_type, KnowledgeRequestType, "request_type")
        _enum_instance(self.output_mode, OutputMode, "output_mode")
        _enum_instance(self.boundary, BoundaryTruth, "boundary")
        _enum_instance(self.clarification, ClarificationTruth, "clarification")
        _enum_instance(self.verification, VerificationTruth, "verification")
        _identity_tuple(self.ranked_evidence_ids, "ranked_evidence_ids")
        _text_tuple(self.missing_information, "missing_information")
        _text_tuple(self.trace_fields, "trace_fields")
        _identity_tuple(self.source_evidence_ids, "source_evidence_ids")
        _identity_tuple(self.lineage_evidence_ids, "lineage_evidence_ids")
        if not set(self.source_evidence_ids) <= set(self.ranked_evidence_ids):
            raise EvaluationValidationError("source Evidence must be ranked Evidence")
        if not set(self.lineage_evidence_ids) <= set(self.ranked_evidence_ids):
            raise EvaluationValidationError("lineage Evidence must be ranked Evidence")
        if type(self.final_business_decision) is not bool:
            raise EvaluationValidationError("final_business_decision must be boolean")
        if self.comparison_outcome is not None:
            _text(self.comparison_outcome, "comparison_outcome")
        if type(self.observed_forbidden_behaviors) is not tuple or any(
            not isinstance(item, ForbiddenBehavior)
            for item in self.observed_forbidden_behaviors
        ):
            raise EvaluationValidationError(
                "observed_forbidden_behaviors must contain ForbiddenBehavior"
            )
        if len(set(self.observed_forbidden_behaviors)) != len(
            self.observed_forbidden_behaviors
        ):
            raise EvaluationValidationError("duplicate forbidden behavior")


@dataclass(frozen=True)
class RunMetadata:
    run_id: str
    system_id: str
    dataset_version: str
    corpus_version: str
    selected_expression_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        _identity(self.run_id, "run_id")
        _identity(self.system_id, "system_id")
        _identity(self.dataset_version, "dataset_version")
        _identity(self.corpus_version, "corpus_version")
        _identity_tuple(self.selected_expression_ids, "selected_expression_ids")
        if self.selected_expression_ids != tuple(sorted(self.selected_expression_ids)):
            raise EvaluationValidationError(
                "selected_expression_ids must be stably sorted"
            )
        if not self.selected_expression_ids:
            raise EvaluationValidationError("selection must not be empty")


@dataclass(frozen=True)
class RecallMetric:
    applicable: bool
    relevant_count: int
    retrieved_relevant_count: int
    value: float | None
    minimum: float | None
    passed: bool | None

    def __post_init__(self) -> None:
        _metric_shape(self.applicable, self.value, self.passed, "recall")
        _count(self.relevant_count, "relevant_count")
        _count(self.retrieved_relevant_count, "retrieved_relevant_count")
        if self.retrieved_relevant_count > self.relevant_count:
            raise EvaluationValidationError("retrieved relevant exceeds relevant")
        _optional_ratio(self.minimum, "minimum")
        if self.applicable != (self.relevant_count > 0):
            raise EvaluationValidationError("recall applicability is inconsistent")
        if self.applicable:
            expected = _q(self.retrieved_relevant_count / self.relevant_count)
            if self.value != expected or self.minimum is None or self.passed != (expected >= self.minimum):
                raise EvaluationValidationError("recall result is inconsistent")
        elif self.minimum is not None:
            raise EvaluationValidationError("N/A recall cannot have a minimum")


@dataclass(frozen=True)
class RankingMetric:
    applicable: bool
    preferred_count: int
    first_relevant_rank: int | None
    mrr: float | None
    ndcg: float | None
    order_compliant: bool | None

    def __post_init__(self) -> None:
        _count(self.preferred_count, "preferred_count")
        _optional_ratio(self.mrr, "mrr")
        _optional_ratio(self.ndcg, "ndcg")
        if self.applicable != (self.preferred_count > 0):
            raise EvaluationValidationError("ranking applicability is inconsistent")
        if self.applicable:
            if self.mrr is None or self.ndcg is None or type(self.order_compliant) is not bool:
                raise EvaluationValidationError("applicable ranking metrics are incomplete")
            if self.first_relevant_rank is not None and (type(self.first_relevant_rank) is not int or self.first_relevant_rank < 1):
                raise EvaluationValidationError("first relevant rank is invalid")
            expected_mrr = 0.0 if self.first_relevant_rank is None else _q(1.0 / self.first_relevant_rank)
            if self.mrr != expected_mrr:
                raise EvaluationValidationError("ranking MRR is inconsistent")
        elif any(item is not None for item in (self.first_relevant_rank, self.mrr, self.ndcg, self.order_compliant)):
            raise EvaluationValidationError("N/A ranking must not contain scores")


@dataclass(frozen=True)
class BrowseMetric:
    applicable: bool
    expected_count: int
    returned_count: int
    matched_count: int
    coverage: float | None
    precision: float | None
    minimum_coverage: float | None
    top1_only: bool | None
    passed: bool | None

    def __post_init__(self) -> None:
        for name in ("expected_count", "returned_count", "matched_count"):
            _count(getattr(self, name), name)
        if self.matched_count > self.expected_count or self.matched_count > self.returned_count:
            raise EvaluationValidationError("Browse matched count is inconsistent")
        _optional_ratio(self.coverage, "coverage")
        _optional_ratio(self.precision, "precision")
        _optional_ratio(self.minimum_coverage, "minimum_coverage")
        if self.applicable:
            if self.expected_count < 2 or any(item is None for item in (self.coverage, self.precision, self.minimum_coverage)) or type(self.top1_only) is not bool or type(self.passed) is not bool:
                raise EvaluationValidationError("applicable Browse metrics are incomplete")
            expected_coverage = _q(self.matched_count / self.expected_count)
            expected_precision = _q(self.matched_count / self.returned_count) if self.returned_count else 0.0
            if self.coverage != expected_coverage or self.precision != expected_precision or self.top1_only != (self.returned_count <= 1) or self.passed != (expected_coverage >= self.minimum_coverage and not self.top1_only):
                raise EvaluationValidationError("Browse result is inconsistent")
        elif self.expected_count != 0 or any(item is not None for item in (self.coverage, self.precision, self.minimum_coverage, self.top1_only, self.passed)):
            raise EvaluationValidationError("N/A Browse must not contain scores")


@dataclass(frozen=True)
class BehaviorMetric:
    check_count: int
    passed_count: int
    passed: bool
    failures: tuple[str, ...]

    def __post_init__(self) -> None:
        _count(self.check_count, "check_count")
        _count(self.passed_count, "passed_count")
        if self.passed_count > self.check_count or type(self.passed) is not bool:
            raise EvaluationValidationError("behavior counts are invalid")
        _text_tuple(self.failures, "behavior.failures")
        if self.check_count - self.passed_count != len(self.failures) or self.passed != (not self.failures and self.passed_count == self.check_count):
            raise EvaluationValidationError("behavior pass flag is inconsistent")


@dataclass(frozen=True)
class SafetyMetric:
    passed: bool
    violations: tuple[str, ...]

    def __post_init__(self) -> None:
        if type(self.passed) is not bool:
            raise EvaluationValidationError("safety passed must be boolean")
        _text_tuple(self.violations, "safety.violations")
        if self.passed != (not self.violations):
            raise EvaluationValidationError("safety pass flag is inconsistent")


@dataclass(frozen=True)
class ExpressionReport:
    expression_id: str
    case_id: str
    request_type: KnowledgeRequestType
    tags: tuple[str, ...]
    missing_prediction: bool
    recall: RecallMetric
    ranking: RankingMetric
    browse: BrowseMetric
    behavior: BehaviorMetric
    safety: SafetyMetric

    def __post_init__(self) -> None:
        _identity(self.expression_id, "expression_id")
        _identity(self.case_id, "case_id")
        _enum_instance(self.request_type, KnowledgeRequestType, "request_type")
        _identity_tuple(self.tags, "tags")
        if self.tags != tuple(sorted(self.tags)):
            raise EvaluationValidationError("expression tags are not stable")
        if type(self.missing_prediction) is not bool:
            raise EvaluationValidationError("missing_prediction must be boolean")
        if not all(isinstance(item, expected) for item, expected in ((self.recall, RecallMetric), (self.ranking, RankingMetric), (self.browse, BrowseMetric), (self.behavior, BehaviorMetric), (self.safety, SafetyMetric))):
            raise EvaluationValidationError("expression metric type is invalid")


@dataclass(frozen=True)
class MetricAggregate:
    applicable_count: int
    value: float | None

    def __post_init__(self) -> None:
        _count(self.applicable_count, "applicable_count")
        _optional_ratio(self.value, "aggregate.value")
        if (self.applicable_count == 0) != (self.value is None):
            raise EvaluationValidationError("aggregate N/A denominator is inconsistent")


@dataclass(frozen=True)
class SliceReport:
    slice_type: str
    slice_value: str
    count: int
    missing: int
    recall: MetricAggregate
    recall_pass_rate: MetricAggregate
    ranking_mrr: MetricAggregate
    ranking_ndcg: MetricAggregate
    ranking_compliance_rate: MetricAggregate
    browse_coverage: MetricAggregate
    browse_precision: MetricAggregate
    browse_pass_rate: MetricAggregate
    behavior_pass_rate: MetricAggregate
    safety_pass_rate: MetricAggregate

    def __post_init__(self) -> None:
        _text(self.slice_type, "slice_type")
        _text(self.slice_value, "slice_value")
        _count(self.count, "slice.count")
        _count(self.missing, "slice.missing")
        if self.missing > self.count:
            raise EvaluationValidationError("slice missing exceeds count")
        metrics = (
            self.recall, self.recall_pass_rate,
            self.ranking_mrr, self.ranking_ndcg, self.ranking_compliance_rate,
            self.browse_coverage, self.browse_precision, self.browse_pass_rate,
            self.behavior_pass_rate, self.safety_pass_rate,
        )
        if any(not isinstance(item, MetricAggregate) or item.applicable_count > self.count for item in metrics):
            raise EvaluationValidationError("slice metric denominator is invalid")
        if self.behavior_pass_rate.applicable_count != self.count or self.safety_pass_rate.applicable_count != self.count:
            raise EvaluationValidationError("behavior/safety denominator must equal slice count")
        if self.recall_pass_rate.applicable_count != self.recall.applicable_count or self.ranking_compliance_rate.applicable_count != self.ranking_mrr.applicable_count or self.browse_pass_rate.applicable_count != self.browse_coverage.applicable_count:
            raise EvaluationValidationError("metric pass-rate denominator is inconsistent")


@dataclass(frozen=True)
class CaseReport:
    case_id: str
    expression_ids: tuple[str, ...]
    summary: SliceReport

    def __post_init__(self) -> None:
        _identity(self.case_id, "case_id")
        _identity_tuple(self.expression_ids, "case.expression_ids")
        if self.expression_ids != tuple(sorted(self.expression_ids)):
            raise EvaluationValidationError("case expressions are not stable")
        if not isinstance(self.summary, SliceReport) or self.summary.slice_type != "case" or self.summary.slice_value != self.case_id or self.summary.count != len(self.expression_ids):
            raise EvaluationValidationError("case summary is inconsistent")


@dataclass(frozen=True)
class EvaluationReport:
    schema: str
    schema_version: int
    metadata: RunMetadata
    expressions: tuple[ExpressionReport, ...]
    cases: tuple[CaseReport, ...]
    aggregate: SliceReport
    request_type_slices: tuple[SliceReport, ...]
    tag_slices: tuple[SliceReport, ...]

    def __post_init__(self) -> None:
        if self.schema != REPORT_SCHEMA or type(self.schema_version) is not int or self.schema_version != REPORT_SCHEMA_VERSION:
            raise EvaluationValidationError("unknown report schema/version")
        if not isinstance(self.metadata, RunMetadata):
            raise EvaluationValidationError("report metadata is invalid")
        for name in ("expressions", "cases", "request_type_slices", "tag_slices"):
            if type(getattr(self, name)) is not tuple:
                raise EvaluationValidationError(f"report {name} must be a tuple")
        expression_ids = tuple(item.expression_id for item in self.expressions)
        if expression_ids != self.metadata.selected_expression_ids or len(set(expression_ids)) != len(expression_ids):
            raise EvaluationValidationError("report expressions do not match selection")
        if any(not isinstance(item, ExpressionReport) for item in self.expressions):
            raise EvaluationValidationError("report expression type is invalid")
        if not isinstance(self.aggregate, SliceReport) or self.aggregate.slice_type != "aggregate" or self.aggregate.slice_value != "all" or self.aggregate.count != len(self.expressions):
            raise EvaluationValidationError("report aggregate is inconsistent")
        for collection, slice_type in ((self.request_type_slices, "request_type"), (self.tag_slices, "tag")):
            values = tuple(item.slice_value for item in collection)
            if values != tuple(sorted(values)) or len(set(values)) != len(values) or any(item.slice_type != slice_type for item in collection):
                raise EvaluationValidationError(f"{slice_type} slices are not stable")
        if sum(item.count for item in self.request_type_slices) != len(self.expressions):
            raise EvaluationValidationError("request type slices do not partition selection")
        if any(item.slice_value not in {request.value for request in KnowledgeRequestType} for item in self.request_type_slices):
            raise EvaluationValidationError("request type slice is unknown")
        if any(not isinstance(item, CaseReport) for item in self.cases):
            raise EvaluationValidationError("case report type is invalid")
        case_ids = tuple(item.case_id for item in self.cases)
        if case_ids != tuple(sorted(case_ids)) or len(set(case_ids)) != len(case_ids):
            raise EvaluationValidationError("case reports are not stable")
        case_expression_ids = tuple(sorted(item for case in self.cases for item in case.expression_ids))
        if case_expression_ids != tuple(sorted(expression_ids)):
            raise EvaluationValidationError("case reports do not close over selection")

    def to_dict(self) -> dict[str, Any]:
        return _json_value(self)

    def to_json(self) -> str:
        return json.dumps(
            self.to_dict(), ensure_ascii=False, sort_keys=True, separators=(",", ":")
        )

    @classmethod
    def from_json(cls, payload: str) -> "EvaluationReport":
        value = _strict_json(payload)
        _keys(value, {"schema", "schema_version", "metadata", "expressions", "cases", "aggregate", "request_type_slices", "tag_slices"}, "report")
        if value["schema"] != REPORT_SCHEMA or type(value["schema_version"]) is not int or value["schema_version"] != REPORT_SCHEMA_VERSION:
            raise EvaluationValidationError("unknown report schema/version")
        return cls(
            REPORT_SCHEMA,
            REPORT_SCHEMA_VERSION,
            _metadata_from(value["metadata"]),
            tuple(_expression_from(item) for item in _list(value["expressions"], "expressions")),
            tuple(_case_from(item) for item in _list(value["cases"], "cases")),
            _slice_from(value["aggregate"]),
            tuple(_slice_from(item) for item in _list(value["request_type_slices"], "request_type_slices")),
            tuple(_slice_from(item) for item in _list(value["tag_slices"], "tag_slices")),
        )


@dataclass(frozen=True)
class MetricDelta:
    metric: str
    baseline: float | None
    current: float | None
    delta: float | None
    regression: bool | None

    def __post_init__(self) -> None:
        _text(self.metric, "delta.metric")
        for name in ("baseline", "current"):
            _optional_ratio(getattr(self, name), f"delta.{name}")
        if self.delta is not None and (type(self.delta) not in {int, float} or not math.isfinite(self.delta) or not -1.0 <= self.delta <= 1.0):
            raise EvaluationValidationError("delta is invalid")
        expected = None if self.baseline is None or self.current is None else _q(self.current - self.baseline)
        if self.delta != expected or self.regression != (None if expected is None else expected < 0):
            raise EvaluationValidationError("delta regression flag is inconsistent")


@dataclass(frozen=True)
class EvaluationComparison:
    schema: str
    schema_version: int
    baseline_run_id: str
    current_run_id: str
    dataset_version: str
    corpus_version: str
    selected_expression_ids: tuple[str, ...]
    aggregate_deltas: tuple[MetricDelta, ...]
    request_type_deltas: tuple[tuple[str, tuple[MetricDelta, ...]], ...]
    tag_deltas: tuple[tuple[str, tuple[MetricDelta, ...]], ...]
    regressed_expression_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        if self.schema != COMPARISON_SCHEMA or type(self.schema_version) is not int or self.schema_version != COMPARISON_SCHEMA_VERSION:
            raise EvaluationValidationError("unknown comparison schema/version")
        if any(type(item) is not tuple for item in (self.aggregate_deltas, self.request_type_deltas, self.tag_deltas)):
            raise EvaluationValidationError("comparison collections must be tuples")
        for name in ("baseline_run_id", "current_run_id", "dataset_version", "corpus_version"):
            _identity(getattr(self, name), name)
        _identity_tuple(self.selected_expression_ids, "selected_expression_ids")
        if self.selected_expression_ids != tuple(sorted(self.selected_expression_ids)):
            raise EvaluationValidationError("comparison selection is not stable")
        _identity_tuple(self.regressed_expression_ids, "regressed_expression_ids")
        if not set(self.regressed_expression_ids) <= set(self.selected_expression_ids):
            raise EvaluationValidationError("regressed expressions are outside selection")
        if tuple(item.metric for item in self.aggregate_deltas) != _DELTA_FIELDS:
            raise EvaluationValidationError("comparison aggregate metrics are incomplete")
        for collection in (self.request_type_deltas, self.tag_deltas):
            names = tuple(item[0] for item in collection)
            if names != tuple(sorted(names)) or len(set(names)) != len(names):
                raise EvaluationValidationError("comparison slices are not stable")
            if any(tuple(delta.metric for delta in deltas) != _DELTA_FIELDS for _, deltas in collection):
                raise EvaluationValidationError("comparison slice metrics are incomplete")

    def to_dict(self) -> dict[str, Any]:
        return _json_value(self)

    def to_json(self) -> str:
        return json.dumps(
            self.to_dict(), ensure_ascii=False, sort_keys=True, separators=(",", ":")
        )

    @classmethod
    def from_json(cls, payload: str) -> "EvaluationComparison":
        value = _strict_json(payload)
        _keys(value, {"schema", "schema_version", "baseline_run_id", "current_run_id", "dataset_version", "corpus_version", "selected_expression_ids", "aggregate_deltas", "request_type_deltas", "tag_deltas", "regressed_expression_ids"}, "comparison")
        if value["schema"] != COMPARISON_SCHEMA or type(value["schema_version"]) is not int or value["schema_version"] != COMPARISON_SCHEMA_VERSION:
            raise EvaluationValidationError("unknown comparison schema/version")
        return cls(
            COMPARISON_SCHEMA,
            COMPARISON_SCHEMA_VERSION,
            _identity(value["baseline_run_id"], "baseline_run_id"),
            _identity(value["current_run_id"], "current_run_id"),
            _identity(value["dataset_version"], "dataset_version"),
            _identity(value["corpus_version"], "corpus_version"),
            tuple(_identity(item, "selected_expression_id") for item in _list(value["selected_expression_ids"], "selected_expression_ids")),
            tuple(_delta_from(item) for item in _list(value["aggregate_deltas"], "aggregate_deltas")),
            tuple(_named_deltas_from(item) for item in _list(value["request_type_deltas"], "request_type_deltas")),
            tuple(_named_deltas_from(item) for item in _list(value["tag_deltas"], "tag_deltas")),
            tuple(_identity(item, "regressed_expression_id") for item in _list(value["regressed_expression_ids"], "regressed_expression_ids")),
        )


def _text(value: object, path: str) -> str:
    if type(value) is not str or not value or value != value.strip():
        raise EvaluationValidationError(f"{path} must be non-empty trimmed text")
    return value


def _identity(value: object, path: str) -> str:
    result = _text(value, path)
    if not _ID_RE.fullmatch(result):
        raise EvaluationValidationError(f"{path} must be a stable identifier")
    return result


def _identity_tuple(value: object, path: str) -> None:
    if type(value) is not tuple:
        raise EvaluationValidationError(f"{path} must be a tuple")
    checked = tuple(_identity(item, path) for item in value)
    if len(set(checked)) != len(checked):
        raise EvaluationValidationError(f"{path} contains duplicates")


def _text_tuple(value: object, path: str) -> None:
    if type(value) is not tuple:
        raise EvaluationValidationError(f"{path} must be a tuple")
    checked = tuple(_text(item, path) for item in value)
    if len(set(checked)) != len(checked):
        raise EvaluationValidationError(f"{path} contains duplicates")


def _enum_instance(value: object, enum_type: type[Enum], path: str) -> None:
    if not isinstance(value, enum_type):
        raise EvaluationValidationError(f"{path} must be {enum_type.__name__}")


def _q(value: float) -> float:
    return round(value, 12)


def _count(value: object, path: str) -> None:
    if type(value) is not int or value < 0:
        raise EvaluationValidationError(f"{path} must be a non-negative integer")


def _optional_ratio(value: object, path: str) -> None:
    if value is not None and (
        type(value) not in {int, float}
        or not math.isfinite(value)
        or not 0.0 <= value <= 1.0
    ):
        raise EvaluationValidationError(f"{path} must be N/A or a finite ratio")


def _metric_shape(
    applicable: object, value: object, passed: object, path: str
) -> None:
    if type(applicable) is not bool:
        raise EvaluationValidationError(f"{path}.applicable must be boolean")
    _optional_ratio(value, f"{path}.value")
    if applicable:
        if value is None or type(passed) is not bool:
            raise EvaluationValidationError(f"applicable {path} is incomplete")
    elif value is not None or passed is not None:
        raise EvaluationValidationError(f"N/A {path} must not contain results")


def _recall(case: GoldenCase, ranked: tuple[str, ...]) -> RecallMetric:
    relevant = set(case.truth.recall.relevant_evidence_ids)
    if not relevant:
        return RecallMetric(False, 0, 0, None, None, None)
    found = len(relevant & set(ranked))
    value = _q(found / len(relevant))
    return RecallMetric(True, len(relevant), found, value, case.truth.recall.minimum_recall, value >= case.truth.recall.minimum_recall)


def _ranking(case: GoldenCase, ranked: tuple[str, ...]) -> RankingMetric:
    tiers = case.truth.ranking.preferred_tiers
    preferred = {item for tier in tiers for item in tier}
    if not preferred:
        return RankingMetric(False, 0, None, None, None, None)
    rank = next((index for index, item in enumerate(ranked, 1) if item in preferred), None)
    mrr = 0.0 if rank is None else _q(1.0 / rank)
    gains = {item: len(tiers) - tier_index for tier_index, tier in enumerate(tiers) for item in tier}
    dcg = sum(gains.get(item, 0) / math.log2(index + 1) for index, item in enumerate(ranked, 1))
    ideal = sorted(gains.values(), reverse=True)
    idcg = sum(gain / math.log2(index + 1) for index, gain in enumerate(ideal, 1))
    ndcg = _q(dcg / idcg) if idcg else None
    observed = [item for item in ranked if item in preferred]
    tier_of = {item: index for index, tier in enumerate(tiers) for item in tier}
    compliant = all(tier_of[left] <= tier_of[right] for left, right in zip(observed, observed[1:]))
    if case.truth.ranking.order_required_within_tier:
        canonical = {item: index for tier in tiers for index, item in enumerate(tier)}
        compliant = compliant and all(
            tier_of[left] != tier_of[right] or canonical[left] <= canonical[right]
            for left, right in zip(observed, observed[1:])
        )
    return RankingMetric(True, len(preferred), rank, mrr, ndcg, compliant)


def _browse(case: GoldenCase, ranked: tuple[str, ...]) -> BrowseMetric:
    truth = case.truth.browse
    if truth is None:
        return BrowseMetric(False, 0, 0, 0, None, None, None, None, None)
    expected = set(truth.expected_evidence_ids)
    matched = len(expected & set(ranked))
    coverage = _q(matched / len(expected))
    precision = _q(matched / len(ranked)) if ranked else 0.0
    top1 = len(expected) > 1 and len(ranked) <= 1
    return BrowseMetric(True, len(expected), len(ranked), matched, coverage, precision, truth.minimum_coverage, top1, coverage >= truth.minimum_coverage and not top1)


def _behavior_and_safety(
    case: GoldenCase,
    prediction: Prediction | None,
    catalog: Mapping[str, Any],
) -> tuple[BehaviorMetric, SafetyMetric]:
    if prediction is None:
        return BehaviorMetric(1, 0, False, ("missing_prediction",)), SafetyMetric(False, ("MISSING_PREDICTION",))
    failures: list[str] = []
    checks: list[tuple[str, bool]] = [
        ("request_type", prediction.request_type is case.request_type),
        ("output_mode", prediction.output_mode in case.truth.allowed_output_modes),
        ("boundary", prediction.boundary is case.truth.boundary),
        ("clarification", prediction.clarification is case.truth.clarification),
        ("verification", prediction.verification is case.truth.verification),
        ("trace_fields", set(case.truth.provenance.required_trace_fields) <= set(prediction.trace_fields)),
    ]
    known_ranked = tuple(item for item in prediction.ranked_evidence_ids if item in catalog)
    if case.truth.provenance.require_source_reference:
        checks.append(("source_reference_coverage", set(known_ranked) <= set(prediction.source_evidence_ids) and bool(known_ranked)))
    if case.truth.provenance.require_lineage:
        checks.append(("lineage_coverage", set(known_ranked) <= set(prediction.lineage_evidence_ids) and bool(known_ranked)))
    comparison = case.truth.comparison
    if comparison is None:
        checks.append(("comparison_absent", prediction.comparison_outcome is None and not prediction.missing_information))
    else:
        checks.append(("comparison_outcome", prediction.comparison_outcome in comparison.acceptable_outcomes))
        checks.append(("missing_information", set(comparison.required_missing_information) <= set(prediction.missing_information)))
    failures.extend(name for name, passed in checks if not passed)

    unknown = sorted(set(prediction.ranked_evidence_ids) - set(catalog))
    forbidden = sorted(set(prediction.ranked_evidence_ids) & set(case.truth.forbidden_evidence_ids))
    hard_filter = sorted(
        item for item in prediction.ranked_evidence_ids
        if item in catalog and (catalog[item].status != "active" or catalog[item].permission != "public")
    )
    disallowed_behaviors = sorted(
        item.value for item in prediction.observed_forbidden_behaviors
        if item in case.truth.forbidden_behaviors
    )
    violations: list[str] = []
    violations.extend(f"UNKNOWN_EVIDENCE:{item}" for item in unknown)
    violations.extend(f"FORBIDDEN_EVIDENCE:{item}" for item in forbidden)
    violations.extend(f"HARD_FILTER_VIOLATION:{item}" for item in hard_filter)
    violations.extend(f"FORBIDDEN_BEHAVIOR:{item}" for item in disallowed_behaviors)
    if prediction.final_business_decision:
        violations.append("FINAL_BUSINESS_DECISION")
    return (
        BehaviorMetric(len(checks), len(checks) - len(failures), not failures, tuple(sorted(failures))),
        SafetyMetric(not violations, tuple(sorted(set(violations)))),
    )


def _missing_metrics(case: GoldenCase) -> tuple[RecallMetric, RankingMetric, BrowseMetric]:
    return _recall(case, ()), _ranking(case, ()), _browse(case, ())


def _mean(values: Iterable[float | None]) -> MetricAggregate:
    applicable = tuple(value for value in values if value is not None)
    return MetricAggregate(len(applicable), _q(sum(applicable) / len(applicable)) if applicable else None)


def _slice(slice_type: str, slice_value: str, reports: Iterable[ExpressionReport]) -> SliceReport:
    items = tuple(sorted(reports, key=lambda item: item.expression_id))
    return SliceReport(
        slice_type,
        slice_value,
        len(items),
        sum(item.missing_prediction for item in items),
        _mean(item.recall.value for item in items),
        _mean(None if item.recall.passed is None else 1.0 if item.recall.passed else 0.0 for item in items),
        _mean(item.ranking.mrr for item in items),
        _mean(item.ranking.ndcg for item in items),
        _mean(None if item.ranking.order_compliant is None else 1.0 if item.ranking.order_compliant else 0.0 for item in items),
        _mean(item.browse.coverage for item in items),
        _mean(item.browse.precision for item in items),
        _mean(None if item.browse.passed is None else 1.0 if item.browse.passed else 0.0 for item in items),
        _mean(1.0 if item.behavior.passed else 0.0 for item in items),
        _mean(1.0 if item.safety.passed else 0.0 for item in items),
    )


def evaluate_predictions(
    dataset: GoldenDataset,
    metadata: RunMetadata,
    predictions: Iterable[Prediction],
) -> EvaluationReport:
    """Evaluate already-produced predictions without executing product behavior."""

    if not isinstance(dataset, GoldenDataset) or not isinstance(metadata, RunMetadata):
        raise TypeError("dataset and metadata must be GoldenDataset and RunMetadata")
    if metadata.dataset_version != dataset.dataset_version or metadata.corpus_version != dataset.corpus_version:
        raise EvaluationValidationError("run metadata dataset/corpus identity mismatch")
    expression_index: dict[str, tuple[GoldenCase, Any]] = {}
    for case in dataset.cases:
        for expression in case.expressions:
            expression_index[expression.expression_id] = (case, expression)
    selected = set(metadata.selected_expression_ids)
    unknown_selection = selected - set(expression_index)
    if unknown_selection:
        raise EvaluationValidationError(f"selection contains unknown expression: {sorted(unknown_selection)}")
    supplied: dict[str, Prediction] = {}
    for prediction in predictions:
        if not isinstance(prediction, Prediction):
            raise TypeError("predictions must contain Prediction items")
        if prediction.expression_id in supplied:
            raise EvaluationValidationError("duplicate prediction expression")
        if prediction.expression_id not in selected:
            raise EvaluationValidationError("prediction is outside selected expressions")
        supplied[prediction.expression_id] = prediction
    catalog = MappingProxyType({item.evidence.evidence_id: item for item in dataset.evidence_catalog})
    reports: list[ExpressionReport] = []
    for expression_id in metadata.selected_expression_ids:
        case, _ = expression_index[expression_id]
        prediction = supplied.get(expression_id)
        ranked = () if prediction is None else prediction.ranked_evidence_ids
        recall, ranking, browse = (_missing_metrics(case) if prediction is None else (_recall(case, ranked), _ranking(case, ranked), _browse(case, ranked)))
        behavior, safety = _behavior_and_safety(case, prediction, catalog)
        reports.append(ExpressionReport(expression_id, case.case_id, case.request_type, case.tags, prediction is None, recall, ranking, browse, behavior, safety))
    expression_reports = tuple(reports)
    by_case: dict[str, list[ExpressionReport]] = defaultdict(list)
    by_request: dict[str, list[ExpressionReport]] = defaultdict(list)
    by_tag: dict[str, list[ExpressionReport]] = defaultdict(list)
    for report in expression_reports:
        by_case[report.case_id].append(report)
        by_request[report.request_type.value].append(report)
        for tag in report.tags:
            by_tag[tag].append(report)
    cases = tuple(CaseReport(case_id, tuple(item.expression_id for item in sorted(items, key=lambda entry: entry.expression_id)), _slice("case", case_id, items)) for case_id, items in sorted(by_case.items()))
    return EvaluationReport(
        REPORT_SCHEMA,
        REPORT_SCHEMA_VERSION,
        metadata,
        expression_reports,
        cases,
        _slice("aggregate", "all", expression_reports),
        tuple(_slice("request_type", value, items) for value, items in sorted(by_request.items())),
        tuple(_slice("tag", value, items) for value, items in sorted(by_tag.items())),
    )


_DELTA_FIELDS = (
    "recall", "recall_pass_rate", "ranking_mrr", "ranking_ndcg",
    "ranking_compliance_rate", "browse_coverage", "browse_precision",
    "browse_pass_rate", "behavior_pass_rate", "safety_pass_rate",
)


def _deltas(baseline: SliceReport, current: SliceReport) -> tuple[MetricDelta, ...]:
    result = []
    for name in _DELTA_FIELDS:
        left = getattr(baseline, name).value
        right = getattr(current, name).value
        delta = None if left is None or right is None else _q(right - left)
        result.append(MetricDelta(name, left, right, delta, None if delta is None else delta < 0))
    return tuple(result)


def compare_reports(baseline: EvaluationReport, current: EvaluationReport) -> EvaluationComparison:
    if not isinstance(baseline, EvaluationReport) or not isinstance(current, EvaluationReport):
        raise TypeError("baseline and current must be EvaluationReport")
    left, right = baseline.metadata, current.metadata
    if left.run_id == right.run_id:
        raise EvaluationValidationError("baseline and current run identity conflict")
    if (left.dataset_version, left.corpus_version, left.selected_expression_ids) != (right.dataset_version, right.corpus_version, right.selected_expression_ids):
        raise EvaluationValidationError("reports have incompatible dataset/corpus/selection")
    def indexed(items: tuple[SliceReport, ...]) -> dict[str, SliceReport]:
        return {item.slice_value: item for item in items}
    left_request, right_request = indexed(baseline.request_type_slices), indexed(current.request_type_slices)
    left_tags, right_tags = indexed(baseline.tag_slices), indexed(current.tag_slices)
    if set(left_request) != set(right_request) or set(left_tags) != set(right_tags):
        raise EvaluationValidationError("reports have incompatible slices")
    baseline_expressions = {item.expression_id: item for item in baseline.expressions}
    regressed: list[str] = []
    for item in current.expressions:
        old = baseline_expressions[item.expression_id]
        numeric_pairs = ((old.recall.value, item.recall.value), (old.ranking.mrr, item.ranking.mrr), (old.ranking.ndcg, item.ranking.ndcg), (old.browse.coverage, item.browse.coverage), (old.browse.precision, item.browse.precision))
        if any(a is not None and b is not None and b < a for a, b in numeric_pairs) or (old.ranking.order_compliant is True and item.ranking.order_compliant is False) or (old.browse.passed is True and item.browse.passed is False) or (old.behavior.passed and not item.behavior.passed) or (old.safety.passed and not item.safety.passed) or (not old.missing_prediction and item.missing_prediction):
            regressed.append(item.expression_id)
    return EvaluationComparison(
        COMPARISON_SCHEMA,
        COMPARISON_SCHEMA_VERSION,
        left.run_id,
        right.run_id,
        left.dataset_version,
        left.corpus_version,
        left.selected_expression_ids,
        _deltas(baseline.aggregate, current.aggregate),
        tuple((name, _deltas(left_request[name], right_request[name])) for name in sorted(left_request)),
        tuple((name, _deltas(left_tags[name], right_tags[name])) for name in sorted(left_tags)),
        tuple(sorted(regressed)),
    )


def _json_value(value: Any) -> Any:
    if isinstance(value, Enum): return value.value
    if is_dataclass(value): return {field.name: _json_value(getattr(value, field.name)) for field in fields(value)}
    if isinstance(value, Mapping): return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)): return [_json_value(item) for item in value]
    return value


def _strict_json(payload: str) -> dict[str, Any]:
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result: raise EvaluationValidationError(f"duplicate JSON key: {key}")
            result[key] = value
        return result
    try:
        value = json.loads(payload, object_pairs_hook=pairs, parse_constant=lambda item: (_ for _ in ()).throw(EvaluationValidationError(f"invalid constant: {item}")))
    except EvaluationValidationError: raise
    except (TypeError, ValueError, json.JSONDecodeError) as error: raise EvaluationValidationError("invalid report JSON") from error
    if type(value) is not dict: raise EvaluationValidationError("report JSON must be an object")
    return value


def _keys(value: object, expected: set[str], path: str) -> dict[str, Any]:
    if type(value) is not dict or set(value) != expected: raise EvaluationValidationError(f"{path} fields do not match schema")
    return value


def _list(value: object, path: str) -> list[Any]:
    if type(value) is not list: raise EvaluationValidationError(f"{path} must be an array")
    return value


def _enum(enum_type, value, path):
    try: return enum_type(value)
    except (TypeError, ValueError) as error: raise EvaluationValidationError(f"{path} has unknown value") from error


def _metric_from(value, kind):
    classes = {"recall": RecallMetric, "ranking": RankingMetric, "browse": BrowseMetric, "behavior": BehaviorMetric, "safety": SafetyMetric, "aggregate": MetricAggregate}
    cls = classes[kind]; expected = {field.name for field in fields(cls)}; item = _keys(value, expected, kind)
    if kind in {"behavior", "safety"}: item = dict(item); item["failures" if kind == "behavior" else "violations"] = tuple(item["failures" if kind == "behavior" else "violations"])
    return cls(**item)


def _metadata_from(value):
    item = _keys(value, {"run_id", "system_id", "dataset_version", "corpus_version", "selected_expression_ids"}, "metadata")
    return RunMetadata(item["run_id"], item["system_id"], item["dataset_version"], item["corpus_version"], tuple(item["selected_expression_ids"]))


def _expression_from(value):
    item = _keys(value, {"expression_id", "case_id", "request_type", "tags", "missing_prediction", "recall", "ranking", "browse", "behavior", "safety"}, "expression")
    return ExpressionReport(item["expression_id"], item["case_id"], _enum(KnowledgeRequestType, item["request_type"], "request_type"), tuple(item["tags"]), item["missing_prediction"], _metric_from(item["recall"], "recall"), _metric_from(item["ranking"], "ranking"), _metric_from(item["browse"], "browse"), _metric_from(item["behavior"], "behavior"), _metric_from(item["safety"], "safety"))


def _slice_from(value):
    expected = {"slice_type", "slice_value", "count", "missing", "recall", "recall_pass_rate", "ranking_mrr", "ranking_ndcg", "ranking_compliance_rate", "browse_coverage", "browse_precision", "browse_pass_rate", "behavior_pass_rate", "safety_pass_rate"}; item = _keys(value, expected, "slice")
    return SliceReport(item["slice_type"], item["slice_value"], item["count"], item["missing"], *(_metric_from(item[name], "aggregate") for name in _DELTA_FIELDS))


def _case_from(value):
    item = _keys(value, {"case_id", "expression_ids", "summary"}, "case")
    return CaseReport(item["case_id"], tuple(item["expression_ids"]), _slice_from(item["summary"]))


def _delta_from(value):
    item = _keys(value, {"metric", "baseline", "current", "delta", "regression"}, "delta")
    return MetricDelta(**item)


def _named_deltas_from(value):
    if type(value) is not list or len(value) != 2: raise EvaluationValidationError("named deltas must be a pair")
    return _text(value[0], "slice name"), tuple(_delta_from(item) for item in _list(value[1], "deltas"))
