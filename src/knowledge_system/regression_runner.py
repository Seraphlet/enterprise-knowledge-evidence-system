"""Explicit, reproducible execution plans for Golden Dataset regression.

Importing this module performs no selection or execution.  ``FULL_GOLDEN``
means all Golden expressions only; it never means the project's unit tests.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, fields, is_dataclass
from enum import Enum
from typing import Any, Callable, Iterable

from .evaluation_metrics import (
    EvaluationComparison,
    EvaluationReport,
    Prediction,
    RunMetadata,
    compare_reports,
    evaluate_predictions,
)
from .golden_dataset import (
    BoundaryTruth,
    ClarificationTruth,
    ForbiddenBehavior,
    GoldenDataset,
    OutputMode,
    VerificationTruth,
)
from .request_classification import KnowledgeRequestType


PLAN_SCHEMA = "knowledge-system.regression-plan"
PLAN_SCHEMA_VERSION = 1
RUN_SCHEMA = "knowledge-system.regression-run"
RUN_SCHEMA_VERSION = 1
PAIR_SCHEMA = "knowledge-system.regression-pair"
PAIR_SCHEMA_VERSION = 1
FAILURE_SCHEMA = "knowledge-system.execution-failure"
FAILURE_SCHEMA_VERSION = 1
_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,127}$")
_FINGERPRINT_RE = re.compile(r"^[0-9a-f]{64}$")


class RunnerValidationError(ValueError):
    """Selection, plan, backend result, or artifact is structurally invalid."""


class SelectionKind(str, Enum):
    CASE_IDS = "CASE_IDS"
    TAGS = "TAGS"
    FULL_GOLDEN = "FULL_GOLDEN"


class TagMatch(str, Enum):
    ANY = "ANY"
    ALL = "ALL"


class FailureKind(str, Enum):
    BACKEND_EXCEPTION = "BACKEND_EXCEPTION"
    NO_RETURN = "NO_RETURN"
    INVALID_RETURN = "INVALID_RETURN"
    EXPRESSION_MISMATCH = "EXPRESSION_MISMATCH"


@dataclass(frozen=True)
class Selection:
    kind: SelectionKind
    case_ids: tuple[str, ...] = ()
    tags: tuple[str, ...] = ()
    tag_match: TagMatch | None = None
    explicit_full: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.kind, SelectionKind):
            raise RunnerValidationError("selection kind is invalid")
        _ids(self.case_ids, "case_ids", sorted_required=True)
        _ids(self.tags, "tags", sorted_required=True)
        if type(self.explicit_full) is not bool:
            raise RunnerValidationError("explicit_full must be boolean")
        if self.kind is SelectionKind.CASE_IDS:
            if not self.case_ids or self.tags or self.tag_match is not None or self.explicit_full:
                raise RunnerValidationError("CASE_IDS selection fields are inconsistent")
        elif self.kind is SelectionKind.TAGS:
            if not self.tags or self.case_ids or not isinstance(self.tag_match, TagMatch) or self.explicit_full:
                raise RunnerValidationError("TAGS selection fields are inconsistent")
        elif self.case_ids or self.tags or self.tag_match is not None or not self.explicit_full:
            raise RunnerValidationError("FULL_GOLDEN must be explicitly constructed")

    @classmethod
    def for_cases(cls, case_ids: Iterable[str]) -> "Selection":
        return cls(SelectionKind.CASE_IDS, tuple(sorted(case_ids)))

    @classmethod
    def for_tags(cls, tags: Iterable[str], match: TagMatch) -> "Selection":
        return cls(SelectionKind.TAGS, (), tuple(sorted(tags)), match)

    @classmethod
    def full_golden(cls) -> "Selection":
        return cls(SelectionKind.FULL_GOLDEN, explicit_full=True)


@dataclass(frozen=True)
class RegressionPlan:
    schema: str
    schema_version: int
    dataset_version: str
    corpus_version: str
    selection: Selection
    case_ids: tuple[str, ...]
    expression_ids: tuple[str, ...]
    fingerprint: str

    def __post_init__(self) -> None:
        if self.schema != PLAN_SCHEMA or type(self.schema_version) is not int or self.schema_version != PLAN_SCHEMA_VERSION:
            raise RunnerValidationError("unknown plan schema/version")
        _identity(self.dataset_version, "dataset_version")
        _identity(self.corpus_version, "corpus_version")
        if not isinstance(self.selection, Selection):
            raise RunnerValidationError("plan selection is invalid")
        _ids(self.case_ids, "plan.case_ids", sorted_required=True)
        _ids(self.expression_ids, "plan.expression_ids", sorted_required=True)
        if not self.case_ids or not self.expression_ids:
            raise RunnerValidationError("plan must not be empty")
        _fingerprint(self.fingerprint, "plan.fingerprint")
        if self.fingerprint != _plan_fingerprint(
            self.dataset_version, self.corpus_version, self.selection,
            self.case_ids, self.expression_ids,
        ):
            raise RunnerValidationError("plan fingerprint mismatch")

    def to_dict(self) -> dict[str, Any]:
        return _json_value(self)

    def to_json(self) -> str:
        return _canonical(self)

    @classmethod
    def from_json(cls, payload: str) -> "RegressionPlan":
        return _plan_from(_strict_json(payload))


@dataclass(frozen=True)
class BackendIdentity:
    system_id: str
    backend_version: str
    config_fingerprint: str

    def __post_init__(self) -> None:
        _identity(self.system_id, "system_id")
        if len(self.system_id) > 100:
            raise RunnerValidationError("system_id is too long for bound metadata")
        _identity(self.backend_version, "backend_version")
        _fingerprint(self.config_fingerprint, "config_fingerprint")

    @property
    def bound_system_id(self) -> str:
        digest = _sha256(_canonical(self))[:16]
        return f"{self.system_id}:{digest}"


@dataclass(frozen=True)
class RunnerTurn:
    turn_id: str
    text: str
    relation: str

    def __post_init__(self) -> None:
        _identity(self.turn_id, "turn_id")
        _text(self.text, "turn.text")
        if self.relation not in {"NEW", "CONTINUE", "UPDATE", "AMBIGUOUS"}:
            raise RunnerValidationError("turn relation is invalid")


@dataclass(frozen=True)
class CorpusEvidenceView:
    evidence_id: str
    original_content: str
    source_name: str
    source_url: str | None
    source_section: str | None
    source_page: int | None
    source_sheet: str | None
    document_id: str
    parser_version: str
    chunker_version: str
    processing_version: str
    status: str
    permission: str
    reliable: bool

    def __post_init__(self) -> None:
        _identity(self.evidence_id, "evidence_id")
        _text(self.original_content, "original_content")
        _text(self.source_name, "source_name")
        for name in ("source_url", "source_section", "source_sheet"):
            value = getattr(self, name)
            if value is not None:
                _text(value, name)
        if self.source_page is not None and (type(self.source_page) is not int or self.source_page < 1):
            raise RunnerValidationError("source_page is invalid")
        for name in ("document_id", "parser_version", "chunker_version", "processing_version"):
            _identity(getattr(self, name), name)
        if self.status not in {"active", "inactive"} or self.permission not in {"public", "restricted"}:
            raise RunnerValidationError("catalog status/permission is invalid")
        if type(self.reliable) is not bool:
            raise RunnerValidationError("catalog reliable must be boolean")


@dataclass(frozen=True)
class BackendInput:
    expression_id: str
    query: str
    turns: tuple[RunnerTurn, ...]
    synthetic_catalog: tuple[CorpusEvidenceView, ...]

    def __post_init__(self) -> None:
        _identity(self.expression_id, "expression_id")
        _text(self.query, "query")
        if type(self.turns) is not tuple or not self.turns or any(not isinstance(item, RunnerTurn) for item in self.turns):
            raise RunnerValidationError("turns must be a non-empty frozen tuple")
        if self.query != self.turns[-1].text:
            raise RunnerValidationError("query must equal final turn")
        if type(self.synthetic_catalog) is not tuple or any(not isinstance(item, CorpusEvidenceView) for item in self.synthetic_catalog):
            raise RunnerValidationError("synthetic_catalog must be a frozen tuple")
        evidence_ids = tuple(item.evidence_id for item in self.synthetic_catalog)
        if evidence_ids != tuple(sorted(evidence_ids)) or len(set(evidence_ids)) != len(evidence_ids):
            raise RunnerValidationError("synthetic catalog is not stable")


Backend = Callable[[BackendInput], Prediction | None]


@dataclass(frozen=True)
class ExecutionFailure:
    schema: str
    schema_version: int
    expression_id: str
    kind: FailureKind
    error_type: str
    message: str

    def __post_init__(self) -> None:
        if self.schema != FAILURE_SCHEMA or type(self.schema_version) is not int or self.schema_version != FAILURE_SCHEMA_VERSION:
            raise RunnerValidationError("unknown failure schema/version")
        _identity(self.expression_id, "failure.expression_id")
        if not isinstance(self.kind, FailureKind):
            raise RunnerValidationError("failure kind is invalid")
        _identity(self.error_type, "failure.error_type")
        _text(self.message, "failure.message")


@dataclass(frozen=True)
class RunManifest:
    run_id: str
    plan_fingerprint: str
    backend: BackendIdentity
    selected_count: int
    prediction_count: int
    failure_count: int

    def __post_init__(self) -> None:
        _identity(self.run_id, "run_id")
        _fingerprint(self.plan_fingerprint, "plan_fingerprint")
        if not isinstance(self.backend, BackendIdentity):
            raise RunnerValidationError("manifest backend is invalid")
        for name in ("selected_count", "prediction_count", "failure_count"):
            _count(getattr(self, name), name)
        if self.prediction_count + self.failure_count != self.selected_count:
            raise RunnerValidationError("manifest counts do not close")


@dataclass(frozen=True)
class RunArtifact:
    schema: str
    schema_version: int
    plan: RegressionPlan
    manifest: RunManifest
    predictions: tuple[Prediction, ...]
    failures: tuple[ExecutionFailure, ...]
    report: EvaluationReport

    def __post_init__(self) -> None:
        if self.schema != RUN_SCHEMA or type(self.schema_version) is not int or self.schema_version != RUN_SCHEMA_VERSION:
            raise RunnerValidationError("unknown run schema/version")
        if not isinstance(self.plan, RegressionPlan) or not isinstance(self.manifest, RunManifest) or not isinstance(self.report, EvaluationReport):
            raise RunnerValidationError("run artifact model is invalid")
        if type(self.predictions) is not tuple or type(self.failures) is not tuple:
            raise RunnerValidationError("run result collections must be tuples")
        prediction_ids = tuple(item.expression_id for item in self.predictions)
        failure_ids = tuple(item.expression_id for item in self.failures)
        if prediction_ids != tuple(sorted(prediction_ids)) or failure_ids != tuple(sorted(failure_ids)):
            raise RunnerValidationError("run results are not stably sorted")
        if len(set(prediction_ids)) != len(prediction_ids) or len(set(failure_ids)) != len(failure_ids) or set(prediction_ids) & set(failure_ids):
            raise RunnerValidationError("run result expression identity conflict")
        if set(prediction_ids) | set(failure_ids) != set(self.plan.expression_ids):
            raise RunnerValidationError("run results do not close over plan")
        if self.manifest.plan_fingerprint != self.plan.fingerprint or self.manifest.selected_count != len(self.plan.expression_ids) or self.manifest.prediction_count != len(self.predictions) or self.manifest.failure_count != len(self.failures):
            raise RunnerValidationError("run manifest mismatch")
        metadata = self.report.metadata
        if metadata.run_id != self.manifest.run_id or metadata.system_id != self.manifest.backend.bound_system_id or metadata.dataset_version != self.plan.dataset_version or metadata.corpus_version != self.plan.corpus_version or metadata.selected_expression_ids != self.plan.expression_ids:
            raise RunnerValidationError("run report identity mismatch")

    def to_dict(self) -> dict[str, Any]:
        return _json_value(self)

    def to_json(self) -> str:
        return _canonical(self)

    @classmethod
    def from_json(cls, payload: str) -> "RunArtifact":
        value = _strict_json(payload)
        return _run_artifact_from(value)


@dataclass(frozen=True)
class PairBundle:
    schema: str
    schema_version: int
    plan: RegressionPlan
    baseline: RunArtifact
    current: RunArtifact
    comparison: EvaluationComparison

    def __post_init__(self) -> None:
        if self.schema != PAIR_SCHEMA or type(self.schema_version) is not int or self.schema_version != PAIR_SCHEMA_VERSION:
            raise RunnerValidationError("unknown pair schema/version")
        if self.baseline.plan != self.plan or self.current.plan != self.plan:
            raise RunnerValidationError("pair runs do not share the exact plan")
        expected = compare_reports(self.baseline.report, self.current.report)
        if self.comparison != expected:
            raise RunnerValidationError("pair comparison does not match reports")

    def to_dict(self) -> dict[str, Any]:
        return _json_value(self)

    def to_json(self) -> str:
        return _canonical(self)

    @classmethod
    def from_json(cls, payload: str) -> "PairBundle":
        value = _strict_json(payload)
        _keys(value, {"schema", "schema_version", "plan", "baseline", "current", "comparison"}, "pair")
        if value["schema"] != PAIR_SCHEMA or type(value["schema_version"]) is not int or value["schema_version"] != PAIR_SCHEMA_VERSION:
            raise RunnerValidationError("unknown pair schema/version")
        return cls(
            PAIR_SCHEMA,
            PAIR_SCHEMA_VERSION,
            _plan_from(value["plan"]),
            _run_artifact_from(value["baseline"]),
            _run_artifact_from(value["current"]),
            EvaluationComparison.from_json(_canonical(value["comparison"])),
        )


def fingerprint_config(config: Any) -> str:
    """Fingerprint a JSON-compatible frozen/config value deterministically."""

    try:
        payload = json.dumps(config, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)
    except (TypeError, ValueError) as error:
        raise RunnerValidationError("backend config must be finite JSON data") from error
    return _sha256(payload)


def build_plan(dataset: GoldenDataset, selection: Selection) -> RegressionPlan:
    if not isinstance(dataset, GoldenDataset) or not isinstance(selection, Selection):
        raise TypeError("dataset and selection must be GoldenDataset and Selection")
    by_id = {case.case_id: case for case in dataset.cases}
    if selection.kind is SelectionKind.CASE_IDS:
        unknown = set(selection.case_ids) - set(by_id)
        if unknown:
            raise RunnerValidationError(f"unknown case IDs: {sorted(unknown)}")
        chosen = tuple(by_id[item] for item in selection.case_ids)
    elif selection.kind is SelectionKind.TAGS:
        known_tags = {tag for case in dataset.cases for tag in case.tags}
        unknown = set(selection.tags) - known_tags
        if unknown:
            raise RunnerValidationError(f"unknown tags: {sorted(unknown)}")
        selected_tags = set(selection.tags)
        chosen = tuple(
            case for case in dataset.cases
            if (
                bool(selected_tags & set(case.tags))
                if selection.tag_match is TagMatch.ANY
                else selected_tags <= set(case.tags)
            )
        )
    else:
        chosen = dataset.cases
    if not chosen:
        raise RunnerValidationError("selection matched no cases")
    case_ids = tuple(sorted(case.case_id for case in chosen))
    expression_ids = tuple(sorted(expression.expression_id for case in chosen for expression in case.expressions))
    fingerprint = _plan_fingerprint(dataset.dataset_version, dataset.corpus_version, selection, case_ids, expression_ids)
    return RegressionPlan(PLAN_SCHEMA, PLAN_SCHEMA_VERSION, dataset.dataset_version, dataset.corpus_version, selection, case_ids, expression_ids, fingerprint)


def _validate_plan(dataset: GoldenDataset, plan: RegressionPlan) -> None:
    if not isinstance(dataset, GoldenDataset) or not isinstance(plan, RegressionPlan):
        raise TypeError("dataset and plan must be GoldenDataset and RegressionPlan")
    if plan.dataset_version != dataset.dataset_version or plan.corpus_version != dataset.corpus_version or build_plan(dataset, plan.selection) != plan:
        raise RunnerValidationError("plan is incompatible with dataset")


def _catalog(dataset: GoldenDataset) -> tuple[CorpusEvidenceView, ...]:
    result = []
    for item in dataset.evidence_catalog:
        evidence, source, lineage = item.evidence, item.evidence.source_reference, item.evidence.lineage
        result.append(CorpusEvidenceView(evidence.evidence_id, evidence.original_content, source.name, source.url, source.section, source.page, source.sheet, lineage.document_id, lineage.parser_version, lineage.chunker_version, lineage.processing_version, item.status, item.permission, item.reliable))
    return tuple(sorted(result, key=lambda item: item.evidence_id))


def _inputs(dataset: GoldenDataset, plan: RegressionPlan) -> tuple[BackendInput, ...]:
    catalog = _catalog(dataset)
    expressions = {
        expression.expression_id: expression
        for case in dataset.cases for expression in case.expressions
    }
    return tuple(
        BackendInput(expression_id, expressions[expression_id].query, tuple(RunnerTurn(turn.turn_id, turn.text, turn.relation) for turn in expressions[expression_id].turns), catalog)
        for expression_id in plan.expression_ids
    )


def execute_plan(
    dataset: GoldenDataset,
    plan: RegressionPlan,
    backend_identity: BackendIdentity,
    backend: Backend,
    *,
    run_id: str,
) -> RunArtifact:
    """Call the supplied backend exactly once per selected expression."""

    _validate_plan(dataset, plan)
    if not isinstance(backend_identity, BackendIdentity) or not callable(backend):
        raise TypeError("backend_identity and callable backend are required")
    _identity(run_id, "run_id")
    predictions: list[Prediction] = []
    failures: list[ExecutionFailure] = []
    for backend_input in _inputs(dataset, plan):
        try:
            value = backend(backend_input)
        except Exception as error:  # backend isolation is the runner boundary
            failures.append(ExecutionFailure(FAILURE_SCHEMA, FAILURE_SCHEMA_VERSION, backend_input.expression_id, FailureKind.BACKEND_EXCEPTION, "BackendException", _exception_message(error)))
            continue
        if value is None:
            failures.append(ExecutionFailure(FAILURE_SCHEMA, FAILURE_SCHEMA_VERSION, backend_input.expression_id, FailureKind.NO_RETURN, "NoneReturn", "backend returned no Prediction"))
        elif not isinstance(value, Prediction):
            failures.append(ExecutionFailure(FAILURE_SCHEMA, FAILURE_SCHEMA_VERSION, backend_input.expression_id, FailureKind.INVALID_RETURN, "InvalidReturn", f"backend returned non-Prediction type {type(value).__name__}"))
        elif value.expression_id != backend_input.expression_id:
            failures.append(ExecutionFailure(FAILURE_SCHEMA, FAILURE_SCHEMA_VERSION, backend_input.expression_id, FailureKind.EXPRESSION_MISMATCH, "PredictionExpressionMismatch", f"backend returned Prediction for {value.expression_id}"))
        else:
            predictions.append(value)
    predictions_tuple = tuple(sorted(predictions, key=lambda item: item.expression_id))
    failures_tuple = tuple(sorted(failures, key=lambda item: item.expression_id))
    metadata = RunMetadata(run_id, backend_identity.bound_system_id, dataset.dataset_version, dataset.corpus_version, plan.expression_ids)
    report = evaluate_predictions(dataset, metadata, predictions_tuple)
    manifest = RunManifest(run_id, plan.fingerprint, backend_identity, len(plan.expression_ids), len(predictions_tuple), len(failures_tuple))
    return RunArtifact(RUN_SCHEMA, RUN_SCHEMA_VERSION, plan, manifest, predictions_tuple, failures_tuple, report)


def execute_pair(
    dataset: GoldenDataset,
    plan: RegressionPlan,
    baseline_identity: BackendIdentity,
    baseline_backend: Backend,
    current_identity: BackendIdentity,
    current_backend: Backend,
    *,
    baseline_run_id: str,
    current_run_id: str,
) -> PairBundle:
    """Run two independent backends against the same immutable plan."""

    if baseline_run_id == current_run_id:
        raise RunnerValidationError("baseline and current run IDs must differ")
    baseline = execute_plan(dataset, plan, baseline_identity, baseline_backend, run_id=baseline_run_id)
    current = execute_plan(dataset, plan, current_identity, current_backend, run_id=current_run_id)
    comparison = compare_reports(baseline.report, current.report)
    return PairBundle(PAIR_SCHEMA, PAIR_SCHEMA_VERSION, plan, baseline, current, comparison)


def _plan_fingerprint(dataset_version, corpus_version, selection, case_ids, expression_ids) -> str:
    payload = {
        "dataset_version": dataset_version,
        "corpus_version": corpus_version,
        "selection": _json_value(selection),
        "case_ids": list(case_ids),
        "expression_ids": list(expression_ids),
    }
    return _sha256(_canonical(payload))


def _sha256(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _exception_message(error: Exception) -> str:
    try:
        detail = str(error).strip()
    except Exception:
        detail = "message unavailable"
    name = f"{type(error).__module__}.{type(error).__qualname__}"
    return f"{name}: {detail or 'backend raised without message'}"


def _canonical(value: Any) -> str:
    return json.dumps(_json_value(value), ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _json_value(value: Any) -> Any:
    if isinstance(value, Enum): return value.value
    if is_dataclass(value): return {field.name: _json_value(getattr(value, field.name)) for field in fields(value)}
    if isinstance(value, dict): return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)): return [_json_value(item) for item in value]
    return value


def _text(value: object, path: str) -> str:
    if type(value) is not str or not value or value != value.strip():
        raise RunnerValidationError(f"{path} must be non-empty trimmed text")
    return value


def _identity(value: object, path: str) -> str:
    result = _text(value, path)
    if not _ID_RE.fullmatch(result): raise RunnerValidationError(f"{path} must be a stable identifier")
    return result


def _ids(value: object, path: str, *, sorted_required: bool) -> None:
    if type(value) is not tuple: raise RunnerValidationError(f"{path} must be a tuple")
    checked = tuple(_identity(item, path) for item in value)
    if len(set(checked)) != len(checked): raise RunnerValidationError(f"{path} contains duplicates")
    if sorted_required and checked != tuple(sorted(checked)): raise RunnerValidationError(f"{path} must be sorted")


def _fingerprint(value: object, path: str) -> str:
    result = _text(value, path)
    if not _FINGERPRINT_RE.fullmatch(result): raise RunnerValidationError(f"{path} must be a SHA-256 fingerprint")
    return result


def _count(value: object, path: str) -> None:
    if type(value) is not int or value < 0: raise RunnerValidationError(f"{path} must be a non-negative integer")


def _strict_json(payload: str) -> dict[str, Any]:
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result: raise RunnerValidationError(f"duplicate JSON key: {key}")
            result[key] = value
        return result
    try:
        value = json.loads(payload, object_pairs_hook=pairs, parse_constant=lambda item: (_ for _ in ()).throw(RunnerValidationError(f"invalid constant: {item}")))
    except RunnerValidationError: raise
    except (TypeError, ValueError, json.JSONDecodeError) as error: raise RunnerValidationError("invalid artifact JSON") from error
    if type(value) is not dict: raise RunnerValidationError("artifact JSON must be an object")
    return value


def _keys(value: object, expected: set[str], path: str) -> dict[str, Any]:
    if type(value) is not dict or set(value) != expected: raise RunnerValidationError(f"{path} fields do not match schema")
    return value


def _array(value: object, path: str) -> list[Any]:
    if type(value) is not list: raise RunnerValidationError(f"{path} must be an array")
    return value


def _enum(enum_type, value, path):
    try: return enum_type(value)
    except (TypeError, ValueError) as error: raise RunnerValidationError(f"{path} has unknown value") from error


def _selection_from(value):
    item = _keys(value, {"kind", "case_ids", "tags", "tag_match", "explicit_full"}, "selection")
    match = None if item["tag_match"] is None else _enum(TagMatch, item["tag_match"], "tag_match")
    return Selection(_enum(SelectionKind, item["kind"], "selection.kind"), tuple(_array(item["case_ids"], "case_ids")), tuple(_array(item["tags"], "tags")), match, item["explicit_full"])


def _plan_from(value):
    item = _keys(value, {"schema", "schema_version", "dataset_version", "corpus_version", "selection", "case_ids", "expression_ids", "fingerprint"}, "plan")
    return RegressionPlan(item["schema"], item["schema_version"], item["dataset_version"], item["corpus_version"], _selection_from(item["selection"]), tuple(_array(item["case_ids"], "case_ids")), tuple(_array(item["expression_ids"], "expression_ids")), item["fingerprint"])


def _backend_from(value):
    item = _keys(value, {"system_id", "backend_version", "config_fingerprint"}, "backend")
    return BackendIdentity(**item)


def _manifest_from(value):
    item = _keys(value, {"run_id", "plan_fingerprint", "backend", "selected_count", "prediction_count", "failure_count"}, "manifest")
    return RunManifest(item["run_id"], item["plan_fingerprint"], _backend_from(item["backend"]), item["selected_count"], item["prediction_count"], item["failure_count"])


def _prediction_from(value):
    item = _keys(value, {"expression_id", "request_type", "ranked_evidence_ids", "output_mode", "boundary", "clarification", "verification", "comparison_outcome", "missing_information", "trace_fields", "source_evidence_ids", "lineage_evidence_ids", "observed_forbidden_behaviors", "final_business_decision"}, "prediction")
    return Prediction(item["expression_id"], _enum(KnowledgeRequestType, item["request_type"], "request_type"), tuple(_array(item["ranked_evidence_ids"], "ranked_evidence_ids")), _enum(OutputMode, item["output_mode"], "output_mode"), _enum(BoundaryTruth, item["boundary"], "boundary"), _enum(ClarificationTruth, item["clarification"], "clarification"), _enum(VerificationTruth, item["verification"], "verification"), item["comparison_outcome"], tuple(_array(item["missing_information"], "missing_information")), tuple(_array(item["trace_fields"], "trace_fields")), tuple(_array(item["source_evidence_ids"], "source_evidence_ids")), tuple(_array(item["lineage_evidence_ids"], "lineage_evidence_ids")), tuple(_enum(ForbiddenBehavior, behavior, "forbidden_behavior") for behavior in _array(item["observed_forbidden_behaviors"], "observed_forbidden_behaviors")), item["final_business_decision"])


def _failure_from(value):
    item = _keys(value, {"schema", "schema_version", "expression_id", "kind", "error_type", "message"}, "failure")
    return ExecutionFailure(item["schema"], item["schema_version"], item["expression_id"], _enum(FailureKind, item["kind"], "failure.kind"), item["error_type"], item["message"])


def _run_artifact_from(value):
    item = _keys(value, {"schema", "schema_version", "plan", "manifest", "predictions", "failures", "report"}, "run")
    if item["schema"] != RUN_SCHEMA or type(item["schema_version"]) is not int or item["schema_version"] != RUN_SCHEMA_VERSION: raise RunnerValidationError("unknown run schema/version")
    return RunArtifact(RUN_SCHEMA, RUN_SCHEMA_VERSION, _plan_from(item["plan"]), _manifest_from(item["manifest"]), tuple(_prediction_from(entry) for entry in _array(item["predictions"], "predictions")), tuple(_failure_from(entry) for entry in _array(item["failures"], "failures")), EvaluationReport.from_json(_canonical(item["report"])))
