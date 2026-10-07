"""Strict loader for the synthetic, behavior-oriented Golden Dataset.

Loading validates frozen evaluation truth only.  It performs no retrieval,
scoring, reporting, model call, network access, or test execution.
"""

from __future__ import annotations

import json
import os
import re
import stat
import unicodedata
from collections import Counter
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from types import MappingProxyType
from typing import Any
from urllib.parse import urlsplit

from .contracts import Evidence, KnowledgeLineage, SourceReference
from .request_classification import KnowledgeRequestType


GOLDEN_SCHEMA = "knowledge-system.synthetic-golden-dataset"
GOLDEN_SCHEMA_VERSION = 1
MIN_CASES = 20
MAX_CASES = 30
MIN_EXPRESSIONS_PER_CASE = 3
MIN_CASES_PER_REQUEST_TYPE = 4


class GoldenDatasetValidationError(ValueError):
    """The evaluation fixture violates its closed schema or product boundary."""


class OutputMode(str, Enum):
    EVIDENCE = "EVIDENCE"
    CLARIFICATION = "CLARIFICATION"
    VERIFICATION = "VERIFICATION"
    OUT_OF_SCOPE = "OUT_OF_SCOPE"


class BoundaryTruth(str, Enum):
    NONE = "NONE"
    CASE_INFORMATION_MISSING = "CASE_INFORMATION_MISSING"
    KNOWLEDGE_MISSING = "KNOWLEDGE_MISSING"
    KNOWLEDGE_AMBIGUOUS = "KNOWLEDGE_AMBIGUOUS"
    SOURCE_FAILURE = "SOURCE_FAILURE"
    RETRIEVAL_FAILURE = "RETRIEVAL_FAILURE"
    PROCESSING_FAILURE = "PROCESSING_FAILURE"
    RECOVERY_FAILURE = "RECOVERY_FAILURE"
    OUT_OF_SCOPE = "OUT_OF_SCOPE"
    QUERY_INSUFFICIENT = "QUERY_INSUFFICIENT"


class ClarificationTruth(str, Enum):
    NONE = "NONE"
    REQUIRED = "REQUIRED"
    ALLOWED_AFTER_EVIDENCE = "ALLOWED_AFTER_EVIDENCE"


class VerificationTruth(str, Enum):
    FORBIDDEN = "FORBIDDEN"
    REQUIRED = "REQUIRED"


class ForbiddenBehavior(str, Enum):
    FIXED_ANSWER_MATCH = "FIXED_ANSWER_MATCH"
    INVENT_FACT = "INVENT_FACT"
    INVENT_SOURCE = "INVENT_SOURCE"
    BYPASS_PERMISSION = "BYPASS_PERMISSION"
    RETURN_INACTIVE = "RETURN_INACTIVE"
    NOT_FOUND_MEANS_NOT_EXIST = "NOT_FOUND_MEANS_NOT_EXIST"
    CASE_MISSING_AS_KNOWLEDGE_GAP = "CASE_MISSING_AS_KNOWLEDGE_GAP"
    RETRIEVED_MEANS_ANSWERABLE = "RETRIEVED_MEANS_ANSWERABLE"
    FINAL_BUSINESS_DECISION = "FINAL_BUSINESS_DECISION"
    BROWSE_TOP1_ONLY = "BROWSE_TOP1_ONLY"
    FAILURE_CLASS_COLLAPSE = "FAILURE_CLASS_COLLAPSE"


@dataclass(frozen=True)
class GoldenTurn:
    turn_id: str
    text: str
    relation: str


@dataclass(frozen=True)
class GoldenExpression:
    expression_id: str
    query: str
    turns: tuple[GoldenTurn, ...]


@dataclass(frozen=True)
class GoldenEvidence:
    evidence: Evidence
    status: str
    permission: str
    reliable: bool
    page_is_verified: bool


@dataclass(frozen=True)
class RecallTruth:
    relevant_evidence_ids: tuple[str, ...]
    minimum_recall: float


@dataclass(frozen=True)
class RankingTruth:
    preferred_tiers: tuple[tuple[str, ...], ...]
    order_required_within_tier: bool


@dataclass(frozen=True)
class BrowseTruth:
    expected_evidence_ids: tuple[str, ...]
    minimum_coverage: float
    top1_sufficient: bool


@dataclass(frozen=True)
class ProvenanceTruth:
    require_source_reference: bool
    require_lineage: bool
    required_trace_fields: tuple[str, ...]


@dataclass(frozen=True)
class ComparisonTruth:
    acceptable_outcomes: tuple[str, ...]
    required_missing_information: tuple[str, ...]
    final_decision_forbidden: bool


@dataclass(frozen=True)
class BehaviorTruth:
    allowed_output_modes: tuple[OutputMode, ...]
    recall: RecallTruth
    ranking: RankingTruth
    browse: BrowseTruth | None
    forbidden_evidence_ids: tuple[str, ...]
    forbidden_behaviors: tuple[ForbiddenBehavior, ...]
    boundary: BoundaryTruth
    clarification: ClarificationTruth
    verification: VerificationTruth
    provenance: ProvenanceTruth
    comparison: ComparisonTruth | None


@dataclass(frozen=True)
class GoldenCase:
    case_id: str
    request_type: KnowledgeRequestType
    tags: tuple[str, ...]
    expressions: tuple[GoldenExpression, ...]
    truth: BehaviorTruth


@dataclass(frozen=True)
class GoldenDataset:
    schema: str
    schema_version: int
    dataset_version: str
    corpus_version: str
    authority: str
    fixture_notice: str
    evidence_catalog: tuple[GoldenEvidence, ...]
    cases: tuple[GoldenCase, ...]
    request_type_counts: MappingProxyType
    expression_count: int


_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,127}$")
_INTERNAL_PATH_RE = re.compile(r"^(?:[A-Za-z]:[\\/]|\\\\|//|/)")
_TRACE_FIELDS = frozenset(
    {
        "raw_query",
        "retrieval_query",
        "candidate_evidence_ids",
        "final_evidence_ids",
        "applied_filters",
        "knowledge_version",
        "source_references",
        "processing_versions",
        "retrieval_paths",
        "fusion_scores",
        "rerank_summary",
    }
)
_RELATIONS = frozenset({"NEW", "CONTINUE", "UPDATE", "AMBIGUOUS"})
_COMPARISON_OUTCOMES = frozenset(
    {"SATISFIED", "NOT_SATISFIED", "MISSING", "UNKNOWN", "HUMAN_REQUIRED"}
)
_REQUIRED_SCENARIO_TAGS = frozenset(
    {
        "normal_hit",
        "exact",
        "set",
        "fuzzy",
        "multi_turn",
        "threshold",
        "case_missing",
        "knowledge_missing",
        "ambiguity",
        "source_failure",
        "processing_failure",
        "retrieval_failure",
        "recovery_failure",
        "clarification",
        "permission",
        "inactive",
    }
)
_REQUIRED_BOUNDARIES = frozenset(
    {
        BoundaryTruth.NONE,
        BoundaryTruth.CASE_INFORMATION_MISSING,
        BoundaryTruth.KNOWLEDGE_MISSING,
        BoundaryTruth.KNOWLEDGE_AMBIGUOUS,
        BoundaryTruth.SOURCE_FAILURE,
        BoundaryTruth.RETRIEVAL_FAILURE,
        BoundaryTruth.PROCESSING_FAILURE,
        BoundaryTruth.RECOVERY_FAILURE,
        BoundaryTruth.OUT_OF_SCOPE,
        BoundaryTruth.QUERY_INSUFFICIENT,
    }
)


def _object(value: object, fields: frozenset[str], path: str) -> dict[str, Any]:
    if type(value) is not dict or set(value) != fields:
        raise GoldenDatasetValidationError(f"{path} fields do not match schema")
    return value


def _array(value: object, path: str) -> list[Any]:
    if type(value) is not list:
        raise GoldenDatasetValidationError(f"{path} must be an array")
    return value


def _text(value: object, path: str) -> str:
    if type(value) is not str or not value.strip() or value != value.strip():
        raise GoldenDatasetValidationError(f"{path} must be non-empty trimmed text")
    return value


def _identity(value: object, path: str) -> str:
    result = _text(value, path)
    if not _ID_RE.fullmatch(result):
        raise GoldenDatasetValidationError(f"{path} must be a stable identifier")
    return result


def _bool(value: object, path: str) -> bool:
    if type(value) is not bool:
        raise GoldenDatasetValidationError(f"{path} must be boolean")
    return value


def _ratio(value: object, path: str, *, allow_zero: bool = True) -> float:
    if type(value) not in {int, float}:
        raise GoldenDatasetValidationError(f"{path} must be numeric")
    result = float(value)
    minimum = 0.0 if allow_zero else 0.0000001
    if not minimum <= result <= 1.0:
        raise GoldenDatasetValidationError(f"{path} must be within [0, 1]")
    return result


def _unique_texts(value: object, path: str, *, identities: bool = False) -> tuple[str, ...]:
    items = _array(value, path)
    checked = tuple(
        (_identity(item, f"{path}[{index}]") if identities else _text(item, f"{path}[{index}]"))
        for index, item in enumerate(items)
    )
    if len(set(checked)) != len(checked):
        raise GoldenDatasetValidationError(f"{path} contains duplicates")
    return checked


def _enum(enum_type: type[Enum], value: object, path: str) -> Enum:
    try:
        return enum_type(value)
    except (TypeError, ValueError) as error:
        raise GoldenDatasetValidationError(f"{path} has unknown value") from error


def _source(value: object, page_is_verified: bool, path: str) -> SourceReference:
    item = _object(
        value,
        frozenset({"name", "url", "section", "page", "sheet"}),
        path,
    )
    name = _text(item["name"], f"{path}.name")
    if _INTERNAL_PATH_RE.match(name) or "/" in name or "\\" in name:
        raise GoldenDatasetValidationError(f"{path}.name cannot be an internal path")
    url = item["url"]
    if url is not None:
        url = _text(url, f"{path}.url")
        parsed = urlsplit(url)
        if parsed.scheme.casefold() not in {"http", "https"} or not parsed.netloc:
            raise GoldenDatasetValidationError(f"{path}.url must be a stable HTTP(S) URL")
    section = item["section"]
    if section is not None:
        section = _text(section, f"{path}.section")
    sheet = item["sheet"]
    if sheet is not None:
        sheet = _text(sheet, f"{path}.sheet")
    page = item["page"]
    if page is not None and (type(page) is not int or page < 1):
        raise GoldenDatasetValidationError(f"{path}.page must be a positive integer")
    # This self-contained corpus has no paginated source artifact.  A boolean
    # assertion inside the same JSON cannot independently prove a page number,
    # so v1 rejects pages rather than accepting a self-attested locator.
    if page is not None or page_is_verified:
        raise GoldenDatasetValidationError(
            f"{path}.page cannot be verified by this synthetic corpus"
        )
    return SourceReference(name, url, section, page, sheet)


def _lineage(value: object, path: str) -> KnowledgeLineage:
    item = _object(
        value,
        frozenset(
            {
                "document_id",
                "parser_version",
                "chunker_version",
                "processing_version",
                "source_position",
                "parent_section_id",
            }
        ),
        path,
    )
    required = {
        name: _identity(item[name], f"{path}.{name}")
        for name in ("document_id", "parser_version", "chunker_version", "processing_version")
    }
    optional: dict[str, str | None] = {}
    for name in ("source_position", "parent_section_id"):
        optional[name] = None if item[name] is None else _text(item[name], f"{path}.{name}")
        if optional[name] is not None and _INTERNAL_PATH_RE.match(optional[name]):
            raise GoldenDatasetValidationError(f"{path}.{name} cannot be an internal path")
    return KnowledgeLineage(**required, **optional)


def _catalog(value: object) -> tuple[GoldenEvidence, ...]:
    result: list[GoldenEvidence] = []
    ids: set[str] = set()
    unit_ids: set[str] = set()
    for index, raw in enumerate(_array(value, "evidence_catalog")):
        path = f"evidence_catalog[{index}]"
        item = _object(
            raw,
            frozenset(
                {
                    "evidence_id",
                    "unit_id",
                    "original_content",
                    "source_reference",
                    "lineage",
                    "status",
                    "permission",
                    "reliable",
                    "page_is_verified",
                }
            ),
            path,
        )
        evidence_id = _identity(item["evidence_id"], f"{path}.evidence_id")
        if evidence_id in ids:
            raise GoldenDatasetValidationError("duplicate evidence_id")
        ids.add(evidence_id)
        page_verified = _bool(item["page_is_verified"], f"{path}.page_is_verified")
        status = _text(item["status"], f"{path}.status")
        permission = _text(item["permission"], f"{path}.permission")
        if status not in {"active", "inactive"}:
            raise GoldenDatasetValidationError(f"{path}.status is invalid")
        if permission not in {"public", "restricted"}:
            raise GoldenDatasetValidationError(f"{path}.permission is invalid")
        original_content = _text(item["original_content"], f"{path}.original_content")
        if not original_content.startswith("合成"):
            raise GoldenDatasetValidationError(
                f"{path}.original_content must identify synthetic policy content"
            )
        unit_id = _identity(item["unit_id"], f"{path}.unit_id")
        if unit_id in unit_ids:
            raise GoldenDatasetValidationError("duplicate unit_id")
        unit_ids.add(unit_id)
        result.append(
            GoldenEvidence(
                Evidence(
                    evidence_id,
                    unit_id,
                    original_content,
                    _source(item["source_reference"], page_verified, f"{path}.source_reference"),
                    _lineage(item["lineage"], f"{path}.lineage"),
                ),
                status,
                permission,
                _bool(item["reliable"], f"{path}.reliable"),
                page_verified,
            )
        )
    if not result:
        raise GoldenDatasetValidationError("evidence_catalog must not be empty")
    return tuple(sorted(result, key=lambda entry: entry.evidence.evidence_id))


def _expressions(value: object, path: str) -> tuple[GoldenExpression, ...]:
    result: list[GoldenExpression] = []
    for index, raw in enumerate(_array(value, path)):
        item_path = f"{path}[{index}]"
        item = _object(raw, frozenset({"expression_id", "query", "turns"}), item_path)
        turns: list[GoldenTurn] = []
        for turn_index, raw_turn in enumerate(_array(item["turns"], f"{item_path}.turns")):
            turn_path = f"{item_path}.turns[{turn_index}]"
            turn = _object(raw_turn, frozenset({"turn_id", "text", "relation"}), turn_path)
            relation = _text(turn["relation"], f"{turn_path}.relation")
            if relation not in _RELATIONS:
                raise GoldenDatasetValidationError(f"{turn_path}.relation is invalid")
            turns.append(GoldenTurn(_identity(turn["turn_id"], f"{turn_path}.turn_id"), _text(turn["text"], f"{turn_path}.text"), relation))
        if not turns:
            raise GoldenDatasetValidationError(f"{item_path}.turns must not be empty")
        result.append(
            GoldenExpression(
                _identity(item["expression_id"], f"{item_path}.expression_id"),
                _text(item["query"], f"{item_path}.query"),
                tuple(turns),
            )
        )
    if len(result) < MIN_EXPRESSIONS_PER_CASE:
        raise GoldenDatasetValidationError(f"{path} requires at least three expressions")
    return tuple(sorted(result, key=lambda expression: expression.expression_id))


def _normalized_expression(value: str) -> str:
    """Canonical duplicate key without changing the preserved user query."""

    return " ".join(unicodedata.normalize("NFKC", value).casefold().split())


def _truth(value: object, request_type: KnowledgeRequestType, catalog_ids: frozenset[str], path: str) -> BehaviorTruth:
    item = _object(
        value,
        frozenset(
            {
                "allowed_output_modes", "recall", "ranking", "browse",
                "forbidden_evidence_ids", "forbidden_behaviors", "boundary",
                "clarification", "verification", "provenance", "comparison",
            }
        ),
        path,
    )
    modes = tuple(_enum(OutputMode, raw, f"{path}.allowed_output_modes") for raw in _array(item["allowed_output_modes"], f"{path}.allowed_output_modes"))
    if not modes or len(set(modes)) != len(modes):
        raise GoldenDatasetValidationError(f"{path}.allowed_output_modes is empty or duplicated")

    recall_item = _object(item["recall"], frozenset({"relevant_evidence_ids", "minimum_recall"}), f"{path}.recall")
    relevant = _unique_texts(recall_item["relevant_evidence_ids"], f"{path}.recall.relevant_evidence_ids", identities=True)
    recall = RecallTruth(tuple(sorted(relevant)), _ratio(recall_item["minimum_recall"], f"{path}.recall.minimum_recall"))
    if (relevant and recall.minimum_recall == 0.0) or (
        not relevant and recall.minimum_recall != 0.0
    ):
        raise GoldenDatasetValidationError(
            f"{path}.recall minimum is inconsistent with relevant Evidence"
        )

    ranking_item = _object(item["ranking"], frozenset({"preferred_tiers", "order_required_within_tier"}), f"{path}.ranking")
    tiers = tuple(
        tuple(sorted(_unique_texts(tier, f"{path}.ranking.preferred_tiers[{index}]", identities=True)))
        for index, tier in enumerate(_array(ranking_item["preferred_tiers"], f"{path}.ranking.preferred_tiers"))
    )
    ranked = tuple(evidence_id for tier in tiers for evidence_id in tier)
    if len(set(ranked)) != len(ranked):
        raise GoldenDatasetValidationError(f"{path}.ranking contains duplicate Evidence")
    ranking = RankingTruth(tiers, _bool(ranking_item["order_required_within_tier"], f"{path}.ranking.order_required_within_tier"))

    forbidden_ids = tuple(sorted(_unique_texts(item["forbidden_evidence_ids"], f"{path}.forbidden_evidence_ids", identities=True)))
    behaviors = tuple(_enum(ForbiddenBehavior, raw, f"{path}.forbidden_behaviors") for raw in _array(item["forbidden_behaviors"], f"{path}.forbidden_behaviors"))
    if not behaviors or len(set(behaviors)) != len(behaviors):
        raise GoldenDatasetValidationError(f"{path}.forbidden_behaviors is empty or duplicated")
    boundary = _enum(BoundaryTruth, item["boundary"], f"{path}.boundary")
    clarification = _enum(ClarificationTruth, item["clarification"], f"{path}.clarification")
    verification = _enum(VerificationTruth, item["verification"], f"{path}.verification")

    provenance_item = _object(item["provenance"], frozenset({"require_source_reference", "require_lineage", "required_trace_fields"}), f"{path}.provenance")
    trace_fields = tuple(sorted(_unique_texts(provenance_item["required_trace_fields"], f"{path}.provenance.required_trace_fields")))
    if not set(trace_fields) <= _TRACE_FIELDS or not {"raw_query", "retrieval_query", "candidate_evidence_ids", "final_evidence_ids", "knowledge_version"} <= set(trace_fields):
        raise GoldenDatasetValidationError(f"{path}.provenance has invalid or incomplete trace requirements")
    provenance = ProvenanceTruth(
        _bool(provenance_item["require_source_reference"], f"{path}.provenance.require_source_reference"),
        _bool(provenance_item["require_lineage"], f"{path}.provenance.require_lineage"),
        trace_fields,
    )
    browse: BrowseTruth | None = None
    if item["browse"] is not None:
        browse_item = _object(item["browse"], frozenset({"expected_evidence_ids", "minimum_coverage", "top1_sufficient"}), f"{path}.browse")
        expected = tuple(sorted(_unique_texts(browse_item["expected_evidence_ids"], f"{path}.browse.expected_evidence_ids", identities=True)))
        browse = BrowseTruth(expected, _ratio(browse_item["minimum_coverage"], f"{path}.browse.minimum_coverage", allow_zero=False), _bool(browse_item["top1_sufficient"], f"{path}.browse.top1_sufficient"))

    comparison: ComparisonTruth | None = None
    if item["comparison"] is not None:
        comparison_item = _object(item["comparison"], frozenset({"acceptable_outcomes", "required_missing_information", "final_decision_forbidden"}), f"{path}.comparison")
        outcomes = tuple(sorted(_unique_texts(comparison_item["acceptable_outcomes"], f"{path}.comparison.acceptable_outcomes")))
        if not set(outcomes) <= _COMPARISON_OUTCOMES or not outcomes:
            raise GoldenDatasetValidationError(f"{path}.comparison outcomes are invalid")
        comparison = ComparisonTruth(
            outcomes,
            tuple(sorted(_unique_texts(comparison_item["required_missing_information"], f"{path}.comparison.required_missing_information"))),
            _bool(comparison_item["final_decision_forbidden"], f"{path}.comparison.final_decision_forbidden"),
        )

    all_references = set(relevant) | set(ranked) | set(forbidden_ids)
    if browse is not None:
        all_references.update(browse.expected_evidence_ids)
    dangling = all_references - catalog_ids
    if dangling:
        raise GoldenDatasetValidationError(f"{path} has dangling Evidence IDs: {sorted(dangling)}")
    if set(relevant) & set(forbidden_ids):
        raise GoldenDatasetValidationError(f"{path} marks Evidence both relevant and forbidden")
    if not set(ranked) <= set(relevant):
        raise GoldenDatasetValidationError(f"{path} mixes ranking preference into non-recall Evidence")
    if request_type is KnowledgeRequestType.BROWSE:
        if (
            browse is None
            or len(browse.expected_evidence_ids) < 2
            or browse.top1_sufficient
            or set(browse.expected_evidence_ids) != set(relevant)
        ):
            raise GoldenDatasetValidationError(f"{path} Browse truth must be set coverage, never Top-1")
        if ForbiddenBehavior.BROWSE_TOP1_ONLY not in behaviors:
            raise GoldenDatasetValidationError(f"{path} Browse must forbid Top-1-only evaluation")
    elif browse is not None:
        raise GoldenDatasetValidationError(f"{path}.browse is only valid for Browse")
    if request_type is KnowledgeRequestType.APPLY:
        if comparison is None or not comparison.final_decision_forbidden or ForbiddenBehavior.FINAL_BUSINESS_DECISION not in behaviors:
            raise GoldenDatasetValidationError(f"{path} Apply must retain human final responsibility")
    elif comparison is not None:
        raise GoldenDatasetValidationError(f"{path}.comparison is only valid for Apply")
    if request_type is KnowledgeRequestType.OUT_OF_SCOPE:
        if (
            relevant
            or ranked
            or modes != (OutputMode.OUT_OF_SCOPE,)
            or boundary is not BoundaryTruth.OUT_OF_SCOPE
        ):
            raise GoldenDatasetValidationError(f"{path} Out-of-Scope truth is inconsistent")
    if boundary is BoundaryTruth.KNOWLEDGE_MISSING and relevant:
        raise GoldenDatasetValidationError(f"{path} Knowledge Missing cannot claim relevant Evidence")
    if boundary in {
        BoundaryTruth.KNOWLEDGE_AMBIGUOUS,
        BoundaryTruth.SOURCE_FAILURE,
        BoundaryTruth.KNOWLEDGE_MISSING,
        BoundaryTruth.RETRIEVAL_FAILURE,
        BoundaryTruth.PROCESSING_FAILURE,
        BoundaryTruth.RECOVERY_FAILURE,
    } and verification is not VerificationTruth.REQUIRED:
        raise GoldenDatasetValidationError(
            f"{path} unreliable Evidence boundary requires Verification"
        )
    if boundary is BoundaryTruth.QUERY_INSUFFICIENT and clarification is not ClarificationTruth.REQUIRED:
        raise GoldenDatasetValidationError(
            f"{path} insufficient query requires clarification"
        )
    if clarification is ClarificationTruth.REQUIRED and OutputMode.CLARIFICATION not in modes:
        raise GoldenDatasetValidationError(f"{path} required clarification mode is missing")
    if verification is VerificationTruth.REQUIRED and OutputMode.VERIFICATION not in modes:
        raise GoldenDatasetValidationError(f"{path} required Verification mode is missing")
    if verification is VerificationTruth.FORBIDDEN and OutputMode.VERIFICATION in modes:
        raise GoldenDatasetValidationError(f"{path} forbids but allows Verification")
    if ForbiddenBehavior.FIXED_ANSWER_MATCH not in behaviors:
        raise GoldenDatasetValidationError(
            f"{path} must reject fixed-answer-only evaluation"
        )
    if not {
        ForbiddenBehavior.INVENT_FACT,
        ForbiddenBehavior.INVENT_SOURCE,
    } <= set(behaviors):
        raise GoldenDatasetValidationError(f"{path} must forbid invented grounding")
    if relevant and (
        not provenance.require_source_reference or not provenance.require_lineage
    ):
        raise GoldenDatasetValidationError(
            f"{path} relevant Evidence requires source and lineage"
        )
    return BehaviorTruth(modes, recall, ranking, browse, forbidden_ids, behaviors, boundary, clarification, verification, provenance, comparison)


def _cases(value: object, catalog: tuple[GoldenEvidence, ...]) -> tuple[GoldenCase, ...]:
    catalog_ids = frozenset(entry.evidence.evidence_id for entry in catalog)
    catalog_by_id = {entry.evidence.evidence_id: entry for entry in catalog}
    result: list[GoldenCase] = []
    case_ids: set[str] = set()
    expression_ids: set[str] = set()
    queries: set[str] = set()
    turn_ids: set[str] = set()
    for index, raw in enumerate(_array(value, "cases")):
        path = f"cases[{index}]"
        item = _object(raw, frozenset({"case_id", "request_type", "tags", "expressions", "truth"}), path)
        case_id = _identity(item["case_id"], f"{path}.case_id")
        if case_id in case_ids:
            raise GoldenDatasetValidationError("duplicate case_id")
        case_ids.add(case_id)
        request_type = _enum(KnowledgeRequestType, item["request_type"], f"{path}.request_type")
        tags = tuple(sorted(_unique_texts(item["tags"], f"{path}.tags", identities=True)))
        expressions = _expressions(item["expressions"], f"{path}.expressions")
        for expression in expressions:
            if expression.expression_id in expression_ids:
                raise GoldenDatasetValidationError("duplicate expression_id")
            expression_ids.add(expression.expression_id)
            query_key = _normalized_expression(expression.query)
            if query_key in queries:
                raise GoldenDatasetValidationError("duplicate expression query")
            queries.add(query_key)
            if expression.query != expression.turns[-1].text:
                raise GoldenDatasetValidationError(
                    "expression query must equal its final turn text"
                )
            for turn in expression.turns:
                if turn.turn_id in turn_ids:
                    raise GoldenDatasetValidationError("duplicate turn_id")
                turn_ids.add(turn.turn_id)
        truth = _truth(item["truth"], request_type, catalog_ids, f"{path}.truth")
        for evidence_id in truth.recall.relevant_evidence_ids:
            evidence = catalog_by_id[evidence_id]
            if evidence.status != "active" or evidence.permission != "public":
                raise GoldenDatasetValidationError(
                    f"{path} recall truth cannot bypass status or permission filters"
                )
            if not evidence.reliable and truth.verification is not VerificationTruth.REQUIRED:
                raise GoldenDatasetValidationError(
                    f"{path} unreliable Evidence requires Verification"
                )
        unavailable_ids = {
            evidence_id
            for evidence_id, evidence in catalog_by_id.items()
            if evidence.status != "active" or evidence.permission != "public"
        }
        if not unavailable_ids <= set(truth.forbidden_evidence_ids):
            raise GoldenDatasetValidationError(
                f"{path} must forbid all inaccessible or inactive Evidence"
            )
        result.append(GoldenCase(case_id, request_type, tags, expressions, truth))
    if not MIN_CASES <= len(result) <= MAX_CASES:
        raise GoldenDatasetValidationError(f"dataset must contain {MIN_CASES}-{MAX_CASES} cases")
    counts = Counter(item.request_type for item in result)
    missing = {request_type.value: counts[request_type] for request_type in KnowledgeRequestType if counts[request_type] < MIN_CASES_PER_REQUEST_TYPE}
    if missing:
        raise GoldenDatasetValidationError(f"request type coverage is insufficient: {missing}")
    tags = frozenset(tag for case in result for tag in case.tags)
    missing_tags = _REQUIRED_SCENARIO_TAGS - tags
    if missing_tags:
        raise GoldenDatasetValidationError(
            f"scenario coverage is insufficient: {sorted(missing_tags)}"
        )
    boundaries = frozenset(case.truth.boundary for case in result)
    missing_boundaries = _REQUIRED_BOUNDARIES - boundaries
    if missing_boundaries:
        raise GoldenDatasetValidationError(
            "boundary coverage is insufficient: "
            f"{sorted(item.value for item in missing_boundaries)}"
        )
    return tuple(sorted(result, key=lambda case: case.case_id))


def _strict_json(text: str) -> object:
    def pairs(items: list[tuple[str, object]]) -> dict[str, object]:
        result: dict[str, object] = {}
        for key, value in items:
            if key in result:
                raise GoldenDatasetValidationError(f"duplicate JSON key: {key}")
            result[key] = value
        return result
    try:
        return json.loads(
            text,
            object_pairs_hook=pairs,
            parse_constant=lambda value: (_ for _ in ()).throw(GoldenDatasetValidationError(f"invalid JSON constant: {value}")),
        )
    except GoldenDatasetValidationError:
        raise
    except (TypeError, ValueError, json.JSONDecodeError) as error:
        raise GoldenDatasetValidationError("invalid Golden Dataset JSON") from error


def load_golden_dataset(path: str | os.PathLike[str]) -> GoldenDataset:
    """Load one explicit local fixture into a validated, immutable model."""

    if not isinstance(path, (str, os.PathLike)):
        raise TypeError("path must be a string or path-like value")
    source = Path(path)
    try:
        mode = source.lstat().st_mode
    except OSError as error:
        raise GoldenDatasetValidationError("Golden Dataset path is not a readable file") from error
    if stat.S_ISLNK(mode) or not stat.S_ISREG(mode):
        raise GoldenDatasetValidationError("Golden Dataset path must be a regular file")
    try:
        text = source.read_bytes().decode("utf-8", errors="strict")
    except (OSError, UnicodeDecodeError) as error:
        raise GoldenDatasetValidationError("Golden Dataset must be strict UTF-8") from error
    root = _object(
        _strict_json(text),
        frozenset({"schema", "schema_version", "dataset_version", "corpus_version", "authority", "fixture_notice", "evidence_catalog", "cases"}),
        "dataset",
    )
    if root["schema"] != GOLDEN_SCHEMA:
        raise GoldenDatasetValidationError("unknown Golden Dataset schema")
    if type(root["schema_version"]) is not int or root["schema_version"] != GOLDEN_SCHEMA_VERSION:
        raise GoldenDatasetValidationError("unknown Golden Dataset schema version")
    dataset_version = _identity(root["dataset_version"], "dataset.dataset_version")
    corpus_version = _identity(root["corpus_version"], "dataset.corpus_version")
    authority = _text(root["authority"], "dataset.authority")
    notice = _text(root["fixture_notice"], "dataset.fixture_notice")
    if authority != "SYNTHETIC_EVALUATION_ONLY_NOT_FORMAL_POLICY" or "synthetic" not in notice.casefold() or "not formal" not in notice.casefold():
        raise GoldenDatasetValidationError("fixture must explicitly disclaim formal policy authority")
    catalog = _catalog(root["evidence_catalog"])
    cases = _cases(root["cases"], catalog)
    counts = Counter(case.request_type.value for case in cases)
    return GoldenDataset(
        GOLDEN_SCHEMA,
        GOLDEN_SCHEMA_VERSION,
        dataset_version,
        corpus_version,
        authority,
        notice,
        catalog,
        cases,
        MappingProxyType(dict(sorted(counts.items()))),
        sum(len(case.expressions) for case in cases),
    )
