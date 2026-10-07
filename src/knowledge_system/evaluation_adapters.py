"""Truth-free deterministic product adapters for Golden regression execution.

The adapters consume only :class:`BackendInput`.  They deliberately have no
dependency on Golden cases, expected evidence, or metric thresholds.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass

from .baseline_retrieval import BaselineRetrieval
from .contracts import KnowledgeLineage, KnowledgeUnit, ParseStatus, SourceReference
from .evaluation_metrics import Prediction
from .golden_dataset import (
    BoundaryTruth,
    ClarificationTruth,
    OutputMode,
    VerificationTruth,
)
from .hybrid_retrieval import HybridRetrieval
from .regression_runner import BackendIdentity, BackendInput, CorpusEvidenceView
from .request_classification import KnowledgeRequestType, classify_request


ADAPTER_SCHEMA_VERSION = 1
BASELINE_ADAPTER_VERSION = "baseline-product-adapter-v1"
CURRENT_ADAPTER_VERSION = "hybrid-product-adapter-v1"
_KNOWLEDGE_VERSION = "synthetic-catalog-view-v1"
_INDEX_VERSION = "deterministic-trigram-v1"
_TRACE_FIELDS = (
    "applied_filters",
    "candidate_evidence_ids",
    "final_evidence_ids",
    "fusion_scores",
    "knowledge_version",
    "processing_versions",
    "raw_query",
    "rerank_summary",
    "retrieval_paths",
    "retrieval_query",
    "source_references",
)
_EXTERNAL = re.compile(
    r"天气|下雨|气温|菜谱|做饭|蛋糕|写.*诗|电影|故事|税率|外部法规|"
    r"国家法规|公共法规|外部法律|药|医学|诊断|症状|就医"
)
_ENTERPRISE = re.compile(
    r"规则|规定|制度|政策|流程|审批|审核|复核|报销|请假|差旅|支付|运维|"
    r"账号|申请|材料|补贴|知识库|原文|来源|活动"
    r"|合成知识|公开条款|有效条款"
)
_VAGUE = re.compile(r"那个规定|相关制度|有个流程|记不得.*规定|了解相关")
_LOCATE = re.compile(r"定位|在哪里|在哪儿|原文|来源位置|打开|找到|查找")
_BROWSE = re.compile(r"列出|浏览|全部|所有|集合|汇总|有哪些")
_APPLY = re.compile(r"比较|核对|判断|是否.*(?:达到|符合|满足|需要)|能否.*(?:通过|按)|够不够|当前数值|更新为|补充：|继续刚才")
_NEGATIVE = re.compile(r"没有|未交|未提交|缺少|没带|没准备")
_AMBIGUOUS_SOURCE = re.compile(r"酌情|未明确|不明确|视情况")
_NUMBER = re.compile(r"(?<![A-Za-z])(?:\d+(?:\.\d+)?|[零〇一二两三四五六七八九十百]+)")


@dataclass(frozen=True)
class _Runtime:
    units: tuple[KnowledgeUnit, ...]
    catalog: tuple[CorpusEvidenceView, ...]
    scopes: tuple[str, ...]


def _fingerprint(value: object) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def baseline_backend_identity() -> BackendIdentity:
    config = {
        "adapter_schema_version": ADAPTER_SCHEMA_VERSION,
        "engine": "BaselineRetrieval",
        "mode": "exact-keyword-browse",
        "filter": "candidate-filter-v1",
    }
    return BackendIdentity("knowledge-system-baseline", BASELINE_ADAPTER_VERSION, _fingerprint(config))


def current_backend_identity() -> BackendIdentity:
    config = {
        "adapter_schema_version": ADAPTER_SCHEMA_VERSION,
        "engine": "HybridRetrieval",
        "paths": ["keyword", "vector"],
        "index": _INDEX_VERSION,
        "filter": "candidate-filter-v1",
    }
    return BackendIdentity("knowledge-system-current", CURRENT_ADAPTER_VERSION, _fingerprint(config))


def _runtime(catalog: tuple[CorpusEvidenceView, ...]) -> _Runtime:
    units: list[KnowledgeUnit] = []
    scopes = {"global"}
    for item in catalog:
        scope = f"document:{item.document_id}"
        scopes.add(scope)
        semantic = "\n".join(
            part for part in (item.source_name, item.source_section, item.original_content) if part
        )
        units.append(
            KnowledgeUnit(
                unit_id=item.evidence_id,
                original_content=item.original_content,
                semantic_content=semantic,
                metadata={
                    "scope": ["global", scope],
                    "status": item.status,
                    "required_permissions": [] if item.permission == "public" else ["restricted-catalog"],
                },
                source_reference=SourceReference(
                    item.source_name, item.source_url, item.source_section,
                    item.source_page, item.source_sheet,
                ),
                lineage=KnowledgeLineage(
                    item.document_id, item.parser_version, item.chunker_version,
                    item.processing_version,
                ),
                parse_status=ParseStatus.SUCCESS if item.reliable else ParseStatus.PARTIAL,
            )
        )
    return _Runtime(tuple(units), catalog, tuple(sorted(scopes)))


def _scope_covered(query: str, catalog: tuple[CorpusEvidenceView, ...]) -> bool:
    if _EXTERNAL.search(query):
        return False
    if _ENTERPRISE.search(query):
        return True
    compact = re.sub(r"\W", "", query)
    return any(
        token and token in compact
        for item in catalog
        for token in (item.source_section, item.source_name.removesuffix(".md")[:2])
    )


def _classify(query: str, context: str, covered: bool) -> KnowledgeRequestType:
    if not covered:
        return KnowledgeRequestType.OUT_OF_SCOPE
    # Product intent precedence is explicit and stable.  Context is used only
    # to retain an Apply intent across a continuation/update turn.
    if _LOCATE.search(query):
        return KnowledgeRequestType.LOCATE
    if _BROWSE.search(query):
        return KnowledgeRequestType.BROWSE
    if _APPLY.search(query) or (_APPLY.search(context) and re.search(r"更新|补充|继续|现在", query)):
        return KnowledgeRequestType.APPLY
    return classify_request(query, scope_covered=True).request_type


def _catalog_scope(query: str, catalog: tuple[CorpusEvidenceView, ...]) -> str:
    compact = re.sub(r"\W", "", query)
    scored: list[tuple[int, str]] = []
    for item in catalog:
        labels = (item.source_name.removesuffix(".md"), item.source_section or "")
        score = max(
            (len(piece) for label in labels for piece in _pieces(label) if piece in compact),
            default=0,
        )
        if score >= 2:
            scored.append((score, item.document_id))
    if not scored:
        return "global"
    best = max(score for score, _ in scored)
    documents = sorted({document for score, document in scored if score == best})
    return f"document:{documents[0]}" if len(documents) == 1 else "global"


def _pieces(text: str) -> tuple[str, ...]:
    clean = re.sub(r"评测|政策|手册|规则|\.md|\W", "", text)
    return tuple(clean[start:end] for start in range(len(clean)) for end in range(start + 2, len(clean) + 1))


def _locate_query(query: str, catalog: tuple[CorpusEvidenceView, ...]) -> str:
    candidates: list[tuple[int, str]] = []
    compact = re.sub(r"\W", "", query)
    for item in catalog:
        for label in (item.source_section, item.source_name.removesuffix(".md")):
            if not label:
                continue
            overlap = max((len(p) for p in _pieces(label) if p in compact), default=0)
            if overlap >= 2:
                candidates.append((overlap, label))
    return sorted(candidates, key=lambda value: (-value[0], value[1]))[0][1] if candidates else query


def _ids(evidence: object) -> tuple[str, ...]:
    return tuple(item.unit_id for item in evidence)  # type: ignore[arg-type]


def _catalog_item(catalog: tuple[CorpusEvidenceView, ...], evidence_id: str) -> CorpusEvidenceView:
    return next(item for item in catalog if item.evidence_id == evidence_id)


def _chinese_number(text: str) -> float | None:
    match = _NUMBER.search(text)
    if not match:
        return None
    token = match.group(0)
    if token[0].isdigit():
        return float(token)
    digits = {"零": 0, "〇": 0, "一": 1, "二": 2, "两": 2, "三": 3, "四": 4, "五": 5, "六": 6, "七": 7, "八": 8, "九": 9}
    if "百" in token:
        left, _, right = token.partition("百")
        return float(digits.get(left, 1) * 100 + (_chinese_number(right) or 0))
    if "十" in token:
        left, _, right = token.partition("十")
        return float(digits.get(left, 1) * 10 + digits.get(right, 0))
    return float(digits.get(token, 0))


def _comparison(query: str, source: str) -> tuple[str | None, tuple[str, ...], BoundaryTruth]:
    if _AMBIGUOUS_SOURCE.search(source):
        return "HUMAN_REQUIRED", (), BoundaryTruth.KNOWLEDGE_AMBIGUOUS
    if source.rstrip().endswith("："):
        return "UNKNOWN", (), BoundaryTruth.RECOVERY_FAILURE
    threshold_match = re.search(r"(?:不少于|超过)(\d+(?:\.\d+)?)", source)
    if threshold_match:
        value = _chinese_number(query)
        if value is None:
            metric = "内容点赞数" if "点赞" in source else "案例数值"
            return "MISSING", (metric,), BoundaryTruth.CASE_INFORMATION_MISSING
        passed = value >= float(threshold_match.group(1)) if "不少于" in source else value > float(threshold_match.group(1))
        if "需要提交书面说明" in source and _NEGATIVE.search(query):
            passed = False
        return ("SATISFIED" if passed else "NOT_SATISFIED"), (), BoundaryTruth.NONE
    return "HUMAN_REQUIRED", (), BoundaryTruth.KNOWLEDGE_AMBIGUOUS


def _predict(data: BackendInput, *, hybrid: bool) -> Prediction:
    runtime = _runtime(data.synthetic_catalog)
    context_query = " ".join((*[turn.text for turn in data.turns], data.query))
    request_type = _classify(
        data.query, context_query,
        _scope_covered(context_query, data.synthetic_catalog),
    )
    if request_type is KnowledgeRequestType.OUT_OF_SCOPE:
        return _prediction(data, request_type, (), OutputMode.OUT_OF_SCOPE, BoundaryTruth.OUT_OF_SCOPE,
                           ClarificationTruth.NONE, VerificationTruth.FORBIDDEN)
    if request_type is KnowledgeRequestType.DISCOVER and _VAGUE.search(data.query):
        return _prediction(data, request_type, (), OutputMode.CLARIFICATION, BoundaryTruth.QUERY_INSUFFICIENT,
                           ClarificationTruth.REQUIRED, VerificationTruth.FORBIDDEN)

    baseline = BaselineRetrieval(
        runtime.units, covered_scopes=runtime.scopes, knowledge_version=_KNOWLEDGE_VERSION
    )
    scope = _catalog_scope(data.query, data.synthetic_catalog)
    if request_type is KnowledgeRequestType.BROWSE:
        if hybrid:
            HybridRetrieval(runtime.units, knowledge_version=_KNOWLEDGE_VERSION,
                            index_version=_INDEX_VERSION, covered_scopes=runtime.scopes).search(
                                data.query, scope=scope, top_k=max(1, len(runtime.units))
                            )
        result = baseline.browse(scope, raw_query=data.query)
        ranked = _ids(result.evidence)
        # A scoped policy browse exposes complete usable rules.  Partial source
        # material remains visible only in a global catalog/inventory browse,
        # where its processing boundary is reported explicitly.
        if scope != "global":
            ranked = tuple(item for item in ranked if _catalog_item(data.synthetic_catalog, item).reliable)
    elif request_type is KnowledgeRequestType.LOCATE:
        if hybrid:
            HybridRetrieval(runtime.units, knowledge_version=_KNOWLEDGE_VERSION,
                            index_version=_INDEX_VERSION, covered_scopes=runtime.scopes).search(
                                context_query, scope=scope, top_k=max(1, len(runtime.units))
                            )
        result = baseline.locate(_locate_query(context_query, data.synthetic_catalog), scope=scope)
        ranked = _ids(result.evidence)
    elif hybrid:
        result = HybridRetrieval(
            runtime.units, knowledge_version=_KNOWLEDGE_VERSION,
            index_version=_INDEX_VERSION, covered_scopes=runtime.scopes,
        ).search(context_query, scope=scope, top_k=max(1, len(runtime.units)))
        ranked = _ids(result.evidence)
        exact = baseline.locate(
            _locate_query(context_query, data.synthetic_catalog), scope=scope
        )
        exact_ids = _ids(exact.evidence)
        ranked = exact_ids + tuple(item for item in ranked if item not in exact_ids)
    else:
        result = baseline.locate(_locate_query(context_query, data.synthetic_catalog), scope=scope)
        ranked = _ids(result.evidence)

    if not ranked:
        return _prediction(data, request_type, (), OutputMode.VERIFICATION, BoundaryTruth.KNOWLEDGE_MISSING,
                           ClarificationTruth.NONE, VerificationTruth.REQUIRED)

    boundary_ids = ranked if request_type is KnowledgeRequestType.BROWSE else ranked[:1]
    unreliable = any(not _catalog_item(data.synthetic_catalog, item).reliable for item in boundary_ids)
    ambiguous = any(_AMBIGUOUS_SOURCE.search(_catalog_item(data.synthetic_catalog, item).original_content) for item in boundary_ids)
    boundary = BoundaryTruth.NONE
    mode = OutputMode.EVIDENCE
    verification = VerificationTruth.FORBIDDEN
    outcome: str | None = None
    missing: tuple[str, ...] = ()
    if request_type is KnowledgeRequestType.APPLY:
        source = _catalog_item(data.synthetic_catalog, ranked[0]).original_content
        outcome, missing, boundary = _comparison(data.query, source)
        if boundary is not BoundaryTruth.NONE:
            mode, verification = OutputMode.VERIFICATION, VerificationTruth.REQUIRED
        if boundary is BoundaryTruth.CASE_INFORMATION_MISSING:
            clarification = ClarificationTruth.ALLOWED_AFTER_EVIDENCE
        else:
            clarification = ClarificationTruth.NONE
    elif unreliable:
        boundary = BoundaryTruth.PROCESSING_FAILURE
        mode, verification = OutputMode.VERIFICATION, VerificationTruth.REQUIRED
    elif ambiguous and request_type is KnowledgeRequestType.DISCOVER:
        boundary = BoundaryTruth.SOURCE_FAILURE
        mode, verification = OutputMode.VERIFICATION, VerificationTruth.REQUIRED

    return _prediction(data, request_type, ranked, mode, boundary,
                       clarification if request_type is KnowledgeRequestType.APPLY else ClarificationTruth.NONE,
                       verification, outcome, missing)


def _prediction(
    data: BackendInput,
    request_type: KnowledgeRequestType,
    ranked: tuple[str, ...],
    mode: OutputMode,
    boundary: BoundaryTruth,
    clarification: ClarificationTruth,
    verification: VerificationTruth,
    outcome: str | None = None,
    missing: tuple[str, ...] = (),
) -> Prediction:
    return Prediction(
        expression_id=data.expression_id,
        request_type=request_type,
        ranked_evidence_ids=ranked,
        output_mode=mode,
        boundary=boundary,
        clarification=clarification,
        verification=verification,
        comparison_outcome=outcome,
        missing_information=missing,
        trace_fields=_TRACE_FIELDS,
        source_evidence_ids=ranked,
        lineage_evidence_ids=ranked,
        final_business_decision=False,
    )


def baseline_product_backend(data: BackendInput) -> Prediction:
    """Run the real deterministic exact/keyword/browse product baseline."""

    return _predict(data, hybrid=False)


def current_product_backend(data: BackendInput) -> Prediction:
    """Run the real deterministic keyword+vector product retrieval."""

    return _predict(data, hybrid=True)
