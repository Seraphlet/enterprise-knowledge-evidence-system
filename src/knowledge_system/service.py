"""UI-independent, deterministic application surface for Knowledge System V0."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Iterable

from .agent_state import AgentState
from .apply_compare import (
    ApplyCompareResult,
    LogicalOutcome,
    organize_apply_comparisons,
)
from .baseline_retrieval import BaselineRetrieval, RetrievalStatus
from .candidate_filter import filter_candidates
from .clarification import (
    ClarificationDecision,
    KnowledgeBoundary,
    RetrievalOutcome,
    decide_clarification,
)
from .context import CaseContext, QueryContext
from .contracts import Evidence, KnowledgeUnit, ParseStatus, Trace
from .evidence_quality import EvidenceQualitySignal, detect_evidence_quality
from .fact_extraction import extract_required_facts
from .hybrid_retrieval import HybridRetrieval
from .information_requirements import generate_information_requirements
from .keyword_index import evidence_from_unit
from .limited_recovery import (
    RecoveryAction,
    RecoveryExecutionResult,
    RecoveryRequest,
    RecoveryStatus,
    run_limited_recovery,
)
from .request_classification import (
    KnowledgeRequestType,
    RequestClassification,
    classify_request,
)
from .deterministic_comparison import ComparisonOutcome, compare_condition
from .rule_structure import RuleExtractionStatus, extract_rule_structure
from .session_snapshot import (
    SessionResumeResult,
    SessionSnapshotMetadata,
    resume_session_snapshot,
    save_session_snapshot,
)


SERVICE_SCHEMA = "knowledge-system.query-result"
SERVICE_SCHEMA_VERSION = 1
HUMAN_RESPONSIBILITY = "结果仅用于知识核验辅助，最终业务判断与责任由 Human 承担。"
NOT_FOUND_MESSAGE = "当前知识库未检索到相关内容；这不代表企业没有相关规定。"
OUT_OF_SCOPE_MESSAGE = "当前知识库不覆盖该范围；系统不会回退为通用问答。"


class ServiceInputError(ValueError):
    """Stable user/data error safe to expose without filesystem details."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code
        self.message = message


@dataclass(frozen=True)
class VerificationView:
    required: bool
    reasons: tuple[str, ...] = ()
    guidance: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "required": self.required,
            "reasons": list(self.reasons),
            "guidance": self.guidance,
        }


@dataclass(frozen=True)
class QueryResult:
    session_id: str
    request_type: KnowledgeRequestType
    boundary: KnowledgeBoundary
    message: str
    evidence: tuple[Evidence, ...]
    quality_signals: tuple[EvidenceQualitySignal, ...]
    verification: VerificationView
    trace: Trace
    state: AgentState

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": SERVICE_SCHEMA,
            "schema_version": SERVICE_SCHEMA_VERSION,
            "session_id": self.session_id,
            "request_type": self.request_type.value,
            "boundary": self.boundary.value,
            "message": self.message,
            "evidence": [
                {
                    "evidence_id": item.evidence_id,
                    "original_content": item.original_content,
                    "source": item.source_reference.to_dict(),
                    "lineage": item.lineage.to_dict(),
                }
                for item in self.evidence
            ],
            "quality_signals": [item.to_dict() for item in self.quality_signals],
            "verification": self.verification.to_dict(),
            "trace": self.trace.to_dict(),
            "human_responsibility": HUMAN_RESPONSIBILITY,
        }

    def to_json(self) -> str:
        return json.dumps(
            self.to_dict(), ensure_ascii=False, sort_keys=True, separators=(",", ":")
        )


def _strict_object_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ServiceInputError("DUPLICATE_JSON_FIELD", "知识数据包含重复字段。")
        result[key] = value
    return result


def load_knowledge_units(path: str | Path) -> tuple[KnowledgeUnit, ...]:
    """Load a local JSON array of canonical KnowledgeUnit objects."""

    try:
        payload = Path(path).read_text(encoding="utf-8", errors="strict")
    except (OSError, UnicodeError) as error:
        raise ServiceInputError("CORPUS_READ_ERROR", "无法读取本地知识数据。") from error
    try:
        value = json.loads(payload, object_pairs_hook=_strict_object_pairs)
    except ServiceInputError:
        raise
    except (json.JSONDecodeError, ValueError, TypeError) as error:
        raise ServiceInputError("INVALID_CORPUS_JSON", "知识数据不是有效 JSON。") from error
    if not isinstance(value, list):
        raise ServiceInputError("INVALID_CORPUS_SCHEMA", "知识数据根节点必须是数组。")
    try:
        units = tuple(KnowledgeUnit.from_dict(item) for item in value if isinstance(item, dict))
    except (KeyError, TypeError, ValueError) as error:
        raise ServiceInputError("INVALID_CORPUS_SCHEMA", "知识数据不符合 KnowledgeUnit schema。") from error
    if len(units) != len(value):
        raise ServiceInputError("INVALID_CORPUS_SCHEMA", "知识数据数组只能包含对象。")
    if len({item.unit_id for item in units}) != len(units):
        raise ServiceInputError("DUPLICATE_UNIT_ID", "知识数据包含重复 unit_id。")
    return units


def _scope_values(unit: KnowledgeUnit) -> tuple[str, ...]:
    raw = unit.metadata.get("scope")
    if isinstance(raw, str) and raw:
        return (raw,)
    if isinstance(raw, list):
        return tuple(value for value in raw if isinstance(value, str) and value)
    return ()


def _surface_eligible(unit: KnowledgeUnit) -> bool:
    # The shared candidate filter handles permissions and inactive states.  The
    # product surface additionally refuses explicit unknown state and failed
    # parses, which must never become employee-facing Evidence.
    status = unit.metadata.get("status")
    if isinstance(status, str) and status.strip().casefold() in {
        "unknown", "未知", "待确认", "unverified"
    }:
        return False
    return unit.parse_status is not ParseStatus.FAILED


def _source_order_key(unit: KnowledgeUnit) -> tuple[int, int, str, str]:
    """Order source-grounded recovery independently of corpus input order."""

    start_line = unit.metadata.get("start_line")
    if type(start_line) is int and start_line >= 0:
        return (0, start_line, unit.lineage.source_position or "", unit.unit_id)
    position = unit.lineage.source_position or ""
    _, separator, span = position.partition(":")
    first = span.split("-", 1)[0] if separator else ""
    if first.isdigit():
        return (0, int(first), position, unit.unit_id)
    return (1, 0, position, unit.unit_id)


def _stable_trace(trace: Trace, *, session_id: str) -> Trace:
    identity = {
        "session_id": session_id,
        "raw_query": trace.raw_query,
        "retrieval_query": trace.retrieval_query,
        "final": trace.final_evidence_ids,
        "knowledge_version": trace.knowledge_version,
        "index_version": trace.index_version,
    }
    digest = hashlib.sha256(
        json.dumps(identity, ensure_ascii=False, sort_keys=True).encode("utf-8")
    ).hexdigest()[:24]
    return replace(trace, trace_id=f"trace-{digest}")


class KnowledgeService:
    """Reusable product facade; performs no network, LLM, or persistent writes."""

    def __init__(
        self,
        units: Iterable[KnowledgeUnit],
        *,
        knowledge_version: str = "local",
        index_version: str = "local-derived",
        granted_permissions: Iterable[str] = (),
    ) -> None:
        self._units = tuple(item for item in units if _surface_eligible(item))
        self._by_unit_id = {item.unit_id: item for item in self._units}
        if len(self._by_unit_id) != len(self._units):
            raise ServiceInputError("DUPLICATE_UNIT_ID", "知识数据包含重复 unit_id。")
        self._scopes = frozenset(
            scope for item in self._units for scope in _scope_values(item)
        )
        self._knowledge_version = knowledge_version
        self._index_version = index_version
        self._permissions = tuple(granted_permissions)
        self._recovery_units = filter_candidates(
            self._units, granted_permissions=self._permissions
        ).candidates

    def _recovery_pool(
        self, original: KnowledgeUnit, scope: str | None
    ) -> tuple[KnowledgeUnit, ...]:
        original_scopes = frozenset(_scope_values(original))
        return tuple(sorted((
            unit
            for unit in self._recovery_units
            if unit.lineage.document_id == original.lineage.document_id
            and unit.source_reference.section == original.source_reference.section
            and unit.lineage.parent_section_id
            == original.lineage.parent_section_id
            and (
                scope in _scope_values(unit)
                if scope is not None
                else frozenset(_scope_values(unit)) == original_scopes
            )
        ), key=_source_order_key))

    def _recovery_executor(
        self, original: KnowledgeUnit, scope: str | None
    ):
        pool = self._recovery_pool(original, scope)

        def execute(request: RecoveryRequest) -> RecoveryExecutionResult:
            try:
                index = next(
                    position
                    for position, unit in enumerate(pool)
                    if unit.unit_id == original.unit_id
                )
            except StopIteration:
                return RecoveryExecutionResult(
                    (), request.trigger_signals, "filtered_source_unit_unavailable"
                )

            selected: tuple[KnowledgeUnit, ...]
            if request.action is RecoveryAction.NEIGHBOR_EXPANSION:
                selected = tuple(
                    pool[position]
                    for position in (index - 1, index + 1)
                    if 0 <= position < len(pool)
                )
            elif request.action in {
                RecoveryAction.PARENT_UNIT,
                RecoveryAction.SECTION_EXPANSION,
            }:
                selected = tuple(
                    unit for unit in pool if unit.unit_id != original.unit_id
                )[:4]
            else:
                return RecoveryExecutionResult(
                    (), request.trigger_signals, "action_not_available_locally"
                )

            checked = tuple(
                (evidence_from_unit(unit), unit) for unit in selected
            )
            recovered = tuple(
                item
                for item, unit in checked
                if not detect_evidence_quality(item, knowledge_unit=unit)
            )
            if not recovered:
                return RecoveryExecutionResult(
                    (),
                    request.trigger_signals,
                    "bounded_context_not_sufficient",
                )
            return RecoveryExecutionResult(recovered, ())

        return execute

    def _recover_context_fragments(
        self,
        evidence: tuple[Evidence, ...],
        signals_by_evidence: dict[str, tuple[EvidenceQualitySignal, ...]],
        *,
        scope: str | None,
        trace: Trace,
    ) -> tuple[
        tuple[Evidence, ...],
        tuple[EvidenceQualitySignal, ...],
        tuple[EvidenceQualitySignal, ...],
        Trace,
    ]:
        final = list(evidence)
        known_ids = {item.evidence_id for item in evidence}
        observed: list[EvidenceQualitySignal] = []
        unresolved: list[EvidenceQualitySignal] = []
        recovery_paths = list(trace.retrieval_paths)

        for item in evidence:
            item_signals = signals_by_evidence.get(item.evidence_id, ())
            observed.extend(item_signals)
            fragment_signals = tuple(
                signal
                for signal in item_signals
                if signal.reason == "context_dependent_fragment"
            )
            unresolved.extend(
                signal for signal in item_signals if signal not in fragment_signals
            )
            if not fragment_signals:
                continue
            unit = self._by_unit_id.get(item.unit_id)
            if unit is None:
                unresolved.extend(fragment_signals)
                continue
            recovery = run_limited_recovery(
                item,
                fragment_signals,
                self._recovery_executor(unit, scope),
                max_rounds=2,
            )
            if recovery.status is RecoveryStatus.RESOLVED:
                for recovered in recovery.recovered_evidence:
                    if recovered.evidence_id not in known_ids:
                        known_ids.add(recovered.evidence_id)
                        final.append(recovered)
            else:
                unresolved.extend(fragment_signals)
            for round_trace in recovery.trace:
                recovery_paths.append(
                    {
                        "path": "bounded_recovery",
                        "round": round_trace.round_number,
                        "action": round_trace.action.value,
                        "trigger_evidence_id": item.evidence_id,
                        "recovered_evidence_ids": [
                            recovered.evidence_id
                            for recovered in round_trace.recovered_evidence
                        ],
                        "failure_reason": round_trace.failure_reason,
                    }
                )

        final_tuple = tuple(final)
        updated_trace = replace(
            trace,
            final_evidence_ids=[item.evidence_id for item in final_tuple],
            source_references=[
                item.source_reference.to_dict() for item in final_tuple
            ],
            processing_versions=[
                {
                    "evidence_id": item.evidence_id,
                    "parser_version": item.lineage.parser_version,
                    "chunker_version": item.lineage.chunker_version,
                    "processing_version": item.lineage.processing_version,
                }
                for item in final_tuple
            ],
            retrieval_paths=recovery_paths,
        )
        return final_tuple, tuple(observed), tuple(unresolved), updated_trace

    @staticmethod
    def _apply_comparison(
        evidence: tuple[Evidence, ...], query: str
    ) -> tuple[
        CaseContext,
        ApplyCompareResult | None,
        VerificationView,
        KnowledgeBoundary,
        str,
    ]:
        """Run the existing conservative rule pipeline for one Apply request.

        Source Evidence remains authoritative.  Only a fully structured source
        rule is eligible for deterministic comparison; ordinary prose is not
        promoted into a rule merely because the request is an Apply request.
        """

        case_context = CaseContext(raw_descriptions=(query,))
        comparison_result: ApplyCompareResult | None = None
        for item in evidence:
            extraction = extract_rule_structure(item)
            if (
                extraction.status is not RuleExtractionStatus.STRUCTURED
                or extraction.structure is None
            ):
                continue
            requirements = generate_information_requirements(extraction)
            if not requirements:
                continue
            facts = extract_required_facts(requirements, case_context)
            comparisons = tuple(
                compare_condition(
                    requirement.threshold
                    if requirement.threshold is not None
                    else requirement.condition,
                    requirement,
                    fact,
                    item,
                    case_context,
                )
                for requirement, fact in zip(requirements, facts, strict=True)
            )
            comparison_result = organize_apply_comparisons(
                extraction, comparisons, case_context
            )
            break

        if comparison_result is None:
            return (
                case_context,
                None,
                VerificationView(
                    True,
                    ("rule_structure_unavailable",),
                    "已保留 Source Evidence，但原文不能可靠派生可比较的 Rule Structure；请由 Human 核验正式规则。",
                ),
                KnowledgeBoundary.EVIDENCE_INSUFFICIENT,
                "已检索到 Source Evidence，但无法从原文可靠派生 Rule Structure；未执行 Case Comparison。",
            )

        reasons: list[str] = []
        outcomes = {
            item.outcome for item in comparison_result.comparisons
        }
        if comparison_result.missing:
            reasons.append("case_information_missing")
        if ComparisonOutcome.UNKNOWN in outcomes:
            reasons.append("comparison_unknown")
        if ComparisonOutcome.HUMAN_REQUIRED in outcomes:
            reasons.append("human_required")
        if comparison_result.unevaluated_conditions:
            reasons.append("comparison_incomplete")
        if (
            comparison_result.logical_outcome is LogicalOutcome.INDETERMINATE
            and not reasons
        ):
            reasons.append("comparison_indeterminate")

        if reasons:
            return (
                case_context,
                comparison_result,
                VerificationView(
                    True,
                    tuple(reasons),
                    "Rule Source 已定位，但 Case 事实或机械比较不足；请补充 Case 信息或由 Human 核验。",
                ),
                KnowledgeBoundary.EVIDENCE_INSUFFICIENT,
                "已检索并解析可核验规则，但无法完成可靠的 Case Comparison。",
            )

        return (
            case_context,
            comparison_result,
            VerificationView(False),
            KnowledgeBoundary.NONE,
            "已基于 Source Rule 完成确定性 Case Comparison："
            f"{comparison_result.logical_outcome.value}。",
        )

    def query(
        self,
        query: str,
        *,
        session_id: str,
        scope: str | None = None,
        top_k: int = 10,
    ) -> QueryResult:
        if not isinstance(query, str) or not query.strip():
            raise ServiceInputError("EMPTY_QUERY", "query 不能为空。")
        if not isinstance(session_id, str) or not session_id.strip():
            raise ServiceInputError("EMPTY_SESSION_ID", "session_id 不能为空。")
        if top_k <= 0:
            raise ServiceInputError("INVALID_TOP_K", "top_k 必须是正整数。")

        scope_covered = None if scope is None else scope in self._scopes
        classification = classify_request(query, scope_covered=scope_covered)
        query_context = QueryContext(
            query, scope, classification.request_type, classification.signals
        )
        if classification.request_type is KnowledgeRequestType.OUT_OF_SCOPE:
            return self._boundary_result(
                query, session_id, classification, query_context,
                KnowledgeBoundary.OUT_OF_SCOPE, OUT_OF_SCOPE_MESSAGE,
            )

        baseline = BaselineRetrieval(
            self._units,
            covered_scopes=self._scopes,
            granted_permissions=self._permissions,
            knowledge_version=self._knowledge_version,
        )
        if classification.request_type is KnowledgeRequestType.BROWSE:
            if scope is None:
                document = baseline.resolve_document_identity(query)
                if document is None:
                    return self._boundary_result(
                        query, session_id, classification, query_context,
                        KnowledgeBoundary.RETRIEVAL_QUERY_INSUFFICIENT,
                        "浏览请求需要显式 scope 或唯一、完整的文档身份；系统未猜测浏览范围。",
                    )
                retrieved = baseline.browse(
                    None, document=document, raw_query=query
                )
            else:
                retrieved = baseline.browse(scope, raw_query=query)
            evidence = retrieved.evidence
            trace = retrieved.trace
        else:
            hybrid = HybridRetrieval(
                self._units,
                knowledge_version=self._knowledge_version,
                index_version=self._index_version,
                covered_scopes=self._scopes,
                granted_permissions=self._permissions,
            )
            result = hybrid.search(query, scope=scope, top_k=top_k)
            evidence = result.evidence
            trace = result.trace
            retrieved = None

        assert trace is not None
        if not evidence or (retrieved is not None and retrieved.status is RetrievalStatus.NOT_FOUND):
            trace = _stable_trace(trace, session_id=session_id)
            return self._result(
                query, session_id, classification, query_context, (), trace,
                KnowledgeBoundary.KNOWLEDGE_MISSING, NOT_FOUND_MESSAGE,
                VerificationView(
                    True, ("knowledge_not_retrieved",),
                    "请按完整资料名称、Section 或知识库维护渠道核验。",
                ),
            )

        signals_by_evidence = {
            item.evidence_id: detect_evidence_quality(
                item, knowledge_unit=self._by_unit_id.get(item.unit_id)
            )
            for item in evidence
        }
        evidence, signals, unresolved_signals, trace = (
            self._recover_context_fragments(
                evidence, signals_by_evidence, scope=scope, trace=trace
            )
        )
        trace = _stable_trace(trace, session_id=session_id)
        case_context = (
            CaseContext(raw_descriptions=(query,))
            if classification.request_type is KnowledgeRequestType.APPLY
            else CaseContext()
        )
        comparison_result: ApplyCompareResult | None = None
        if unresolved_signals:
            boundary = KnowledgeBoundary.EVIDENCE_INSUFFICIENT
            verification = VerificationView(
                True,
                tuple(
                    sorted(
                        {item.signal_type.value for item in unresolved_signals}
                    )
                ),
                "请通过下列 Source 定位回到正式原文核验；系统不会猜测缺失内容。",
            )
            message = "已检索到 Evidence，但其可靠性信号要求进入 Verification Mode。"
        elif classification.request_type is KnowledgeRequestType.APPLY:
            (
                case_context,
                comparison_result,
                verification,
                boundary,
                message,
            ) = self._apply_comparison(evidence, query)
            if comparison_result is not None:
                trace = replace(
                    trace,
                    retrieval_paths=[
                        *trace.retrieval_paths,
                        {
                            "path": "apply_comparison",
                            "evidence_id": comparison_result.extraction.source_evidence.evidence_id,
                            "logical_outcome": comparison_result.logical_outcome.value,
                            "condition_outcomes": [
                                item.outcome.value
                                for item in comparison_result.comparisons
                            ],
                            "verification_required": verification.required,
                        },
                    ],
                )
        else:
            boundary = KnowledgeBoundary.NONE
            verification = VerificationView(False)
            message = (
                "已返回当前范围内的 Evidence 集合。"
                if classification.request_type is KnowledgeRequestType.BROWSE
                else (
                    "已返回可追溯 Evidence，并恢复了必要的原文上下文。"
                    if signals
                    else "已返回可追溯 Evidence。"
                )
            )
        return self._result(
            query, session_id, classification, query_context, evidence, trace,
            boundary, message, verification, signals,
            case_context=case_context,
            comparison_result=comparison_result,
        )

    def _boundary_result(
        self,
        query: str,
        session_id: str,
        classification: RequestClassification,
        query_context: QueryContext,
        boundary: KnowledgeBoundary,
        message: str,
    ) -> QueryResult:
        trace = _stable_trace(
            Trace(
                trace_id="pending", raw_query=query, retrieval_query=query.strip(),
                applied_filters=["permission", "status", "scope"],
                knowledge_version=self._knowledge_version,
                index_version=self._index_version,
            ),
            session_id=session_id,
        )
        verification = VerificationView(
            boundary is not KnowledgeBoundary.OUT_OF_SCOPE,
            (boundary.value.lower(),),
            None if boundary is KnowledgeBoundary.OUT_OF_SCOPE else "请补充可检索的知识范围或正式资料线索。",
        )
        return self._result(
            query, session_id, classification, query_context, (), trace,
            boundary, message, verification,
        )

    @staticmethod
    def _result(
        query: str,
        session_id: str,
        classification: RequestClassification,
        query_context: QueryContext,
        evidence: tuple[Evidence, ...],
        trace: Trace,
        boundary: KnowledgeBoundary,
        message: str,
        verification: VerificationView,
        signals: tuple[EvidenceQualitySignal, ...] = (),
        *,
        case_context: CaseContext | None = None,
        comparison_result: ApplyCompareResult | None = None,
    ) -> QueryResult:
        outcome = {
            KnowledgeBoundary.NONE: RetrievalOutcome.USEFUL,
            KnowledgeBoundary.KNOWLEDGE_MISSING: RetrievalOutcome.NOT_FOUND,
            KnowledgeBoundary.OUT_OF_SCOPE: RetrievalOutcome.OUT_OF_SCOPE,
            KnowledgeBoundary.RETRIEVAL_QUERY_INSUFFICIENT: RetrievalOutcome.QUERY_INSUFFICIENT,
            KnowledgeBoundary.EVIDENCE_INSUFFICIENT: RetrievalOutcome.EVIDENCE_INSUFFICIENT,
        }[boundary]
        clarification: ClarificationDecision = decide_clarification(
            query, outcome, evidence_ids=tuple(item.evidence_id for item in evidence)
        )
        state = AgentState(
            session_id=session_id,
            raw_query=query,
            query_context=query_context,
            case_context=case_context or CaseContext(),
            knowledge_request=classification,
            clarification=clarification,
            candidate_evidence=evidence,
            evidence_quality=signals,
            comparison_result=comparison_result,
        )
        return QueryResult(
            session_id, classification.request_type, boundary, message,
            evidence, signals, verification, trace, state,
        )


def save_query_snapshot(
    result: QueryResult, path: str | Path, *, revision: int = 1
) -> SessionSnapshotMetadata:
    return save_session_snapshot(result.state, path, revision=revision)


def resume_query_snapshot(path: str | Path, session_id: str) -> SessionResumeResult:
    return resume_session_snapshot(path, session_id)
