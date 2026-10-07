"""UI-independent, deterministic application surface for Knowledge System V0."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Iterable

from .agent_state import AgentState
from .baseline_retrieval import BaselineRetrieval, RetrievalStatus
from .clarification import (
    ClarificationDecision,
    KnowledgeBoundary,
    RetrievalOutcome,
    decide_clarification,
)
from .context import CaseContext, QueryContext
from .contracts import Evidence, KnowledgeUnit, ParseStatus, Trace
from .evidence_quality import EvidenceQualitySignal, detect_evidence_quality
from .hybrid_retrieval import HybridRetrieval
from .request_classification import (
    KnowledgeRequestType,
    RequestClassification,
    classify_request,
)
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
                return self._boundary_result(
                    query, session_id, classification, query_context,
                    KnowledgeBoundary.RETRIEVAL_QUERY_INSUFFICIENT,
                    "浏览请求需要显式 scope，系统未猜测浏览范围。",
                )
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
        trace = _stable_trace(trace, session_id=session_id)
        if not evidence or (retrieved is not None and retrieved.status is RetrievalStatus.NOT_FOUND):
            return self._result(
                query, session_id, classification, query_context, (), trace,
                KnowledgeBoundary.KNOWLEDGE_MISSING, NOT_FOUND_MESSAGE,
                VerificationView(
                    True, ("knowledge_not_retrieved",),
                    "请按完整资料名称、Section 或知识库维护渠道核验。",
                ),
            )

        signals = tuple(
            signal
            for item in evidence
            for signal in detect_evidence_quality(
                item, knowledge_unit=self._by_unit_id.get(item.unit_id)
            )
        )
        if signals:
            boundary = KnowledgeBoundary.EVIDENCE_INSUFFICIENT
            verification = VerificationView(
                True,
                tuple(sorted({item.signal_type.value for item in signals})),
                "请通过下列 Source 定位回到正式原文核验；系统不会猜测缺失内容。",
            )
            message = "已检索到 Evidence，但其可靠性信号要求进入 Verification Mode。"
        else:
            boundary = KnowledgeBoundary.NONE
            verification = VerificationView(False)
            message = (
                "已返回当前范围内的 Evidence 集合。"
                if classification.request_type is KnowledgeRequestType.BROWSE
                else "已返回可追溯 Evidence。"
            )
        return self._result(
            query, session_id, classification, query_context, evidence, trace,
            boundary, message, verification, signals,
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
            case_context=CaseContext(),
            knowledge_request=classification,
            clarification=clarification,
            candidate_evidence=evidence,
            evidence_quality=signals,
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

