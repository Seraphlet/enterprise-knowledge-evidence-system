"""Deterministic Locate/Browse behavior over filtered keyword evidence."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from enum import Enum
from uuid import uuid4

from knowledge_system.candidate_filter import filter_candidates
from knowledge_system.contracts import Evidence, KnowledgeUnit, Trace
from knowledge_system.keyword_index import (
    KeywordIndex,
    evidence_from_unit,
    normalize_query,
)
from knowledge_system.source_navigation import SourceNavigator


class RetrievalStatus(str, Enum):
    FOUND = "FOUND"
    NOT_FOUND = "NOT_FOUND"
    OUT_OF_SCOPE = "OUT_OF_SCOPE"


class MatchType(str, Enum):
    EXACT = "EXACT"
    KEYWORD = "KEYWORD"
    BROWSE = "BROWSE"


@dataclass(frozen=True)
class BaselineResult:
    status: RetrievalStatus
    evidence: tuple[Evidence, ...]
    message: str
    match_type: MatchType | None = None
    trace: Trace | None = None


_NOT_FOUND_MESSAGE = (
    "当前知识库未检索到相关内容；这不代表企业没有相关规定。"
)
_OUT_OF_SCOPE_MESSAGE = "当前知识库不覆盖该范围。"


def _unit_in_scope(unit: KnowledgeUnit, scope: str) -> bool:
    value = unit.metadata.get("scope")
    if isinstance(value, str):
        return value == scope
    if isinstance(value, list):
        return scope in value
    return False


class BaselineRetrieval:
    """Permission-safe baseline; contains no scoring, ranking, or semantics."""

    def __init__(
        self,
        units: Iterable[KnowledgeUnit],
        *,
        covered_scopes: Iterable[str],
        granted_permissions: Iterable[str] = (),
        knowledge_version: str | None = None,
    ) -> None:
        self._covered_scopes = frozenset(covered_scopes)
        filtered = filter_candidates(
            units, granted_permissions=granted_permissions
        )
        self._eligible_units = filtered.candidates
        self._knowledge_version = knowledge_version
        rejection_counts: dict[str, int] = {}
        for rejected in filtered.rejected:
            rejection_counts[rejected.reason.value] = (
                rejection_counts.get(rejected.reason.value, 0) + 1
            )
        self._filter_summary = {
            "eligible_count": len(filtered.candidates),
            "rejected_count": len(filtered.rejected),
            "rejection_counts": rejection_counts,
        }

    def _trace(
        self,
        *,
        raw_query: str,
        retrieval_query: str,
        candidates: tuple[Evidence, ...],
        final: tuple[Evidence, ...],
    ) -> Trace:
        return Trace(
            trace_id=f"trace-{uuid4()}",
            raw_query=raw_query,
            retrieval_query=retrieval_query,
            candidate_evidence_ids=[item.evidence_id for item in candidates],
            final_evidence_ids=[item.evidence_id for item in final],
            applied_filters=["permission", "status", "scope"],
            knowledge_version=self._knowledge_version,
            filter_summary=dict(self._filter_summary),
            source_references=[item.source_reference.to_dict() for item in final],
            processing_versions=[
                {
                    "evidence_id": item.evidence_id,
                    "parser_version": item.lineage.parser_version,
                    "chunker_version": item.lineage.chunker_version,
                    "processing_version": item.lineage.processing_version,
                }
                for item in final
            ],
        )

    def _scope_units(self, scope: str | None) -> tuple[KnowledgeUnit, ...] | None:
        if scope is None:
            return self._eligible_units
        if scope not in self._covered_scopes:
            return None
        return tuple(
            unit for unit in self._eligible_units if _unit_in_scope(unit, scope)
        )

    def locate(self, query: str, *, scope: str | None = None) -> BaselineResult:
        """Locate exact identity first, then deterministic keyword containment."""

        units = self._scope_units(scope)
        retrieval_query = normalize_query(query)
        if units is None:
            return BaselineResult(
                RetrievalStatus.OUT_OF_SCOPE,
                (),
                _OUT_OF_SCOPE_MESSAGE,
                trace=self._trace(
                    raw_query=query,
                    retrieval_query=retrieval_query,
                    candidates=(),
                    final=(),
                ),
            )
        index = KeywordIndex(units)
        exact = index.exact(query)
        if exact:
            return BaselineResult(
                RetrievalStatus.FOUND,
                exact,
                "已在当前知识库定位到精确 Evidence。",
                MatchType.EXACT,
                self._trace(
                    raw_query=query,
                    retrieval_query=retrieval_query,
                    candidates=exact,
                    final=exact,
                ),
            )
        keyword = index.search(query)
        if keyword:
            return BaselineResult(
                RetrievalStatus.FOUND,
                keyword,
                "已在当前知识库定位到关键字 Evidence。",
                MatchType.KEYWORD,
                self._trace(
                    raw_query=query,
                    retrieval_query=retrieval_query,
                    candidates=keyword,
                    final=keyword,
                ),
            )
        return BaselineResult(
            RetrievalStatus.NOT_FOUND,
            (),
            _NOT_FOUND_MESSAGE,
            trace=self._trace(
                raw_query=query,
                retrieval_query=retrieval_query,
                candidates=(),
                final=(),
            ),
        )

    def browse(
        self,
        scope: str | None,
        *,
        document: str | None = None,
        section: str | None = None,
        raw_query: str | None = None,
    ) -> BaselineResult:
        """Return an eligible scope, or one explicit document when scope is absent."""

        if scope is None and document is None:
            raise ValueError("scope-less browsing requires a document identity")

        units = self._scope_units(scope)
        raw = raw_query or " ".join(
            part
            for part in (
                f"browse:{scope}",
                f"document:{document}" if document else "",
                f"section:{section}" if section else "",
            )
            if part
        )
        retrieval_query = normalize_query(raw)
        if units is None:
            return BaselineResult(
                RetrievalStatus.OUT_OF_SCOPE,
                (),
                _OUT_OF_SCOPE_MESSAGE,
                trace=self._trace(
                    raw_query=raw,
                    retrieval_query=retrieval_query,
                    candidates=(),
                    final=(),
                ),
            )
        selected = units
        if section is not None and document is None:
            raise ValueError("section browsing requires a document identity")
        if document is not None:
            navigator = SourceNavigator(units)
            selected = (
                navigator.by_section(document, section)
                if section is not None
                else navigator.by_document(document)
            )
        if not selected:
            return BaselineResult(
                RetrievalStatus.NOT_FOUND,
                (),
                _NOT_FOUND_MESSAGE,
                trace=self._trace(
                    raw_query=raw,
                    retrieval_query=retrieval_query,
                    candidates=(),
                    final=(),
                ),
            )
        evidence = tuple(evidence_from_unit(unit) for unit in selected)
        return BaselineResult(
            RetrievalStatus.FOUND,
            evidence,
            "已返回当前知识库范围内的 Evidence 集合。",
            MatchType.BROWSE,
            self._trace(
                raw_query=raw,
                retrieval_query=retrieval_query,
                candidates=evidence,
                final=evidence,
            ),
        )

    def resolve_document_identity(self, raw_query: str) -> str | None:
        """Resolve one complete source name from filtered candidates without inference."""

        matches = {
            (unit.lineage.document_id, unit.source_reference.name)
            for unit in self._eligible_units
            if unit.source_reference.name in raw_query
        }
        if len(matches) != 1:
            return None
        return next(iter(matches))[1]
