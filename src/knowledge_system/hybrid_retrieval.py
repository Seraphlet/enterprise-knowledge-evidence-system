"""Deterministic Keyword + Vector recall fusion with path-level Trace."""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass
import math
from uuid import uuid4

from knowledge_system.candidate_filter import filter_candidates
from knowledge_system.contracts import Evidence, JsonValue, KnowledgeUnit, Trace
from knowledge_system.keyword_index import KeywordIndex, normalize_query
from knowledge_system.vector_index import VectorIndex


_RRF_K = 60


@dataclass(frozen=True)
class HybridResult:
    evidence: tuple[Evidence, ...]
    trace: Trace


RerankFunction = Callable[[str, tuple[Evidence, ...]], Iterable[str]]


@dataclass(frozen=True)
class RerankConfig:
    """Evidence-backed opt-in gate for an externally supplied ordering function."""

    enabled: bool = False
    evaluation_id: str | None = None
    observed_ranking_metric: float | None = None
    minimum_acceptable_metric: float | None = None

    def __post_init__(self) -> None:
        if not self.enabled:
            return
        if not self.evaluation_id or not self.evaluation_id.strip():
            raise ValueError("enabled rerank requires an evaluation_id")
        observed = self.observed_ranking_metric
        minimum = self.minimum_acceptable_metric
        if (
            observed is None
            or minimum is None
            or not math.isfinite(observed)
            or not math.isfinite(minimum)
        ):
            raise ValueError("enabled rerank requires finite evaluation metrics")
        if observed >= minimum:
            raise ValueError("enabled rerank requires an observed ranking gap")


def _in_scope(unit: KnowledgeUnit, scope: str | None) -> bool:
    if scope is None:
        return True
    value = unit.metadata.get("scope")
    return value == scope or (isinstance(value, list) and scope in value)


class HybridRetrieval:
    """Recall and transparent RRF fusion only; no reranking or LLM behavior."""

    def __init__(
        self,
        units: Iterable[KnowledgeUnit],
        *,
        knowledge_version: str,
        index_version: str,
        covered_scopes: Iterable[str] = (),
        granted_permissions: Iterable[str] = (),
        rerank_config: RerankConfig = RerankConfig(),
        reranker: RerankFunction | None = None,
    ) -> None:
        if rerank_config.enabled and reranker is None:
            raise ValueError("enabled rerank requires a reranker")
        all_units = tuple(units)
        permission_set = frozenset(granted_permissions)
        filtered = filter_candidates(
            all_units, granted_permissions=permission_set
        )
        self._eligible_units = filtered.candidates
        self._knowledge_version = knowledge_version
        self._index_version = index_version
        self._covered_scopes = frozenset(covered_scopes)
        self._granted_permissions = permission_set
        self._rerank_config = rerank_config
        self._reranker = reranker
        reason_counts: dict[str, int] = {}
        for item in filtered.rejected:
            reason_counts[item.reason.value] = reason_counts.get(item.reason.value, 0) + 1
        self._filter_summary: dict[str, JsonValue] = {
            "eligible_count": len(filtered.candidates),
            "rejected_count": len(filtered.rejected),
            "rejection_counts": reason_counts,
        }

    def search(
        self,
        query: str,
        *,
        scope: str | None = None,
        top_k: int = 10,
    ) -> HybridResult:
        if top_k <= 0:
            raise ValueError("top_k must be positive")
        if scope is not None and scope not in self._covered_scopes:
            units: tuple[KnowledgeUnit, ...] = ()
        else:
            units = tuple(
                unit for unit in self._eligible_units if _in_scope(unit, scope)
            )

        keyword_index = KeywordIndex(units)
        keyword_hits = keyword_index.exact(query)
        keyword_score = 1.0
        if not keyword_hits:
            keyword_hits = keyword_index.search(query)
            keyword_score = 0.5
        vector_index = VectorIndex.rebuild(
            units,
            knowledge_version=self._knowledge_version,
            index_version=self._index_version,
            granted_permissions=self._granted_permissions,
        )
        vector_hits = vector_index.search(query, top_k=top_k)

        evidence_by_id: dict[str, Evidence] = {}
        fusion_scores: dict[str, float] = {}
        path_records: list[dict[str, JsonValue]] = []
        for rank, evidence in enumerate(keyword_hits, start=1):
            evidence_by_id[evidence.evidence_id] = evidence
            fusion_scores[evidence.evidence_id] = fusion_scores.get(
                evidence.evidence_id, 0.0
            ) + 1.0 / (_RRF_K + rank)
            path_records.append(
                {
                    "path": "keyword",
                    "evidence_id": evidence.evidence_id,
                    "rank": rank,
                    "score": keyword_score,
                }
            )
        for rank, hit in enumerate(vector_hits, start=1):
            evidence = hit.evidence
            evidence_by_id[evidence.evidence_id] = evidence
            fusion_scores[evidence.evidence_id] = fusion_scores.get(
                evidence.evidence_id, 0.0
            ) + 1.0 / (_RRF_K + rank)
            path_records.append(
                {
                    "path": "vector",
                    "evidence_id": evidence.evidence_id,
                    "rank": rank,
                    "score": hit.score,
                }
            )

        ordered_ids = sorted(
            fusion_scores,
            key=lambda evidence_id: (-fusion_scores[evidence_id], evidence_id),
        )[:top_k]
        rerank_summary: dict[str, JsonValue] = {
            "enabled": self._rerank_config.enabled,
            "applied": False,
            "reason": "DISABLED",
        }
        if self._rerank_config.enabled:
            candidates = tuple(
                evidence_by_id[evidence_id] for evidence_id in ordered_ids
            )
            assert self._reranker is not None
            reranked_ids = tuple(self._reranker(query, candidates))
            if len(reranked_ids) != len(ordered_ids) or set(reranked_ids) != set(
                ordered_ids
            ):
                raise ValueError("reranker must return each candidate evidence_id once")
            ordered_ids = list(reranked_ids)
            rerank_summary = {
                "enabled": True,
                "applied": True,
                "reason": "EVALUATION_RANKING_GAP",
                "evaluation_id": self._rerank_config.evaluation_id,
                "observed_ranking_metric": (
                    self._rerank_config.observed_ranking_metric
                ),
                "minimum_acceptable_metric": (
                    self._rerank_config.minimum_acceptable_metric
                ),
            }
        final = tuple(evidence_by_id[evidence_id] for evidence_id in ordered_ids)
        trace = Trace(
            trace_id=f"trace-{uuid4()}",
            raw_query=query,
            retrieval_query=normalize_query(query),
            candidate_evidence_ids=list(evidence_by_id),
            final_evidence_ids=ordered_ids,
            applied_filters=["permission", "status", "scope"],
            knowledge_version=self._knowledge_version,
            index_version=self._index_version,
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
            retrieval_paths=path_records,
            fusion_scores={key: value for key, value in fusion_scores.items()},
            rerank_summary=rerank_summary,
        )
        return HybridResult(final, trace)
