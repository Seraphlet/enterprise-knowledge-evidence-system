"""Evidence-grounded Discover request over the existing Hybrid retrieval path."""

from __future__ import annotations

from dataclasses import dataclass, replace

from knowledge_system.contracts import Evidence, JsonValue, Trace
from knowledge_system.hybrid_retrieval import HybridRetrieval


@dataclass(frozen=True)
class DiscoverMatch:
    """A discovered Evidence item with a formal name copied from its source."""

    formal_name: str
    formal_name_source: str
    evidence: Evidence


@dataclass(frozen=True)
class DiscoverResult:
    matches: tuple[DiscoverMatch, ...]
    trace: Trace


def _formal_name(evidence: Evidence) -> tuple[str, str]:
    source = evidence.source_reference
    if source.section and source.section.strip():
        return source.section, "source_reference.section"
    return source.name, "source_reference.name"


class DiscoverRetrieval:
    """Discover Knowledge Units without generating or rewriting formal names."""

    def __init__(self, hybrid_retrieval: HybridRetrieval) -> None:
        self._hybrid_retrieval = hybrid_retrieval

    def discover(
        self,
        query: str,
        *,
        scope: str | None = None,
        top_k: int = 10,
    ) -> DiscoverResult:
        hybrid = self._hybrid_retrieval.search(query, scope=scope, top_k=top_k)
        matches = tuple(
            DiscoverMatch(*_formal_name(evidence), evidence)
            for evidence in hybrid.evidence
        )
        formal_names: list[dict[str, JsonValue]] = [
            {
                "evidence_id": match.evidence.evidence_id,
                "formal_name": match.formal_name,
                "source_field": match.formal_name_source,
            }
            for match in matches
        ]
        trace = replace(
            hybrid.trace,
            discover_summary={
                "request": "discover",
                "formal_names": formal_names,
            },
        )
        return DiscoverResult(matches, trace)
