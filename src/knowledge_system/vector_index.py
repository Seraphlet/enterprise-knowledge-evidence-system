"""Lightweight, rebuildable local vector index for the V0 retrieval layer."""

from __future__ import annotations

import hashlib
import math
import unicodedata
from collections.abc import Iterable
from dataclasses import dataclass

from knowledge_system.candidate_filter import filter_candidates
from knowledge_system.contracts import Evidence, KnowledgeUnit
from knowledge_system.keyword_index import evidence_from_unit


@dataclass(frozen=True)
class VectorIndexInfo:
    """Index identity kept explicitly separate from knowledge identity."""

    index_version: str
    knowledge_version: str
    dimension: int


@dataclass(frozen=True)
class VectorHit:
    evidence: Evidence
    score: float


def _features(text: str) -> tuple[str, ...]:
    normalized = unicodedata.normalize("NFKC", text).casefold()
    compact = "".join(character for character in normalized if not character.isspace())
    if not compact:
        return ()
    if len(compact) < 3:
        return (compact,)
    return tuple(compact[index : index + 3] for index in range(len(compact) - 2))


def _embed(text: str, dimension: int) -> tuple[float, ...]:
    values = [0.0] * dimension
    for feature in _features(text):
        digest = hashlib.blake2b(feature.encode("utf-8"), digest_size=8).digest()
        number = int.from_bytes(digest, "big")
        index = number % dimension
        values[index] += -1.0 if number & 1 else 1.0
    magnitude = math.sqrt(sum(value * value for value in values))
    if magnitude:
        values = [value / magnitude for value in values]
    return tuple(values)


class VectorIndex:
    """In-memory derived index that can be deterministically rebuilt."""

    def __init__(
        self,
        *,
        info: VectorIndexInfo,
        entries: tuple[tuple[KnowledgeUnit, tuple[float, ...]], ...],
    ) -> None:
        self.info = info
        self._entries = entries

    @classmethod
    def rebuild(
        cls,
        units: Iterable[KnowledgeUnit],
        *,
        knowledge_version: str,
        index_version: str,
        granted_permissions: Iterable[str] = (),
        dimension: int = 128,
    ) -> "VectorIndex":
        """Rebuild all vectors after applying hard eligibility filters."""

        if dimension <= 0:
            raise ValueError("dimension must be positive")
        filtered = filter_candidates(
            units, granted_permissions=granted_permissions
        )
        unique: dict[str, KnowledgeUnit] = {}
        for unit in filtered.candidates:
            unique.setdefault(unit.unit_id, unit)
        entries = tuple(
            (unit, _embed(unit.semantic_content, dimension))
            for unit in unique.values()
        )
        return cls(
            info=VectorIndexInfo(
                index_version=index_version,
                knowledge_version=knowledge_version,
                dimension=dimension,
            ),
            entries=entries,
        )

    def __len__(self) -> int:
        return len(self._entries)

    def snapshot(self) -> tuple[tuple[str, tuple[float, ...]], ...]:
        """Return stable derived data for persistence or rebuild verification."""

        return tuple((unit.unit_id, vector) for unit, vector in self._entries)

    def search(self, query: str, *, top_k: int = 10) -> tuple[VectorHit, ...]:
        """Return vector-nearest eligible Evidence; no hybrid fusion or rerank."""

        if top_k <= 0:
            return ()
        query_vector = _embed(query, self.info.dimension)
        scored = [
            (
                sum(left * right for left, right in zip(query_vector, vector)),
                unit,
            )
            for unit, vector in self._entries
        ]
        scored = [item for item in scored if item[0] > 0.0]
        scored.sort(key=lambda item: (-item[0], item[1].unit_id))
        return tuple(
            VectorHit(evidence_from_unit(unit), score)
            for score, unit in scored[:top_k]
        )

