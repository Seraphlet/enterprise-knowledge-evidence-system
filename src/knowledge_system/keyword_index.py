"""Deterministic exact/keyword index over Knowledge Units."""

from __future__ import annotations

import json
import unicodedata
from collections.abc import Iterable

from knowledge_system.contracts import Evidence, KnowledgeUnit


def normalize_query(value: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", value).casefold().split())


def _search_text(unit: KnowledgeUnit) -> str:
    fields = (
        unit.source_reference.name,
        unit.source_reference.section or "",
        unit.original_content,
        unit.semantic_content,
        json.dumps(unit.metadata, ensure_ascii=False, sort_keys=True),
    )
    return normalize_query("\n".join(fields))


def evidence_from_unit(unit: KnowledgeUnit) -> Evidence:
    return Evidence(
        evidence_id=f"evidence-{unit.unit_id}",
        unit_id=unit.unit_id,
        original_content=unit.original_content,
        source_reference=unit.source_reference,
        lineage=unit.lineage,
        derived_structure=unit.rule_structure,
    )


class KeywordIndex:
    """Immutable baseline index using normalized exact substring matching."""

    def __init__(self, units: Iterable[KnowledgeUnit]) -> None:
        unique_units: dict[str, KnowledgeUnit] = {}
        for unit in units:
            unique_units.setdefault(unit.unit_id, unit)
        self._entries = tuple(
            (unit, _search_text(unit)) for unit in unique_units.values()
        )

    def search(self, query: str) -> tuple[Evidence, ...]:
        """Return source-ordered Evidence containing the normalized query."""

        normalized_query = normalize_query(query)
        if not normalized_query:
            return ()
        return tuple(
            evidence_from_unit(unit)
            for unit, text in self._entries
            if normalized_query in text
        )

    def exact(self, identity: str) -> tuple[Evidence, ...]:
        """Match a complete document name, section path, or section title."""

        normalized_identity = normalize_query(identity)
        if not normalized_identity:
            return ()
        matches: list[Evidence] = []
        for unit, _ in self._entries:
            source = unit.source_reference
            section = source.section or ""
            section_title = section.rsplit(" > ", 1)[-1]
            identities = {
                normalize_query(source.name),
                normalize_query(section),
                normalize_query(section_title),
            }
            if normalized_identity in identities:
                matches.append(evidence_from_unit(unit))
        return tuple(matches)
