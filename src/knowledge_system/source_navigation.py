"""Deterministic document and section navigation over Knowledge Units."""

from __future__ import annotations

from collections.abc import Iterable

from knowledge_system.contracts import KnowledgeUnit


class SourceNavigator:
    """Navigate source structure without retrieval, ranking, or inference."""

    def __init__(self, units: Iterable[KnowledgeUnit]) -> None:
        self._units = tuple(units)

    def by_document(self, document: str) -> tuple[KnowledgeUnit, ...]:
        """Return units matching an exact document ID or complete source name."""

        return tuple(
            unit
            for unit in self._units
            if unit.lineage.document_id == document
            or unit.source_reference.name == document
        )

    def by_section(
        self,
        document: str,
        section: str,
        *,
        include_descendants: bool = True,
    ) -> tuple[KnowledgeUnit, ...]:
        """Return units at an exact section path, optionally including descendants."""

        prefix = f"{section} > "
        return tuple(
            unit
            for unit in self.by_document(document)
            if unit.source_reference.section == section
            or (
                include_descendants
                and unit.source_reference.section is not None
                and unit.source_reference.section.startswith(prefix)
            )
        )

