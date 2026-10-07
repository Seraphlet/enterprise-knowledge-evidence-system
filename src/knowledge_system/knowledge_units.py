"""Structure-aware conversion from parsed documents to knowledge units."""

from __future__ import annotations

import hashlib
import json
from dataclasses import replace

from knowledge_system.contracts import KnowledgeLineage, KnowledgeUnit, SourceReference
from knowledge_system.ingestion import HtmlBlock, HtmlDocument


PARSER_VERSION = "html-stdlib-v1"
CHUNKER_VERSION = "html-structure-v1"
PROCESSING_VERSION = "knowledge-unit-v1"


def _stable_id(prefix: str, *parts: object) -> str:
    canonical = json.dumps(parts, ensure_ascii=False, separators=(",", ":"))
    digest = hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:24]
    return f"{prefix}-{digest}"


def _section_name(path: tuple[str, ...]) -> str | None:
    return " > ".join(path) if path else None


def _semantic_content(
    document_title: str | None, path: tuple[str, ...], content: str
) -> str:
    context = [part for part in (document_title, *path) if part]
    context_text = " > ".join(dict.fromkeys(context))
    return f"{context_text}\n{content}" if context_text else content


def _make_unit(
    document: HtmlDocument,
    *,
    document_id: str,
    kind: str,
    path: tuple[str, ...],
    position: str,
    original_content: str,
) -> KnowledgeUnit:
    section = _section_name(path)
    source: SourceReference = replace(document.source_reference, section=section)
    parent_section_id = (
        _stable_id("section", document_id, path) if path else None
    )
    unit_id = _stable_id(
        "unit", document_id, kind, path, position, original_content
    )
    return KnowledgeUnit(
        unit_id=unit_id,
        original_content=original_content,
        semantic_content=_semantic_content(document.title, path, original_content),
        metadata={"structure_kind": kind, "section_path": list(path)},
        source_reference=source,
        lineage=KnowledgeLineage(
            document_id=document_id,
            source_position=position,
            parent_section_id=parent_section_id,
            parser_version=PARSER_VERSION,
            chunker_version=CHUNKER_VERSION,
            processing_version=PROCESSING_VERSION,
        ),
        parse_status=document.parse_status,
    )


def knowledge_units_from_html(document: HtmlDocument) -> list[KnowledgeUnit]:
    """Split HTML on semantic structure boundaries, never on character counts."""

    if not document.original_text:
        return []
    document_id = _stable_id(
        "document", document.source_reference.name, document.source_reference.url
    )
    units: list[KnowledgeUnit] = []
    pending: list[HtmlBlock] = []
    pending_path: tuple[str, ...] = ()
    start_index = 0

    def flush() -> None:
        nonlocal pending, start_index
        if not pending:
            return
        content = "\n".join(block.text for block in pending if block.text)
        if content:
            units.append(
                _make_unit(
                    document,
                    document_id=document_id,
                    kind="section",
                    path=pending_path,
                    position=f"blocks:{start_index}-{start_index + len(pending) - 1}",
                    original_content=content,
                )
            )
        pending = []

    for index, block in enumerate(document.blocks):
        if block.kind == "table":
            flush()
            units.append(
                _make_unit(
                    document,
                    document_id=document_id,
                    kind="table",
                    path=block.section_path,
                    position=f"table:{block.table_index}",
                    original_content=block.text,
                )
            )
            continue
        if block.kind == "heading":
            flush()
            pending_path = block.section_path
            start_index = index
            pending = [block]
            continue
        if pending and block.section_path != pending_path:
            flush()
        if not pending:
            pending_path = block.section_path
            start_index = index
        pending.append(block)
    flush()

    if not units:
        units.append(
            _make_unit(
                document,
                document_id=document_id,
                kind="document",
                path=(),
                position="document",
                original_content=document.original_text,
            )
        )
    return units

