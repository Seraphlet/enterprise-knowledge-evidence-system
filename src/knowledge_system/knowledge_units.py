"""Structure-aware conversion from parsed documents to knowledge units."""

from __future__ import annotations

import hashlib
import json
from dataclasses import replace

from knowledge_system.contracts import KnowledgeLineage, KnowledgeUnit, SourceReference
from knowledge_system.ingestion import (
    CsvDocument,
    CsvRow,
    HtmlBlock,
    HtmlDocument,
    MarkdownBlock,
    MarkdownDocument,
)


PARSER_VERSION = "html-stdlib-v1"
CHUNKER_VERSION = "html-structure-v1"
PROCESSING_VERSION = "knowledge-unit-v1"
MARKDOWN_PARSER_VERSION = "markdown-lines-v1"
MARKDOWN_CHUNKER_VERSION = "markdown-structure-v1"
MARKDOWN_PROCESSING_VERSION = "knowledge-unit-v1"
CSV_PARSER_VERSION = "csv-stdlib-v1"
CSV_CHUNKER_VERSION = "csv-row-v1"
CSV_PROCESSING_VERSION = "knowledge-unit-v1"


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


def _make_markdown_unit(
    document: MarkdownDocument,
    block: MarkdownBlock,
    *,
    document_id: str,
) -> KnowledgeUnit:
    section = _section_name(block.section_path)
    source: SourceReference = replace(document.source_reference, section=section)
    parent_section_id = (
        _stable_id("section", document_id, block.section_path)
        if block.section_path
        else None
    )
    position = f"lines:{block.start_line}-{block.end_line}"
    metadata: dict[str, object] = {
        "structure_kind": block.kind,
        "section_path": list(block.section_path),
        "start_line": block.start_line,
        "end_line": block.end_line,
    }
    if block.info is not None:
        metadata["code_info"] = block.info
    context = " > ".join(block.section_path)
    semantic_content = (
        f"{context}\n{block.text}" if context and block.text != context else block.text
    )
    return KnowledgeUnit(
        unit_id=_stable_id(
            "unit",
            document_id,
            block.kind,
            block.section_path,
            position,
            block.raw_content,
        ),
        original_content=block.raw_content,
        semantic_content=semantic_content,
        metadata=metadata,  # type: ignore[arg-type]
        source_reference=source,
        lineage=KnowledgeLineage(
            document_id=document_id,
            source_position=position,
            parent_section_id=parent_section_id,
            parser_version=MARKDOWN_PARSER_VERSION,
            chunker_version=MARKDOWN_CHUNKER_VERSION,
            processing_version=MARKDOWN_PROCESSING_VERSION,
        ),
        parse_status=document.parse_status,
    )


def knowledge_units_from_markdown(
    document: MarkdownDocument,
) -> list[KnowledgeUnit]:
    """Create one stable unit per ordered Markdown structural block."""

    if not document.original_text:
        return []
    document_id = _stable_id(
        "document",
        "markdown",
        document.source_reference.name,
        document.source_reference.url,
    )
    return [
        _make_markdown_unit(document, block, document_id=document_id)
        for block in document.blocks
        if block.raw_content.strip()
    ]


def _make_csv_unit(
    document: CsvDocument,
    row: CsvRow,
    *,
    document_id: str,
) -> KnowledgeUnit:
    position = f"lines:{row.start_line}-{row.end_line}"
    source = replace(document.source_reference, section=f"CSV row {row.start_line}")
    fields = [
        {
            "column_index": field.column_index,
            "header": field.header,
            "value": field.value,
        }
        for field in row.fields
    ]
    semantic_content = "\n".join(
        f"{field.header}: {field.value}" for field in row.fields
    )
    return KnowledgeUnit(
        unit_id=_stable_id(
            "unit", document_id, position, document.headers, row.values
        ),
        original_content=row.raw_content,
        semantic_content=semantic_content,
        metadata={
            "structure_kind": "csv_row",
            "headers": list(document.headers),
            "values": list(row.values),
            "fields": fields,
            "start_line": row.start_line,
            "end_line": row.end_line,
        },
        source_reference=source,
        lineage=KnowledgeLineage(
            document_id=document_id,
            source_position=position,
            parent_section_id=None,
            parser_version=CSV_PARSER_VERSION,
            chunker_version=CSV_CHUNKER_VERSION,
            processing_version=CSV_PROCESSING_VERSION,
        ),
        parse_status=document.parse_status,
    )


def knowledge_units_from_csv(document: CsvDocument) -> list[KnowledgeUnit]:
    """Create one stable, ordered knowledge unit for each reliable CSV row."""

    if not document.rows:
        return []
    document_id = _stable_id(
        "document",
        "csv",
        document.source_reference.name,
        document.source_reference.url,
        document.headers,
    )
    return [
        _make_csv_unit(document, row, document_id=document_id)
        for row in document.rows
    ]
