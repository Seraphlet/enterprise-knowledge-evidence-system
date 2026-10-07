"""Dependency-free HTML loader/parser that preserves useful source structure."""

from __future__ import annotations

from dataclasses import dataclass
from html.parser import HTMLParser
from pathlib import Path

from knowledge_system.contracts import ParseStatus, SourceReference


@dataclass(frozen=True)
class HtmlSection:
    """A heading and its structural ancestry."""

    level: int
    title: str
    path: tuple[str, ...]


@dataclass(frozen=True)
class HtmlTable:
    """A table retaining header-to-cell relationships by column order."""

    headers: tuple[str, ...]
    rows: tuple[tuple[str, ...], ...]


@dataclass(frozen=True)
class HtmlBlock:
    """An ordered structural block used for structure-aware unit creation."""

    kind: str
    text: str
    section_path: tuple[str, ...]
    table_index: int | None = None


@dataclass(frozen=True)
class HtmlDocument:
    """Parsed HTML plus the untouched source and a traceable identity."""

    raw_html: str
    original_text: str
    title: str | None
    sections: tuple[HtmlSection, ...]
    tables: tuple[HtmlTable, ...]
    blocks: tuple[HtmlBlock, ...]
    source_reference: SourceReference
    parse_status: ParseStatus
    warnings: tuple[str, ...] = ()


def _clean(parts: list[str]) -> str:
    return " ".join("".join(parts).split())


class _StructureParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.text_parts: list[str] = []
        self.title_parts: list[str] = []
        self.heading_level: int | None = None
        self.heading_parts: list[str] = []
        self.sections: list[HtmlSection] = []
        self.heading_ancestry: list[str] = []
        self.tables: list[HtmlTable] = []
        self.blocks: list[HtmlBlock] = []
        self.table_rows: list[tuple[tuple[str, ...], tuple[bool, ...]]] | None = None
        self.row_cells: list[str] | None = None
        self.row_header_flags: list[bool] | None = None
        self.cell_parts: list[str] | None = None
        self.cell_is_header = False
        self.block_tag: str | None = None
        self.block_parts: list[str] = []
        self.in_title = False
        self.ignored_depth = 0
        self.warnings: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        tag = tag.lower()
        if tag in {"script", "style"}:
            self.ignored_depth += 1
        elif tag == "title":
            self.in_title = True
        elif len(tag) == 2 and tag[0] == "h" and tag[1].isdigit():
            if self.heading_level is not None:
                self.warnings.append("nested or unclosed heading")
            self.heading_level = int(tag[1])
            self.heading_parts = []
        elif tag == "table":
            if self.table_rows is not None:
                self.warnings.append("nested or unclosed table")
            self.table_rows = []
        elif tag == "tr" and self.table_rows is not None:
            self.row_cells = []
            self.row_header_flags = []
        elif tag in {"th", "td"} and self.row_cells is not None:
            self.cell_parts = []
            self.cell_is_header = tag == "th"
        elif tag in {"p", "li", "pre", "blockquote"} and self.table_rows is None:
            self.block_tag = tag
            self.block_parts = []

    def handle_endtag(self, tag: str) -> None:
        tag = tag.lower()
        if tag in {"script", "style"}:
            self.ignored_depth = max(0, self.ignored_depth - 1)
        elif tag == "title":
            self.in_title = False
        elif (
            len(tag) == 2
            and tag[0] == "h"
            and tag[1].isdigit()
            and self.heading_level is not None
        ):
            title = _clean(self.heading_parts)
            if title:
                level = self.heading_level
                self.heading_ancestry = self.heading_ancestry[: level - 1]
                self.heading_ancestry.append(title)
                self.sections.append(
                    HtmlSection(level=level, title=title, path=tuple(self.heading_ancestry))
                )
                self.blocks.append(
                    HtmlBlock(
                        kind="heading",
                        text=title,
                        section_path=tuple(self.heading_ancestry),
                    )
                )
            self.heading_level = None
            self.heading_parts = []
        elif tag in {"th", "td"} and self.cell_parts is not None:
            self.row_cells.append(_clean(self.cell_parts))  # type: ignore[union-attr]
            self.row_header_flags.append(self.cell_is_header)  # type: ignore[union-attr]
            self.cell_parts = None
        elif tag == "tr" and self.row_cells is not None and self.table_rows is not None:
            self.table_rows.append((tuple(self.row_cells), tuple(self.row_header_flags or [])))
            self.row_cells = None
            self.row_header_flags = None
        elif tag == "table" and self.table_rows is not None:
            headers: tuple[str, ...] = ()
            rows: list[tuple[str, ...]] = []
            for cells, flags in self.table_rows:
                if not headers and cells and all(flags):
                    headers = cells
                elif cells:
                    rows.append(cells)
            table = HtmlTable(headers=headers, rows=tuple(rows))
            self.tables.append(table)
            table_text = "\n".join(
                [" | ".join(headers)] if headers else []
            )
            row_text = "\n".join(" | ".join(row) for row in rows)
            combined = "\n".join(part for part in (table_text, row_text) if part)
            self.blocks.append(
                HtmlBlock(
                    kind="table",
                    text=combined,
                    section_path=tuple(self.heading_ancestry),
                    table_index=len(self.tables) - 1,
                )
            )
            self.table_rows = None
        elif tag == self.block_tag:
            text = _clean(self.block_parts)
            if text:
                self.blocks.append(
                    HtmlBlock(
                        kind=self.block_tag,
                        text=text,
                        section_path=tuple(self.heading_ancestry),
                    )
                )
            self.block_tag = None
            self.block_parts = []

    def handle_data(self, data: str) -> None:
        if self.ignored_depth:
            return
        if data.strip():
            self.text_parts.append(data)
        if self.in_title:
            self.title_parts.append(data)
        if self.heading_level is not None:
            self.heading_parts.append(data)
        if self.cell_parts is not None:
            self.cell_parts.append(data)
        if self.block_tag is not None:
            self.block_parts.append(data)

    def structural_warnings(self) -> tuple[str, ...]:
        warnings = list(self.warnings)
        if self.heading_level is not None:
            warnings.append("unclosed heading")
        if self.table_rows is not None:
            warnings.append("unclosed table")
        if self.row_cells is not None:
            warnings.append("unclosed table row")
        if self.cell_parts is not None:
            warnings.append("unclosed table cell")
        if self.block_tag is not None:
            warnings.append(f"unclosed {self.block_tag} block")
        return tuple(dict.fromkeys(warnings))


def parse_html(raw_html: str, *, name: str, url: str | None = None) -> HtmlDocument:
    """Parse HTML while retaining the exact input and reliable source identity."""

    parser = _StructureParser()
    parser.feed(raw_html)
    parser.close()
    original_text = _clean(parser.text_parts)
    warnings = parser.structural_warnings()
    if not original_text:
        status = ParseStatus.FAILED
        warnings = (*warnings, "document contains no readable text")
    elif warnings:
        status = ParseStatus.PARTIAL
    else:
        status = ParseStatus.SUCCESS
    title = _clean(parser.title_parts) or (
        parser.sections[0].title if parser.sections else None
    )
    return HtmlDocument(
        raw_html=raw_html,
        original_text=original_text,
        title=title,
        sections=tuple(parser.sections),
        tables=tuple(parser.tables),
        blocks=tuple(parser.blocks),
        source_reference=SourceReference(name=name, url=url, page=None),
        parse_status=status,
        warnings=warnings,
    )


def load_html(
    path: str | Path, *, url: str | None = None, encoding: str = "utf-8-sig"
) -> HtmlDocument:
    """Load an HTML file and parse it using its full filename as source identity."""

    source_path = Path(path)
    return parse_html(
        source_path.read_text(encoding=encoding), name=source_path.name, url=url
    )
