"""Dependency-free Markdown loader preserving source and structural blocks."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

from knowledge_system.contracts import ParseStatus, SourceReference


_HEADING = re.compile(
    r"^[ ]{0,3}(#{1,6})[ \t]+(.+?)(?:[ \t]+#+[ \t]*)?$"
)
_FENCE = re.compile(r"^[ ]{0,3}(`{3,}|~{3,})(.*)$")
_LIST_ITEM = re.compile(r"^[ ]{0,3}(?:[-+*]|\d+[.)])[ \t]+(.*)$")
_QUOTE = re.compile(r"^[ ]{0,3}>[ \t]?(.*)$")
_TABLE_SEPARATOR_CELL = re.compile(r"^:?-{3,}:?$")


@dataclass(frozen=True)
class MarkdownHeading:
    level: int
    title: str
    path: tuple[str, ...]
    line: int


@dataclass(frozen=True)
class MarkdownBlock:
    """One ordered Markdown structure with both source and readable forms."""

    kind: str
    raw_content: str
    text: str
    section_path: tuple[str, ...]
    start_line: int
    end_line: int
    info: str | None = None


@dataclass(frozen=True)
class MarkdownDocument:
    raw_markdown: str
    original_text: str
    title: str | None
    headings: tuple[MarkdownHeading, ...]
    blocks: tuple[MarkdownBlock, ...]
    source_reference: SourceReference
    parse_status: ParseStatus
    warnings: tuple[str, ...] = ()


def _table_cells(line: str) -> tuple[str, ...]:
    value = line.strip()
    if value.startswith("|"):
        value = value[1:]
    if value.endswith("|"):
        value = value[:-1]
    return tuple(cell.strip() for cell in value.split("|"))


def _is_table_separator(line: str) -> bool:
    cells = _table_cells(line)
    return len(cells) >= 2 and all(
        bool(_TABLE_SEPARATOR_CELL.fullmatch(cell)) for cell in cells
    )


def _is_table_start(lines: list[str], index: int) -> bool:
    return (
        index + 1 < len(lines)
        and "|" in lines[index]
        and _is_table_separator(lines[index + 1])
        and len(_table_cells(lines[index])) == len(_table_cells(lines[index + 1]))
    )


def _is_structure_start(lines: list[str], index: int) -> bool:
    line = lines[index]
    return bool(
        _HEADING.match(line)
        or _FENCE.match(line)
        or _LIST_ITEM.match(line)
        or _QUOTE.match(line)
        or _is_table_start(lines, index)
    )


def parse_markdown(
    raw_markdown: str, *, name: str, url: str | None = None
) -> MarkdownDocument:
    """Parse supported Markdown structures without rewriting the source."""

    lines = raw_markdown.splitlines()
    blocks: list[MarkdownBlock] = []
    headings: list[MarkdownHeading] = []
    warnings: list[str] = []
    ancestry: list[str] = []
    index = 0

    while index < len(lines):
        line = lines[index]
        if not line.strip():
            index += 1
            continue

        fence = _FENCE.match(line)
        if fence:
            start = index
            marker = fence.group(1)
            marker_char = marker[0]
            marker_length = len(marker)
            info = fence.group(2).strip() or None
            index += 1
            content_lines: list[str] = []
            closed = False
            while index < len(lines):
                candidate = lines[index].strip()
                if (
                    candidate
                    and set(candidate) == {marker_char}
                    and len(candidate) >= marker_length
                ):
                    closed = True
                    index += 1
                    break
                content_lines.append(lines[index])
                index += 1
            if not closed:
                warnings.append(f"unclosed fenced code block at line {start + 1}")
            blocks.append(
                MarkdownBlock(
                    kind="code",
                    raw_content="\n".join(lines[start:index]),
                    text="\n".join(content_lines),
                    section_path=tuple(ancestry),
                    start_line=start + 1,
                    end_line=index,
                    info=info,
                )
            )
            continue

        heading = _HEADING.match(line)
        if heading:
            level = len(heading.group(1))
            title = heading.group(2).strip()
            ancestry = ancestry[: level - 1]
            ancestry.append(title)
            path = tuple(ancestry)
            headings.append(MarkdownHeading(level, title, path, index + 1))
            blocks.append(
                MarkdownBlock(
                    kind="heading",
                    raw_content=line,
                    text=title,
                    section_path=path,
                    start_line=index + 1,
                    end_line=index + 1,
                )
            )
            index += 1
            continue

        if _is_table_start(lines, index):
            start = index
            table_lines = [lines[index], lines[index + 1]]
            index += 2
            while index < len(lines) and lines[index].strip() and "|" in lines[index]:
                table_lines.append(lines[index])
                index += 1
            readable_rows = [" | ".join(_table_cells(table_lines[0]))]
            readable_rows.extend(
                " | ".join(_table_cells(row)) for row in table_lines[2:]
            )
            blocks.append(
                MarkdownBlock(
                    kind="table",
                    raw_content="\n".join(table_lines),
                    text="\n".join(readable_rows),
                    section_path=tuple(ancestry),
                    start_line=start + 1,
                    end_line=index,
                )
            )
            continue

        list_item = _LIST_ITEM.match(line)
        if list_item:
            start = index
            raw_lines: list[str] = []
            readable: list[str] = []
            while index < len(lines) and lines[index].strip():
                match = _LIST_ITEM.match(lines[index])
                if match:
                    raw_lines.append(lines[index])
                    readable.append(match.group(1).strip())
                    index += 1
                    continue
                if lines[index].startswith(("    ", "\t")):
                    raw_lines.append(lines[index])
                    readable.append(lines[index].strip())
                    index += 1
                    continue
                break
            blocks.append(
                MarkdownBlock(
                    kind="list",
                    raw_content="\n".join(raw_lines),
                    text="\n".join(readable),
                    section_path=tuple(ancestry),
                    start_line=start + 1,
                    end_line=index,
                )
            )
            continue

        quote = _QUOTE.match(line)
        if quote:
            start = index
            raw_lines = []
            readable = []
            while index < len(lines):
                match = _QUOTE.match(lines[index])
                if not match:
                    break
                raw_lines.append(lines[index])
                readable.append(match.group(1))
                index += 1
            blocks.append(
                MarkdownBlock(
                    kind="quote",
                    raw_content="\n".join(raw_lines),
                    text="\n".join(readable),
                    section_path=tuple(ancestry),
                    start_line=start + 1,
                    end_line=index,
                )
            )
            continue

        start = index
        paragraph: list[str] = []
        while index < len(lines) and lines[index].strip():
            if index != start and _is_structure_start(lines, index):
                break
            paragraph.append(lines[index])
            index += 1
        blocks.append(
            MarkdownBlock(
                kind="paragraph",
                raw_content="\n".join(paragraph),
                text=" ".join(part.strip() for part in paragraph),
                section_path=tuple(ancestry),
                start_line=start + 1,
                end_line=index,
            )
        )

    readable_parts = [block.text for block in blocks if block.text.strip()]
    original_text = "\n".join(readable_parts)
    if not original_text.strip():
        status = ParseStatus.FAILED
        warnings.append("document contains no readable content")
    elif warnings:
        status = ParseStatus.PARTIAL
    else:
        status = ParseStatus.SUCCESS
    return MarkdownDocument(
        raw_markdown=raw_markdown,
        original_text=original_text,
        title=headings[0].title if headings else None,
        headings=tuple(headings),
        blocks=tuple(blocks),
        source_reference=SourceReference(name=name, url=url, page=None, sheet=None),
        parse_status=status,
        warnings=tuple(dict.fromkeys(warnings)),
    )


def load_markdown(
    path: str | Path, *, url: str | None = None, encoding: str = "utf-8-sig"
) -> MarkdownDocument:
    """Read UTF-8/UTF-8-SIG Markdown using its full filename as identity."""

    source_path = Path(path)
    return parse_markdown(
        source_path.read_text(encoding=encoding), name=source_path.name, url=url
    )


__all__ = [
    "MarkdownBlock",
    "MarkdownDocument",
    "MarkdownHeading",
    "load_markdown",
    "parse_markdown",
]
