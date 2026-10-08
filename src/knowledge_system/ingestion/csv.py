"""Dependency-free CSV loader with ordered fields and physical line provenance."""

from __future__ import annotations

import csv as csv_module
import io
from dataclasses import dataclass
from pathlib import Path

from knowledge_system.contracts import ParseStatus, SourceReference


@dataclass(frozen=True)
class CsvField:
    """One positional header/value relation; never stored in a lossy mapping."""

    column_index: int
    header: str
    value: str


@dataclass(frozen=True)
class CsvRow:
    raw_content: str
    values: tuple[str, ...]
    fields: tuple[CsvField, ...]
    start_line: int
    end_line: int


@dataclass(frozen=True)
class CsvDocument:
    raw_csv: str
    headers: tuple[str, ...]
    rows: tuple[CsvRow, ...]
    source_reference: SourceReference
    parse_status: ParseStatus
    warnings: tuple[str, ...] = ()


def _raw_record(physical_lines: list[str], start_line: int, end_line: int) -> str:
    return "".join(physical_lines[start_line - 1 : end_line]).rstrip("\r\n")


def _header_warnings(headers: tuple[str, ...]) -> list[str]:
    warnings: list[str] = []
    empty = [str(index) for index, value in enumerate(headers, start=1) if not value.strip()]
    if empty:
        warnings.append(f"empty CSV header at columns {','.join(empty)}")
    positions: dict[str, list[int]] = {}
    for index, value in enumerate(headers, start=1):
        normalized = value.strip()
        if normalized:
            positions.setdefault(normalized, []).append(index)
    for value, indexes in positions.items():
        if len(indexes) > 1:
            warnings.append(
                f"duplicate CSV header {value!r} at columns "
                f"{','.join(str(index) for index in indexes)}"
            )
    return warnings


def parse_csv(raw_csv: str, *, name: str, url: str | None = None) -> CsvDocument:
    """Parse CSV rows while retaining order, raw records and physical line ranges."""

    source = SourceReference(name=name, url=url, page=None, sheet=None)
    physical_lines = raw_csv.splitlines(keepends=True)
    reader = csv_module.reader(io.StringIO(raw_csv, newline=""), strict=True)
    try:
        first = next(reader)
    except StopIteration:
        return CsvDocument(
            raw_csv=raw_csv,
            headers=(),
            rows=(),
            source_reference=source,
            parse_status=ParseStatus.FAILED,
            warnings=("CSV source is empty",),
        )

    headers = tuple(first)
    warnings = _header_warnings(headers)
    if not headers or not any(value.strip() for value in headers):
        return CsvDocument(
            raw_csv=raw_csv,
            headers=headers,
            rows=(),
            source_reference=source,
            parse_status=ParseStatus.FAILED,
            warnings=tuple((*warnings, "CSV header is completely unusable")),
        )

    rows: list[CsvRow] = []
    previous_end = reader.line_num
    try:
        for values_list in reader:
            start_line = previous_end + 1
            end_line = reader.line_num
            previous_end = end_line
            values = tuple(values_list)
            if not values or not any(value.strip() for value in values):
                warnings.append(
                    f"empty CSV data row skipped at lines {start_line}-{end_line}"
                )
                continue
            if len(values) != len(headers):
                warnings.append(
                    f"CSV column count mismatch at lines {start_line}-{end_line}: "
                    f"expected {len(headers)}, got {len(values)}"
                )
            fields = tuple(
                CsvField(index, headers[index], values[index])
                for index in range(min(len(headers), len(values)))
                if headers[index].strip()
            )
            if not fields:
                warnings.append(
                    f"CSV data row has no safely mapped fields at lines "
                    f"{start_line}-{end_line}; row skipped"
                )
                continue
            rows.append(
                CsvRow(
                    raw_content=_raw_record(physical_lines, start_line, end_line),
                    values=values,
                    fields=fields,
                    start_line=start_line,
                    end_line=end_line,
                )
            )
    except csv_module.Error:
        warnings.append(f"CSV syntax error at or before line {reader.line_num}")

    if not rows:
        status = ParseStatus.FAILED
        warnings.append("CSV source contains no usable data rows")
    elif warnings:
        status = ParseStatus.PARTIAL
    else:
        status = ParseStatus.SUCCESS
    return CsvDocument(
        raw_csv=raw_csv,
        headers=headers,
        rows=tuple(rows),
        source_reference=source,
        parse_status=status,
        warnings=tuple(dict.fromkeys(warnings)),
    )


def load_csv(
    path: str | Path, *, url: str | None = None, encoding: str = "utf-8-sig"
) -> CsvDocument:
    """Read UTF-8/UTF-8-SIG CSV using its full filename as source identity."""

    source_path = Path(path)
    return parse_csv(
        source_path.read_text(encoding=encoding), name=source_path.name, url=url
    )


__all__ = [
    "CsvDocument",
    "CsvField",
    "CsvRow",
    "load_csv",
    "parse_csv",
]
