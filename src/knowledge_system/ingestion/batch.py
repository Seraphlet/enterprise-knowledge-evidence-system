"""Deterministic batch ingestion and atomic canonical corpus output."""

from __future__ import annotations

import json
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from knowledge_system.contracts import KnowledgeUnit, ParseStatus
from knowledge_system.ingestion.router import (
    IngestionIssue,
    IngestionResult,
    identify_format,
    ingest_file,
)


BATCH_SCHEMA = "knowledge-system.ingest-result"
BATCH_SCHEMA_VERSION = 1


class BatchIngestionError(ValueError):
    """Stable input/write failure safe for the CLI boundary."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code
        self.message = message


@dataclass(frozen=True)
class SourceIngestionSummary:
    source_path: str
    status: ParseStatus
    source_format: str | None
    unit_count: int
    warnings: tuple[IngestionIssue, ...]
    errors: tuple[IngestionIssue, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "source_path": self.source_path,
            "status": self.status.value,
            "source_format": self.source_format,
            "unit_count": self.unit_count,
            "warnings": [item.to_dict() for item in self.warnings],
            "errors": [item.to_dict() for item in self.errors],
        }


@dataclass(frozen=True)
class BatchIngestionResult:
    status: ParseStatus
    sources: tuple[SourceIngestionSummary, ...]
    units: tuple[KnowledgeUnit, ...]
    input_file_count: int
    written_unit_count: int
    deduplicated_unit_count: int
    output_written: bool

    def to_dict(self) -> dict[str, Any]:
        counts = {
            status.value: sum(1 for item in self.sources if item.status is status)
            for status in ParseStatus
        }
        return {
            "schema": BATCH_SCHEMA,
            "schema_version": BATCH_SCHEMA_VERSION,
            "status": self.status.value,
            "input_file_count": self.input_file_count,
            "written_unit_count": self.written_unit_count,
            "deduplicated_unit_count": self.deduplicated_unit_count,
            "output_written": self.output_written,
            "source_status_counts": counts,
            "sources": [item.to_dict() for item in self.sources],
        }


def _sort_key(root: Path, path: Path) -> tuple[str, str]:
    relative = path.relative_to(root).as_posix()
    return relative.casefold(), relative


def discover_source_files(source: str | Path) -> tuple[Path, ...]:
    """Discover only supported files with stable, case-insensitive path order."""

    root = Path(source)
    if not root.exists():
        raise BatchIngestionError("SOURCE_NOT_FOUND", "Import source does not exist.")
    if root.is_file():
        if identify_format(root) is None:
            raise BatchIngestionError(
                "UNSUPPORTED_SOURCE_FILE",
                "Import source file extension is not supported.",
            )
        return (root,)
    if not root.is_dir():
        raise BatchIngestionError(
            "INVALID_SOURCE_TYPE", "Import source must be a file or directory."
        )
    try:
        files = tuple(
            sorted(
                (
                    path
                    for path in root.rglob("*")
                    if path.is_file() and identify_format(path) is not None
                ),
                key=lambda path: _sort_key(root, path),
            )
        )
    except OSError as error:
        raise BatchIngestionError(
            "SOURCE_DIRECTORY_READ_ERROR", "Import source directory could not be read."
        ) from error
    if not files:
        raise BatchIngestionError(
            "NO_SUPPORTED_SOURCE_FILES",
            "Import source directory contains no supported files.",
        )
    return files


def _display_path(source: Path, path: Path) -> str:
    return path.relative_to(source).as_posix() if source.is_dir() else path.name


def _canonical_unit(unit: KnowledgeUnit) -> str:
    return json.dumps(
        unit.to_dict(), ensure_ascii=False, sort_keys=True, separators=(",", ":")
    )


def ingest_sources(
    source: str | Path, *, _files: tuple[Path, ...] | None = None
) -> BatchIngestionResult:
    """Ingest and strictly merge one file or a deterministic directory walk."""

    source_root = Path(source)
    files = _files if _files is not None else discover_source_files(source_root)
    summaries: list[SourceIngestionSummary] = []
    units: list[KnowledgeUnit] = []
    by_id: dict[str, str] = {}
    duplicate_count = 0

    for path in files:
        result: IngestionResult = ingest_file(path)
        summaries.append(
            SourceIngestionSummary(
                source_path=_display_path(source_root, path),
                status=result.status,
                source_format=(
                    result.source_format.value
                    if result.source_format is not None
                    else None
                ),
                unit_count=len(result.units),
                warnings=result.warnings,
                errors=result.errors,
            )
        )
        for unit in result.units:
            canonical = _canonical_unit(unit)
            existing = by_id.get(unit.unit_id)
            if existing is None:
                by_id[unit.unit_id] = canonical
                units.append(unit)
            elif existing == canonical:
                duplicate_count += 1
            else:
                raise BatchIngestionError(
                    "UNIT_ID_CONFLICT",
                    "The same unit_id refers to different knowledge content or provenance.",
                )

    if not units:
        status = ParseStatus.FAILED
    elif any(item.status is not ParseStatus.SUCCESS for item in summaries):
        status = ParseStatus.PARTIAL
    else:
        status = ParseStatus.SUCCESS
    return BatchIngestionResult(
        status=status,
        sources=tuple(summaries),
        units=tuple(units),
        input_file_count=len(files),
        written_unit_count=len(units),
        deduplicated_unit_count=duplicate_count,
        output_written=False,
    )


def canonical_corpus_json(units: tuple[KnowledgeUnit, ...]) -> str:
    """Serialize the canonical KnowledgeUnit array accepted by query loading."""

    return json.dumps(
        [unit.to_dict() for unit in units],
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ) + "\n"


def write_corpus_atomic(
    units: tuple[KnowledgeUnit, ...], output: str | Path
) -> None:
    """Write in the target directory and replace only after a complete fsync."""

    target = Path(output)
    parent = target.parent
    if not parent.exists() or not parent.is_dir():
        raise BatchIngestionError(
            "OUTPUT_DIRECTORY_NOT_FOUND", "Output directory does not exist."
        )
    try:
        payload = canonical_corpus_json(units).encode("utf-8")
    except (TypeError, ValueError) as error:
        raise BatchIngestionError(
            "CORPUS_SERIALIZATION_ERROR", "Knowledge corpus could not be serialized."
        ) from error

    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            prefix=f".{target.name}.",
            suffix=".tmp",
            dir=parent,
            delete=False,
        ) as handle:
            temporary = Path(handle.name)
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, target)
        temporary = None
    except OSError as error:
        raise BatchIngestionError(
            "CORPUS_WRITE_ERROR", "Knowledge corpus could not be written atomically."
        ) from error
    finally:
        if temporary is not None:
            try:
                temporary.unlink(missing_ok=True)
            except OSError:
                pass


def ingest_to_corpus(
    source: str | Path, output: str | Path
) -> BatchIngestionResult:
    """Ingest sources and atomically write only a non-empty reliable corpus."""

    files = discover_source_files(source)
    target = Path(output)
    try:
        target_identity = target.resolve(strict=False)
        source_identities = {path.resolve(strict=False) for path in files}
    except OSError as error:
        raise BatchIngestionError(
            "PATH_RESOLUTION_ERROR", "Import or output path could not be resolved."
        ) from error
    if target_identity in source_identities:
        raise BatchIngestionError(
            "OUTPUT_OVERLAPS_SOURCE", "Output must not overwrite an input source file."
        )

    result = ingest_sources(source, _files=files)
    if result.status is ParseStatus.FAILED or not result.units:
        return result
    write_corpus_atomic(result.units, target)
    return BatchIngestionResult(
        status=result.status,
        sources=result.sources,
        units=result.units,
        input_file_count=result.input_file_count,
        written_unit_count=result.written_unit_count,
        deduplicated_unit_count=result.deduplicated_unit_count,
        output_written=True,
    )


def format_batch_text(result: BatchIngestionResult) -> str:
    lines = [
        f"Status: {result.status.value}",
        f"Input files: {result.input_file_count}",
        f"Written units: {result.written_unit_count}",
        f"Deduplicated units: {result.deduplicated_unit_count}",
        f"Output written: {'YES' if result.output_written else 'NO'}",
    ]
    lines.extend(
        f"[{item.status.value}] {item.source_path} ({item.unit_count} units)"
        for item in result.sources
    )
    return "\n".join(lines)


def batch_diagnostics(result: BatchIngestionResult) -> tuple[dict[str, Any], ...]:
    diagnostics: list[dict[str, Any]] = []
    for source in result.sources:
        for severity, issues in (("warning", source.warnings), ("error", source.errors)):
            diagnostics.extend(
                {
                    "source_path": source.source_path,
                    "severity": severity,
                    **issue.to_dict(),
                }
                for issue in issues
            )
    return tuple(diagnostics)


__all__ = [
    "BATCH_SCHEMA",
    "BATCH_SCHEMA_VERSION",
    "BatchIngestionError",
    "BatchIngestionResult",
    "SourceIngestionSummary",
    "batch_diagnostics",
    "canonical_corpus_json",
    "discover_source_files",
    "format_batch_text",
    "ingest_sources",
    "ingest_to_corpus",
    "write_corpus_atomic",
]
