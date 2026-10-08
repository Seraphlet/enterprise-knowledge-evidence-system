"""Deterministic format routing and unified ingestion results.

This module owns dispatch only.  Format-specific parsing remains in each
loader, and the existing HTML parser/converter is reused without duplication.
"""

from __future__ import annotations

import csv as csv_module
from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any

from knowledge_system.contracts import KnowledgeUnit, ParseStatus, SourceReference
from knowledge_system.ingestion.csv import load_csv
from knowledge_system.ingestion.html import load_html
from knowledge_system.ingestion.markdown import load_markdown


class IngestionFormat(str, Enum):
    """Source formats recognized by the Phase 14 ingestion surface."""

    HTML = "HTML"
    MARKDOWN = "MARKDOWN"
    CSV = "CSV"


EXTENSION_FORMATS: dict[str, IngestionFormat] = {
    ".html": IngestionFormat.HTML,
    ".htm": IngestionFormat.HTML,
    ".md": IngestionFormat.MARKDOWN,
    ".markdown": IngestionFormat.MARKDOWN,
    ".csv": IngestionFormat.CSV,
}


@dataclass(frozen=True)
class IngestionIssue:
    """Stable, employee-safe warning or error emitted at the ingestion boundary."""

    code: str
    message: str

    def to_dict(self) -> dict[str, str]:
        return {"code": self.code, "message": self.message}


@dataclass(frozen=True)
class IngestionResult:
    """Common result returned by every registered format loader."""

    status: ParseStatus
    source_format: IngestionFormat | None
    source_reference: SourceReference
    units: tuple[KnowledgeUnit, ...] = ()
    warnings: tuple[IngestionIssue, ...] = ()
    errors: tuple[IngestionIssue, ...] = ()

    def __post_init__(self) -> None:
        if self.status is ParseStatus.SUCCESS and self.errors:
            raise ValueError("successful ingestion cannot contain errors")
        if self.status is ParseStatus.FAILED and self.units:
            raise ValueError("failed ingestion cannot contain knowledge units")

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status.value,
            "source_format": (
                self.source_format.value if self.source_format is not None else None
            ),
            "source_reference": self.source_reference.to_dict(),
            "units": [unit.to_dict() for unit in self.units],
            "warnings": [warning.to_dict() for warning in self.warnings],
            "errors": [error.to_dict() for error in self.errors],
        }


IngestionLoader = Callable[[Path, str | None], IngestionResult]


def identify_format(path: str | Path) -> IngestionFormat | None:
    """Identify a supported extension deterministically and case-insensitively."""

    return EXTENSION_FORMATS.get(Path(path).suffix.casefold())


def _source_for(path: Path, url: str | None) -> SourceReference:
    return SourceReference(name=path.name, url=url, page=None)


def _failed(
    path: Path,
    *,
    source_format: IngestionFormat | None,
    code: str,
    message: str,
    url: str | None,
) -> IngestionResult:
    return IngestionResult(
        status=ParseStatus.FAILED,
        source_format=source_format,
        source_reference=_source_for(path, url),
        errors=(IngestionIssue(code, message),),
    )


def _load_html(path: Path, url: str | None) -> IngestionResult:
    """Adapt the existing HTML loader and unit converter to the common result."""

    try:
        document = load_html(path, url=url)
    except UnicodeError:
        return _failed(
            path,
            source_format=IngestionFormat.HTML,
            code="SOURCE_ENCODING_ERROR",
            message="HTML source is not valid UTF-8/UTF-8-SIG text.",
            url=url,
        )
    except OSError:
        return _failed(
            path,
            source_format=IngestionFormat.HTML,
            code="SOURCE_READ_ERROR",
            message="HTML source could not be read.",
            url=url,
        )
    except Exception:
        return _failed(
            path,
            source_format=IngestionFormat.HTML,
            code="HTML_PARSE_ERROR",
            message="HTML source could not be parsed.",
            url=url,
        )

    try:
        # Local import prevents the established knowledge_units -> ingestion
        # dependency from becoming a package initialization cycle.
        from knowledge_system.knowledge_units import knowledge_units_from_html

        units = tuple(knowledge_units_from_html(document))
    except Exception:
        return _failed(
            path,
            source_format=IngestionFormat.HTML,
            code="KNOWLEDGE_UNIT_BUILD_ERROR",
            message="HTML source could not be converted to knowledge units.",
            url=url,
        )

    warnings = tuple(
        IngestionIssue("HTML_PARSE_WARNING", warning)
        for warning in document.warnings
    )
    if document.parse_status is ParseStatus.FAILED:
        return IngestionResult(
            status=ParseStatus.FAILED,
            source_format=IngestionFormat.HTML,
            source_reference=document.source_reference,
            warnings=warnings,
            errors=(
                IngestionIssue(
                    "HTML_PARSE_FAILED",
                    "HTML source contains no usable knowledge content.",
                ),
            ),
        )
    if not units:
        return _failed(
            path,
            source_format=IngestionFormat.HTML,
            code="NO_KNOWLEDGE_UNITS",
            message="HTML source produced no knowledge units.",
            url=url,
        )
    return IngestionResult(
        status=document.parse_status,
        source_format=IngestionFormat.HTML,
        source_reference=document.source_reference,
        units=units,
        warnings=warnings,
    )


def _load_markdown(path: Path, url: str | None) -> IngestionResult:
    """Load Markdown and adapt its structural units to the common result."""

    try:
        document = load_markdown(path, url=url)
    except UnicodeError:
        return _failed(
            path,
            source_format=IngestionFormat.MARKDOWN,
            code="SOURCE_ENCODING_ERROR",
            message="Markdown source is not valid UTF-8/UTF-8-SIG text.",
            url=url,
        )
    except OSError:
        return _failed(
            path,
            source_format=IngestionFormat.MARKDOWN,
            code="SOURCE_READ_ERROR",
            message="Markdown source could not be read.",
            url=url,
        )
    except Exception:
        return _failed(
            path,
            source_format=IngestionFormat.MARKDOWN,
            code="MARKDOWN_PARSE_ERROR",
            message="Markdown source could not be parsed.",
            url=url,
        )

    try:
        from knowledge_system.knowledge_units import knowledge_units_from_markdown

        units = tuple(knowledge_units_from_markdown(document))
    except Exception:
        return _failed(
            path,
            source_format=IngestionFormat.MARKDOWN,
            code="KNOWLEDGE_UNIT_BUILD_ERROR",
            message="Markdown source could not be converted to knowledge units.",
            url=url,
        )

    warnings = tuple(
        IngestionIssue("MARKDOWN_PARSE_WARNING", warning)
        for warning in document.warnings
    )
    if document.parse_status is ParseStatus.FAILED:
        return IngestionResult(
            status=ParseStatus.FAILED,
            source_format=IngestionFormat.MARKDOWN,
            source_reference=document.source_reference,
            warnings=warnings,
            errors=(
                IngestionIssue(
                    "MARKDOWN_PARSE_FAILED",
                    "Markdown source contains no usable knowledge content.",
                ),
            ),
        )
    if not units:
        return _failed(
            path,
            source_format=IngestionFormat.MARKDOWN,
            code="NO_KNOWLEDGE_UNITS",
            message="Markdown source produced no knowledge units.",
            url=url,
        )
    return IngestionResult(
        status=document.parse_status,
        source_format=IngestionFormat.MARKDOWN,
        source_reference=document.source_reference,
        units=units,
        warnings=warnings,
    )


def _load_csv(path: Path, url: str | None) -> IngestionResult:
    """Load CSV and adapt reliable rows to the common result."""

    try:
        document = load_csv(path, url=url)
    except UnicodeError:
        return _failed(
            path,
            source_format=IngestionFormat.CSV,
            code="SOURCE_ENCODING_ERROR",
            message="CSV source is not valid UTF-8/UTF-8-SIG text.",
            url=url,
        )
    except OSError:
        return _failed(
            path,
            source_format=IngestionFormat.CSV,
            code="SOURCE_READ_ERROR",
            message="CSV source could not be read.",
            url=url,
        )
    except csv_module.Error:
        return _failed(
            path,
            source_format=IngestionFormat.CSV,
            code="CSV_PARSE_ERROR",
            message="CSV source contains invalid CSV syntax.",
            url=url,
        )
    except Exception:
        return _failed(
            path,
            source_format=IngestionFormat.CSV,
            code="CSV_PARSE_ERROR",
            message="CSV source could not be parsed.",
            url=url,
        )

    try:
        from knowledge_system.knowledge_units import knowledge_units_from_csv

        units = tuple(knowledge_units_from_csv(document))
    except Exception:
        return _failed(
            path,
            source_format=IngestionFormat.CSV,
            code="KNOWLEDGE_UNIT_BUILD_ERROR",
            message="CSV source could not be converted to knowledge units.",
            url=url,
        )

    warnings = tuple(
        IngestionIssue("CSV_PARSE_WARNING", warning) for warning in document.warnings
    )
    if document.parse_status is ParseStatus.FAILED:
        error_code = (
            "CSV_PARSE_ERROR"
            if any(warning.startswith("CSV syntax error") for warning in document.warnings)
            else "CSV_PARSE_FAILED"
        )
        return IngestionResult(
            status=ParseStatus.FAILED,
            source_format=IngestionFormat.CSV,
            source_reference=document.source_reference,
            warnings=warnings,
            errors=(
                IngestionIssue(
                    error_code,
                    "CSV source contains no usable header and data rows.",
                ),
            ),
        )
    if not units:
        return _failed(
            path,
            source_format=IngestionFormat.CSV,
            code="NO_KNOWLEDGE_UNITS",
            message="CSV source produced no knowledge units.",
            url=url,
        )
    return IngestionResult(
        status=document.parse_status,
        source_format=IngestionFormat.CSV,
        source_reference=document.source_reference,
        units=units,
        warnings=warnings,
    )


class LoaderRegistry:
    """Explicit registry that separates extension routing from format parsing."""

    def __init__(self) -> None:
        self._loaders: dict[IngestionFormat, IngestionLoader] = {}

    def register(
        self, source_format: IngestionFormat, loader: IngestionLoader
    ) -> None:
        if not isinstance(source_format, IngestionFormat):
            raise TypeError("source_format must be an IngestionFormat")
        if not callable(loader):
            raise TypeError("loader must be callable")
        self._loaders[source_format] = loader

    def is_registered(self, source_format: IngestionFormat) -> bool:
        return source_format in self._loaders

    def ingest(
        self, path: str | Path, *, url: str | None = None
    ) -> IngestionResult:
        source_path = Path(path)
        source_format = identify_format(source_path)
        if source_format is None:
            extension = source_path.suffix.casefold() or "<none>"
            return _failed(
                source_path,
                source_format=None,
                code="UNSUPPORTED_EXTENSION",
                message=f"Unsupported source extension: {extension}.",
                url=url,
            )
        loader = self._loaders.get(source_format)
        if loader is None:
            return _failed(
                source_path,
                source_format=source_format,
                code="LOADER_NOT_REGISTERED",
                message=f"No loader is registered for {source_format.value}.",
                url=url,
            )
        try:
            result = loader(source_path, url)
        except Exception:
            return _failed(
                source_path,
                source_format=source_format,
                code="LOADER_EXECUTION_ERROR",
                message=f"The {source_format.value} loader failed.",
                url=url,
            )
        if not isinstance(result, IngestionResult):
            return _failed(
                source_path,
                source_format=source_format,
                code="INVALID_LOADER_RESULT",
                message=f"The {source_format.value} loader returned an invalid result.",
                url=url,
            )
        return result


def create_default_registry() -> LoaderRegistry:
    """Create a registry containing the format loaders implemented so far."""

    registry = LoaderRegistry()
    registry.register(IngestionFormat.HTML, _load_html)
    registry.register(IngestionFormat.MARKDOWN, _load_markdown)
    registry.register(IngestionFormat.CSV, _load_csv)
    return registry


def ingest_file(
    path: str | Path,
    *,
    url: str | None = None,
    registry: LoaderRegistry | None = None,
) -> IngestionResult:
    """Ingest one recognized source through an explicit or default registry."""

    active_registry = registry if registry is not None else create_default_registry()
    return active_registry.ingest(path, url=url)


__all__ = [
    "EXTENSION_FORMATS",
    "IngestionFormat",
    "IngestionIssue",
    "IngestionLoader",
    "IngestionResult",
    "LoaderRegistry",
    "create_default_registry",
    "identify_format",
    "ingest_file",
]
