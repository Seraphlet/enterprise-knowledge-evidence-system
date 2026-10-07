"""Core, dependency-free data contracts for the knowledge system."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, ClassVar, TypeVar


JsonValue = None | bool | int | float | str | list["JsonValue"] | dict[str, "JsonValue"]
ContractT = TypeVar("ContractT", bound="JsonContract")


class ParseStatus(str, Enum):
    """Outcome of parsing source material."""

    SUCCESS = "SUCCESS"
    PARTIAL = "PARTIAL"
    FAILED = "FAILED"


def _json_value(value: Any) -> JsonValue:
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, dict):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    return value


class JsonContract:
    """Small JSON round-trip interface shared by V0 contracts."""

    required_fields: ClassVar[tuple[str, ...]] = ()

    def to_dict(self) -> dict[str, JsonValue]:
        return _json_value(asdict(self))  # type: ignore[return-value]

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, sort_keys=True)

    @classmethod
    def from_json(cls: type[ContractT], payload: str) -> ContractT:
        value = json.loads(payload)
        if not isinstance(value, dict):
            raise ValueError(f"{cls.__name__} JSON must contain an object")
        return cls.from_dict(value)

    @classmethod
    def from_dict(cls: type[ContractT], value: dict[str, Any]) -> ContractT:
        raise NotImplementedError


@dataclass(frozen=True)
class SourceReference(JsonContract):
    """Employee-usable identity and optional location of original source."""

    name: str
    url: str | None = None
    section: str | None = None
    page: int | None = None
    sheet: str | None = None

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "SourceReference":
        return cls(
            name=value["name"],
            url=value.get("url"),
            section=value.get("section"),
            page=value.get("page"),
            sheet=value.get("sheet"),
        )


@dataclass(frozen=True)
class KnowledgeLineage(JsonContract):
    """Processing provenance from a knowledge unit back to its document."""

    document_id: str
    parser_version: str
    chunker_version: str
    processing_version: str
    source_position: str | None = None
    parent_section_id: str | None = None

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "KnowledgeLineage":
        return cls(
            document_id=value["document_id"],
            parser_version=value["parser_version"],
            chunker_version=value["chunker_version"],
            processing_version=value["processing_version"],
            source_position=value.get("source_position"),
            parent_section_id=value.get("parent_section_id"),
        )


@dataclass(frozen=True)
class KnowledgeUnit(JsonContract):
    """Retrievable knowledge with source facts separated from derived structure."""

    unit_id: str
    original_content: str
    semantic_content: str
    metadata: dict[str, JsonValue]
    source_reference: SourceReference
    lineage: KnowledgeLineage
    parse_status: ParseStatus
    rule_structure: dict[str, JsonValue] | None = None

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "KnowledgeUnit":
        return cls(
            unit_id=value["unit_id"],
            original_content=value["original_content"],
            semantic_content=value["semantic_content"],
            metadata=dict(value.get("metadata", {})),
            source_reference=SourceReference.from_dict(value["source_reference"]),
            lineage=KnowledgeLineage.from_dict(value["lineage"]),
            parse_status=ParseStatus(value["parse_status"]),
            rule_structure=(
                dict(value["rule_structure"])
                if value.get("rule_structure") is not None
                else None
            ),
        )


@dataclass(frozen=True)
class Evidence(JsonContract):
    """Evidence selected from a knowledge unit without rewriting source facts."""

    evidence_id: str
    unit_id: str
    original_content: str
    source_reference: SourceReference
    lineage: KnowledgeLineage
    derived_structure: dict[str, JsonValue] | None = None

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "Evidence":
        return cls(
            evidence_id=value["evidence_id"],
            unit_id=value["unit_id"],
            original_content=value["original_content"],
            source_reference=SourceReference.from_dict(value["source_reference"]),
            lineage=KnowledgeLineage.from_dict(value["lineage"]),
            derived_structure=(
                dict(value["derived_structure"])
                if value.get("derived_structure") is not None
                else None
            ),
        )


@dataclass(frozen=True)
class Trace(JsonContract):
    """Trace of retrieval inputs and evidence selection; separate from state."""

    trace_id: str
    raw_query: str
    retrieval_query: str
    candidate_evidence_ids: list[str] = field(default_factory=list)
    final_evidence_ids: list[str] = field(default_factory=list)
    applied_filters: list[str] = field(default_factory=list)
    knowledge_version: str | None = None
    filter_summary: dict[str, JsonValue] = field(default_factory=dict)
    source_references: list[dict[str, JsonValue]] = field(default_factory=list)
    processing_versions: list[dict[str, JsonValue]] = field(default_factory=list)
    index_version: str | None = None
    retrieval_paths: list[dict[str, JsonValue]] = field(default_factory=list)
    fusion_scores: dict[str, JsonValue] = field(default_factory=dict)
    rerank_summary: dict[str, JsonValue] = field(default_factory=dict)
    discover_summary: dict[str, JsonValue] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "Trace":
        return cls(
            trace_id=value["trace_id"],
            raw_query=value["raw_query"],
            retrieval_query=value["retrieval_query"],
            candidate_evidence_ids=list(value.get("candidate_evidence_ids", [])),
            final_evidence_ids=list(value.get("final_evidence_ids", [])),
            applied_filters=list(value.get("applied_filters", [])),
            knowledge_version=value.get("knowledge_version"),
            filter_summary=dict(value.get("filter_summary", {})),
            source_references=[
                dict(item) for item in value.get("source_references", [])
            ],
            processing_versions=[
                dict(item) for item in value.get("processing_versions", [])
            ],
            index_version=value.get("index_version"),
            retrieval_paths=[
                dict(item) for item in value.get("retrieval_paths", [])
            ],
            fusion_scores=dict(value.get("fusion_scores", {})),
            rerank_summary=dict(value.get("rerank_summary", {})),
            discover_summary=dict(value.get("discover_summary", {})),
        )
