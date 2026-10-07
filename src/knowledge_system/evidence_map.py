"""Deterministic hydration of validated organization plans."""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from enum import Enum
from typing import Any

from .contracts import Evidence, JsonValue, KnowledgeLineage, SourceReference
from .reference_guard import (
    CandidateEvidenceSnapshot,
    CandidateSnapshotValidationError,
    ValidatedPlan,
)


class HydrationStatus(str, Enum):
    """Outcome of deterministic Evidence Map hydration."""

    SUCCESS = "SUCCESS"
    HYDRATION_ERROR = "HYDRATION_ERROR"


class EvidenceMapValidationError(ValueError):
    """A serialized hydrated Evidence Map is malformed."""


def _object(
    value: object, expected: frozenset[str], path: str
) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise EvidenceMapValidationError(f"{path} must be an object")
    keys = set(value)
    if any(not isinstance(item, str) for item in keys) or keys != expected:
        raise EvidenceMapValidationError(f"{path} fields do not match schema")
    return value


def _array(value: object, path: str) -> list[Any]:
    if type(value) is not list:
        raise EvidenceMapValidationError(f"{path} must be an array")
    return value


def _text(value: object, path: str) -> str:
    if not isinstance(value, str) or not value:
        raise EvidenceMapValidationError(f"{path} must be non-empty text")
    return value


def _order(value: object, path: str) -> int:
    if type(value) is not int or value < 0:
        raise EvidenceMapValidationError(f"{path} must be a non-negative integer")
    return value


@dataclass(frozen=True)
class EvidenceMapEntry:
    """An immutable exact snapshot of authoritative Evidence fields."""

    evidence_id: str
    unit_id: str
    original_content: str
    source_reference: SourceReference
    lineage: KnowledgeLineage
    _derived_structure_json: str | None = None

    @property
    def derived_structure(self) -> dict[str, JsonValue] | None:
        if self._derived_structure_json is None:
            return None
        return json.loads(self._derived_structure_json)

    @classmethod
    def from_evidence(cls, evidence: Evidence) -> "EvidenceMapEntry":
        structure_json = (
            json.dumps(
                evidence.derived_structure,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            )
            if evidence.derived_structure is not None
            else None
        )
        return cls(
            evidence_id=evidence.evidence_id,
            unit_id=evidence.unit_id,
            original_content=evidence.original_content,
            source_reference=evidence.source_reference,
            lineage=evidence.lineage,
            _derived_structure_json=structure_json,
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "evidence_id": self.evidence_id,
            "unit_id": self.unit_id,
            "original_content": self.original_content,
            "source_reference": self.source_reference.to_dict(),
            "lineage": self.lineage.to_dict(),
            "derived_structure": self.derived_structure,
        }

    @classmethod
    def from_dict(cls, value: object, path: str) -> "EvidenceMapEntry":
        item = _object(
            value,
            frozenset(
                {
                    "evidence_id",
                    "unit_id",
                    "original_content",
                    "source_reference",
                    "lineage",
                    "derived_structure",
                }
            ),
            path,
        )
        return cls.from_evidence(Evidence.from_dict(dict(item)))


@dataclass(frozen=True)
class HydratedEvidenceGroup:
    group_id: str
    order: int
    entries: tuple[EvidenceMapEntry, ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "group_id": self.group_id,
            "order": self.order,
            "entries": [item.to_dict() for item in self.entries],
        }

    @classmethod
    def from_dict(cls, value: object, path: str) -> "HydratedEvidenceGroup":
        item = _object(
            value, frozenset({"group_id", "order", "entries"}), path
        )
        entries = _array(item["entries"], f"{path}.entries")
        if not entries:
            raise EvidenceMapValidationError(f"{path}.entries must not be empty")
        return cls(
            _text(item["group_id"], f"{path}.group_id"),
            _order(item["order"], f"{path}.order"),
            tuple(
                EvidenceMapEntry.from_dict(entry, f"{path}.entries[{index}]")
                for index, entry in enumerate(entries)
            ),
        )


@dataclass(frozen=True)
class HydratedEvidenceSection:
    section_id: str
    order: int
    groups: tuple[HydratedEvidenceGroup, ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "section_id": self.section_id,
            "order": self.order,
            "groups": [item.to_dict() for item in self.groups],
        }

    @classmethod
    def from_dict(cls, value: object, path: str) -> "HydratedEvidenceSection":
        item = _object(
            value, frozenset({"section_id", "order", "groups"}), path
        )
        groups = _array(item["groups"], f"{path}.groups")
        if not groups:
            raise EvidenceMapValidationError(f"{path}.groups must not be empty")
        return cls(
            _text(item["section_id"], f"{path}.section_id"),
            _order(item["order"], f"{path}.order"),
            tuple(
                HydratedEvidenceGroup.from_dict(
                    group, f"{path}.groups[{index}]"
                )
                for index, group in enumerate(groups)
            ),
        )


@dataclass(frozen=True)
class EvidenceMap:
    """Immutable, source-grounded result of deterministic hydration."""

    plan_id: str
    request_id: str
    snapshot_id: str
    candidate_fingerprint: str
    sections: tuple[HydratedEvidenceSection, ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "plan_id": self.plan_id,
            "request_id": self.request_id,
            "snapshot_id": self.snapshot_id,
            "candidate_fingerprint": self.candidate_fingerprint,
            "sections": [item.to_dict() for item in self.sections],
        }

    def to_json(self) -> str:
        return json.dumps(
            self.to_dict(),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )

    @classmethod
    def from_dict(cls, value: object) -> "EvidenceMap":
        item = _object(
            value,
            frozenset(
                {
                    "plan_id",
                    "request_id",
                    "snapshot_id",
                    "candidate_fingerprint",
                    "sections",
                }
            ),
            "evidence_map",
        )
        sections = _array(item["sections"], "evidence_map.sections")
        if not sections:
            raise EvidenceMapValidationError(
                "evidence_map.sections must not be empty"
            )
        return cls(
            plan_id=_text(item["plan_id"], "evidence_map.plan_id"),
            request_id=_text(item["request_id"], "evidence_map.request_id"),
            snapshot_id=_text(item["snapshot_id"], "evidence_map.snapshot_id"),
            candidate_fingerprint=_text(
                item["candidate_fingerprint"],
                "evidence_map.candidate_fingerprint",
            ),
            sections=tuple(
                HydratedEvidenceSection.from_dict(
                    section, f"evidence_map.sections[{index}]"
                )
                for index, section in enumerate(sections)
            ),
        )

    @classmethod
    def from_json(cls, payload: str) -> "EvidenceMap":
        if not isinstance(payload, str):
            raise TypeError("payload must be a JSON string")
        try:
            value = json.loads(payload)
        except (TypeError, ValueError, json.JSONDecodeError) as error:
            raise EvidenceMapValidationError("invalid Evidence Map JSON") from error
        return cls.from_dict(value)


@dataclass(frozen=True)
class HydrationBinding:
    request_id: str
    snapshot_id: str
    candidate_fingerprint: str


@dataclass(frozen=True)
class HydrationResult:
    """All-or-nothing hydration result with safe binding information."""

    status: HydrationStatus
    evidence_map: EvidenceMap | None
    reason: str
    failed_evidence_ids: tuple[str, ...] = ()
    binding: HydrationBinding | None = None


def _binding(validated_plan: ValidatedPlan) -> HydrationBinding:
    return HydrationBinding(
        validated_plan.request_id,
        validated_plan.snapshot_id,
        validated_plan.candidate_fingerprint,
    )


def _references(validated_plan: ValidatedPlan) -> tuple[str, ...]:
    return tuple(
        evidence_id
        for section in validated_plan.plan.sections
        for group in section.groups
        for evidence_id in group.evidence_ids
    )


def hydrate_evidence_map(
    validated_plan: ValidatedPlan,
    candidate_snapshot: CandidateEvidenceSnapshot,
) -> HydrationResult:
    """Hydrate only a T-802 plan from its exact authoritative snapshot."""

    if not isinstance(validated_plan, ValidatedPlan):
        return HydrationResult(
            HydrationStatus.HYDRATION_ERROR, None, "validated_plan_required"
        )
    binding = _binding(validated_plan)
    if not isinstance(candidate_snapshot, CandidateEvidenceSnapshot):
        return HydrationResult(
            HydrationStatus.HYDRATION_ERROR,
            None,
            "candidate_snapshot_required",
            binding=binding,
        )

    try:
        snapshot = CandidateEvidenceSnapshot(
            request_id=candidate_snapshot.request_id,
            snapshot_id=candidate_snapshot.snapshot_id,
            fingerprint=candidate_snapshot.fingerprint,
            candidates=candidate_snapshot.candidates,
            candidate_ids=candidate_snapshot.candidate_ids,
        )
    except (CandidateSnapshotValidationError, TypeError):
        return HydrationResult(
            HydrationStatus.HYDRATION_ERROR,
            None,
            "candidate_snapshot_integrity_failed",
            binding=binding,
        )

    if validated_plan.request_id != snapshot.request_id:
        return HydrationResult(
            HydrationStatus.HYDRATION_ERROR,
            None,
            "request_binding_mismatch",
            binding=binding,
        )
    if validated_plan.snapshot_id != snapshot.snapshot_id:
        return HydrationResult(
            HydrationStatus.HYDRATION_ERROR,
            None,
            "snapshot_binding_mismatch",
            binding=binding,
        )
    if validated_plan.candidate_fingerprint != snapshot.fingerprint:
        return HydrationResult(
            HydrationStatus.HYDRATION_ERROR,
            None,
            "fingerprint_binding_mismatch",
            binding=binding,
        )

    references = _references(validated_plan)
    if references != validated_plan.referenced_evidence_ids:
        return HydrationResult(
            HydrationStatus.HYDRATION_ERROR,
            None,
            "validated_reference_binding_mismatch",
            binding=binding,
        )

    evidence_by_id = {item.evidence_id: item for item in snapshot.candidates}
    missing = tuple(item for item in references if item not in evidence_by_id)
    if missing:
        return HydrationResult(
            HydrationStatus.HYDRATION_ERROR,
            None,
            "authoritative_evidence_missing",
            failed_evidence_ids=missing,
            binding=binding,
        )

    sections = tuple(
        HydratedEvidenceSection(
            section_id=section.section_id,
            order=section.order,
            groups=tuple(
                HydratedEvidenceGroup(
                    group_id=group.group_id,
                    order=group.order,
                    entries=tuple(
                        EvidenceMapEntry.from_evidence(
                            evidence_by_id[evidence_id]
                        )
                        for evidence_id in group.evidence_ids
                    ),
                )
                for group in section.groups
            ),
        )
        for section in validated_plan.plan.sections
    )
    return HydrationResult(
        HydrationStatus.SUCCESS,
        EvidenceMap(
            plan_id=validated_plan.plan.plan_id,
            request_id=validated_plan.request_id,
            snapshot_id=validated_plan.snapshot_id,
            candidate_fingerprint=validated_plan.candidate_fingerprint,
            sections=sections,
        ),
        "evidence_map_hydrated",
        binding=binding,
    )
