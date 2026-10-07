"""Request-scoped, exact Evidence ID validation for canonical plans."""

from __future__ import annotations

import json
import math
import re
from collections.abc import Iterable
from dataclasses import dataclass
from hashlib import sha256

from .contracts import Evidence, KnowledgeLineage, SourceReference
from .evidence_map_plan import EvidenceMapPlan
from .grounded_organizer import OrganizationStatus


class CandidateSnapshotValidationError(ValueError):
    """The supplied request-scoped Candidate Evidence is not trustworthy."""


_IDENTITY_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:/-]{0,127}$")


def _identity(value: object, path: str) -> str:
    if not isinstance(value, str) or not _IDENTITY_RE.fullmatch(value):
        raise CandidateSnapshotValidationError(
            f"{path} must be a non-empty stable identity"
        )
    return value


def _optional_text(value: object, path: str) -> None:
    if value is not None and not isinstance(value, str):
        raise CandidateSnapshotValidationError(f"{path} must be text or null")


def _validate_source(source: object, path: str) -> SourceReference:
    if not isinstance(source, SourceReference):
        raise CandidateSnapshotValidationError(
            f"{path} must be a SourceReference"
        )
    if not isinstance(source.name, str) or not source.name.strip():
        raise CandidateSnapshotValidationError(f"{path}.name must identify source")
    for field_name in ("url", "section", "sheet"):
        _optional_text(getattr(source, field_name), f"{path}.{field_name}")
    if source.page is not None and (
        type(source.page) is not int or source.page < 1
    ):
        raise CandidateSnapshotValidationError(
            f"{path}.page must be a positive integer or null"
        )
    return source


def _validate_lineage(lineage: object, path: str) -> KnowledgeLineage:
    if not isinstance(lineage, KnowledgeLineage):
        raise CandidateSnapshotValidationError(
            f"{path} must be KnowledgeLineage"
        )
    for field_name in (
        "document_id",
        "parser_version",
        "chunker_version",
        "processing_version",
    ):
        value = getattr(lineage, field_name)
        if not isinstance(value, str) or not value.strip():
            raise CandidateSnapshotValidationError(
                f"{path}.{field_name} must be non-empty"
            )
    for field_name in ("source_position", "parent_section_id"):
        _optional_text(getattr(lineage, field_name), f"{path}.{field_name}")
    return lineage


def _validate_json_value(value: object, path: str) -> None:
    if value is None or type(value) in {bool, int, str}:
        return
    if type(value) is float:
        if not math.isfinite(value):
            raise CandidateSnapshotValidationError(
                f"{path} must not contain non-finite numbers"
            )
        return
    if type(value) is list:
        for index, item in enumerate(value):
            _validate_json_value(item, f"{path}[{index}]")
        return
    if type(value) is dict:
        for key, item in value.items():
            if not isinstance(key, str):
                raise CandidateSnapshotValidationError(
                    f"{path} keys must be strings"
                )
            _validate_json_value(item, f"{path}.{key}")
        return
    raise CandidateSnapshotValidationError(f"{path} must contain JSON values")


def _validate_candidates(candidates: tuple[Evidence, ...]) -> None:
    if not candidates:
        raise CandidateSnapshotValidationError(
            "Candidate Evidence collection must not be empty"
        )
    seen: set[str] = set()
    for index, evidence in enumerate(candidates):
        path = f"candidates[{index}]"
        if not isinstance(evidence, Evidence):
            raise CandidateSnapshotValidationError(f"{path} must be Evidence")
        evidence_id = _identity(evidence.evidence_id, f"{path}.evidence_id")
        _identity(evidence.unit_id, f"{path}.unit_id")
        if not isinstance(evidence.original_content, str):
            raise CandidateSnapshotValidationError(
                f"{path}.original_content must be text"
            )
        _validate_source(evidence.source_reference, f"{path}.source_reference")
        _validate_lineage(evidence.lineage, f"{path}.lineage")
        if evidence.derived_structure is not None:
            if type(evidence.derived_structure) is not dict:
                raise CandidateSnapshotValidationError(
                    f"{path}.derived_structure must be an object or null"
                )
            _validate_json_value(
                evidence.derived_structure, f"{path}.derived_structure"
            )
        if evidence_id in seen:
            raise CandidateSnapshotValidationError(
                "duplicate Candidate Evidence ID"
            )
        seen.add(evidence_id)


def _evidence_identity(evidence: Evidence) -> dict[str, object]:
    source = evidence.source_reference
    lineage = evidence.lineage
    return {
        "evidence_id": evidence.evidence_id,
        "unit_id": evidence.unit_id,
        "original_content": evidence.original_content,
        "derived_structure": evidence.derived_structure,
        "source_reference": {
            "name": source.name,
            "url": source.url,
            "section": source.section,
            "page": source.page,
            "sheet": source.sheet,
        },
        "lineage": {
            "document_id": lineage.document_id,
            "parser_version": lineage.parser_version,
            "chunker_version": lineage.chunker_version,
            "processing_version": lineage.processing_version,
            "source_position": lineage.source_position,
            "parent_section_id": lineage.parent_section_id,
        },
    }


def _fingerprint(request_id: str, candidates: tuple[Evidence, ...]) -> str:
    payload = {
        "request_id": request_id,
        "candidates": sorted(
            (_evidence_identity(item) for item in candidates),
            key=lambda item: item["evidence_id"],
        ),
    }
    canonical = json.dumps(
        payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    )
    return sha256(canonical.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class CandidateEvidenceSnapshot:
    """Immutable request binding for one exact Candidate Evidence collection."""

    request_id: str
    snapshot_id: str
    fingerprint: str
    candidates: tuple[Evidence, ...]
    candidate_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        request_id = _identity(self.request_id, "request_id")
        _identity(self.snapshot_id, "snapshot_id")
        if type(self.candidates) is not tuple:
            raise CandidateSnapshotValidationError("candidates must be a tuple")
        _validate_candidates(self.candidates)
        expected_ids = tuple(sorted(item.evidence_id for item in self.candidates))
        if self.candidate_ids != expected_ids:
            raise CandidateSnapshotValidationError(
                "candidate_ids do not match Candidate Evidence"
            )
        expected_fingerprint = _fingerprint(request_id, self.candidates)
        if self.fingerprint != expected_fingerprint:
            raise CandidateSnapshotValidationError(
                "Candidate Evidence fingerprint does not match snapshot"
            )
        expected_snapshot_id = f"candidate-snapshot-{expected_fingerprint[:24]}"
        if self.snapshot_id != expected_snapshot_id:
            raise CandidateSnapshotValidationError(
                "snapshot_id does not match Candidate Evidence fingerprint"
            )

    @classmethod
    def create(
        cls, request_id: str, candidates: Iterable[Evidence]
    ) -> "CandidateEvidenceSnapshot":
        request_id = _identity(request_id, "request_id")
        candidate_tuple = tuple(candidates)
        _validate_candidates(candidate_tuple)
        fingerprint = _fingerprint(request_id, candidate_tuple)
        return cls(
            request_id=request_id,
            snapshot_id=f"candidate-snapshot-{fingerprint[:24]}",
            fingerprint=fingerprint,
            candidates=candidate_tuple,
            candidate_ids=tuple(
                sorted(item.evidence_id for item in candidate_tuple)
            ),
        )


@dataclass(frozen=True)
class ValidatedPlan:
    """Canonical plan bound to the exact request snapshot for T-803."""

    plan: EvidenceMapPlan
    request_id: str
    referenced_evidence_ids: tuple[str, ...]
    candidate_fingerprint: str
    snapshot_id: str


@dataclass(frozen=True)
class ReferenceGuardResult:
    """All-or-nothing Reference Guard result."""

    status: OrganizationStatus
    validated_plan: ValidatedPlan | None
    reason: str
    invalid_evidence_ids: tuple[str, ...] = ()


def _plan_references(plan: EvidenceMapPlan) -> tuple[str, ...]:
    return tuple(
        evidence_id
        for section in plan.sections
        for group in section.groups
        for evidence_id in group.evidence_ids
    )


def validate_evidence_references(
    plan: EvidenceMapPlan,
    request_id: str,
    candidate_evidence: CandidateEvidenceSnapshot | Iterable[Evidence],
    *,
    expected_snapshot_id: str | None = None,
) -> ReferenceGuardResult:
    """Validate every plan reference exactly against one request snapshot."""

    if not isinstance(plan, EvidenceMapPlan):
        raise TypeError("plan must be a canonical EvidenceMapPlan")

    try:
        checked_request_id = _identity(request_id, "request_id")
        if isinstance(candidate_evidence, CandidateEvidenceSnapshot):
            snapshot = candidate_evidence
            # Rebuild to detect any post-construction mutation of nested inputs.
            snapshot = CandidateEvidenceSnapshot(
                request_id=snapshot.request_id,
                snapshot_id=snapshot.snapshot_id,
                fingerprint=snapshot.fingerprint,
                candidates=snapshot.candidates,
                candidate_ids=snapshot.candidate_ids,
            )
        else:
            snapshot = CandidateEvidenceSnapshot.create(
                checked_request_id, candidate_evidence
            )
    except (CandidateSnapshotValidationError, TypeError):
        return ReferenceGuardResult(
            OrganizationStatus.ORGANIZATION_ERROR,
            None,
            "candidate_snapshot_invalid",
        )

    if snapshot.request_id != checked_request_id:
        return ReferenceGuardResult(
            OrganizationStatus.ORGANIZATION_ERROR,
            None,
            "request_identity_mismatch",
        )
    if expected_snapshot_id is not None and (
        not isinstance(expected_snapshot_id, str)
        or expected_snapshot_id != snapshot.snapshot_id
    ):
        return ReferenceGuardResult(
            OrganizationStatus.ORGANIZATION_ERROR,
            None,
            "stale_candidate_snapshot",
        )

    references = _plan_references(plan)
    candidate_ids = frozenset(snapshot.candidate_ids)
    invalid = tuple(item for item in references if item not in candidate_ids)
    if invalid:
        return ReferenceGuardResult(
            OrganizationStatus.ORGANIZATION_ERROR,
            None,
            "invalid_evidence_references",
            invalid,
        )

    return ReferenceGuardResult(
        OrganizationStatus.SUCCESS,
        ValidatedPlan(
            plan=plan,
            request_id=checked_request_id,
            referenced_evidence_ids=references,
            candidate_fingerprint=snapshot.fingerprint,
            snapshot_id=snapshot.snapshot_id,
        ),
        "evidence_references_validated",
    )
