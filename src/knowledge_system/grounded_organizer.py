"""Organization-only boundary over grounded, read-only knowledge views."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from enum import Enum

from .contracts import Evidence, KnowledgeLineage, SourceReference
from .deterministic_comparison import ConditionComparison
from .evidence_map_plan import (
    EvidenceMapPlan,
    EvidenceMapPlanValidationError,
)
from .rule_authority import validate_rule_authority
from .rule_structure import (
    RuleExtractionResult,
    RuleExtractionStatus,
    RuleStructure,
)


class OrganizationStatus(str, Enum):
    """Outcome of one organization-only backend call."""

    SUCCESS = "SUCCESS"
    ORGANIZATION_ERROR = "ORGANIZATION_ERROR"


class OrganizationErrorReason(str, Enum):
    """Stable failure reasons; backend prose is never used as a fallback."""

    BACKEND_EXCEPTION = "backend_exception"
    OUTPUT_TYPE_NOT_ALLOWED = "output_type_not_allowed"
    STRICT_PLAN_VALIDATION_FAILED = "strict_plan_validation_failed"


@dataclass(frozen=True)
class OrganizerEvidenceView:
    """Immutable references to the grounded fields an organizer may read."""

    evidence_id: str
    unit_id: str
    original_content: str
    source_reference: SourceReference
    lineage: KnowledgeLineage
    rule_structure: RuleStructure | None = None
    rule_extraction_status: RuleExtractionStatus | None = None


@dataclass(frozen=True)
class OrganizerInput:
    """Read-only organizer state with no backend or runtime dependencies."""

    candidates: tuple[OrganizerEvidenceView, ...]
    comparison_results: tuple[ConditionComparison, ...] = ()

    def __post_init__(self) -> None:
        if type(self.candidates) is not tuple or not self.candidates:
            raise ValueError("candidates must be a non-empty tuple")
        if any(not isinstance(item, OrganizerEvidenceView) for item in self.candidates):
            raise TypeError("candidates must contain OrganizerEvidenceView items")
        if type(self.comparison_results) is not tuple or any(
            not isinstance(item, ConditionComparison)
            for item in self.comparison_results
        ):
            raise TypeError(
                "comparison_results must be a tuple of ConditionComparison items"
            )
        identities = tuple(item.evidence_id for item in self.candidates)
        if len(set(identities)) != len(identities):
            raise ValueError("candidate evidence_id values must be unique")

    @classmethod
    def from_authoritative(
        cls,
        candidate_evidence: Iterable[Evidence],
        *,
        rule_extractions: Iterable[RuleExtractionResult] = (),
        comparison_results: Iterable[ConditionComparison] = (),
    ) -> "OrganizerInput":
        """Create a view without copying or rewriting authoritative objects."""

        evidence_items = tuple(candidate_evidence)
        if any(not isinstance(item, Evidence) for item in evidence_items):
            raise TypeError("candidate_evidence must contain Evidence items")

        extraction_by_id: dict[str, RuleExtractionResult] = {}
        for extraction in rule_extractions:
            if not isinstance(extraction, RuleExtractionResult):
                raise TypeError(
                    "rule_extractions must contain RuleExtractionResult items"
                )
            validate_rule_authority(extraction)
            evidence_id = extraction.source_evidence.evidence_id
            if evidence_id in extraction_by_id:
                raise ValueError("duplicate RuleExtractionResult for evidence_id")
            extraction_by_id[evidence_id] = extraction

        candidate_ids = {item.evidence_id for item in evidence_items}
        if set(extraction_by_id) - candidate_ids:
            raise ValueError("RuleExtractionResult is not for Candidate Evidence")

        views: list[OrganizerEvidenceView] = []
        for evidence in evidence_items:
            extraction = extraction_by_id.get(evidence.evidence_id)
            if extraction is not None and extraction.source_evidence != evidence:
                raise ValueError(
                    "RuleExtractionResult source does not match Candidate Evidence"
                )
            views.append(
                OrganizerEvidenceView(
                    evidence_id=evidence.evidence_id,
                    unit_id=evidence.unit_id,
                    original_content=evidence.original_content,
                    source_reference=evidence.source_reference,
                    lineage=evidence.lineage,
                    rule_structure=(
                        extraction.structure if extraction is not None else None
                    ),
                    rule_extraction_status=(
                        extraction.status if extraction is not None else None
                    ),
                )
            )
        return cls(tuple(views), tuple(comparison_results))


ORGANIZATION_ONLY_AUTHORITY = (
    "ORGANIZATION_ONLY: arrange candidate evidence IDs; do not add, remove, "
    "rewrite, or decide knowledge."
)
ALLOWED_OUTPUT_SCHEMA = (
    "plan:{plan_id,sections}",
    "section:{section_id,order,groups}",
    "group:{group_id,order,evidence_ids}",
)


@dataclass(frozen=True)
class OrganizerBackendRequest:
    """Explicit authority envelope sent to an injected organizer backend."""

    organizer_input: OrganizerInput
    authority: str = ORGANIZATION_ONLY_AUTHORITY
    allowed_output_schema: tuple[str, ...] = ALLOWED_OUTPUT_SCHEMA


OrganizerBackend = Callable[
    [OrganizerBackendRequest], Mapping[str, object] | str
]


@dataclass(frozen=True)
class OrganizationResult:
    """A canonical plan or an explicit organization error, never prose."""

    status: OrganizationStatus
    plan: EvidenceMapPlan | None
    reason: str


def organize_evidence_map(
    organizer_input: OrganizerInput,
    backend: OrganizerBackend,
) -> OrganizationResult:
    """Call an injected organizer once and immediately enforce the T-800 schema."""

    if not isinstance(organizer_input, OrganizerInput):
        raise TypeError("organizer_input must be an OrganizerInput")
    if not callable(backend):
        raise TypeError("backend must be callable")

    request = OrganizerBackendRequest(organizer_input)
    try:
        output = backend(request)
    except Exception:
        return OrganizationResult(
            OrganizationStatus.ORGANIZATION_ERROR,
            None,
            OrganizationErrorReason.BACKEND_EXCEPTION.value,
        )

    if not isinstance(output, (str, Mapping)):
        return OrganizationResult(
            OrganizationStatus.ORGANIZATION_ERROR,
            None,
            OrganizationErrorReason.OUTPUT_TYPE_NOT_ALLOWED.value,
        )

    try:
        plan = (
            EvidenceMapPlan.from_json(output)
            if isinstance(output, str)
            else EvidenceMapPlan.from_dict(output)
        )
    except (EvidenceMapPlanValidationError, TypeError, ValueError):
        return OrganizationResult(
            OrganizationStatus.ORGANIZATION_ERROR,
            None,
            OrganizationErrorReason.STRICT_PLAN_VALIDATION_FAILED.value,
        )

    return OrganizationResult(
        OrganizationStatus.SUCCESS,
        plan,
        "organization_plan_validated",
    )
