"""Deterministic clarification policy over an explicit retrieval outcome."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class RetrievalOutcome(str, Enum):
    """Upstream retrieval assessment consumed by the policy."""

    USEFUL = "USEFUL"
    QUERY_INSUFFICIENT = "QUERY_INSUFFICIENT"
    NOT_FOUND = "NOT_FOUND"
    OUT_OF_SCOPE = "OUT_OF_SCOPE"
    KNOWLEDGE_AMBIGUOUS = "KNOWLEDGE_AMBIGUOUS"
    EVIDENCE_INSUFFICIENT = "EVIDENCE_INSUFFICIENT"


class ClarificationAction(str, Enum):
    """Control decision; this module does not generate an answer."""

    PROCEED_WITH_EVIDENCE = "PROCEED_WITH_EVIDENCE"
    BLOCK_FOR_CLARIFICATION = "BLOCK_FOR_CLARIFICATION"
    RETURN_BOUNDARY = "RETURN_BOUNDARY"


class KnowledgeBoundary(str, Enum):
    """Why the request cannot yet produce a complete evidence-backed result."""

    NONE = "NONE"
    CASE_INFORMATION_MISSING = "CASE_INFORMATION_MISSING"
    KNOWLEDGE_MISSING = "KNOWLEDGE_MISSING"
    OUT_OF_SCOPE = "OUT_OF_SCOPE"
    KNOWLEDGE_AMBIGUOUS = "KNOWLEDGE_AMBIGUOUS"
    EVIDENCE_INSUFFICIENT = "EVIDENCE_INSUFFICIENT"
    RETRIEVAL_QUERY_INSUFFICIENT = "RETRIEVAL_QUERY_INSUFFICIENT"


@dataclass(frozen=True)
class ClarificationDecision:
    """Auditable decision that retains caller inputs verbatim."""

    original_query: str
    retrieval_outcome: RetrievalOutcome
    action: ClarificationAction
    boundary: KnowledgeBoundary
    evidence_ids: tuple[str, ...] = field(default_factory=tuple)
    missing_case_information: tuple[str, ...] = field(default_factory=tuple)
    audit_reasons: tuple[str, ...] = field(default_factory=tuple)

    @property
    def blocks_for_clarification(self) -> bool:
        return self.action is ClarificationAction.BLOCK_FOR_CLARIFICATION


def _validate_inputs(
    original_query: str,
    outcome: RetrievalOutcome,
    evidence_ids: tuple[str, ...],
    missing_case_information: tuple[str, ...],
) -> None:
    if not isinstance(original_query, str):
        raise TypeError("original_query must be a string")
    if not isinstance(outcome, RetrievalOutcome):
        raise TypeError("retrieval_outcome must be a RetrievalOutcome")
    if any(not isinstance(item, str) for item in evidence_ids):
        raise TypeError("evidence_ids must contain strings")
    if any(not isinstance(item, str) for item in missing_case_information):
        raise TypeError("missing_case_information must contain strings")
    if outcome is RetrievalOutcome.USEFUL and not evidence_ids:
        raise ValueError("USEFUL retrieval requires at least one evidence_id")
    if outcome in {
        RetrievalOutcome.QUERY_INSUFFICIENT,
        RetrievalOutcome.NOT_FOUND,
        RetrievalOutcome.OUT_OF_SCOPE,
    } and evidence_ids:
        raise ValueError(f"{outcome.value} cannot include evidence_ids")


def decide_clarification(
    original_query: str,
    retrieval_outcome: RetrievalOutcome,
    *,
    evidence_ids: tuple[str, ...] = (),
    missing_case_information: tuple[str, ...] = (),
) -> ClarificationDecision:
    """Apply the V0 clarification boundary without rewriting or inference.

    Only an insufficient retrieval query blocks for clarification. Missing case
    facts never hide useful Evidence, and knowledge boundaries are returned as
    distinct outcomes rather than being mislabeled as user-information gaps.
    """

    _validate_inputs(
        original_query,
        retrieval_outcome,
        evidence_ids,
        missing_case_information,
    )

    if retrieval_outcome is RetrievalOutcome.QUERY_INSUFFICIENT:
        return ClarificationDecision(
            original_query,
            retrieval_outcome,
            ClarificationAction.BLOCK_FOR_CLARIFICATION,
            KnowledgeBoundary.RETRIEVAL_QUERY_INSUFFICIENT,
            missing_case_information=missing_case_information,
            audit_reasons=("no_useful_retrieval_can_be_formed",),
        )

    if retrieval_outcome is RetrievalOutcome.USEFUL:
        boundary = (
            KnowledgeBoundary.CASE_INFORMATION_MISSING
            if missing_case_information
            else KnowledgeBoundary.NONE
        )
        reasons = ["reliable_evidence_available"]
        if missing_case_information:
            reasons.append("case_information_missing_does_not_block")
        return ClarificationDecision(
            original_query,
            retrieval_outcome,
            ClarificationAction.PROCEED_WITH_EVIDENCE,
            boundary,
            evidence_ids,
            missing_case_information,
            tuple(reasons),
        )

    boundary_by_outcome = {
        RetrievalOutcome.NOT_FOUND: KnowledgeBoundary.KNOWLEDGE_MISSING,
        RetrievalOutcome.OUT_OF_SCOPE: KnowledgeBoundary.OUT_OF_SCOPE,
        RetrievalOutcome.KNOWLEDGE_AMBIGUOUS: (
            KnowledgeBoundary.KNOWLEDGE_AMBIGUOUS
        ),
        RetrievalOutcome.EVIDENCE_INSUFFICIENT: (
            KnowledgeBoundary.EVIDENCE_INSUFFICIENT
        ),
    }
    boundary = boundary_by_outcome[retrieval_outcome]
    return ClarificationDecision(
        original_query,
        retrieval_outcome,
        ClarificationAction.RETURN_BOUNDARY,
        boundary,
        evidence_ids,
        missing_case_information,
        (f"boundary:{boundary.value}",),
    )
