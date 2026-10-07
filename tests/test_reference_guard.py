"""Focused tests for the request-scoped Evidence reference guard."""

import copy
import unittest
from dataclasses import FrozenInstanceError, fields

from knowledge_system import (
    CandidateEvidenceSnapshot,
    CandidateSnapshotValidationError,
    Evidence,
    EvidenceMapPlan,
    KnowledgeLineage,
    OrganizationStatus,
    SourceReference,
    validate_evidence_references,
)


def _evidence(
    evidence_id: str,
    *,
    content: str | None = None,
    source_name: str | None = None,
    document_id: str | None = None,
    derived_structure=None,
) -> Evidence:
    suffix = evidence_id.replace("/", "-")
    return Evidence(
        evidence_id=evidence_id,
        unit_id=f"unit-{suffix}",
        original_content=content or f"original content for {evidence_id}",
        source_reference=SourceReference(
            name=source_name or f"{suffix}.md",
            url=f"https://example.invalid/{suffix}",
            section="rules",
            page=1,
        ),
        lineage=KnowledgeLineage(
            document_id=document_id or f"document-{suffix}",
            parser_version="parser-1",
            chunker_version="chunker-1",
            processing_version="processing-1",
            source_position="line-1",
            parent_section_id="section-rules",
        ),
        derived_structure=derived_structure,
    )


def _plan(*evidence_ids: str) -> EvidenceMapPlan:
    midpoint = max(1, len(evidence_ids) // 2)
    groups = [
        {
            "group_id": "group-1",
            "order": 0,
            "evidence_ids": list(evidence_ids[:midpoint]),
        }
    ]
    if evidence_ids[midpoint:]:
        groups.append(
            {
                "group_id": "group-2",
                "order": 1,
                "evidence_ids": list(evidence_ids[midpoint:]),
            }
        )
    return EvidenceMapPlan.from_dict(
        {
            "plan_id": "plan-1",
            "sections": [
                {"section_id": "section-1", "order": 0, "groups": groups}
            ],
        }
    )


class ReferenceGuardTest(unittest.TestCase):
    def test_all_references_validate_and_bind_the_original_canonical_plan(self) -> None:
        evidence = (_evidence("evidence-1"), _evidence("evidence-2"))
        snapshot = CandidateEvidenceSnapshot.create("request-1", evidence)
        plan = _plan("evidence-2", "evidence-1")

        result = validate_evidence_references(
            plan,
            "request-1",
            snapshot,
            expected_snapshot_id=snapshot.snapshot_id,
        )

        self.assertEqual(result.status, OrganizationStatus.SUCCESS)
        self.assertIs(result.validated_plan.plan, plan)
        self.assertEqual(
            result.validated_plan.referenced_evidence_ids,
            ("evidence-2", "evidence-1"),
        )
        self.assertEqual(
            result.validated_plan.candidate_fingerprint, snapshot.fingerprint
        )
        self.assertEqual(result.validated_plan.snapshot_id, snapshot.snapshot_id)
        self.assertEqual(result.invalid_evidence_ids, ())

    def test_single_and_multiple_unknown_ids_are_all_or_nothing(self) -> None:
        candidate = _evidence("evidence-1")
        for references, expected_invalid in (
            (("unknown-1",), ("unknown-1",)),
            (
                ("evidence-1", "unknown-1", "unknown-2"),
                ("unknown-1", "unknown-2"),
            ),
        ):
            with self.subTest(references=references):
                result = validate_evidence_references(
                    _plan(*references), "request-1", (candidate,)
                )
                self.assertEqual(
                    result.status, OrganizationStatus.ORGANIZATION_ERROR
                )
                self.assertEqual(result.reason, "invalid_evidence_references")
                self.assertEqual(result.invalid_evidence_ids, expected_invalid)
                self.assertIsNone(result.validated_plan)

    def test_membership_is_exact_without_case_alias_or_fuzzy_matching(self) -> None:
        candidate = _evidence("Evidence-Alpha")
        variants = ("evidence-Alpha", "Evidence_Alpha", "Evidence-Alph")
        for variant in variants:
            with self.subTest(variant=variant):
                result = validate_evidence_references(
                    _plan(variant), "request-1", (candidate,)
                )
                self.assertEqual(
                    result.status, OrganizationStatus.ORGANIZATION_ERROR
                )
                self.assertEqual(result.invalid_evidence_ids, (variant,))

        invalid_whitespace_candidate = _evidence("Evidence-Alpha ")
        whitespace = validate_evidence_references(
            _plan("Evidence-Alpha"),
            "request-1",
            (invalid_whitespace_candidate,),
        )
        self.assertEqual(whitespace.status, OrganizationStatus.ORGANIZATION_ERROR)
        self.assertEqual(whitespace.reason, "candidate_snapshot_invalid")

    def test_cross_request_and_stale_snapshot_are_rejected_without_leakage(self) -> None:
        private_candidate = _evidence(
            "private-evidence", content="content from another request"
        )
        snapshot = CandidateEvidenceSnapshot.create(
            "request-private", (private_candidate,)
        )

        cross_request = validate_evidence_references(
            _plan("private-evidence"), "request-public", snapshot
        )
        stale = validate_evidence_references(
            _plan("private-evidence"),
            "request-private",
            snapshot,
            expected_snapshot_id="candidate-snapshot-stale",
        )

        self.assertEqual(cross_request.reason, "request_identity_mismatch")
        self.assertEqual(stale.reason, "stale_candidate_snapshot")
        for result in (cross_request, stale):
            self.assertEqual(result.status, OrganizationStatus.ORGANIZATION_ERROR)
            self.assertIsNone(result.validated_plan)
            self.assertEqual(result.invalid_evidence_ids, ())
            self.assertNotIn("content from another request", repr(result))

    def test_duplicate_candidate_ids_are_rejected_even_for_same_object(self) -> None:
        first = _evidence("evidence-1")
        conflicting = _evidence(
            "evidence-1",
            content="different content",
            source_name="different.md",
            document_id="different-document",
        )
        for candidates in ((first, first), (first, conflicting)):
            with self.subTest(conflicting=candidates[0] is not candidates[1]):
                result = validate_evidence_references(
                    _plan("evidence-1"), "request-1", candidates
                )
                self.assertEqual(
                    result.status, OrganizationStatus.ORGANIZATION_ERROR
                )
                self.assertEqual(result.reason, "candidate_snapshot_invalid")
                self.assertIsNone(result.validated_plan)
                with self.assertRaisesRegex(
                    CandidateSnapshotValidationError, "duplicate"
                ):
                    CandidateEvidenceSnapshot.create("request-1", candidates)

    def test_snapshot_validates_request_and_required_provenance(self) -> None:
        valid = _evidence("evidence-1")
        bad_source = Evidence(
            evidence_id="evidence-2",
            unit_id="unit-2",
            original_content="content",
            source_reference=SourceReference(name=" "),
            lineage=valid.lineage,
        )
        bad_lineage = Evidence(
            evidence_id="evidence-3",
            unit_id="unit-3",
            original_content="content",
            source_reference=valid.source_reference,
            lineage=KnowledgeLineage("", "p", "c", "v"),
        )

        for request_id, candidates in (
            (" request-1", (valid,)),
            ("request-1", (bad_source,)),
            ("request-1", (bad_lineage,)),
        ):
            with self.subTest(request_id=request_id, evidence=candidates[0].evidence_id):
                result = validate_evidence_references(
                    _plan(valid.evidence_id), request_id, candidates
                )
                self.assertEqual(
                    result.status, OrganizationStatus.ORGANIZATION_ERROR
                )
                self.assertEqual(result.reason, "candidate_snapshot_invalid")

    def test_fingerprint_is_stable_order_independent_and_provenance_sensitive(self) -> None:
        first = _evidence("evidence-1")
        second = _evidence("evidence-2")
        forward = CandidateEvidenceSnapshot.create(
            "request-1", (first, second)
        )
        reverse = CandidateEvidenceSnapshot.create(
            "request-1", (second, first)
        )
        changed_content = CandidateEvidenceSnapshot.create(
            "request-1", (_evidence("evidence-1", content="changed"), second)
        )
        changed_source = CandidateEvidenceSnapshot.create(
            "request-1", (_evidence("evidence-1", source_name="other.md"), second)
        )
        changed_lineage = CandidateEvidenceSnapshot.create(
            "request-1", (_evidence("evidence-1", document_id="other-doc"), second)
        )
        changed_structure = CandidateEvidenceSnapshot.create(
            "request-1",
            (_evidence("evidence-1", derived_structure={"threshold": 10}), second),
        )
        changed_request = CandidateEvidenceSnapshot.create(
            "request-2", (first, second)
        )

        self.assertEqual(forward.fingerprint, reverse.fingerprint)
        self.assertEqual(forward.snapshot_id, reverse.snapshot_id)
        for changed in (
            changed_content,
            changed_source,
            changed_lineage,
            changed_structure,
            changed_request,
        ):
            self.assertNotEqual(forward.fingerprint, changed.fingerprint)

    def test_inputs_and_validated_wrapper_are_immutable_and_not_hydrated(self) -> None:
        evidence = _evidence("evidence-1")
        plan = _plan("evidence-1")
        evidence_snapshot = copy.deepcopy(evidence)
        plan_snapshot = copy.deepcopy(plan)

        result = validate_evidence_references(
            plan, "request-1", (evidence,)
        )

        self.assertEqual(evidence, evidence_snapshot)
        self.assertEqual(plan, plan_snapshot)
        with self.assertRaises(FrozenInstanceError):
            result.validated_plan.request_id = "request-2"
        self.assertEqual(
            {item.name for item in fields(type(result.validated_plan))},
            {
                "plan",
                "request_id",
                "referenced_evidence_ids",
                "candidate_fingerprint",
                "snapshot_id",
            },
        )
        self.assertFalse(hasattr(result.validated_plan, "evidence"))
        self.assertFalse(hasattr(result.validated_plan, "hydrate"))


if __name__ == "__main__":
    unittest.main()
