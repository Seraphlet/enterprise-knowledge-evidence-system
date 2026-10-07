"""Focused tests for deterministic authoritative Evidence Map hydration."""

import copy
import json
import unittest
from dataclasses import FrozenInstanceError, fields, replace

from knowledge_system import (
    CandidateEvidenceSnapshot,
    Evidence,
    EvidenceMap,
    EvidenceMapPlan,
    HydrationStatus,
    KnowledgeLineage,
    SourceReference,
    ValidatedPlan,
    hydrate_evidence_map,
    validate_evidence_references,
)


def _evidence(evidence_id: str, marker: str) -> Evidence:
    return Evidence(
        evidence_id=evidence_id,
        unit_id=f"unit-{marker}",
        original_content=f"Original {marker} 原文。",
        source_reference=SourceReference(
            name=f"Policy-{marker}.md",
            url=f"https://example.invalid/{marker}",
            section=f"Section {marker}",
            page=len(marker),
            sheet=None,
        ),
        lineage=KnowledgeLineage(
            document_id=f"document-{marker}",
            parser_version="parser-1",
            chunker_version="chunker-1",
            processing_version="processing-1",
            source_position=f"line-{marker}",
            parent_section_id=f"parent-{marker}",
        ),
        derived_structure={
            "conditions": [f"condition-{marker}"],
            "threshold": {"value": len(marker), "unit": "个"},
        },
    )


def _multi_plan() -> EvidenceMapPlan:
    return EvidenceMapPlan.from_dict(
        {
            "plan_id": "plan-1",
            "sections": [
                {
                    "section_id": "section-b",
                    "order": 1,
                    "groups": [
                        {
                            "group_id": "group-b",
                            "order": 0,
                            "evidence_ids": ["evidence-c"],
                        }
                    ],
                },
                {
                    "section_id": "section-a",
                    "order": 0,
                    "groups": [
                        {
                            "group_id": "group-a2",
                            "order": 1,
                            "evidence_ids": ["evidence-b"],
                        },
                        {
                            "group_id": "group-a1",
                            "order": 0,
                            "evidence_ids": ["evidence-a"],
                        },
                    ],
                },
            ],
        }
    )


def _validated_fixture():
    candidates = (
        _evidence("evidence-a", "a"),
        _evidence("evidence-b", "bb"),
        _evidence("evidence-c", "ccc"),
    )
    snapshot = CandidateEvidenceSnapshot.create("request-1", candidates)
    guard = validate_evidence_references(
        _multi_plan(),
        "request-1",
        snapshot,
        expected_snapshot_id=snapshot.snapshot_id,
    )
    return candidates, snapshot, guard.validated_plan


class EvidenceMapHydrationTest(unittest.TestCase):
    def test_hydrates_canonical_section_group_and_evidence_order(self) -> None:
        _, snapshot, validated = _validated_fixture()

        result = hydrate_evidence_map(validated, snapshot)

        self.assertEqual(result.status, HydrationStatus.SUCCESS)
        hydrated = result.evidence_map
        self.assertEqual(
            tuple(section.section_id for section in hydrated.sections),
            ("section-a", "section-b"),
        )
        self.assertEqual(
            tuple(group.group_id for group in hydrated.sections[0].groups),
            ("group-a1", "group-a2"),
        )
        self.assertEqual(
            tuple(
                entry.evidence_id
                for section in hydrated.sections
                for group in section.groups
                for entry in group.entries
            ),
            ("evidence-a", "evidence-b", "evidence-c"),
        )

    def test_entries_are_exact_authoritative_evidence_without_rewriting(self) -> None:
        candidates, snapshot, validated = _validated_fixture()

        result = hydrate_evidence_map(validated, snapshot)
        entries = {
            entry.evidence_id: entry
            for section in result.evidence_map.sections
            for group in section.groups
            for entry in group.entries
        }

        for evidence in candidates:
            with self.subTest(evidence=evidence.evidence_id):
                entry = entries[evidence.evidence_id]
                self.assertIs(entry.original_content, evidence.original_content)
                self.assertEqual(entry.derived_structure, evidence.derived_structure)
                self.assertIs(entry.source_reference, evidence.source_reference)
                self.assertIs(entry.lineage, evidence.lineage)
                self.assertEqual(entry.unit_id, evidence.unit_id)
                self.assertEqual(entry.to_dict(), evidence.to_dict())

    def test_raw_plan_is_rejected_instead_of_hydrated(self) -> None:
        _, snapshot, _ = _validated_fixture()

        result = hydrate_evidence_map(_multi_plan(), snapshot)

        self.assertEqual(result.status, HydrationStatus.HYDRATION_ERROR)
        self.assertEqual(result.reason, "validated_plan_required")
        self.assertIsNone(result.evidence_map)
        self.assertEqual(result.failed_evidence_ids, ())

    def test_request_snapshot_and_fingerprint_bindings_are_exact(self) -> None:
        _, snapshot, validated = _validated_fixture()
        variants = (
            (
                replace(validated, request_id="request-2"),
                "request_binding_mismatch",
            ),
            (
                replace(validated, snapshot_id="candidate-snapshot-other"),
                "snapshot_binding_mismatch",
            ),
            (
                replace(validated, candidate_fingerprint="0" * 64),
                "fingerprint_binding_mismatch",
            ),
            (
                replace(validated, referenced_evidence_ids=("evidence-a",)),
                "validated_reference_binding_mismatch",
            ),
        )
        for changed, reason in variants:
            with self.subTest(reason=reason):
                result = hydrate_evidence_map(changed, snapshot)
                self.assertEqual(result.status, HydrationStatus.HYDRATION_ERROR)
                self.assertEqual(result.reason, reason)
                self.assertIsNone(result.evidence_map)
                self.assertEqual(result.binding.request_id, changed.request_id)

    def test_toctou_content_source_lineage_and_structure_changes_fail(self) -> None:
        mutators = (
            lambda evidence: object.__setattr__(
                evidence, "original_content", "tampered content"
            ),
            lambda evidence: object.__setattr__(
                evidence.source_reference, "name", "tampered-source.md"
            ),
            lambda evidence: object.__setattr__(
                evidence.lineage, "document_id", "tampered-document"
            ),
            lambda evidence: evidence.derived_structure.__setitem__(
                "invented", True
            ),
        )
        for index, mutate in enumerate(mutators):
            with self.subTest(index=index):
                candidates, snapshot, validated = _validated_fixture()
                mutate(candidates[0])

                result = hydrate_evidence_map(validated, snapshot)

                self.assertEqual(result.status, HydrationStatus.HYDRATION_ERROR)
                self.assertEqual(
                    result.reason, "candidate_snapshot_integrity_failed"
                )
                self.assertIsNone(result.evidence_map)

    def test_missing_authoritative_id_is_all_or_nothing(self) -> None:
        evidence = _evidence("evidence-a", "a")
        snapshot = CandidateEvidenceSnapshot.create("request-1", (evidence,))
        plan = EvidenceMapPlan.from_dict(
            {
                "plan_id": "plan-missing",
                "sections": [
                    {
                        "section_id": "section-1",
                        "order": 0,
                        "groups": [
                            {
                                "group_id": "group-1",
                                "order": 0,
                                "evidence_ids": ["evidence-a", "evidence-missing"],
                            }
                        ],
                    }
                ],
            }
        )
        references = ("evidence-a", "evidence-missing")
        forged_binding = ValidatedPlan(
            plan=plan,
            request_id=snapshot.request_id,
            referenced_evidence_ids=references,
            candidate_fingerprint=snapshot.fingerprint,
            snapshot_id=snapshot.snapshot_id,
        )

        result = hydrate_evidence_map(forged_binding, snapshot)

        self.assertEqual(result.status, HydrationStatus.HYDRATION_ERROR)
        self.assertEqual(result.reason, "authoritative_evidence_missing")
        self.assertEqual(result.failed_evidence_ids, ("evidence-missing",))
        self.assertIsNone(result.evidence_map)

    def test_serialization_is_deterministic_round_trip_and_inputs_unchanged(self) -> None:
        candidates, snapshot, validated = _validated_fixture()
        candidates_before = copy.deepcopy(candidates)
        snapshot_before = copy.deepcopy(snapshot)
        validated_before = copy.deepcopy(validated)

        first = hydrate_evidence_map(validated, snapshot)
        second = hydrate_evidence_map(validated, snapshot)
        encoded = first.evidence_map.to_json()
        decoded = EvidenceMap.from_json(encoded)

        self.assertEqual(first, second)
        self.assertEqual(decoded, first.evidence_map)
        self.assertEqual(decoded.to_json(), encoded)
        self.assertEqual(json.loads(encoded), first.evidence_map.to_dict())
        self.assertEqual(candidates, candidates_before)
        self.assertEqual(snapshot, snapshot_before)
        self.assertEqual(validated, validated_before)

    def test_output_is_immutable_and_contains_no_generation_or_decision_fields(self) -> None:
        _, snapshot, validated = _validated_fixture()
        result = hydrate_evidence_map(validated, snapshot)
        hydrated = result.evidence_map
        entry = hydrated.sections[0].groups[0].entries[0]

        with self.assertRaises(FrozenInstanceError):
            hydrated.plan_id = "changed"
        with self.assertRaises(FrozenInstanceError):
            entry.original_content = "changed"
        structure = entry.derived_structure
        structure["invented"] = True
        self.assertNotIn("invented", entry.derived_structure)
        self.assertEqual(
            {item.name for item in fields(type(entry))},
            {
                "evidence_id",
                "unit_id",
                "original_content",
                "source_reference",
                "lineage",
                "_derived_structure_json",
            },
        )
        serialized = hydrated.to_dict()
        self.assertNotIn("summary", repr(serialized))
        self.assertNotIn("answer", repr(serialized))
        self.assertNotIn("decision", repr(serialized))
        self.assertNotIn("comparison", repr(serialized))
        self.assertFalse(hasattr(hydrated, "verification"))
        self.assertFalse(hasattr(hydrated, "feedback"))


if __name__ == "__main__":
    unittest.main()
