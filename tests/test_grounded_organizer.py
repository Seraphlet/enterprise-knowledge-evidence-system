"""Focused tests for the organization-only grounded organizer boundary."""

import copy
import json
import unittest
from dataclasses import FrozenInstanceError, fields

from knowledge_system import (
    ALLOWED_OUTPUT_SCHEMA,
    ORGANIZATION_ONLY_AUTHORITY,
    CaseSourceSpan,
    ComparisonOutcome,
    ConditionComparison,
    Evidence,
    ExtractedRequirementFact,
    FactExtractionStatus,
    FactObservation,
    InformationRequirement,
    KnowledgeLineage,
    OrganizationStatus,
    OrganizerBackendRequest,
    OrganizerInput,
    RequirementCertainty,
    RuleExtractionResult,
    RuleExtractionStatus,
    RuleStructure,
    SourceFragment,
    SourceReference,
    Threshold,
    organize_evidence_map,
)


def _grounded_objects():
    text = "点赞数至少50个"
    source_reference = SourceReference(
        name="policy.md", url="https://example.invalid/policy", section="A", page=2
    )
    lineage = KnowledgeLineage(
        document_id="doc-1",
        parser_version="parser-1",
        chunker_version="chunker-1",
        processing_version="processing-1",
        source_position="line-4",
        parent_section_id="section-A",
    )
    evidence = Evidence(
        evidence_id="evidence-1",
        unit_id="unit-1",
        original_content=text,
        source_reference=source_reference,
        lineage=lineage,
    )
    source = SourceFragment(text, 0, len(text))
    threshold = Threshold("点赞数", "至少", 50, "个", source)
    structure = RuleStructure(conditions=(source,), thresholds=(threshold,))
    extraction = RuleExtractionResult(
        evidence, RuleExtractionStatus.STRUCTURED, structure
    )
    requirement = InformationRequirement(
        requirement_id="requirement-1",
        fact_type="numeric",
        question="当前点赞数是多少？",
        certainty=RequirementCertainty.RELIABLE,
        metric_hints=("点赞数",),
        condition=source,
        evidence_id=evidence.evidence_id,
        source_reference=source_reference,
        lineage=lineage,
        threshold=threshold,
    )
    case_source = CaseSourceSpan(0, 4, 7, "52个")
    observation = FactObservation(52, case_source, "个")
    fact = ExtractedRequirementFact(
        requirement_id=requirement.requirement_id,
        evidence_id=evidence.evidence_id,
        fact_type="numeric",
        status=FactExtractionStatus.FOUND,
        value=52,
        observations=(observation,),
    )
    comparison = ConditionComparison(
        threshold,
        requirement,
        fact,
        ComparisonOutcome.SATISFIED,
        "numeric_condition_satisfied",
    )
    return evidence, extraction, comparison


def _plan(evidence_id="evidence-1"):
    return {
        "plan_id": "plan-1",
        "sections": [
            {
                "section_id": "section-1",
                "order": 0,
                "groups": [
                    {
                        "group_id": "group-1",
                        "order": 0,
                        "evidence_ids": [evidence_id],
                    }
                ],
            }
        ],
    }


class GroundedOrganizerTest(unittest.TestCase):
    def test_complete_input_view_is_grounded_and_read_only(self) -> None:
        evidence, extraction, comparison = _grounded_objects()

        organizer_input = OrganizerInput.from_authoritative(
            (evidence,),
            rule_extractions=(extraction,),
            comparison_results=(comparison,),
        )

        view = organizer_input.candidates[0]
        self.assertEqual(view.evidence_id, evidence.evidence_id)
        self.assertIs(view.original_content, evidence.original_content)
        self.assertIs(view.source_reference, evidence.source_reference)
        self.assertIs(view.lineage, evidence.lineage)
        self.assertIs(view.rule_structure, extraction.structure)
        self.assertIs(organizer_input.comparison_results[0], comparison)
        with self.assertRaises(FrozenInstanceError):
            view.original_content = "rewritten"
        with self.assertRaises(FrozenInstanceError):
            organizer_input.candidates = ()

    def test_backend_receives_explicit_authority_and_valid_plan_is_canonical(self) -> None:
        evidence, extraction, comparison = _grounded_objects()
        organizer_input = OrganizerInput.from_authoritative(
            (evidence,),
            rule_extractions=(extraction,),
            comparison_results=(comparison,),
        )
        captured = []

        def backend(request):
            captured.append(request)
            return _plan()

        first = organize_evidence_map(organizer_input, backend)
        second = organize_evidence_map(organizer_input, backend)

        self.assertEqual(first.status, OrganizationStatus.SUCCESS)
        self.assertEqual(first, second)
        self.assertEqual(first.plan.to_json(), second.plan.to_json())
        self.assertEqual(captured[0].authority, ORGANIZATION_ONLY_AUTHORITY)
        self.assertEqual(captured[0].allowed_output_schema, ALLOWED_OUTPUT_SCHEMA)
        self.assertIs(captured[0].organizer_input, organizer_input)
        with self.assertRaises(FrozenInstanceError):
            captured[0].authority = "WRITE_KNOWLEDGE"

    def test_mapping_and_json_outputs_are_both_strictly_parsed(self) -> None:
        evidence, extraction, _ = _grounded_objects()
        organizer_input = OrganizerInput.from_authoritative(
            (evidence,), rule_extractions=(extraction,)
        )

        mapping_result = organize_evidence_map(
            organizer_input, lambda request: _plan()
        )
        json_result = organize_evidence_map(
            organizer_input,
            lambda request: json.dumps(_plan(), ensure_ascii=False),
        )

        self.assertEqual(mapping_result.plan, json_result.plan)

    def test_knowledge_injection_is_an_explicit_organization_error(self) -> None:
        evidence, extraction, _ = _grounded_objects()
        organizer_input = OrganizerInput.from_authoritative(
            (evidence,), rule_extractions=(extraction,)
        )
        injections = (
            ((), "content"),
            ((), "comparison_result"),
            (("sections", 0), "source"),
            (("sections", 0), "page"),
            (("sections", 0, "groups", 0), "facts"),
            (("sections", 0, "groups", 0), "threshold"),
            (("sections", 0, "groups", 0), "condition"),
            (("sections", 0, "groups", 0), "url"),
            (("sections", 0, "groups", 0), "free_text"),
        )
        for path, field_name in injections:
            with self.subTest(path=path, field=field_name):
                output = _plan()
                target = output
                for part in path:
                    target = target[part]
                target[field_name] = "invented"
                result = organize_evidence_map(
                    organizer_input, lambda request, value=output: value
                )
                self.assertEqual(
                    result.status, OrganizationStatus.ORGANIZATION_ERROR
                )
                self.assertIsNone(result.plan)
                self.assertEqual(result.reason, "strict_plan_validation_failed")

    def test_malformed_wrong_type_and_backend_failure_never_fall_back_to_text(self) -> None:
        evidence, _, _ = _grounded_objects()
        organizer_input = OrganizerInput.from_authoritative((evidence,))

        def failing_backend(request):
            raise RuntimeError("backend details must not become output")

        cases = (
            (lambda request: "organize this however you want", "strict_plan_validation_failed"),
            (lambda request: ["evidence-1"], "output_type_not_allowed"),
            (failing_backend, "backend_exception"),
        )
        for backend, expected_reason in cases:
            with self.subTest(reason=expected_reason):
                result = organize_evidence_map(organizer_input, backend)
                self.assertEqual(
                    result.status, OrganizationStatus.ORGANIZATION_ERROR
                )
                self.assertEqual(result.reason, expected_reason)
                self.assertIsNone(result.plan)

    def test_inputs_are_unchanged_and_runtime_dependency_is_not_state(self) -> None:
        evidence, extraction, comparison = _grounded_objects()
        organizer_input = OrganizerInput.from_authoritative(
            (evidence,),
            rule_extractions=(extraction,),
            comparison_results=(comparison,),
        )
        evidence_snapshot = copy.deepcopy(evidence)
        extraction_snapshot = copy.deepcopy(extraction)
        comparison_snapshot = copy.deepcopy(comparison)

        result = organize_evidence_map(organizer_input, lambda request: _plan())

        self.assertEqual(result.status, OrganizationStatus.SUCCESS)
        self.assertEqual(evidence, evidence_snapshot)
        self.assertEqual(extraction, extraction_snapshot)
        self.assertEqual(comparison, comparison_snapshot)
        self.assertEqual(
            {item.name for item in fields(OrganizerInput)},
            {"candidates", "comparison_results"},
        )
        self.assertNotIn("backend", {item.name for item in fields(OrganizerInput)})

    def test_unknown_but_well_shaped_evidence_id_passes_without_hydration(self) -> None:
        evidence, _, _ = _grounded_objects()
        organizer_input = OrganizerInput.from_authoritative((evidence,))

        result = organize_evidence_map(
            organizer_input,
            lambda request: _plan("evidence-not-in-candidate-set"),
        )

        self.assertEqual(result.status, OrganizationStatus.SUCCESS)
        self.assertEqual(
            result.plan.sections[0].groups[0].evidence_ids,
            ("evidence-not-in-candidate-set",),
        )
        self.assertEqual(
            {item.name for item in fields(type(result))},
            {"status", "plan", "reason"},
        )
        self.assertFalse(hasattr(result, "hydrated_evidence"))
        self.assertFalse(hasattr(result.plan, "hydrate"))


if __name__ == "__main__":
    unittest.main()
