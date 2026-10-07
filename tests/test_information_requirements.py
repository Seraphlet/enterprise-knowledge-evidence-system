"""Targeted tests for rule-driven Information Requirements."""

import unittest

from knowledge_system import (
    Evidence,
    KnowledgeLineage,
    RequirementCertainty,
    RuleExtractionResult,
    RuleExtractionStatus,
    RuleStructure,
    SourceFragment,
    SourceReference,
    extract_rule_structure,
    generate_information_requirements,
)


def _evidence(content: str) -> Evidence:
    return Evidence(
        evidence_id="e-rule-501",
        unit_id="u-rule-501",
        original_content=content,
        source_reference=SourceReference(
            name="审核规则.html",
            url="https://kb.example/rules",
            section="升级规则",
        ),
        lineage=KnowledgeLineage(
            document_id="doc-rule-501",
            parser_version="p1",
            chunker_version="c1",
            processing_version="v1",
            source_position="section:upgrade",
        ),
    )


class InformationRequirementsTest(unittest.TestCase):
    def test_condition_generates_traceable_fact_type_and_question(self) -> None:
        evidence = _evidence(
            "条件：点赞数不少于50个 且 账号状态正常\n结果：可以申请升级"
        )
        extraction = extract_rule_structure(evidence)

        requirements = generate_information_requirements(extraction)

        self.assertEqual(len(requirements), 1)
        requirement = requirements[0]
        self.assertEqual(requirement.fact_type, "点赞数不少于50个 且 账号状态正常")
        self.assertIn(requirement.fact_type, requirement.question)
        self.assertEqual(requirement.metric_hints, ("点赞数",))
        self.assertEqual(requirement.certainty, RequirementCertainty.RELIABLE)
        self.assertEqual(requirement.evidence_id, evidence.evidence_id)
        self.assertIs(requirement.source_reference, evidence.source_reference)
        self.assertIs(requirement.lineage, evidence.lineage)
        self.assertEqual(
            evidence.original_content[
                requirement.condition.start : requirement.condition.end
            ],
            requirement.condition.text,
        )

    def test_requirement_id_is_stable_for_same_evidence_and_condition(self) -> None:
        evidence = _evidence("条件：账号状态正常\n结果：允许提交")
        extraction = extract_rule_structure(evidence)

        first = generate_information_requirements(extraction)
        second = generate_information_requirements(extraction)

        self.assertEqual(first, second)
        self.assertEqual(first[0].requirement_id, second[0].requirement_id)

    def test_empty_or_conditionless_partial_structure_generates_nothing(self) -> None:
        empty = extract_rule_structure(_evidence("普通说明文字。"))
        threshold_only = extract_rule_structure(_evidence("阈值：金额不超过1000元"))

        self.assertEqual(generate_information_requirements(empty), ())
        self.assertEqual(generate_information_requirements(threshold_only), ())

    def test_ambiguous_rule_marks_requirement_human_required(self) -> None:
        evidence = _evidence(
            "条件：原则上由负责人确认\n歧义：负责人范围不明确"
        )

        requirements = generate_information_requirements(
            extract_rule_structure(evidence)
        )

        self.assertEqual(len(requirements), 1)
        self.assertEqual(
            requirements[0].certainty, RequirementCertainty.HUMAN_REQUIRED
        )
        self.assertEqual(requirements[0].fact_type, "原则上由负责人确认")

    def test_multiple_conditions_remain_separate_open_requirements(self) -> None:
        content = "条件：账号状态正常\n条件：申请材料完整\n结果：允许提交"

        requirements = generate_information_requirements(
            extract_rule_structure(_evidence(content))
        )

        self.assertEqual(
            tuple(item.fact_type for item in requirements),
            ("账号状态正常", "申请材料完整"),
        )
        self.assertEqual(len({item.requirement_id for item in requirements}), 2)

    def test_generation_does_not_modify_source_or_extract_case_facts(self) -> None:
        evidence = _evidence("条件：点赞数至少50个\n结果：可以升级")
        original = evidence.original_content

        requirements = generate_information_requirements(
            extract_rule_structure(evidence)
        )

        self.assertEqual(evidence.original_content, original)
        self.assertIsNone(evidence.derived_structure)
        self.assertFalse(hasattr(requirements[0], "extracted_value"))
        self.assertFalse(hasattr(requirements[0], "comparison_result"))

    def test_non_source_condition_is_rejected_instead_of_becoming_requirement(self) -> None:
        evidence = _evidence("原始资料没有这个条件。")
        invalid = RuleExtractionResult(
            source_evidence=evidence,
            status=RuleExtractionStatus.PARTIAL,
            structure=RuleStructure(
                conditions=(SourceFragment("发明的条件", 0, 5),)
            ),
        )

        with self.assertRaises(ValueError):
            generate_information_requirements(invalid)

    def test_matching_slice_with_out_of_range_end_is_rejected(self) -> None:
        evidence = _evidence("abc")
        invalid = RuleExtractionResult(
            source_evidence=evidence,
            status=RuleExtractionStatus.PARTIAL,
            structure=RuleStructure(
                conditions=(SourceFragment("abc", 0, 999),)
            ),
        )

        with self.assertRaisesRegex(ValueError, "invalid source offsets"):
            generate_information_requirements(invalid)


if __name__ == "__main__":
    unittest.main()
