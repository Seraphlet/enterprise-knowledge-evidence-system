"""Targeted tests for conservative derived Rule Structure extraction."""

import unittest

from knowledge_system import (
    Evidence,
    KnowledgeLineage,
    RuleExtractionStatus,
    SourceReference,
    extract_rule_structure,
)


def _evidence(content: str) -> Evidence:
    return Evidence(
        evidence_id="e-rule",
        unit_id="u-rule",
        original_content=content,
        source_reference=SourceReference(
            name="审核规则.html", section="升级规则"
        ),
        lineage=KnowledgeLineage(
            document_id="doc-rule",
            parser_version="p1",
            chunker_version="c1",
            processing_version="v1",
        ),
    )


class RuleStructureExtractionTest(unittest.TestCase):
    def test_extracts_supported_fields_from_explicit_source(self) -> None:
        content = (
            "条件：点赞数不少于50个 且 账号状态正常\n"
            "逻辑关系：AND\n"
            "阈值：点赞数不少于50个\n"
            "例外：测试账号除外\n"
            "结果：可以申请升级\n"
            "歧义：‘正常’的判断标准未明确"
        )
        evidence = _evidence(content)

        result = extract_rule_structure(evidence)

        self.assertIs(result.source_evidence, evidence)
        self.assertEqual(result.status, RuleExtractionStatus.STRUCTURED)
        self.assertIsNotNone(result.structure)
        structure = result.structure
        assert structure is not None
        self.assertIn("点赞数不少于50个", structure.conditions[0].text)
        self.assertTrue(
            any(item.relation == "AND" for item in structure.logical_relations)
        )
        self.assertEqual(structure.thresholds[0].metric, "点赞数")
        self.assertEqual(structure.thresholds[0].operator, "不少于")
        self.assertEqual(structure.thresholds[0].value, 50)
        self.assertEqual(structure.thresholds[0].unit, "个")
        self.assertTrue(any("除外" in item.text for item in structure.exceptions))
        self.assertEqual(structure.consequences[0].text, "可以申请升级")
        self.assertTrue(any("未明确" in item.text for item in structure.ambiguity))

    def test_natural_high_confidence_sentence_can_be_structured(self) -> None:
        evidence = _evidence(
            "如果点赞数 >= 50 且账号状态正常，则可以申请升级；测试账号除外。"
        )

        result = extract_rule_structure(evidence)

        self.assertEqual(result.status, RuleExtractionStatus.STRUCTURED)
        assert result.structure is not None
        self.assertEqual(result.structure.conditions[0].text, "点赞数 >= 50 且账号状态正常")
        self.assertEqual(result.structure.consequences[0].text, "可以申请升级")
        self.assertEqual(result.structure.thresholds[0].value, 50)
        self.assertTrue(
            any(item.relation == "AND" for item in result.structure.logical_relations)
        )

    def test_unreliable_unstructured_text_returns_empty(self) -> None:
        evidence = _evidence("本章节介绍相关事项，详情请参阅正式资料。")

        result = extract_rule_structure(evidence)

        self.assertEqual(result.status, RuleExtractionStatus.EMPTY)
        self.assertIsNone(result.structure)

    def test_incidental_or_in_explanatory_prose_does_not_create_structure(self) -> None:
        evidence = _evidence(
            "本章节介绍申请或报销事项，详情请参阅正式资料。"
        )

        result = extract_rule_structure(evidence)

        self.assertEqual(result.status, RuleExtractionStatus.EMPTY)
        self.assertIsNone(result.structure)

    def test_single_reliable_field_is_partial_not_forced_complete(self) -> None:
        evidence = _evidence("阈值：金额不超过1000元")

        result = extract_rule_structure(evidence)

        self.assertEqual(result.status, RuleExtractionStatus.PARTIAL)
        assert result.structure is not None
        self.assertEqual(result.structure.conditions, ())
        self.assertEqual(result.structure.consequences, ())
        self.assertEqual(result.structure.thresholds[0].metric, "金额")

    def test_ambiguity_is_retained_without_forced_interpretation(self) -> None:
        content = "条件：原则上由负责人确认\n歧义：负责人范围不明确"

        result = extract_rule_structure(_evidence(content))

        self.assertEqual(result.status, RuleExtractionStatus.PARTIAL)
        assert result.structure is not None
        ambiguity_text = tuple(item.text for item in result.structure.ambiguity)
        self.assertIn("负责人范围不明确", ambiguity_text)
        self.assertIn("原则上", ambiguity_text)
        self.assertIn("不明确", ambiguity_text)

    def test_every_derived_fragment_is_verbatim_and_points_into_source(self) -> None:
        content = "当金额 <= 1000元时，可以自动提交；特殊项目除外。"
        evidence = _evidence(content)

        result = extract_rule_structure(evidence)

        self.assertEqual(evidence.original_content, content)
        self.assertIsNone(evidence.derived_structure)
        assert result.structure is not None
        fragments = [
            *result.structure.conditions,
            *(item.source for item in result.structure.logical_relations),
            *(item.source for item in result.structure.thresholds),
            *result.structure.exceptions,
            *result.structure.consequences,
            *result.structure.ambiguity,
        ]
        for fragment in fragments:
            self.assertEqual(content[fragment.start : fragment.end], fragment.text)

    def test_structure_serializes_as_separate_derived_mapping(self) -> None:
        evidence = _evidence("条件：点赞数至少50个\n结果：可以升级")
        result = extract_rule_structure(evidence)

        assert result.structure is not None
        value = result.structure.to_dict()

        self.assertEqual(
            set(value),
            {
                "conditions",
                "logical_relations",
                "thresholds",
                "exceptions",
                "consequence",
                "ambiguity",
            },
        )
        self.assertNotIn("original_content", value)
        self.assertEqual(result.source_evidence.original_content, evidence.original_content)


if __name__ == "__main__":
    unittest.main()
