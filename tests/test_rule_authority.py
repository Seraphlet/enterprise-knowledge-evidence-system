"""Targeted tests for Derived Rule Structure authority checks."""

import unittest
from dataclasses import replace

from knowledge_system import (
    AuthorityViolation,
    Evidence,
    KnowledgeLineage,
    LogicalRelation,
    RuleAuthority,
    RuleExtractionResult,
    RuleExtractionStatus,
    RuleStructure,
    SourceFragment,
    SourceReference,
    Threshold,
    extract_rule_structure,
    validate_rule_authority,
)


def _evidence(content: str) -> Evidence:
    return Evidence(
        evidence_id="e-authority",
        unit_id="u-authority",
        original_content=content,
        source_reference=SourceReference(
            name="正式审核规则.html",
            url="https://kb.example/formal-rule",
            section="升级规则",
        ),
        lineage=KnowledgeLineage(
            document_id="doc-authority",
            parser_version="p1",
            chunker_version="c1",
            processing_version="v1",
            source_position="section:upgrade",
        ),
    )


class RuleAuthorityTest(unittest.TestCase):
    def test_all_derived_fields_validate_and_source_remains_authoritative(self) -> None:
        evidence = _evidence(
            "条件：点赞数不少于50个 且 账号状态正常\n"
            "例外：测试账号除外\n"
            "结果：可以申请升级\n"
            "歧义：账号状态标准未明确"
        )
        extraction = extract_rule_structure(evidence)
        original_evidence = evidence.to_dict()
        assert extraction.structure is not None
        original_structure = extraction.structure.to_dict()

        result = validate_rule_authority(extraction)

        self.assertEqual(
            result.authority_order,
            (
                RuleAuthority.SOURCE_EVIDENCE,
                RuleAuthority.DERIVED_RULE_STRUCTURE,
            ),
        )
        self.assertIs(result.source_reference, evidence.source_reference)
        self.assertIs(result.lineage, evidence.lineage)
        self.assertEqual(result.evidence_id, evidence.evidence_id)
        self.assertTrue(
            {
                "conditions[0]",
                "logical_relations[0]",
                "thresholds[0]",
                "exceptions[0]",
                "consequences[0]",
                "ambiguity[0]",
            }.issubset(result.checked_fields)
        )
        self.assertEqual(evidence.to_dict(), original_evidence)
        self.assertEqual(extraction.structure.to_dict(), original_structure)

    def test_each_fragment_group_rejects_untraceable_source(self) -> None:
        evidence = _evidence("abc")
        bad = SourceFragment("forged", 0, 3)
        structures = {
            "conditions": RuleStructure(conditions=(bad,)),
            "logical_relations": RuleStructure(
                logical_relations=(LogicalRelation("AND", bad),)
            ),
            "thresholds": RuleStructure(
                thresholds=(Threshold("x", ">=", 1, None, bad),)
            ),
            "exceptions": RuleStructure(exceptions=(bad,)),
            "consequences": RuleStructure(consequences=(bad,)),
            "ambiguity": RuleStructure(ambiguity=(bad,)),
        }

        for name, structure in structures.items():
            with self.subTest(name=name):
                extraction = RuleExtractionResult(
                    evidence, RuleExtractionStatus.PARTIAL, structure
                )
                with self.assertRaises(AuthorityViolation):
                    validate_rule_authority(extraction)

    def test_invalid_offsets_are_rejected_before_slice_truncation(self) -> None:
        evidence = _evidence("abc")
        extraction = RuleExtractionResult(
            evidence,
            RuleExtractionStatus.PARTIAL,
            RuleStructure(conditions=(SourceFragment("abc", 0, 999),)),
        )

        with self.assertRaisesRegex(AuthorityViolation, "invalid source offsets"):
            validate_rule_authority(extraction)

    def test_logical_relation_must_match_its_source_marker(self) -> None:
        evidence = _evidence("且")
        extraction = RuleExtractionResult(
            evidence,
            RuleExtractionStatus.PARTIAL,
            RuleStructure(
                logical_relations=(
                    LogicalRelation("OR", SourceFragment("且", 0, 1)),
                )
            ),
        )

        with self.assertRaisesRegex(AuthorityViolation, "contradicts"):
            validate_rule_authority(extraction)

    def test_threshold_values_must_match_literal_source(self) -> None:
        evidence = _evidence("金额不超过1000元")
        extraction = extract_rule_structure(evidence)
        assert extraction.structure is not None
        valid = extraction.structure.thresholds[0]
        forged = replace(valid, value=999)
        invalid = replace(
            extraction,
            structure=replace(extraction.structure, thresholds=(forged,)),
        )

        with self.assertRaisesRegex(AuthorityViolation, "value contradicts"):
            validate_rule_authority(invalid)

    def test_incomplete_source_reference_or_lineage_is_rejected(self) -> None:
        evidence = _evidence("条件：账号正常")
        extraction = extract_rule_structure(evidence)

        cases = (
            replace(evidence, source_reference=SourceReference(name="")),
            replace(evidence, lineage=replace(evidence.lineage, document_id="")),
        )
        for invalid_evidence in cases:
            with self.subTest(evidence=invalid_evidence):
                invalid = replace(extraction, source_evidence=invalid_evidence)
                with self.assertRaisesRegex(AuthorityViolation, "identity is empty"):
                    validate_rule_authority(invalid)

    def test_empty_extraction_is_valid_but_cannot_contain_structure(self) -> None:
        evidence = _evidence("普通说明。")
        empty = extract_rule_structure(evidence)

        result = validate_rule_authority(empty)

        self.assertEqual(result.checked_fields, ())
        invalid = RuleExtractionResult(
            evidence, RuleExtractionStatus.EMPTY, RuleStructure()
        )
        with self.assertRaises(AuthorityViolation):
            validate_rule_authority(invalid)


if __name__ == "__main__":
    unittest.main()
