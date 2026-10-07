"""Targeted tests for deterministic keyword/exact lookup."""

import unittest

from knowledge_system import (
    KeywordIndex,
    KnowledgeLineage,
    KnowledgeUnit,
    ParseStatus,
    SourceReference,
)


def _unit(
    unit_id: str,
    *,
    name: str,
    section: str,
    original: str,
    semantic: str,
) -> KnowledgeUnit:
    return KnowledgeUnit(
        unit_id=unit_id,
        original_content=original,
        semantic_content=semantic,
        metadata={"category": "示例规则"},
        source_reference=SourceReference(
            name=name,
            url=f"https://kb.example/{unit_id}",
            section=section,
        ),
        lineage=KnowledgeLineage(
            document_id=f"document-{name}",
            parser_version="p1",
            chunker_version="c1",
            processing_version="v1",
        ),
        parse_status=ParseStatus.SUCCESS,
    )


class KeywordIndexTest(unittest.TestCase):
    def setUp(self) -> None:
        self.rule = _unit(
            "u1",
            name="业务手册.html",
            section="审核规范 > 示例分级规则",
            original="关键术语甲需要人工核验。",
            semantic="说明关键术语甲的处理方式。",
        )
        self.other = _unit(
            "u2",
            name="其他说明.html",
            section="常见问题",
            original="无关内容。",
            semantic="其他说明。",
        )
        self.index = KeywordIndex([self.rule, self.other, self.rule])

    def test_exact_rule_name_locates_evidence(self) -> None:
        result = self.index.exact("示例分级规则")

        self.assertEqual([item.evidence_id for item in result], ["evidence-u1"])

    def test_keyword_locates_evidence(self) -> None:
        result = self.index.search("关键术语甲")

        self.assertEqual([item.unit_id for item in result], ["u1"])

    def test_document_name_locates_all_document_evidence(self) -> None:
        result = self.index.exact("业务手册.html")

        self.assertEqual([item.unit_id for item in result], ["u1"])

    def test_evidence_retains_complete_source_identity(self) -> None:
        evidence = self.index.search("关键术语甲")[0]

        self.assertEqual(evidence.evidence_id, "evidence-u1")
        self.assertEqual(evidence.source_reference.name, "业务手册.html")
        self.assertEqual(evidence.source_reference.url, "https://kb.example/u1")
        self.assertEqual(
            evidence.source_reference.section, "审核规范 > 示例分级规则"
        )
        self.assertIsNone(evidence.source_reference.page)
        self.assertEqual(evidence.lineage.document_id, "document-业务手册.html")

    def test_matching_is_normalized_but_not_semantic(self) -> None:
        self.assertEqual(self.index.search("  关键术语甲  ")[0].unit_id, "u1")
        self.assertEqual(self.index.search("不存在的近义词"), ())
        self.assertEqual(self.index.search(""), ())


if __name__ == "__main__":
    unittest.main()

