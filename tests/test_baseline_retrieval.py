"""Targeted tests for deterministic Locate/Browse behavior."""

import unittest

from knowledge_system import (
    BaselineRetrieval,
    KnowledgeLineage,
    KnowledgeUnit,
    MatchType,
    ParseStatus,
    RetrievalStatus,
    SourceReference,
)


def _unit(
    unit_id: str,
    *,
    section: str,
    content: str,
    scope: str = "payments",
    status: str = "active",
    permissions: list[str] | None = None,
) -> KnowledgeUnit:
    metadata: dict[str, object] = {"scope": scope, "status": status}
    if permissions is not None:
        metadata["required_permissions"] = permissions
    return KnowledgeUnit(
        unit_id=unit_id,
        original_content=content,
        semantic_content=content,
        metadata=metadata,
        source_reference=SourceReference(
            name="规则手册.html",
            url="https://kb.example/rules",
            section=section,
        ),
        lineage=KnowledgeLineage(
            document_id="doc-rules",
            parser_version="p1",
            chunker_version="c1",
            processing_version="v1",
        ),
        parse_status=ParseStatus.SUCCESS,
    )


class BaselineRetrievalTest(unittest.TestCase):
    def setUp(self) -> None:
        self.first = _unit(
            "u1", section="审核规范 > 精确规则名", content="关键术语甲的说明。"
        )
        self.second = _unit(
            "u2", section="审核规范 > 第二规则", content="关键术语乙的说明。"
        )
        self.denied = _unit(
            "u3",
            section="审核规范 > 受限规则",
            content="受限关键术语。",
            permissions=["secret.read"],
        )
        self.expired = _unit(
            "u4",
            section="审核规范 > 失效规则",
            content="失效关键术语。",
            status="已失效",
        )
        self.service = BaselineRetrieval(
            [self.first, self.second, self.denied, self.expired],
            covered_scopes=["payments", "hr"],
        )

    def test_locate_exact_then_keyword(self) -> None:
        exact = self.service.locate("精确规则名", scope="payments")
        keyword = self.service.locate("关键术语乙", scope="payments")

        self.assertEqual(exact.status, RetrievalStatus.FOUND)
        self.assertEqual(exact.match_type, MatchType.EXACT)
        self.assertEqual([item.unit_id for item in exact.evidence], ["u1"])
        self.assertEqual(keyword.match_type, MatchType.KEYWORD)
        self.assertEqual([item.unit_id for item in keyword.evidence], ["u2"])

    def test_browse_returns_scope_collection_not_top_one(self) -> None:
        result = self.service.browse("payments")

        self.assertEqual(result.status, RetrievalStatus.FOUND)
        self.assertEqual(result.match_type, MatchType.BROWSE)
        self.assertEqual([item.unit_id for item in result.evidence], ["u1", "u2"])

    def test_permission_and_status_filter_are_hard_candidate_boundaries(self) -> None:
        self.assertEqual(
            self.service.locate("受限关键术语", scope="payments").evidence, ()
        )
        self.assertEqual(
            self.service.locate("失效关键术语", scope="payments").evidence, ()
        )
        self.assertNotIn(
            "u3", [item.unit_id for item in self.service.browse("payments").evidence]
        )
        self.assertNotIn(
            "u4", [item.unit_id for item in self.service.browse("payments").evidence]
        )

    def test_browse_can_narrow_by_document_and_section_without_top_one(self) -> None:
        result = self.service.browse(
            "payments", document="规则手册.html", section="审核规范"
        )

        self.assertEqual([item.unit_id for item in result.evidence], ["u1", "u2"])

    def test_not_found_and_out_of_scope_messages_preserve_boundary(self) -> None:
        not_found = self.service.locate("未收录术语", scope="payments")
        out_of_scope = self.service.browse("legal")

        self.assertEqual(not_found.status, RetrievalStatus.NOT_FOUND)
        self.assertIn("当前知识库未检索到", not_found.message)
        self.assertIn("不代表企业没有", not_found.message)
        self.assertEqual(out_of_scope.status, RetrievalStatus.OUT_OF_SCOPE)
        self.assertIn("当前知识库不覆盖", out_of_scope.message)
        self.assertNotIn("企业没有", out_of_scope.message)


if __name__ == "__main__":
    unittest.main()

