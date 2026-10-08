"""Targeted tests for deterministic Knowledge Request classification."""

import unittest

from knowledge_system import KnowledgeRequestType, classify_request


class RequestClassificationTest(unittest.TestCase):
    def test_supports_all_five_request_types(self) -> None:
        cases = (
            ("请找到《差旅报销制度》的原文链接", KnowledgeRequestType.LOCATE),
            ("列出全部差旅相关制度", KnowledgeRequestType.BROWSE),
            ("出差回来怎么报销", KnowledgeRequestType.DISCOVER),
            ("我现在有52个赞，是否符合升级条件？", KnowledgeRequestType.APPLY),
        )

        for query, expected in cases:
            with self.subTest(query=query):
                result = classify_request(query)
                self.assertEqual(result.request_type, expected)
                self.assertTrue(result.signals)
                self.assertTrue(result.signals[0].rule)

        out_of_scope = classify_request("帮我看明天天气", scope_covered=False)
        self.assertEqual(
            out_of_scope.request_type, KnowledgeRequestType.OUT_OF_SCOPE
        )
        self.assertEqual(out_of_scope.signals[0].rule, "scope_covered_false")

    def test_original_query_is_preserved_exactly(self) -> None:
        query = "  请找到\n《差旅报销制度》原文  "

        result = classify_request(query)

        self.assertEqual(result.original_query, query)
        self.assertEqual(result.request_type, KnowledgeRequestType.LOCATE)

    def test_case_comparison_has_priority_over_location_wording(self) -> None:
        result = classify_request("请找到适用于我这个 case 的规则，是否符合？")

        self.assertEqual(result.request_type, KnowledgeRequestType.APPLY)
        self.assertEqual(result.signals[0].rule, "case_comparison")

    def test_collection_request_has_priority_over_location_wording(self) -> None:
        result = classify_request("查找并列出所有差旅制度")

        self.assertEqual(result.request_type, KnowledgeRequestType.BROWSE)

    def test_topic_question_is_not_browse_due_to_collection_words(self) -> None:
        result = classify_request(
            "一个规范的 Python 项目通常应该包含哪些基本目录和配置文件？"
        )

        self.assertEqual(result.request_type, KnowledgeRequestType.DISCOVER)

    def test_explicit_scoped_collection_request_remains_browse(self) -> None:
        result = classify_request("浏览 python-learning 范围内的全部知识")

        self.assertEqual(result.request_type, KnowledgeRequestType.BROWSE)

    def test_unknown_wording_does_not_invent_out_of_scope_boundary(self) -> None:
        result = classify_request("差旅费票据")

        self.assertEqual(result.request_type, KnowledgeRequestType.DISCOVER)
        self.assertEqual(result.signals[0].rule, "discover_fallback")

    def test_explicit_out_of_scope_statement_is_auditable(self) -> None:
        result = classify_request("这个问题与企业知识库无关")

        self.assertEqual(result.request_type, KnowledgeRequestType.OUT_OF_SCOPE)
        self.assertEqual(result.signals[0].matched_text, "与企业知识库无关")

    def test_rejects_non_string_query(self) -> None:
        with self.assertRaises(TypeError):
            classify_request(None)  # type: ignore[arg-type]


if __name__ == "__main__":
    unittest.main()
