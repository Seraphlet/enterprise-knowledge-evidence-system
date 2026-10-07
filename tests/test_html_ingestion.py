"""Targeted tests for dependency-free HTML ingestion."""

import unittest
from pathlib import Path

from knowledge_system import ParseStatus
from knowledge_system.ingestion import load_html, parse_html


FIXTURE = (
    Path(__file__).resolve().parents[1]
    / "data"
    / "fixtures"
    / "structured_document.html"
)


class HtmlIngestionTest(unittest.TestCase):
    def test_load_preserves_source_text_structure_and_no_page(self) -> None:
        raw_html = FIXTURE.read_text(encoding="utf-8-sig")
        document = load_html(FIXTURE)

        self.assertEqual(document.raw_html, raw_html)
        self.assertIn("不包含企业业务规则", document.original_text)
        self.assertEqual(document.title, "示例知识文档")
        self.assertEqual(
            [(section.level, section.path) for section in document.sections],
            [
                (1, ("文档结构示例",)),
                (2, ("文档结构示例", "示例章节")),
            ],
        )
        self.assertEqual(document.tables[0].headers, ("字段", "说明"))
        self.assertEqual(document.tables[0].rows, (("示例项", "中性测试内容"),))
        self.assertEqual(document.source_reference.name, FIXTURE.name)
        self.assertIsNone(document.source_reference.page)
        self.assertEqual(document.parse_status, ParseStatus.SUCCESS)

    def test_empty_document_fails_without_inventing_location(self) -> None:
        document = parse_html("<html><body></body></html>", name="empty.html")

        self.assertEqual(document.parse_status, ParseStatus.FAILED)
        self.assertEqual(document.original_text, "")
        self.assertIsNone(document.source_reference.page)

    def test_unclosed_structure_is_partial(self) -> None:
        document = parse_html("<html><body><h1>Incomplete", name="partial.html")

        self.assertEqual(document.parse_status, ParseStatus.PARTIAL)
        self.assertIn("unclosed heading", document.warnings)
        self.assertIsNone(document.source_reference.page)


if __name__ == "__main__":
    unittest.main()

