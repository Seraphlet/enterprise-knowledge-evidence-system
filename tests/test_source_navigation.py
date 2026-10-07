"""Targeted source-navigation tests."""

import unittest
from pathlib import Path

from knowledge_system.ingestion import load_html, parse_html
from knowledge_system.knowledge_units import knowledge_units_from_html
from knowledge_system.source_navigation import SourceNavigator


FIXTURE = (
    Path(__file__).resolve().parents[1]
    / "data"
    / "fixtures"
    / "structured_document.html"
)


class SourceNavigationTest(unittest.TestCase):
    def setUp(self) -> None:
        first_document = load_html(FIXTURE, url="https://kb.example/structured")
        second_document = parse_html(
            "<html><body><h1>Other</h1><p>Other content.</p></body></html>",
            name="other.html",
            url="https://kb.example/other",
        )
        self.first_units = knowledge_units_from_html(first_document)
        self.navigator = SourceNavigator(
            [*self.first_units, *knowledge_units_from_html(second_document)]
        )

    def test_navigate_by_complete_document_name(self) -> None:
        result = self.navigator.by_document(FIXTURE.name)

        self.assertEqual(result, tuple(self.first_units))
        self.assertTrue(result)
        self.assertTrue(all(unit.source_reference.name == FIXTURE.name for unit in result))

    def test_navigate_by_document_id(self) -> None:
        document_id = self.first_units[0].lineage.document_id

        result = self.navigator.by_document(document_id)

        self.assertEqual(result, tuple(self.first_units))

    def test_navigate_section_and_descendants(self) -> None:
        root = self.navigator.by_section(FIXTURE.name, "文档结构示例")
        exact = self.navigator.by_section(
            FIXTURE.name,
            "文档结构示例",
            include_descendants=False,
        )

        self.assertGreater(len(root), len(exact))
        self.assertTrue(
            all(
                unit.source_reference.section == "文档结构示例"
                or unit.source_reference.section.startswith("文档结构示例 > ")
                for unit in root
            )
        )

    def test_result_retains_complete_source_identity(self) -> None:
        result = self.navigator.by_section(
            FIXTURE.name, "文档结构示例 > 示例章节"
        )

        self.assertTrue(result)
        for unit in result:
            source = unit.source_reference
            self.assertEqual(source.name, FIXTURE.name)
            self.assertEqual(source.url, "https://kb.example/structured")
            self.assertEqual(source.section, "文档结构示例 > 示例章节")
            self.assertIsNone(source.page)
            self.assertIsNone(source.sheet)


if __name__ == "__main__":
    unittest.main()

