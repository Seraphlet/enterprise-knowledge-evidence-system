"""Targeted tests for structure-aware knowledge unit creation."""

import unittest

from knowledge_system.ingestion import parse_html
from knowledge_system.knowledge_units import knowledge_units_from_html


HTML = """<!doctype html><html><head><title>Example</title></head><body>
<h1>Guide</h1><p>Introduction text.</p>
<h2>Details</h2><p>Detailed explanation.</p>
<table><tr><th>Field</th><th>Meaning</th></tr><tr><td>A</td><td>Alpha</td></tr></table>
</body></html>"""


class KnowledgeUnitCreationTest(unittest.TestCase):
    def test_units_follow_sections_and_tables(self) -> None:
        document = parse_html(HTML, name="example.html")
        units = knowledge_units_from_html(document)

        self.assertEqual([unit.metadata["structure_kind"] for unit in units], [
            "section", "section", "table"
        ])
        self.assertEqual(units[0].original_content, "Guide\nIntroduction text.")
        self.assertEqual(units[1].source_reference.section, "Guide > Details")
        self.assertEqual(units[2].original_content, "Field | Meaning\nA | Alpha")
        self.assertIsNone(units[2].source_reference.page)

    def test_ids_are_stable_and_every_unit_has_lineage_and_source(self) -> None:
        first = knowledge_units_from_html(parse_html(HTML, name="example.html"))
        second = knowledge_units_from_html(parse_html(HTML, name="example.html"))

        self.assertEqual([unit.unit_id for unit in first], [unit.unit_id for unit in second])
        self.assertEqual(len({unit.unit_id for unit in first}), len(first))
        for unit in first:
            self.assertEqual(unit.source_reference.name, "example.html")
            self.assertTrue(unit.lineage.document_id)
            self.assertTrue(unit.lineage.source_position)
            self.assertEqual(unit.lineage.chunker_version, "html-structure-v1")

    def test_original_and_semantic_content_are_separate(self) -> None:
        unit = knowledge_units_from_html(parse_html(HTML, name="example.html"))[1]

        self.assertEqual(unit.original_content, "Details\nDetailed explanation.")
        self.assertIn("Example > Guide > Details", unit.semantic_content)
        self.assertNotEqual(unit.original_content, unit.semantic_content)

    def test_long_structural_block_is_not_split_by_character_count(self) -> None:
        paragraph = "x" * 20_000
        html = f"<html><body><h1>One section</h1><p>{paragraph}</p></body></html>"

        units = knowledge_units_from_html(parse_html(html, name="long.html"))

        self.assertEqual(len(units), 1)
        self.assertEqual(units[0].original_content, f"One section\n{paragraph}")


if __name__ == "__main__":
    unittest.main()

