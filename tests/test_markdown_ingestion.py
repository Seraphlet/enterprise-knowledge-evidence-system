"""Focused T-1401 tests; execution is deferred to P14-GATE."""

import tempfile
import unittest
from pathlib import Path

from knowledge_system.contracts import ParseStatus
from knowledge_system.ingestion import IngestionFormat, ingest_file, parse_markdown
from knowledge_system.knowledge_units import knowledge_units_from_markdown


MARKDOWN = """# Guide

Intro paragraph.

## Structures

- first item
- second item

> quoted source

```markdown
# not a heading
- not a list
| not | a table |
```

| Field | Meaning |
| --- | --- |
| A | Alpha |
"""


class MarkdownIngestionTest(unittest.TestCase):
    def test_document_preserves_source_paths_order_and_structures(self) -> None:
        document = parse_markdown(MARKDOWN, name="guide.md")

        self.assertEqual(document.raw_markdown, MARKDOWN)
        self.assertEqual(document.parse_status, ParseStatus.SUCCESS)
        self.assertEqual(document.title, "Guide")
        self.assertEqual(
            [(heading.level, heading.path) for heading in document.headings],
            [(1, ("Guide",)), (2, ("Guide", "Structures"))],
        )
        self.assertEqual(
            [block.kind for block in document.blocks],
            ["heading", "paragraph", "heading", "list", "quote", "code", "table"],
        )
        code = next(block for block in document.blocks if block.kind == "code")
        self.assertIn("# not a heading", code.text)
        self.assertEqual(len(document.headings), 2)
        self.assertIsNone(document.source_reference.page)
        self.assertIsNone(document.source_reference.sheet)

    def test_units_are_structural_stable_traceable_and_source_grounded(self) -> None:
        first = knowledge_units_from_markdown(parse_markdown(MARKDOWN, name="guide.md"))
        second = knowledge_units_from_markdown(parse_markdown(MARKDOWN, name="guide.md"))

        self.assertEqual(
            [unit.unit_id for unit in first], [unit.unit_id for unit in second]
        )
        self.assertEqual(len(first), 7)
        self.assertEqual(
            [unit.metadata["structure_kind"] for unit in first],
            ["heading", "paragraph", "heading", "list", "quote", "code", "table"],
        )
        table = first[-1]
        self.assertIn("| Field | Meaning |", table.original_content)
        self.assertIn("Guide > Structures", table.semantic_content)
        for unit in first:
            self.assertEqual(unit.source_reference.name, "guide.md")
            self.assertTrue(unit.lineage.source_position.startswith("lines:"))
            self.assertEqual(unit.lineage.parser_version, "markdown-lines-v1")
            self.assertEqual(unit.lineage.chunker_version, "markdown-structure-v1")

    def test_default_registry_supports_both_markdown_extensions(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            results = []
            for name in ("guide.MD", "guide.MarkDown"):
                path = root / name
                path.write_text(MARKDOWN, encoding="utf-8-sig")
                results.append(ingest_file(path, url="https://example.test/guide"))

        for result in results:
            self.assertEqual(result.status, ParseStatus.SUCCESS)
            self.assertEqual(result.source_format, IngestionFormat.MARKDOWN)
            self.assertTrue(result.units)
            self.assertEqual(result.source_reference.url, "https://example.test/guide")

    def test_empty_and_unclosed_fence_have_explicit_status(self) -> None:
        empty = parse_markdown("\n  \n", name="empty.md")
        self.assertEqual(empty.parse_status, ParseStatus.FAILED)
        self.assertIn("document contains no readable content", empty.warnings)

        partial = parse_markdown("# Guide\n\n```text\nvalue", name="partial.md")
        self.assertEqual(partial.parse_status, ParseStatus.PARTIAL)
        self.assertIn("unclosed fenced code block at line 3", partial.warnings)
        units = knowledge_units_from_markdown(partial)
        self.assertTrue(units)
        self.assertTrue(all(unit.parse_status is ParseStatus.PARTIAL for unit in units))

    def test_read_and_decode_failures_use_safe_common_result(self) -> None:
        missing = ingest_file("missing.markdown")
        self.assertEqual(missing.errors[0].code, "SOURCE_READ_ERROR")

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "invalid.md"
            path.write_bytes(b"\xff\xfe\x00")
            invalid = ingest_file(path)

        self.assertEqual(invalid.status, ParseStatus.FAILED)
        self.assertEqual(invalid.errors[0].code, "SOURCE_ENCODING_ERROR")
        self.assertNotIn("Traceback", invalid.errors[0].message)


if __name__ == "__main__":
    unittest.main()
