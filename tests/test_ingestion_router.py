"""Focused T-1400 contract tests; execution is deferred to P14-GATE."""

import tempfile
import unittest
from pathlib import Path

from knowledge_system.contracts import ParseStatus
from knowledge_system.ingestion import (
    IngestionFormat,
    LoaderRegistry,
    identify_format,
    ingest_file,
)


HTML = """<!doctype html><html><head><title>Example</title></head><body>
<h1>Guide</h1><p>Traceable content.</p>
</body></html>"""


class IngestionRouterTest(unittest.TestCase):
    def test_extension_identification_is_deterministic_and_case_insensitive(self) -> None:
        expected = {
            "a.html": IngestionFormat.HTML,
            "a.HTM": IngestionFormat.HTML,
            "a.md": IngestionFormat.MARKDOWN,
            "a.MARKDOWN": IngestionFormat.MARKDOWN,
            "a.CsV": IngestionFormat.CSV,
        }
        self.assertEqual(
            {name: identify_format(name) for name in expected}, expected
        )

    def test_unknown_extension_is_a_stable_failure(self) -> None:
        result = ingest_file("rules.txt")

        self.assertEqual(result.status, ParseStatus.FAILED)
        self.assertIsNone(result.source_format)
        self.assertEqual(result.source_reference.name, "rules.txt")
        self.assertEqual(result.units, ())
        self.assertEqual(result.errors[0].code, "UNSUPPORTED_EXTENSION")

    def test_registry_is_the_explicit_future_loader_extension_point(self) -> None:
        registry = LoaderRegistry()
        self.assertFalse(registry.is_registered(IngestionFormat.MARKDOWN))
        self.assertFalse(registry.is_registered(IngestionFormat.CSV))

    def test_existing_html_pipeline_is_reused_through_unified_result(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "Example.HTML"
            path.write_text(HTML, encoding="utf-8")

            first = ingest_file(path, url="https://example.test/rules")
            second = ingest_file(path, url="https://example.test/rules")

        self.assertEqual(first.status, ParseStatus.SUCCESS)
        self.assertEqual(first.source_format, IngestionFormat.HTML)
        self.assertEqual(first.source_reference.name, "Example.HTML")
        self.assertEqual(first.source_reference.url, "https://example.test/rules")
        self.assertTrue(first.units)
        self.assertEqual(
            [unit.unit_id for unit in first.units],
            [unit.unit_id for unit in second.units],
        )
        for unit in first.units:
            self.assertEqual(unit.source_reference.name, "Example.HTML")
            self.assertEqual(unit.parse_status, ParseStatus.SUCCESS)
            self.assertTrue(unit.lineage.document_id)
            self.assertEqual(unit.lineage.parser_version, "html-stdlib-v1")

    def test_html_read_and_empty_content_fail_without_traceback_text(self) -> None:
        missing = ingest_file("missing.html")
        self.assertEqual(missing.status, ParseStatus.FAILED)
        self.assertEqual(missing.errors[0].code, "SOURCE_READ_ERROR")
        self.assertNotIn("Traceback", missing.errors[0].message)

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "empty.htm"
            path.write_text("<html><body></body></html>", encoding="utf-8")
            empty = ingest_file(path)

        self.assertEqual(empty.status, ParseStatus.FAILED)
        self.assertEqual(empty.units, ())
        self.assertEqual(empty.errors[0].code, "HTML_PARSE_FAILED")
        self.assertTrue(empty.warnings)


if __name__ == "__main__":
    unittest.main()
