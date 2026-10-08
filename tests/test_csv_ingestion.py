"""Focused T-1402 tests; execution is deferred to P14-GATE."""

import tempfile
import unittest
from pathlib import Path

from knowledge_system.contracts import ParseStatus
from knowledge_system.ingestion import IngestionFormat, ingest_file, parse_csv
from knowledge_system.knowledge_units import knowledge_units_from_csv


class CsvIngestionTest(unittest.TestCase):
    def test_ordered_headers_rows_and_stable_units(self) -> None:
        raw = "Name,Rule,Value\nAlpha,Threshold,52\nBeta,Status,active\n"
        document = parse_csv(raw, name="rules.csv", url="https://example.test/rules")
        first = knowledge_units_from_csv(document)
        second = knowledge_units_from_csv(parse_csv(raw, name="rules.csv", url="https://example.test/rules"))

        self.assertEqual(document.parse_status, ParseStatus.SUCCESS)
        self.assertEqual(document.headers, ("Name", "Rule", "Value"))
        self.assertEqual([row.start_line for row in document.rows], [2, 3])
        self.assertEqual(document.rows[0].values, ("Alpha", "Threshold", "52"))
        self.assertEqual(
            [(field.header, field.value) for field in document.rows[0].fields],
            [("Name", "Alpha"), ("Rule", "Threshold"), ("Value", "52")],
        )
        self.assertEqual([unit.unit_id for unit in first], [unit.unit_id for unit in second])
        self.assertEqual(first[0].original_content, "Alpha,Threshold,52")
        self.assertIn("Name: Alpha", first[0].semantic_content)
        self.assertEqual(first[0].source_reference.name, "rules.csv")
        self.assertEqual(first[0].source_reference.section, "CSV row 2")
        self.assertIsNone(first[0].source_reference.page)
        self.assertIsNone(first[0].source_reference.sheet)
        self.assertEqual(first[0].lineage.source_position, "lines:2-2")
        self.assertEqual(first[0].lineage.parser_version, "csv-stdlib-v1")

    def test_bom_and_multiline_record_keep_physical_line_range(self) -> None:
        raw = "Name,Description\nAlpha,\"line one\nline two\"\n"
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "rules.CSV"
            path.write_text(raw, encoding="utf-8-sig")
            result = ingest_file(path)

        self.assertEqual(result.status, ParseStatus.SUCCESS)
        self.assertEqual(result.source_format, IngestionFormat.CSV)
        self.assertEqual(len(result.units), 1)
        self.assertEqual(result.units[0].lineage.source_position, "lines:2-3")
        self.assertIn("line one\nline two", result.units[0].original_content)

    def test_empty_and_duplicate_headers_are_positional_and_audited(self) -> None:
        document = parse_csv("Name,,Name\nA,ignored,B\n", name="partial.csv")

        self.assertEqual(document.parse_status, ParseStatus.PARTIAL)
        self.assertEqual(document.headers, ("Name", "", "Name"))
        self.assertEqual(
            [(field.column_index, field.header, field.value) for field in document.rows[0].fields],
            [(0, "Name", "A"), (2, "Name", "B")],
        )
        self.assertTrue(any("empty CSV header" in item for item in document.warnings))
        self.assertTrue(any("duplicate CSV header" in item for item in document.warnings))

    def test_short_and_long_rows_never_invent_or_shift_fields(self) -> None:
        document = parse_csv("A,B,C\n1,2\n3,4,5,EXTRA\n", name="width.csv")

        self.assertEqual(document.parse_status, ParseStatus.PARTIAL)
        self.assertEqual(
            [(field.header, field.value) for field in document.rows[0].fields],
            [("A", "1"), ("B", "2")],
        )
        self.assertEqual(
            [(field.header, field.value) for field in document.rows[1].fields],
            [("A", "3"), ("B", "4"), ("C", "5")],
        )
        self.assertEqual(document.rows[1].values[-1], "EXTRA")
        self.assertEqual(sum("column count mismatch" in item for item in document.warnings), 2)

    def test_empty_bad_header_and_no_data_are_failed(self) -> None:
        empty = parse_csv("", name="empty.csv")
        bad_header = parse_csv(",,\n1,2,3\n", name="bad.csv")
        no_data = parse_csv("A,B\n", name="header-only.csv")

        self.assertEqual(empty.parse_status, ParseStatus.FAILED)
        self.assertEqual(bad_header.parse_status, ParseStatus.FAILED)
        self.assertEqual(no_data.parse_status, ParseStatus.FAILED)
        self.assertEqual(knowledge_units_from_csv(empty), [])
        self.assertEqual(knowledge_units_from_csv(bad_header), [])
        self.assertEqual(knowledge_units_from_csv(no_data), [])

    def test_decode_and_csv_syntax_errors_use_safe_common_result(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            invalid_encoding = root / "encoding.csv"
            invalid_encoding.write_bytes(b"\xff\xfe\x00")
            encoding_result = ingest_file(invalid_encoding)

            invalid_syntax = root / "syntax.csv"
            invalid_syntax.write_text('A,B\n"unterminated,B\n', encoding="utf-8")
            syntax_result = ingest_file(invalid_syntax)

            partial_syntax = root / "partial-syntax.csv"
            partial_syntax.write_text(
                'A,B\n1,2\n"unterminated,B\n', encoding="utf-8"
            )
            partial_result = ingest_file(partial_syntax)

        self.assertEqual(encoding_result.errors[0].code, "SOURCE_ENCODING_ERROR")
        self.assertEqual(syntax_result.errors[0].code, "CSV_PARSE_ERROR")
        self.assertNotIn("Traceback", syntax_result.errors[0].message)
        self.assertEqual(partial_result.status, ParseStatus.PARTIAL)
        self.assertEqual(len(partial_result.units), 1)
        self.assertTrue(any(
            "CSV syntax error" in warning.message
            for warning in partial_result.warnings
        ))


if __name__ == "__main__":
    unittest.main()
