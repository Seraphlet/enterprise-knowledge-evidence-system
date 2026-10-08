"""Focused T-1403/P14 integration tests; execution is deferred to P14-GATE."""

import contextlib
import io
import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

from knowledge_system.cli import EXIT_INPUT_ERROR, EXIT_OK, main
from knowledge_system.contracts import (
    KnowledgeLineage,
    KnowledgeUnit,
    ParseStatus,
    SourceReference,
)
from knowledge_system.ingestion import IngestionFormat, IngestionResult
from knowledge_system.ingestion.batch import (
    BatchIngestionError,
    discover_source_files,
    ingest_to_corpus,
    write_corpus_atomic,
)
from knowledge_system.service import KnowledgeService, load_knowledge_units


HTML = "<html><body><h1>HTML Guide</h1><p>Traceable HTML.</p></body></html>"
MARKDOWN = "# Markdown Guide\n\nTraceable Markdown.\n"
CSV = "Name,Description\nCSV Guide,Traceable CSV\n"


def _run_ingest_cli(
    source: Path, output: Path, scope: str
) -> tuple[int, str, str]:
    stdout = io.StringIO()
    stderr = io.StringIO()
    with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
        code = main(
            [
                "ingest",
                "--source",
                str(source),
                "--output",
                str(output),
                "--scope",
                scope,
                "--format",
                "text",
            ]
        )
    return code, stdout.getvalue(), stderr.getvalue()


class BatchIngestionTest(unittest.TestCase):
    def test_mixed_directory_is_stable_canonical_and_query_readable(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "sources"
            source.mkdir()
            nested = source / "nested"
            nested.mkdir()
            (source / "b.MD").write_text(MARKDOWN, encoding="utf-8")
            (source / "a.HTML").write_text(HTML, encoding="utf-8")
            (nested / "c.CSV").write_text(CSV, encoding="utf-8-sig")
            (source / "ignored.txt").write_text("not supported", encoding="utf-8")
            first_output = root / "first.json"
            second_output = root / "second.json"

            first = ingest_to_corpus(source, first_output)
            second = ingest_to_corpus(source, second_output)
            units = load_knowledge_units(first_output)
            query = KnowledgeService(units).query(
                "Markdown Guide", session_id="p14-gate", top_k=3
            )

            self.assertEqual(first.status, ParseStatus.SUCCESS)
            self.assertEqual(first.input_file_count, 3)
            self.assertEqual(
                [item.source_path for item in first.sources],
                ["a.HTML", "b.MD", "nested/c.CSV"],
            )
            self.assertEqual(first_output.read_bytes(), second_output.read_bytes())
            self.assertEqual(first.to_dict(), second.to_dict())
            self.assertEqual(len(units), first.written_unit_count)
            self.assertEqual(
                {unit.source_reference.name for unit in units},
                {"a.HTML", "b.MD", "c.CSV"},
            )
            self.assertEqual(
                {unit.lineage.parser_version for unit in units},
                {"html-stdlib-v1", "markdown-lines-v1", "csv-stdlib-v1"},
            )
            self.assertTrue(query.evidence)

    def test_single_supported_file_uses_the_same_canonical_pipeline(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "single.markdown"
            source.write_text(MARKDOWN, encoding="utf-8")
            output = root / "corpus.json"

            result = ingest_to_corpus(source, output)
            units = load_knowledge_units(output)

            self.assertEqual(result.status, ParseStatus.SUCCESS)
            self.assertEqual(result.input_file_count, 1)
            self.assertEqual([item.source_path for item in result.sources], [source.name])
            self.assertEqual(result.written_unit_count, len(units))
            self.assertTrue(all(unit.source_reference.name == source.name for unit in units))

    def test_explicit_scope_is_deterministic_metadata_only(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "guide.md"
            source.write_text(MARKDOWN, encoding="utf-8")
            unscoped_output = root / "unscoped.json"
            first_output = root / "first.json"
            second_output = root / "second.json"

            ingest_to_corpus(source, unscoped_output)
            first = ingest_to_corpus(source, first_output, scope="python-learning")
            second = ingest_to_corpus(source, second_output, scope="python-learning")
            unscoped = load_knowledge_units(unscoped_output)
            scoped = load_knowledge_units(first_output)
            browsed = KnowledgeService(scoped).query(
                "浏览全部知识",
                session_id="scoped-ingest-browse",
                scope="python-learning",
            )

            self.assertEqual(first_output.read_bytes(), second_output.read_bytes())
            self.assertEqual(first.to_dict(), second.to_dict())
            self.assertTrue(scoped)
            self.assertTrue(
                all(unit.metadata.get("scope") == "python-learning" for unit in scoped)
            )
            self.assertEqual(
                [unit.unit_id for unit in scoped], [unit.unit_id for unit in unscoped]
            )
            self.assertEqual(
                [unit.source_reference for unit in scoped],
                [unit.source_reference for unit in unscoped],
            )
            self.assertEqual(
                [unit.lineage for unit in scoped], [unit.lineage for unit in unscoped]
            )
            self.assertEqual(
                [unit.original_content for unit in scoped],
                [unit.original_content for unit in unscoped],
            )
            self.assertEqual(browsed.boundary.value, "NONE")
            self.assertEqual(
                [item.unit_id for item in browsed.evidence],
                [unit.unit_id for unit in scoped],
            )

    def test_cli_ingest_accepts_explicit_scope(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "guide.md"
            source.write_text(MARKDOWN, encoding="utf-8")
            output = root / "corpus.json"

            code, stdout, stderr = _run_ingest_cli(
                source, output, "python-learning"
            )

            self.assertEqual(code, EXIT_OK)
            self.assertEqual(stderr, "")
            self.assertIn("Status: SUCCESS", stdout)
            self.assertTrue(
                all(
                    unit.metadata.get("scope") == "python-learning"
                    for unit in load_knowledge_units(output)
                )
            )

    def test_directory_discovery_uses_relative_casefolded_order_only(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "z").mkdir()
            (root / "A.md").write_text(MARKDOWN, encoding="utf-8")
            (root / "a.CSV").write_text(CSV, encoding="utf-8")
            (root / "z" / "B.HTML").write_text(HTML, encoding="utf-8")
            (root / "00.txt").write_text("ignored", encoding="utf-8")

            discovered = discover_source_files(root)

            self.assertEqual(
                [path.relative_to(root).as_posix() for path in discovered],
                ["a.CSV", "A.md", "z/B.HTML"],
            )

    def test_identical_units_deduplicate_in_first_seen_order(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "sources"
            (source / "a").mkdir(parents=True)
            (source / "b").mkdir()
            (source / "a" / "same.md").write_text(MARKDOWN, encoding="utf-8")
            (source / "b" / "same.md").write_text(MARKDOWN, encoding="utf-8")
            output = root / "corpus.json"

            result = ingest_to_corpus(source, output)

            self.assertGreater(result.deduplicated_unit_count, 0)
            self.assertEqual(len(load_knowledge_units(output)), result.written_unit_count)

    def test_conflicting_unit_id_fails_before_replacing_existing_output(self) -> None:
        lineage = KnowledgeLineage("document", "parser", "chunker", "processing")
        first = KnowledgeUnit(
            "same-id", "one", "one", {}, SourceReference("one.html"), lineage,
            ParseStatus.SUCCESS,
        )
        conflict = replace(first, original_content="two")
        results = [
            IngestionResult(ParseStatus.SUCCESS, IngestionFormat.HTML, first.source_reference, (first,)),
            IngestionResult(ParseStatus.SUCCESS, IngestionFormat.HTML, conflict.source_reference, (conflict,)),
        ]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "sources"
            source.mkdir()
            (source / "a.html").write_text(HTML, encoding="utf-8")
            (source / "b.html").write_text(HTML, encoding="utf-8")
            output = root / "corpus.json"
            output.write_text("existing", encoding="utf-8")

            with patch("knowledge_system.ingestion.batch.ingest_file", side_effect=results):
                with self.assertRaises(BatchIngestionError) as raised:
                    ingest_to_corpus(source, output)

            self.assertEqual(raised.exception.code, "UNIT_ID_CONFLICT")
            self.assertEqual(output.read_text(encoding="utf-8"), "existing")

    def test_partial_and_failed_sources_keep_only_reliable_units(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "sources"
            source.mkdir()
            (source / "partial.csv").write_text("A,B\n1\n", encoding="utf-8")
            (source / "failed.md").write_text("\n", encoding="utf-8")
            output = root / "corpus.json"

            result = ingest_to_corpus(source, output)

            self.assertEqual(result.status, ParseStatus.PARTIAL)
            self.assertTrue(result.output_written)
            self.assertEqual(
                [item.status for item in result.sources],
                [ParseStatus.FAILED, ParseStatus.PARTIAL],
            )
            self.assertEqual(len(load_knowledge_units(output)), 1)

    def test_all_failed_sources_do_not_replace_existing_output(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "sources"
            source.mkdir()
            (source / "empty.md").write_text("\n", encoding="utf-8")
            (source / "header-only.csv").write_text("A,B\n", encoding="utf-8")
            output = root / "corpus.json"
            output.write_text("existing", encoding="utf-8")

            result = ingest_to_corpus(source, output)

            self.assertEqual(result.status, ParseStatus.FAILED)
            self.assertFalse(result.output_written)
            self.assertEqual(result.written_unit_count, 0)
            self.assertEqual(output.read_text(encoding="utf-8"), "existing")
            self.assertTrue(all(item.errors for item in result.sources))

    def test_bad_source_inputs_do_not_create_or_replace_output(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output = root / "corpus.json"
            output.write_text("existing", encoding="utf-8")
            empty = root / "empty"
            empty.mkdir()
            (empty / "ignored.txt").write_text("ignored", encoding="utf-8")

            for source, code in (
                (root / "missing", "SOURCE_NOT_FOUND"),
                (empty, "NO_SUPPORTED_SOURCE_FILES"),
                (empty / "ignored.txt", "UNSUPPORTED_SOURCE_FILE"),
            ):
                with self.assertRaises(BatchIngestionError) as raised:
                    ingest_to_corpus(source, output)
                self.assertEqual(raised.exception.code, code)
                self.assertEqual(output.read_text(encoding="utf-8"), "existing")

    def test_unreadable_directory_and_output_overlap_are_stable_input_errors(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "sources"
            source.mkdir()
            source_file = source / "rules.md"
            source_file.write_text(MARKDOWN, encoding="utf-8")

            with patch.object(Path, "rglob", side_effect=PermissionError):
                with self.assertRaises(BatchIngestionError) as unreadable:
                    discover_source_files(source)
            self.assertEqual(unreadable.exception.code, "SOURCE_DIRECTORY_READ_ERROR")

            with self.assertRaises(BatchIngestionError) as overlap:
                ingest_to_corpus(source_file, source_file)
            self.assertEqual(overlap.exception.code, "OUTPUT_OVERLAPS_SOURCE")
            self.assertEqual(source_file.read_text(encoding="utf-8"), MARKDOWN)

    def test_atomic_replace_failure_preserves_existing_output_and_cleans_temp(self) -> None:
        unit = KnowledgeUnit(
            "unit", "source", "semantic", {}, SourceReference("source.md"),
            KnowledgeLineage("document", "parser", "chunker", "processing"),
            ParseStatus.SUCCESS,
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output = root / "corpus.json"
            output.write_text("existing", encoding="utf-8")

            with patch("knowledge_system.ingestion.batch.os.replace", side_effect=OSError):
                with self.assertRaises(BatchIngestionError) as raised:
                    write_corpus_atomic((unit,), output)

            self.assertEqual(raised.exception.code, "CORPUS_WRITE_ERROR")
            self.assertEqual(output.read_text(encoding="utf-8"), "existing")
            self.assertEqual(list(root.glob(".corpus.json.*.tmp")), [])

    def test_serialization_failure_preserves_existing_output_without_temp_file(self) -> None:
        unit = KnowledgeUnit(
            "unit", "source", "semantic", {}, SourceReference("source.md"),
            KnowledgeLineage("document", "parser", "chunker", "processing"),
            ParseStatus.SUCCESS,
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output = root / "corpus.json"
            output.write_text("existing", encoding="utf-8")

            with patch(
                "knowledge_system.ingestion.batch.canonical_corpus_json",
                side_effect=TypeError,
            ):
                with self.assertRaises(BatchIngestionError) as raised:
                    write_corpus_atomic((unit,), output)

            self.assertEqual(raised.exception.code, "CORPUS_SERIALIZATION_ERROR")
            self.assertEqual(output.read_text(encoding="utf-8"), "existing")
            self.assertEqual(list(root.glob(".corpus.json.*.tmp")), [])

    def test_cli_json_summary_diagnostics_and_stable_exit_codes(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "sources"
            source.mkdir()
            (source / "partial.csv").write_text("A,B\n1\n", encoding="utf-8")
            output = root / "corpus.json"
            stdout = io.StringIO()
            stderr = io.StringIO()
            with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
                code = main([
                    "ingest", "--source", str(source), "--output", str(output),
                    "--format", "json",
                ])

            payload = json.loads(stdout.getvalue())
            diagnostics = json.loads(stderr.getvalue())
            self.assertEqual(code, EXIT_OK)
            self.assertEqual(payload["schema"], "knowledge-system.ingest-result")
            self.assertEqual(payload["status"], "PARTIAL")
            self.assertTrue(diagnostics["diagnostics"])

            with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                failed_code = main([
                    "ingest", "--source", str(root / "missing"),
                    "--output", str(output), "--format", "text",
                ])
            self.assertEqual(failed_code, EXIT_INPUT_ERROR)

    def test_cli_conflict_has_no_stdout_traceback_or_output_replacement(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output = root / "corpus.json"
            output.write_text("existing", encoding="utf-8")
            stdout = io.StringIO()
            stderr = io.StringIO()
            conflict = BatchIngestionError("UNIT_ID_CONFLICT", "conflicting unit id")

            with patch(
                "knowledge_system.cli.ingest_to_corpus", side_effect=conflict
            ), contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
                code = main([
                    "ingest", "--source", str(root), "--output", str(output),
                    "--format", "text",
                ])

            self.assertEqual(code, EXIT_INPUT_ERROR)
            self.assertEqual(stdout.getvalue(), "")
            error = json.loads(stderr.getvalue())["error"]
            self.assertEqual(error["code"], "UNIT_ID_CONFLICT")
            self.assertNotIn("Traceback", stderr.getvalue())
            self.assertEqual(output.read_text(encoding="utf-8"), "existing")


if __name__ == "__main__":
    unittest.main()
