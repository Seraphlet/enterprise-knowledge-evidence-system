"""Focused T-1300 tests for the application service and thin CLI."""

from __future__ import annotations

import contextlib
import io
import json
import subprocess
import sys
from pathlib import Path

from knowledge_system.cli import EXIT_INPUT_ERROR, EXIT_OK, main
from knowledge_system.contracts import (
    KnowledgeLineage,
    KnowledgeUnit,
    ParseStatus,
    SourceReference,
)
from knowledge_system.service import KnowledgeService, load_knowledge_units


def _unit(
    unit_id: str,
    content: str,
    *,
    status: str = "active",
    permission: list[str] | None = None,
    parse_status: ParseStatus = ParseStatus.SUCCESS,
) -> KnowledgeUnit:
    return KnowledgeUnit(
        unit_id=unit_id,
        original_content=content,
        semantic_content=content,
        metadata={
            "scope": "payments",
            "status": status,
            "required_permissions": permission or [],
        },
        source_reference=SourceReference(
            "支付审核手册.html",
            "https://kb.example/rules",
            "点赞阈值",
        ),
        lineage=KnowledgeLineage("doc-1", "parser-v1", "chunker-v1", "processing-v1"),
        parse_status=parse_status,
    )


def _write_corpus(path: Path, units: tuple[KnowledgeUnit, ...]) -> None:
    path.write_text(
        json.dumps([item.to_dict() for item in units], ensure_ascii=False),
        encoding="utf-8",
    )


def _run_main(argv: list[str]) -> tuple[int, str, str]:
    stdout, stderr = io.StringIO(), io.StringIO()
    with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
        code = main(argv)
    return code, stdout.getvalue(), stderr.getvalue()


def test_service_returns_stable_grounded_json_and_trace() -> None:
    service = KnowledgeService((_unit("u-1", "点赞数不少于 50 才进入复核。"),), knowledge_version="k1", index_version="i1")
    first = service.query("点赞数不少于 50", session_id="session-1", scope="payments")
    second = service.query("点赞数不少于 50", session_id="session-1", scope="payments")
    assert first.to_json() == second.to_json()
    payload = first.to_dict()
    assert payload["schema_version"] == 1
    assert payload["evidence"][0]["source"]["name"] == "支付审核手册.html"
    assert payload["trace"]["final_evidence_ids"] == ["evidence-u-1"]
    assert payload["human_responsibility"].endswith("Human 承担。")


def test_permission_inactive_unknown_and_failed_units_never_surface() -> None:
    units = (
        _unit("ok", "公开规则"),
        _unit("restricted", "秘密规则", permission=["secret"]),
        _unit("inactive", "失效规则", status="inactive"),
        _unit("unknown", "未知规则", status="unknown"),
        _unit("failed", "解析失败规则", parse_status=ParseStatus.FAILED),
    )
    result = KnowledgeService(units).query("浏览全部", session_id="s", scope="payments")
    assert [item.unit_id for item in result.evidence] == ["ok"]


def test_not_found_and_out_of_scope_use_cautious_boundaries() -> None:
    service = KnowledgeService((_unit("u-1", "差旅报销"),))
    missing = service.query("完全不相干词", session_id="s", scope="payments")
    assert missing.boundary.value == "KNOWLEDGE_MISSING"
    assert "不代表企业没有" in missing.message
    outside = service.query("天气如何", session_id="s", scope="weather")
    assert outside.request_type.value == "OUT_OF_SCOPE"
    assert "不覆盖" in outside.message
    assert not outside.evidence


def test_partial_source_enters_verification_without_guessing() -> None:
    service = KnowledgeService((_unit("u-1", "规则片段：", parse_status=ParseStatus.PARTIAL),))
    result = service.query("规则片段", session_id="s", scope="payments")
    assert result.verification.required is True
    assert result.boundary.value == "EVIDENCE_INSUFFICIENT"
    assert "不会猜测" in (result.verification.guidance or "")


def test_cli_json_stdout_and_diagnostics_stderr_are_separate(tmp_path: Path) -> None:
    corpus = tmp_path / "corpus.json"
    _write_corpus(corpus, (_unit("u-1", "点赞规则"),))
    code, stdout, stderr = _run_main([
        "query", "--corpus", str(corpus), "--query", "点赞规则",
        "--session-id", "s", "--scope", "payments", "--format", "json",
    ])
    assert code == EXIT_OK and stderr == ""
    assert json.loads(stdout)["evidence"][0]["evidence_id"] == "evidence-u-1"


def test_cli_human_output_has_evidence_source_trace_and_responsibility(tmp_path: Path) -> None:
    corpus = tmp_path / "corpus.json"
    _write_corpus(corpus, (_unit("u-1", "点赞规则"),))
    code, stdout, stderr = _run_main([
        "query", "--corpus", str(corpus), "--query", "点赞规则",
        "--session-id", "s", "--scope", "payments",
    ])
    assert code == EXIT_OK and stderr == ""
    for marker in ("Evidence:", "Source:", "Trace:", "Verification:", "Human 承担"):
        assert marker in stdout


def test_snapshot_resume_and_cross_session_failure_are_stable(tmp_path: Path) -> None:
    corpus, snapshot = tmp_path / "corpus.json", tmp_path / "session.json"
    _write_corpus(corpus, (_unit("u-1", "点赞规则"),))
    code, _, error = _run_main([
        "query", "--corpus", str(corpus), "--query", "点赞规则",
        "--session-id", "s", "--scope", "payments", "--save-snapshot", str(snapshot),
    ])
    assert code == EXIT_OK and error == ""
    code, output, error = _run_main([
        "resume", "--snapshot", str(snapshot), "--session-id", "s", "--format", "json",
    ])
    assert code == EXIT_OK and error == ""
    assert json.loads(output)["evidence_ids"] == ["evidence-u-1"]
    code, output, error = _run_main([
        "resume", "--snapshot", str(snapshot), "--session-id", "other", "--format", "json",
    ])
    assert code == EXIT_INPUT_ERROR and output == ""
    assert json.loads(error)["error"]["code"] == "SNAPSHOT_CROSS_SESSION"
    assert str(tmp_path) not in error and "Traceback" not in error


def test_tampered_snapshot_is_rejected_without_traceback(tmp_path: Path) -> None:
    corpus, snapshot = tmp_path / "corpus.json", tmp_path / "session.json"
    _write_corpus(corpus, (_unit("u-1", "点赞规则"),))
    _run_main(["query", "--corpus", str(corpus), "--query", "点赞规则", "--session-id", "s", "--scope", "payments", "--save-snapshot", str(snapshot)])
    snapshot.write_text(snapshot.read_text(encoding="utf-8").replace("点赞规则", "篡改规则", 1), encoding="utf-8")
    code, output, error = _run_main(["resume", "--snapshot", str(snapshot), "--session-id", "s"])
    assert code == EXIT_INPUT_ERROR and output == ""
    assert "SNAPSHOT_TAMPERED" in error and "Traceback" not in error


def test_invalid_corpus_has_stable_input_exit_and_no_internal_path(tmp_path: Path) -> None:
    corpus = tmp_path / "bad.json"
    corpus.write_text("{not-json", encoding="utf-8")
    code, output, error = _run_main(["query", "--corpus", str(corpus), "--query", "q", "--session-id", "s"])
    assert code == EXIT_INPUT_ERROR and output == ""
    assert json.loads(error)["error"]["code"] == "INVALID_CORPUS_JSON"
    assert str(tmp_path) not in error and "Traceback" not in error


def test_module_help_version_and_import_have_no_external_side_effects(tmp_path: Path) -> None:
    version = subprocess.run([sys.executable, "-m", "knowledge_system", "--version"], text=True, capture_output=True, check=False)
    assert version.returncode == 0 and "knowledge-system 0.0.0" in version.stdout
    help_result = subprocess.run([sys.executable, "-m", "knowledge_system", "--help"], text=True, capture_output=True, check=False)
    assert help_result.returncode == 0 and "query" in help_result.stdout and "resume" in help_result.stdout
    before = tuple(tmp_path.iterdir())
    imported = subprocess.run([sys.executable, "-c", "import knowledge_system"], cwd=tmp_path, text=True, capture_output=True, check=False)
    assert imported.returncode == 0 and imported.stdout == "" and imported.stderr == ""
    assert tuple(tmp_path.iterdir()) == before


def test_loader_rejects_duplicate_fields(tmp_path: Path) -> None:
    corpus = tmp_path / "dupe.json"
    corpus.write_text('[{"unit_id":"a","unit_id":"b"}]', encoding="utf-8")
    try:
        load_knowledge_units(corpus)
    except Exception as error:
        assert getattr(error, "code", None) == "DUPLICATE_JSON_FIELD"
    else:
        raise AssertionError("duplicate JSON field was accepted")
