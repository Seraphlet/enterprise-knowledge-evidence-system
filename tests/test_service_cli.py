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
    scope: str | None = "payments",
    source_name: str = "支付审核手册.html",
    document_id: str = "doc-1",
    status: str = "active",
    permission: list[str] | None = None,
    parse_status: ParseStatus = ParseStatus.SUCCESS,
    section: str = "点赞阈值",
    structure_kind: str = "paragraph",
    source_position: str | None = None,
    parent_section_id: str | None = None,
) -> KnowledgeUnit:
    return KnowledgeUnit(
        unit_id=unit_id,
        original_content=content,
        semantic_content=content,
        metadata={
            **({"scope": scope} if scope is not None else {}),
            "status": status,
            "required_permissions": permission or [],
            "structure_kind": structure_kind,
        },
        source_reference=SourceReference(
            source_name,
            "https://kb.example/rules",
            section,
        ),
        lineage=KnowledgeLineage(
            document_id,
            "parser-v1",
            "chunker-v1",
            "processing-v1",
            source_position=source_position,
            parent_section_id=parent_section_id,
        ),
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


def test_browse_with_explicit_scope_never_mixes_other_scope() -> None:
    units = (
        _unit("payments", "支付知识", scope="payments"),
        _unit(
            "hr",
            "人事知识",
            scope="hr",
            source_name="人事手册.md",
            document_id="doc-hr",
        ),
    )

    result = KnowledgeService(units).query(
        "浏览全部知识", session_id="scope-browse", scope="payments"
    )

    assert [item.unit_id for item in result.evidence] == ["payments"]


def test_browse_without_scope_allows_only_unique_complete_document_identity() -> None:
    units = (
        _unit(
            "public",
            "公开内容",
            scope=None,
            source_name="Python工程指南.md",
            document_id="doc-python",
        ),
        _unit(
            "restricted",
            "受限内容",
            scope=None,
            source_name="Python工程指南.md",
            document_id="doc-python",
            permission=["secret.read"],
        ),
        _unit(
            "other",
            "其他文档",
            scope="other",
            source_name="其他手册.md",
            document_id="doc-other",
        ),
    )
    service = KnowledgeService(units)

    found = service.query("浏览 Python工程指南.md", session_id="document-browse")
    missing_identity = service.query("浏览全部知识", session_id="missing-scope")
    unknown_document = service.query("浏览 不存在.md", session_id="unknown-document")
    ambiguous = KnowledgeService(
        (
            _unit(
                "duplicate-a",
                "第一份同名文档",
                scope=None,
                source_name="同名手册.md",
                document_id="doc-a",
            ),
            _unit(
                "duplicate-b",
                "第二份同名文档",
                scope=None,
                source_name="同名手册.md",
                document_id="doc-b",
            ),
        )
    ).query("浏览 同名手册.md", session_id="ambiguous-document")

    assert found.boundary.value == "NONE"
    assert [item.unit_id for item in found.evidence] == ["public"]
    assert missing_identity.boundary.value == "RETRIEVAL_QUERY_INSUFFICIENT"
    assert not missing_identity.evidence
    assert unknown_document.boundary.value == "RETRIEVAL_QUERY_INSUFFICIENT"
    assert not unknown_document.evidence
    assert ambiguous.boundary.value == "RETRIEVAL_QUERY_INSUFFICIENT"
    assert not ambiguous.evidence


def test_topic_question_without_scope_uses_discovery_instead_of_browse_block() -> None:
    service = KnowledgeService(
        (
            _unit(
                "python-structure",
                "Python 项目的基本目录和配置文件包括 src、tests 和 pyproject.toml。",
                scope="python-learning",
                source_name="Python工程指南.md",
                document_id="doc-python",
            ),
        )
    )

    result = service.query(
        "一个规范的 Python 项目通常应该包含哪些基本目录和配置文件？",
        session_id="topic-discover",
    )

    assert result.request_type.value == "DISCOVER"
    assert result.boundary.value != "RETRIEVAL_QUERY_INSUFFICIENT"
    assert [item.unit_id for item in result.evidence] == ["python-structure"]


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


def test_context_dependent_fragment_recovers_bounded_same_section_neighbors() -> None:
    units = (
        _unit(
            "after",
            "项目是否同时具备 src、tests 与 pyproject.toml？",
            scope="python-learning",
            source_name="Python工程指南.md",
            document_id="doc-python",
            section="65. Python 工程最小结构",
            source_position="lines:102-102",
            parent_section_id="section-65",
        ),
        _unit(
            "fragment",
            "然后问：",
            scope="python-learning",
            source_name="Python工程指南.md",
            document_id="doc-python",
            section="65. Python 工程最小结构",
            source_position="lines:101-101",
            parent_section_id="section-65",
        ),
        _unit(
            "before",
            "Python 工程至少需要可声明构建配置的 pyproject.toml。",
            scope="python-learning",
            source_name="Python工程指南.md",
            document_id="doc-python",
            section="65. Python 工程最小结构",
            source_position="lines:100-100",
            parent_section_id="section-65",
        ),
        _unit(
            "other-section",
            "其他章节不得作为恢复上下文。",
            scope="python-learning",
            source_name="Python工程指南.md",
            document_id="doc-python",
            section="66. 其他章节",
            source_position="lines:103-103",
            parent_section_id="section-66",
        ),
    )

    first = KnowledgeService(units).query(
        "然后问：", session_id="recover-1", scope="python-learning", top_k=1
    )
    second = KnowledgeService(units).query(
        "然后问：", session_id="recover-1", scope="python-learning", top_k=1
    )

    assert [item.unit_id for item in first.evidence] == [
        "fragment", "before", "after"
    ]
    assert first.boundary.value == "NONE"
    assert first.verification.required is False
    assert first.to_json() == second.to_json()
    assert first.evidence[0].original_content == "然后问："
    assert first.evidence[1].lineage.document_id == "doc-python"
    assert first.evidence[2].source_reference.section == "65. Python 工程最小结构"
    assert "other-section" not in [item.unit_id for item in first.evidence]
    recovery_paths = [
        item for item in first.trace.retrieval_paths
        if item.get("path") == "bounded_recovery"
    ]
    assert recovery_paths
    assert recovery_paths[0]["action"] == "NEIGHBOR_EXPANSION"


def test_fragment_recovery_excludes_partial_neighbor_when_clean_context_remains() -> None:
    units = (
        _unit(
            "before-partial",
            "规则片段：",
            scope="python-learning",
            source_name="Python工程指南.md",
            document_id="doc-python",
            section="65. Python 工程最小结构",
            source_position="lines:100-100",
            parent_section_id="section-65",
            parse_status=ParseStatus.PARTIAL,
        ),
        _unit(
            "fragment",
            "然后问：",
            scope="python-learning",
            source_name="Python工程指南.md",
            document_id="doc-python",
            section="65. Python 工程最小结构",
            source_position="lines:101-101",
            parent_section_id="section-65",
        ),
        _unit(
            "after-good",
            "项目至少需要 pyproject.toml。",
            scope="python-learning",
            source_name="Python工程指南.md",
            document_id="doc-python",
            section="65. Python 工程最小结构",
            source_position="lines:102-102",
            parent_section_id="section-65",
        ),
    )

    result = KnowledgeService(units).query(
        "然后问：", session_id="recover-mixed-quality", scope="python-learning", top_k=1
    )

    assert [item.unit_id for item in result.evidence] == ["fragment", "after-good"]
    assert result.boundary.value == "NONE"
    assert result.verification.required is False
    assert "evidence-before-partial" not in result.trace.final_evidence_ids
    recovery_paths = [
        item for item in result.trace.retrieval_paths
        if item.get("path") == "bounded_recovery"
    ]
    assert recovery_paths[0]["recovered_evidence_ids"] == ["evidence-after-good"]


def test_failed_fragment_recovery_requires_verification_and_never_crosses_filters() -> None:
    units = (
        _unit(
            "fragment",
            "完全一致。",
            scope="python-learning",
            source_name="Python工程指南.md",
            document_id="doc-python",
            section="65. Python 工程最小结构",
            source_position="lines:10-10",
            parent_section_id="section-65",
        ),
        _unit(
            "restricted",
            "秘密上下文",
            scope="python-learning",
            source_name="Python工程指南.md",
            document_id="doc-python",
            section="65. Python 工程最小结构",
            source_position="lines:11-11",
            parent_section_id="section-65",
            permission=["secret.read"],
        ),
        _unit(
            "other-scope",
            "另一范围上下文",
            scope="hr",
            source_name="Python工程指南.md",
            document_id="doc-python",
            section="65. Python 工程最小结构",
            source_position="lines:12-12",
            parent_section_id="section-65",
        ),
        _unit(
            "inactive",
            "失效上下文",
            scope="python-learning",
            source_name="Python工程指南.md",
            document_id="doc-python",
            section="65. Python 工程最小结构",
            source_position="lines:13-13",
            parent_section_id="section-65",
            status="inactive",
        ),
    )

    result = KnowledgeService(units).query(
        "完全一致。", session_id="recover-failed", scope="python-learning", top_k=1
    )

    assert [item.unit_id for item in result.evidence] == ["fragment"]
    assert result.boundary.value == "EVIDENCE_INSUFFICIENT"
    assert result.verification.required is True
    assert "POSSIBLY_INCOMPLETE" in result.verification.reasons
    assert "evidence-restricted" not in result.trace.final_evidence_ids
    assert "evidence-other-scope" not in result.trace.final_evidence_ids
    assert "evidence-inactive" not in result.trace.final_evidence_ids
    recovery_paths = [
        item for item in result.trace.retrieval_paths
        if item.get("path") == "bounded_recovery"
    ]
    assert len(recovery_paths) == 2
    assert all(item["recovered_evidence_ids"] == [] for item in recovery_paths)


def test_apply_weak_markdown_cannot_report_not_required() -> None:
    service = KnowledgeService(
        (
            _unit(
                "python-heading",
                "# 65. Python 工程最小结构",
                scope="python-learning",
                source_name="Python工程基础面试补缺.md",
                document_id="doc-python",
                section="65. Python 工程最小结构",
            ),
        )
    )

    result = service.query(
        "我的 Python 项目只有 main.py，没有 pyproject.toml，是否符合 Python 工程最小结构？",
        session_id="apply-weak-markdown",
        scope="python-learning",
        top_k=1,
    )

    assert result.request_type.value == "APPLY"
    assert [item.unit_id for item in result.evidence] == ["python-heading"]
    assert result.boundary.value == "EVIDENCE_INSUFFICIENT"
    assert result.verification.required is True
    assert "rule_structure_unavailable" in result.verification.reasons
    assert result.state.comparison_result is None
    assert "Knowledge Gap" not in result.message


def test_apply_explicit_numeric_rule_completes_deterministic_comparison() -> None:
    service = KnowledgeService(
        (
            _unit(
                "likes-rule",
                "条件：点赞数不少于50个\n结果：继续处理",
                scope="content-review",
                source_name="内容复核规则.md",
                document_id="doc-content-review",
                section="点赞阈值",
            ),
        )
    )

    result = service.query(
        "我的点赞数是52个，是否符合继续处理规则？",
        session_id="apply-explicit-rule",
        scope="content-review",
        top_k=1,
    )
    repeated = service.query(
        "我的点赞数是52个，是否符合继续处理规则？",
        session_id="apply-explicit-rule",
        scope="content-review",
        top_k=1,
    )

    comparison = result.state.comparison_result
    assert result.request_type.value == "APPLY"
    assert result.boundary.value == "NONE"
    assert result.verification.required is False
    assert comparison is not None
    assert comparison.logical_outcome.value == "SATISFIED"
    assert comparison.comparisons[0].outcome.value == "SATISFIED"
    assert comparison.extraction.source_evidence.evidence_id == "evidence-likes-rule"
    assert result.to_json() == repeated.to_json()
    assert comparison == repeated.state.comparison_result
    observation = comparison.comparisons[0].fact.observations[0]
    assert observation.source.text == "点赞数是52个"


def test_apply_explicit_rule_with_missing_case_fact_requires_verification() -> None:
    service = KnowledgeService(
        (
            _unit(
                "likes-rule",
                "条件：点赞数不少于50个\n结果：继续处理",
                scope="content-review",
                source_name="内容复核规则.md",
                document_id="doc-content-review",
                section="点赞阈值",
            ),
        )
    )

    result = service.query(
        "这个案例是否符合继续处理规则？",
        session_id="apply-missing-case-fact",
        scope="content-review",
        top_k=1,
    )

    comparison = result.state.comparison_result
    assert result.boundary.value == "EVIDENCE_INSUFFICIENT"
    assert result.verification.required is True
    assert "case_information_missing" in result.verification.reasons
    assert comparison is not None
    assert comparison.logical_outcome.value == "INDETERMINATE"
    assert comparison.missing[0].reason == "fact_missing"
    assert comparison.missing[0].fact.reason == "required_fact_not_found"


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
