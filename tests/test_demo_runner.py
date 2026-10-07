"""Focused T-1301 checks for synthetic demo scenarios and CLI."""

from __future__ import annotations

import contextlib
import io
import json
from pathlib import Path

from knowledge_system.cli import EXIT_INPUT_ERROR, EXIT_OK, main
from knowledge_system.demo_runner import DEMO_DISCLAIMER, DemoRunner, default_demo_paths


def _runner() -> DemoRunner:
    manifest, corpus = default_demo_paths()
    return DemoRunner.from_files(manifest, corpus)


def _main(argv: list[str]) -> tuple[int, str, str]:
    stdout, stderr = io.StringIO(), io.StringIO()
    with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
        code = main(argv)
    return code, stdout.getvalue(), stderr.getvalue()


def _query_step(result: dict[str, object]) -> dict[str, object]:
    return result["steps"][0]["result"]  # type: ignore[index,return-value]


def test_manifest_is_versioned_synthetic_and_has_exactly_eight_categories() -> None:
    runner = _runner()
    assert runner.manifest.dataset_version == "demo-v1"
    assert runner.manifest.disclaimer == DEMO_DISCLAIMER
    assert [item.category for item in runner.manifest.scenarios] == [
        "exact_locate", "browse", "fuzzy_discover", "case_threshold_update",
        "duplicate", "knowledge_missing", "source_ambiguity_verification", "out_of_scope",
    ]


def test_exact_locate_returns_grounded_source_and_trace() -> None:
    result = _runner().run("exact-locate")
    query = _query_step(result)
    assert query["request_type"] == "LOCATE"
    assert len(query["evidence"]) == 1
    first = query["evidence"][0]
    assert first["evidence_id"] == "evidence-demo-rule-like-threshold"
    assert first["source"]["name"].startswith("SYNTHETIC-DEMO-")
    assert first["evidence_id"] in query["trace"]["final_evidence_ids"]
    assert query["trace"]["retrieval_paths"][0]["path"] == "keyword"


def test_browse_returns_scope_collection_not_top_one() -> None:
    query = _query_step(_runner().run("browse-scope"))
    assert query["request_type"] == "BROWSE"
    assert len(query["evidence"]) == 3
    assert len(query["trace"]["final_evidence_ids"]) == 3


def test_fuzzy_discover_uses_hybrid_trace_and_corpus_name() -> None:
    query = _query_step(_runner().run("fuzzy-discover"))
    assert query["request_type"] in {"APPLY", "DISCOVER"}
    assert query["evidence"]
    assert query["trace"]["retrieval_paths"]
    sections = {item["source"]["section"] for item in query["evidence"]}
    assert "演示夜间大额复核规则" in sections


def test_case_update_preserves_query_and_moves_missing_to_matched() -> None:
    result = _runner().run("case-threshold-update")
    start, update = result["steps"][1], result["steps"][2]
    assert start["session_id"] == update["session_id"] == "demo-case-threshold-update"
    assert start["original_query"] == update["original_query"]
    assert start["query_context"] == update["query_context"]
    assert update["relation"] == "UPDATE"
    assert update["case_raw_descriptions"] == ["当前尚未提供点赞数。", "现在52个赞"]
    assert any(item["missing"] for item in start["comparison"])
    assert any(item["matched"] for item in update["comparison"])


def test_duplicate_comparison_is_deterministic_and_not_a_business_decision() -> None:
    runner = _runner()
    first = runner.run("duplicate-compare")
    second = runner.run("duplicate-compare")
    assert first == second
    repeated = first["steps"][2]["comparison"]
    assert any(item["duplicate_group_count"] > 0 for item in repeated)
    encoded = json.dumps(repeated, ensure_ascii=False).casefold()
    assert "approval" not in encoded and "eligibility" not in encoded


def test_missing_and_out_of_scope_have_empty_evidence_and_cautious_text() -> None:
    missing = _query_step(_runner().run("knowledge-missing"))
    assert missing["boundary"] == "KNOWLEDGE_MISSING" and not missing["evidence"]
    assert "不代表企业没有" in missing["message"]
    outside = _query_step(_runner().run("out-of-scope"))
    assert outside["boundary"] == "OUT_OF_SCOPE" and not outside["evidence"]
    assert "不会回退" in outside["message"]


def test_source_ambiguity_has_verbatim_source_problem_and_locator() -> None:
    result = _runner().run("source-ambiguity")
    query = _query_step(result)
    assert query["verification"]["required"] is True
    mode = result["verification_mode"]
    assert mode["active"] is True
    record = mode["records"][0]
    assert record["reason"] == "SOURCE_AMBIGUOUS"
    assert record["source_fragments"][0]["text"] == "必要时"
    assert record["verification_locator"]["level"] == "STABLE_URL"


def test_all_order_json_stability_and_per_scenario_status() -> None:
    runner = _runner()
    first, second = runner.run_all(), runner.run_all()
    assert first == second
    assert [item["scenario_id"] for item in first] == [item.scenario_id for item in runner.manifest.scenarios]
    assert all(item["ok"] for item in first)
    assert all(item["result"]["classification"] == "SYNTHETIC/DEMO_ONLY" for item in first)


def test_cli_list_run_all_and_unknown_exit_codes() -> None:
    for args, schema in (
        (["demo", "list", "--format", "json"], "knowledge-system.demo-list"),
        (["demo", "run", "exact-locate", "--format", "json"], "knowledge-system.demo-result"),
        (["demo", "all", "--format", "json"], "knowledge-system.demo-all"),
    ):
        code, stdout, stderr = _main(args)
        assert code == EXIT_OK and stderr == ""
        assert json.loads(stdout)["schema"] == schema
    code, stdout, stderr = _main(["demo", "run", "does-not-exist", "--format", "json"])
    assert code == EXIT_INPUT_ERROR and stdout == ""
    assert json.loads(stderr)["error"]["code"] == "UNKNOWN_DEMO_SCENARIO"


def test_assets_are_under_demo_only_directory() -> None:
    manifest, corpus = default_demo_paths()
    assert manifest.parent.name == "demo" and corpus.parent.name == "demo"
    assert "SYNTHETIC/DEMO_ONLY" in manifest.read_text(encoding="utf-8")
    assert all(item.metadata["classification"] == "SYNTHETIC/DEMO_ONLY" for item in __import__("knowledge_system.service", fromlist=["load_knowledge_units"]).load_knowledge_units(corpus))
