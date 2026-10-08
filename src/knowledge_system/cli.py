"""Thin command-line adapter for the UI-independent KnowledgeService."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path

from . import __version__
from .service import (
    HUMAN_RESPONSIBILITY,
    KnowledgeService,
    QueryResult,
    ServiceInputError,
    load_knowledge_units,
    resume_query_snapshot,
    save_query_snapshot,
)
from .session_snapshot import SessionSnapshotError
from .demo_runner import (
    DEMO_RESULT_SCHEMA_VERSION,
    DemoRunner,
    default_demo_paths,
    format_demo_text,
)
from .ingestion.batch import (
    BatchIngestionError,
    batch_diagnostics,
    format_batch_text,
    ingest_to_corpus,
)


EXIT_OK = 0
EXIT_INPUT_ERROR = 2
EXIT_INTERNAL_ERROR = 3


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="knowledge-system",
        description="Evidence-first enterprise knowledge system",
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    commands = parser.add_subparsers(dest="command")
    query = commands.add_parser("query", help="query a local KnowledgeUnit corpus")
    query.add_argument("--corpus", required=True, help="local KnowledgeUnit JSON array")
    query.add_argument("--query", required=True, help="one explicit knowledge request")
    query.add_argument("--session-id", required=True, help="explicit task session identity")
    query.add_argument("--scope", help="explicit knowledge scope")
    query.add_argument("--permission", action="append", default=[], help="granted permission; repeatable")
    query.add_argument("--knowledge-version", default="local")
    query.add_argument("--index-version", default="local-derived")
    query.add_argument("--top-k", type=int, default=10)
    query.add_argument("--format", choices=("json", "text"), default="text")
    query.add_argument("--save-snapshot", help="save this structured session snapshot")
    query.add_argument("--revision", type=int, default=1)
    resume = commands.add_parser("resume", help="validate and inspect an explicit snapshot")
    resume.add_argument("--snapshot", required=True)
    resume.add_argument("--session-id", required=True)
    resume.add_argument("--format", choices=("json", "text"), default="text")
    ingest = commands.add_parser(
        "ingest", help="ingest one file or directory into a KnowledgeUnit corpus"
    )
    ingest.add_argument("--source", required=True, help="supported file or directory")
    ingest.add_argument("--output", required=True, help="canonical corpus JSON output")
    ingest.add_argument("--scope", help="explicit knowledge scope assigned to imported units")
    ingest.add_argument("--format", choices=("json", "text"), default="text")
    demo = commands.add_parser("demo", help="run the bundled SYNTHETIC/DEMO_ONLY scenarios")
    demo_commands = demo.add_subparsers(dest="demo_command")
    default_manifest, default_corpus = default_demo_paths()
    for name, help_text in (
        ("list", "list available demo scenarios"),
        ("run", "run one demo scenario"),
        ("all", "run all demo scenarios in manifest order"),
    ):
        command = demo_commands.add_parser(name, help=help_text)
        command.add_argument("--manifest", default=str(default_manifest))
        command.add_argument("--corpus", default=str(default_corpus))
        command.add_argument("--format", choices=("json", "text"), default="text")
        if name == "run":
            command.add_argument("scenario_id")
    return parser


def _source_text(source: dict[str, object]) -> str:
    parts = [str(source["name"])]
    for key, label in (("section", "Section"), ("page", "Page"), ("sheet", "Sheet")):
        if source.get(key) is not None:
            parts.append(f"{label}: {source[key]}")
    if source.get("url"):
        parts.append(f"URL: {source['url']}")
    return " | ".join(parts)


def format_text(result: QueryResult) -> str:
    payload = result.to_dict()
    lines = [
        f"Request: {payload['request_type']}",
        f"Boundary: {payload['boundary']}",
        f"Message: {payload['message']}",
    ]
    evidence = payload["evidence"]
    if evidence:
        lines.append("Evidence:")
        for index, item in enumerate(evidence, start=1):
            assert isinstance(item, dict)
            content = str(item["original_content"]).replace("\n", " ").strip()
            lines.append(f"  {index}. [{item['evidence_id']}] {content}")
            lines.append(f"     Source: {_source_text(item['source'])}")
    verification = payload["verification"]
    assert isinstance(verification, dict)
    lines.append(f"Verification: {'REQUIRED' if verification['required'] else 'NOT_REQUIRED'}")
    if verification.get("guidance"):
        lines.append(f"Verification guidance: {verification['guidance']}")
    trace = payload["trace"]
    assert isinstance(trace, dict)
    lines.extend((
        f"Trace: {trace['trace_id']}",
        f"Session: {payload['session_id']}",
        HUMAN_RESPONSIBILITY,
    ))
    return "\n".join(lines)


def _resume_payload(result: object) -> dict[str, object]:
    state = result.state  # type: ignore[attr-defined]
    metadata = result.metadata  # type: ignore[attr-defined]
    return {
        "schema": "knowledge-system.session-resume",
        "schema_version": 1,
        "session_id": state.session_id,
        "revision": metadata.revision,
        "saved_at": metadata.saved_at,
        "raw_query": state.raw_query,
        "request_type": state.knowledge_request.request_type.value if state.knowledge_request else None,
        "evidence_ids": [item.evidence_id for item in state.candidate_evidence],
        "note": "仅恢复显式结构化工作上下文；不包含完整聊天历史或跨 session 个人记忆。",
    }


def _emit_error(code: str, message: str) -> None:
    print(json.dumps({"error": {"code": code, "message": message}}, ensure_ascii=False, sort_keys=True), file=sys.stderr)


def _demo_output(args: argparse.Namespace) -> str:
    if args.demo_command is None:
        raise ServiceInputError("DEMO_COMMAND_REQUIRED", "demo 需要 list、run 或 all 子命令。")
    runner = DemoRunner.from_files(Path(args.manifest), Path(args.corpus))
    if args.demo_command == "list":
        payload: object = {
            "schema": "knowledge-system.demo-list",
            "schema_version": DEMO_RESULT_SCHEMA_VERSION,
            "dataset_id": runner.manifest.dataset_id,
            "dataset_version": runner.manifest.dataset_version,
            "classification": "SYNTHETIC/DEMO_ONLY",
            "disclaimer": runner.manifest.disclaimer,
            "scenarios": list(runner.list_scenarios()),
        }
        if args.format == "text":
            lines = [
                runner.manifest.disclaimer,
                f"Schema: knowledge-system.demo-list/v{DEMO_RESULT_SCHEMA_VERSION}",
                f"Dataset: {runner.manifest.dataset_id}@{runner.manifest.dataset_version}",
            ]
            lines.extend(
                f"{item['scenario_id']}: {item['title']} ({item['category']})"
                for item in runner.list_scenarios()
            )
            return "\n".join(lines)
    elif args.demo_command == "run":
        payload = runner.run(args.scenario_id)
        if args.format == "text":
            return format_demo_text(payload)
    else:
        results = runner.run_all()
        payload = {
            "schema": "knowledge-system.demo-all",
            "schema_version": DEMO_RESULT_SCHEMA_VERSION,
            "dataset_id": runner.manifest.dataset_id,
            "dataset_version": runner.manifest.dataset_version,
            "classification": "SYNTHETIC/DEMO_ONLY",
            "disclaimer": runner.manifest.disclaimer,
            "results": list(results),
        }
        if args.format == "text":
            return format_demo_text(results)
    return json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command is None:
        parser.print_help()
        return EXIT_OK
    try:
        if args.command == "query":
            units = load_knowledge_units(args.corpus)
            service = KnowledgeService(
                units,
                knowledge_version=args.knowledge_version,
                index_version=args.index_version,
                granted_permissions=args.permission,
            )
            result = service.query(args.query, session_id=args.session_id, scope=args.scope, top_k=args.top_k)
            if args.save_snapshot:
                save_query_snapshot(result, args.save_snapshot, revision=args.revision)
            output = result.to_json() if args.format == "json" else format_text(result)
        elif args.command == "resume":
            resumed = resume_query_snapshot(args.snapshot, args.session_id)
            payload = _resume_payload(resumed)
            output = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")) if args.format == "json" else "\n".join(f"{key}: {value}" for key, value in payload.items())
        elif args.command == "ingest":
            result = ingest_to_corpus(args.source, args.output, scope=args.scope)
            diagnostics = batch_diagnostics(result)
            if diagnostics:
                print(
                    json.dumps(
                        {"diagnostics": list(diagnostics)},
                        ensure_ascii=False,
                        sort_keys=True,
                        separators=(",", ":"),
                    ),
                    file=sys.stderr,
                )
            output = (
                json.dumps(
                    result.to_dict(),
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                )
                if args.format == "json"
                else format_batch_text(result)
            )
            print(output)
            return EXIT_INPUT_ERROR if result.status.value == "FAILED" else EXIT_OK
        else:
            output = _demo_output(args)
        print(output)
        return EXIT_OK
    except BatchIngestionError as error:
        _emit_error(error.code, error.message)
        return EXIT_INPUT_ERROR
    except ServiceInputError as error:
        _emit_error(error.code, error.message)
        return EXIT_INPUT_ERROR
    except SessionSnapshotError as error:
        _emit_error(f"SNAPSHOT_{error.code}", "无法安全读取或写入指定 session snapshot。")
        return EXIT_INPUT_ERROR
    except (ValueError, TypeError):
        _emit_error("INVALID_INPUT", "输入或本地数据不符合要求。")
        return EXIT_INPUT_ERROR
    except Exception:
        _emit_error("INTERNAL_ERROR", "请求执行失败；未输出内部路径或 traceback。")
        return EXIT_INTERNAL_ERROR
