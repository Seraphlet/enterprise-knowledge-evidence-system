"""Deterministic, UI-independent runner for the synthetic V0 demonstration."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, is_dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Iterable

from .apply_compare import organize_apply_comparisons
from .context import CaseContext, ContextRelation, apply_context_turn, create_task_context
from .deterministic_comparison import compare_condition
from .fact_extraction import extract_required_facts
from .information_requirements import generate_information_requirements
from .rule_structure import extract_rule_structure
from .service import HUMAN_RESPONSIBILITY, KnowledgeService, ServiceInputError, load_knowledge_units
from .verification_mode import build_verification_mode


DEMO_SCHEMA = "knowledge-system.demo-scenarios"
DEMO_SCHEMA_VERSION = 1
DEMO_RESULT_SCHEMA = "knowledge-system.demo-result"
DEMO_RESULT_SCHEMA_VERSION = 1
DEMO_DISCLAIMER = (
    "SYNTHETIC / DEMO_ONLY：本演示数据并非真实企业制度，不得用于真实业务判断。"
)


class DemoInputError(ServiceInputError):
    """Stable error for invalid demo identifiers or assets."""


@dataclass(frozen=True)
class DemoScenario:
    scenario_id: str
    title: str
    category: str
    scope: str | None
    steps: tuple[dict[str, Any], ...]
    observable_boundaries: tuple[str, ...]

    def summary(self) -> dict[str, Any]:
        return {
            "scenario_id": self.scenario_id,
            "title": self.title,
            "category": self.category,
            "step_count": len(self.steps),
        }


@dataclass(frozen=True)
class DemoManifest:
    dataset_id: str
    dataset_version: str
    disclaimer: str
    scenarios: tuple[DemoScenario, ...]


def _strict_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise DemoInputError("DUPLICATE_DEMO_FIELD", "demo manifest 包含重复字段。")
        result[key] = value
    return result


def load_demo_manifest(path: str | Path) -> DemoManifest:
    try:
        raw = Path(path).read_text(encoding="utf-8", errors="strict")
        value = json.loads(raw, object_pairs_hook=_strict_pairs)
    except DemoInputError:
        raise
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise DemoInputError("INVALID_DEMO_MANIFEST", "无法读取有效 demo manifest。") from error
    if not isinstance(value, dict):
        raise DemoInputError("INVALID_DEMO_MANIFEST", "demo manifest 根节点必须是对象。")
    if value.get("schema") != DEMO_SCHEMA or value.get("schema_version") != DEMO_SCHEMA_VERSION:
        raise DemoInputError("INVALID_DEMO_MANIFEST", "demo manifest schema/version 不受支持。")
    if value.get("classification") != "SYNTHETIC/DEMO_ONLY":
        raise DemoInputError("INVALID_DEMO_MANIFEST", "demo manifest 必须明确标记 SYNTHETIC/DEMO_ONLY。")
    if value.get("disclaimer") != DEMO_DISCLAIMER:
        raise DemoInputError("INVALID_DEMO_MANIFEST", "demo disclaimer 缺失或不匹配。")
    raw_scenarios = value.get("scenarios")
    if not isinstance(raw_scenarios, list) or not raw_scenarios:
        raise DemoInputError("INVALID_DEMO_MANIFEST", "demo scenarios 必须是非空数组。")
    scenarios: list[DemoScenario] = []
    try:
        for item in raw_scenarios:
            if not isinstance(item, dict):
                raise TypeError
            steps = item["steps"]
            boundaries = item["observable_boundaries"]
            if not isinstance(steps, list) or not steps or not all(isinstance(step, dict) for step in steps):
                raise TypeError
            if not isinstance(boundaries, list) or not boundaries or not all(isinstance(entry, str) and entry for entry in boundaries):
                raise TypeError
            scenarios.append(DemoScenario(
                scenario_id=item["scenario_id"], title=item["title"],
                category=item["category"], scope=item.get("scope"),
                steps=tuple(dict(step) for step in steps),
                observable_boundaries=tuple(boundaries),
            ))
        ids = tuple(item.scenario_id for item in scenarios)
        if any(not item for item in ids) or len(set(ids)) != len(ids):
            raise TypeError
        return DemoManifest(
            dataset_id=value["dataset_id"], dataset_version=value["dataset_version"],
            disclaimer=value["disclaimer"], scenarios=tuple(scenarios),
        )
    except (KeyError, TypeError) as error:
        raise DemoInputError("INVALID_DEMO_MANIFEST", "demo scenario schema 不合法。") from error


def _json_value(value: Any) -> Any:
    if isinstance(value, Enum):
        return value.value
    if is_dataclass(value):
        return _json_value(asdict(value))
    if isinstance(value, dict):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_value(item) for item in value]
    return value


class DemoRunner:
    """Run manifest inputs through KnowledgeService and existing core capabilities."""

    def __init__(self, manifest: DemoManifest, service: KnowledgeService) -> None:
        self.manifest = manifest
        self.service = service
        self._by_id = {item.scenario_id: item for item in manifest.scenarios}

    @classmethod
    def from_files(cls, manifest_path: str | Path, corpus_path: str | Path) -> "DemoRunner":
        manifest = load_demo_manifest(manifest_path)
        units = load_knowledge_units(corpus_path)
        if not units or any(item.metadata.get("classification") != "SYNTHETIC/DEMO_ONLY" for item in units):
            raise DemoInputError("INVALID_DEMO_CORPUS", "demo corpus 必须全部标记 SYNTHETIC/DEMO_ONLY。")
        return cls(manifest, KnowledgeService(
            units, knowledge_version=manifest.dataset_version,
            index_version=f"demo-index-{manifest.dataset_version}",
        ))

    def list_scenarios(self) -> tuple[dict[str, Any], ...]:
        return tuple(item.summary() for item in self.manifest.scenarios)

    def run(self, scenario_id: str) -> dict[str, Any]:
        scenario = self._by_id.get(scenario_id)
        if scenario is None:
            raise DemoInputError("UNKNOWN_DEMO_SCENARIO", "未知 demo scenario_id。")
        session_id = f"demo-{scenario.scenario_id}"
        step_results: list[dict[str, Any]] = []
        task_context = None
        latest_query = None
        duplicate_count = 1
        for index, step in enumerate(scenario.steps, start=1):
            action = step.get("action")
            if action == "query":
                query = step.get("query")
                if not isinstance(query, str) or not query:
                    raise DemoInputError("INVALID_DEMO_MANIFEST", "query step 缺少 query。")
                latest_query = self.service.query(
                    query, session_id=session_id, scope=scenario.scope,
                    top_k=int(step.get("top_k", 10)),
                )
                step_results.append({"step": index, "action": action, "result": latest_query.to_dict()})
            elif action in {"case_start", "case_update"}:
                if latest_query is None:
                    raise DemoInputError("INVALID_DEMO_MANIFEST", "case step 必须位于 query 之后。")
                description = step.get("description")
                if not isinstance(description, str) or not description:
                    raise DemoInputError("INVALID_DEMO_MANIFEST", "case step 缺少 description。")
                if action == "case_start":
                    task_context = create_task_context(
                        latest_query.state.query_context.original_query,
                        scope=scenario.scope, scope_covered=True,
                        initial_case_description=description,
                    )
                    relation = "NEW_CASE"
                else:
                    if task_context is None:
                        raise DemoInputError("INVALID_DEMO_MANIFEST", "case_update 缺少现有 CaseContext。")
                    transition = apply_context_turn(task_context, description, ContextRelation.UPDATE)
                    task_context = transition.context
                    relation = transition.relation.value
                step_results.append({
                    "step": index, "action": action, "relation": relation,
                    "session_id": latest_query.session_id,
                    "original_query": task_context.query.original_query,
                    "query_context": _json_value(task_context.query),
                    "case_raw_descriptions": list(task_context.case.raw_descriptions),
                    "comparison": self._compare(latest_query.evidence, task_context.case, duplicate_count),
                })
            elif action == "repeat_comparison_input":
                count = step.get("count")
                if type(count) is not int or count < 2:
                    raise DemoInputError("INVALID_DEMO_MANIFEST", "repeat count 必须至少为 2。")
                duplicate_count = count
                if latest_query is None or task_context is None:
                    raise DemoInputError("INVALID_DEMO_MANIFEST", "repeat step 缺少 query/case。")
                step_results.append({
                    "step": index, "action": action,
                    "comparison": self._compare(latest_query.evidence, task_context.case, duplicate_count),
                })
            else:
                raise DemoInputError("INVALID_DEMO_MANIFEST", "demo step action 不受支持。")

        verification = None
        if latest_query is not None and latest_query.verification.required and latest_query.evidence:
            verification = _json_value(build_verification_mode(
                latest_query.evidence, quality_signals=latest_query.quality_signals,
            ))
        return {
            "schema": DEMO_RESULT_SCHEMA,
            "schema_version": DEMO_RESULT_SCHEMA_VERSION,
            "dataset_id": self.manifest.dataset_id,
            "dataset_version": self.manifest.dataset_version,
            "classification": "SYNTHETIC/DEMO_ONLY",
            "disclaimer": self.manifest.disclaimer,
            "scenario": scenario.summary(),
            "observable_boundaries": list(scenario.observable_boundaries),
            "steps": step_results,
            "verification_mode": verification,
            "human_responsibility": HUMAN_RESPONSIBILITY,
        }

    def run_all(self) -> tuple[dict[str, Any], ...]:
        results: list[dict[str, Any]] = []
        for scenario in self.manifest.scenarios:
            try:
                results.append({"scenario_id": scenario.scenario_id, "ok": True, "result": self.run(scenario.scenario_id)})
            except Exception as error:
                results.append({
                    "scenario_id": scenario.scenario_id, "ok": False,
                    "error": {"type": type(error).__name__, "message": str(error)},
                })
        return tuple(results)

    @staticmethod
    def _compare(evidence: Iterable[Any], case: CaseContext, duplicate_count: int) -> list[dict[str, Any]]:
        outputs: list[dict[str, Any]] = []
        for item in evidence:
            extraction = extract_rule_structure(item)
            if extraction.structure is None:
                continue
            requirements = generate_information_requirements(extraction)
            facts = extract_required_facts(requirements, case)
            comparisons = []
            thresholds = extraction.structure.thresholds
            for requirement, fact in zip(requirements, facts, strict=True):
                condition = requirement.threshold or requirement.condition
                comparisons.append(compare_condition(condition, requirement, fact, item, case))
            supplied = tuple(comparison for comparison in comparisons for _ in range(duplicate_count))
            result = organize_apply_comparisons(extraction, supplied, case)
            outputs.append({
                "evidence_id": item.evidence_id,
                "matched": [entry.requirement.requirement_id for entry in result.matched],
                "missing": [entry.requirement.requirement_id for entry in result.missing],
                "unknown": [entry.requirement.requirement_id for entry in result.unknown],
                "not_satisfied": [entry.requirement.requirement_id for entry in result.not_satisfied],
                "duplicate_group_count": len(result.duplicate_groups),
                "logical_outcome": result.logical_outcome.value,
                "logical_reason": result.logical_reason,
                "comparisons": [_json_value(entry) for entry in result.comparisons],
            })
        return outputs


def default_demo_paths() -> tuple[Path, Path]:
    root = Path(__file__).resolve().parents[2]
    return root / "data" / "demo" / "scenarios.v1.json", root / "data" / "demo" / "corpus.v1.json"


def format_demo_text(payload: dict[str, Any] | tuple[dict[str, Any], ...]) -> str:
    if isinstance(payload, tuple):
        lines = [
            DEMO_DISCLAIMER,
            f"Schema: knowledge-system.demo-all/v{DEMO_RESULT_SCHEMA_VERSION}",
        ]
        for item in payload:
            lines.append(f"[{item['scenario_id']}] {'OK' if item['ok'] else 'FAILED'}")
        lines.append(HUMAN_RESPONSIBILITY)
        return "\n".join(lines)
    scenario = payload["scenario"]
    lines = [
        DEMO_DISCLAIMER,
        f"Schema: {payload['schema']}/v{payload['schema_version']}",
        f"Dataset: {payload['dataset_id']}@{payload['dataset_version']}",
        f"Scenario: {scenario['scenario_id']} — {scenario['title']}",
    ]
    for step in payload["steps"]:
        result = step.get("result")
        if isinstance(result, dict):
            lines.append(f"Step {step['step']}: {result['request_type']} / {result['boundary']}")
            for evidence in result["evidence"]:
                lines.append(f"  [{evidence['evidence_id']}] {evidence['original_content']}")
                lines.append(f"  Source: {evidence['source']['name']} | Section: {evidence['source'].get('section')}")
        if "comparison" in step:
            lines.append(f"Step {step['step']}: comparison={json.dumps(step['comparison'], ensure_ascii=False, sort_keys=True)}")
    if payload.get("verification_mode"):
        lines.append("Verification Mode: REQUIRED")
    lines.append(HUMAN_RESPONSIBILITY)
    return "\n".join(lines)
