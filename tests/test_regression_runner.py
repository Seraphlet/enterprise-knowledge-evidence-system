from dataclasses import FrozenInstanceError, fields, replace
import json
from pathlib import Path

import pytest

from knowledge_system import (
    BackendIdentity,
    BackendInput,
    FailureKind,
    PairBundle,
    Prediction,
    RunArtifact,
    RunnerValidationError,
    Selection,
    SelectionKind,
    TagMatch,
    build_plan,
    execute_pair,
    execute_plan,
    fingerprint_config,
    load_golden_dataset,
)


FIXTURE = Path(__file__).parents[1] / "data" / "golden_dataset_v1.json"


def _dataset():
    return load_golden_dataset(FIXTURE)


def _case(dataset, case_id):
    return next(item for item in dataset.cases if item.case_id == case_id)


def _identity(name):
    return BackendIdentity(name, "v1", fingerprint_config({"name": name, "mode": "test"}))


def _prediction(dataset, expression_id, *, ranked=None):
    case = next(
        case for case in dataset.cases
        if any(item.expression_id == expression_id for item in case.expressions)
    )
    if ranked is None:
        ranked = case.truth.recall.relevant_evidence_ids
    comparison = case.truth.comparison
    return Prediction(
        expression_id,
        case.request_type,
        tuple(ranked),
        case.truth.allowed_output_modes[0],
        case.truth.boundary,
        case.truth.clarification,
        case.truth.verification,
        comparison.acceptable_outcomes[0] if comparison else None,
        comparison.required_missing_information if comparison else (),
        case.truth.provenance.required_trace_fields,
        tuple(ranked),
        tuple(ranked),
    )


def test_case_selection_expands_all_expressions_with_stable_fingerprint():
    dataset = _dataset()
    selection = Selection.for_cases(("C-L02", "C-L01"))
    plan = build_plan(dataset, selection)
    assert plan.case_ids == ("C-L01", "C-L02")
    assert len(plan.expression_ids) == 6
    assert plan.expression_ids == tuple(sorted(plan.expression_ids))
    assert build_plan(dataset, selection) == plan
    assert len(plan.fingerprint) == 64
    assert type(plan).from_json(plan.to_json()) == plan
    with pytest.raises(FrozenInstanceError):
        plan.fingerprint = "changed"


def test_tag_any_all_and_unknown_empty_or_mixed_selection_rejection():
    dataset = _dataset()
    any_plan = build_plan(dataset, Selection.for_tags(("fuzzy",), TagMatch.ANY))
    assert any_plan.case_ids == ("C-D01", "C-D02")
    all_plan = build_plan(
        dataset, Selection.for_tags(("apply", "threshold"), TagMatch.ALL)
    )
    assert all_plan.case_ids == ("C-A01", "C-A06")
    with pytest.raises(RunnerValidationError, match="unknown tags"):
        build_plan(dataset, Selection.for_tags(("absent",), TagMatch.ANY))
    with pytest.raises(RunnerValidationError, match="matched no cases"):
        build_plan(dataset, Selection.for_tags(("exact", "fuzzy"), TagMatch.ALL))
    with pytest.raises(RunnerValidationError, match="inconsistent"):
        Selection(SelectionKind.CASE_IDS, (), ("fuzzy",), TagMatch.ANY)
    with pytest.raises(RunnerValidationError, match="inconsistent"):
        Selection.for_cases(())


def test_full_golden_requires_explicit_construction_and_plan_only_has_72_expressions():
    dataset = _dataset()
    with pytest.raises(RunnerValidationError, match="explicitly"):
        Selection(SelectionKind.FULL_GOLDEN)
    plan = build_plan(dataset, Selection.full_golden())
    assert len(plan.case_ids) == 24
    assert len(plan.expression_ids) == 72
    assert plan.selection.explicit_full


def test_backend_input_is_frozen_truth_free_and_contains_only_synthetic_catalog_view():
    dataset = _dataset()
    plan = build_plan(dataset, Selection.for_cases(("C-L01",)))
    seen = []

    def backend(value):
        seen.append(value)
        assert isinstance(value, BackendInput)
        assert {item.name for item in fields(value)} == {
            "expression_id", "query", "turns", "synthetic_catalog"
        }
        assert not hasattr(value, "truth")
        assert not hasattr(value, "minimum_recall")
        assert len(value.synthetic_catalog) == 10
        assert all(not hasattr(item, "truth") for item in value.synthetic_catalog)
        return _prediction(dataset, value.expression_id)

    before = repr(dataset)
    artifact = execute_plan(dataset, plan, _identity("backend-a"), backend, run_id="run-a")
    assert len(seen) == 3 and len({item.expression_id for item in seen}) == 3
    assert artifact.manifest.prediction_count == 3
    assert artifact.manifest.failure_count == 0
    assert not artifact.report.aggregate.missing
    assert repr(dataset) == before
    with pytest.raises(FrozenInstanceError):
        seen[0].query = "changed"


def test_each_expression_called_once_and_failures_are_isolated_and_scored_missing():
    dataset = _dataset()
    plan = build_plan(dataset, Selection.for_cases(("C-L01", "C-L02")))
    calls = {}

    def backend(value):
        calls[value.expression_id] = calls.get(value.expression_id, 0) + 1
        index = plan.expression_ids.index(value.expression_id)
        if index == 0:
            raise RuntimeError("synthetic failure")
        if index == 1:
            return None
        if index == 2:
            return {"not": "prediction"}
        if index == 3:
            return replace(
                _prediction(dataset, value.expression_id),
                expression_id=plan.expression_ids[4],
            )
        return _prediction(dataset, value.expression_id)

    artifact = execute_plan(dataset, plan, _identity("backend-fail"), backend, run_id="run-fail")
    assert calls == {item: 1 for item in plan.expression_ids}
    assert {item.kind for item in artifact.failures} == {
        FailureKind.BACKEND_EXCEPTION,
        FailureKind.NO_RETURN,
        FailureKind.INVALID_RETURN,
        FailureKind.EXPRESSION_MISMATCH,
    }
    assert artifact.manifest.failure_count == 4
    assert artifact.manifest.prediction_count == 2
    assert all(item.schema == "knowledge-system.execution-failure" for item in artifact.failures)
    assert all(item.schema_version == 1 for item in artifact.failures)
    assert artifact.report.aggregate.missing == 4
    assert sum(item.missing_prediction for item in artifact.report.expressions) == 4


def test_pair_runs_same_plan_independently_and_binds_backend_identity():
    dataset = _dataset()
    plan = build_plan(dataset, Selection.for_cases(("C-B01",)))
    baseline_calls, current_calls = [], []

    def baseline(value):
        baseline_calls.append(value.expression_id)
        return _prediction(dataset, value.expression_id)

    def current(value):
        current_calls.append(value.expression_id)
        return _prediction(dataset, value.expression_id, ranked=("E-PAY-001",))

    pair = execute_pair(
        dataset, plan, _identity("baseline"), baseline,
        _identity("current"), current,
        baseline_run_id="baseline-run", current_run_id="current-run",
    )
    assert tuple(baseline_calls) == plan.expression_ids
    assert tuple(current_calls) == plan.expression_ids
    assert pair.baseline.plan is plan and pair.current.plan is plan
    assert pair.baseline.report.metadata.system_id == pair.baseline.manifest.backend.bound_system_id
    assert pair.current.report.metadata.system_id == pair.current.manifest.backend.bound_system_id
    assert pair.comparison.regressed_expression_ids == plan.expression_ids
    with pytest.raises(RunnerValidationError, match="must differ"):
        execute_pair(dataset, plan, _identity("a"), baseline, _identity("b"), current, baseline_run_id="same", current_run_id="same")


def test_run_and_pair_artifacts_have_strict_deterministic_json_round_trip():
    dataset = _dataset()
    plan = build_plan(dataset, Selection.for_cases(("C-L01",)))
    backend = lambda value: _prediction(dataset, value.expression_id)
    artifact = execute_plan(dataset, plan, _identity("single"), backend, run_id="single-run")
    assert artifact.to_json() == artifact.to_json()
    assert RunArtifact.from_json(artifact.to_json()) == artifact
    pair = execute_pair(dataset, plan, _identity("base"), backend, _identity("curr"), backend, baseline_run_id="base-run", current_run_id="curr-run")
    assert PairBundle.from_json(pair.to_json()) == pair

    value = artifact.to_dict()
    value["extra"] = True
    with pytest.raises(RunnerValidationError, match="fields do not match"):
        RunArtifact.from_json(json.dumps(value))
    value = artifact.to_dict()
    value["schema_version"] = 99
    with pytest.raises(RunnerValidationError, match="schema/version"):
        RunArtifact.from_json(json.dumps(value))
    value = artifact.to_dict()
    value["plan"]["case_ids"] = "C-L01"
    with pytest.raises(RunnerValidationError, match="must be an array"):
        RunArtifact.from_json(json.dumps(value))
    with pytest.raises(RunnerValidationError, match="duplicate JSON key"):
        RunArtifact.from_json('{"schema":"a","schema":"b"}')


def test_plan_dataset_and_pair_plan_mismatch_are_rejected():
    dataset = _dataset()
    left_plan = build_plan(dataset, Selection.for_cases(("C-L01",)))
    right_plan = build_plan(dataset, Selection.for_cases(("C-L02",)))
    backend = lambda value: _prediction(dataset, value.expression_id)
    left = execute_plan(dataset, left_plan, _identity("left"), backend, run_id="left-run")
    right = execute_plan(dataset, right_plan, _identity("right"), backend, run_id="right-run")
    with pytest.raises(RunnerValidationError, match="exact plan"):
        PairBundle(
            "knowledge-system.regression-pair", 1, left_plan, left, right,
            left.report,
        )
    changed_dataset = replace(dataset, corpus_version="other-corpus")
    with pytest.raises(RunnerValidationError, match="incompatible"):
        execute_plan(changed_dataset, left_plan, _identity("x"), backend, run_id="x-run")


def test_invalid_backend_config_and_duplicate_selection_are_rejected():
    with pytest.raises(RunnerValidationError, match="finite JSON"):
        fingerprint_config({"bad": float("nan")})
    with pytest.raises(RunnerValidationError, match="duplicates"):
        Selection.for_cases(("C-L01", "C-L01"))
    with pytest.raises(RunnerValidationError, match="duplicates"):
        Selection.for_tags(("fuzzy", "fuzzy"), TagMatch.ANY)


def test_import_and_plan_construction_do_not_execute_backend():
    dataset = _dataset()
    calls = []

    def backend(value):
        calls.append(value)
        return None

    plan = build_plan(dataset, Selection.for_cases(("C-L01",)))
    identity = _identity("not-run")
    assert plan.expression_ids and identity.system_id == "not-run"
    assert calls == []
    assert callable(backend)
