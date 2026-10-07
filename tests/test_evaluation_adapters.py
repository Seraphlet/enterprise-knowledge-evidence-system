from dataclasses import fields
from pathlib import Path

from knowledge_system import (
    BackendInput,
    BaselineRetrieval,
    HybridRetrieval,
    Selection,
    baseline_backend_identity,
    baseline_product_backend,
    build_plan,
    current_backend_identity,
    current_product_backend,
    execute_pair,
    execute_plan,
    load_golden_dataset,
)


FIXTURE = Path(__file__).parents[1] / "data" / "golden_dataset_v1.json"


def _dataset():
    return load_golden_dataset(FIXTURE)


def _run(case_ids, backend, identity, run_id):
    dataset = _dataset()
    plan = build_plan(dataset, Selection.for_cases(case_ids))
    return execute_plan(dataset, plan, identity, backend, run_id=run_id)


def test_adapter_input_surface_is_truth_free_and_real_components_are_called(monkeypatch):
    seen = []
    baseline_calls = []
    hybrid_calls = []
    original_locate = BaselineRetrieval.locate
    original_search = HybridRetrieval.search

    def locate(self, query, **kwargs):
        baseline_calls.append(query)
        return original_locate(self, query, **kwargs)

    def search(self, query, **kwargs):
        hybrid_calls.append(query)
        return original_search(self, query, **kwargs)

    monkeypatch.setattr(BaselineRetrieval, "locate", locate)
    monkeypatch.setattr(HybridRetrieval, "search", search)

    def baseline(value):
        seen.append(value)
        return baseline_product_backend(value)

    def current(value):
        seen.append(value)
        return current_product_backend(value)

    dataset = _dataset()
    plan = build_plan(dataset, Selection.for_cases(("C-L01", "C-D01")))
    execute_pair(
        dataset, plan,
        baseline_backend_identity(), baseline,
        current_backend_identity(), current,
        baseline_run_id="adapter-wire-baseline", current_run_id="adapter-wire-current",
    )
    assert baseline_calls
    assert hybrid_calls
    assert all(isinstance(value, BackendInput) for value in seen)
    assert {item.name for item in fields(BackendInput)} == {
        "expression_id", "query", "turns", "synthetic_catalog"
    }
    assert all(not hasattr(value, "truth") and not hasattr(value, "minimum_recall") for value in seen)


def test_identity_and_predictions_are_deterministic_and_version_bound():
    first = _run(("C-D01",), current_product_backend, current_backend_identity(), "determinism-1")
    second = _run(("C-D01",), current_product_backend, current_backend_identity(), "determinism-2")
    assert first.predictions == second.predictions
    assert current_backend_identity() == current_backend_identity()
    assert baseline_backend_identity() == baseline_backend_identity()
    assert current_backend_identity().backend_version != baseline_backend_identity().backend_version
    assert len(current_backend_identity().config_fingerprint) == 64


def test_current_discover_has_semantic_recall_value_over_keyword_baseline():
    dataset = _dataset()
    plan = build_plan(dataset, Selection.for_cases(("C-D01", "C-D02", "C-D04")))
    pair = execute_pair(
        dataset, plan,
        baseline_backend_identity(), baseline_product_backend,
        current_backend_identity(), current_product_backend,
        baseline_run_id="discover-baseline", current_run_id="discover-current",
    )
    assert pair.current.report.aggregate.recall.value > pair.baseline.report.aggregate.recall.value


def test_current_locate_does_not_regress_exact_keyword_product_behavior():
    dataset = _dataset()
    plan = build_plan(dataset, Selection.for_cases(("C-L01", "C-L02", "C-L03", "C-L04")))
    pair = execute_pair(
        dataset, plan,
        baseline_backend_identity(), baseline_product_backend,
        current_backend_identity(), current_product_backend,
        baseline_run_id="locate-baseline", current_run_id="locate-current",
    )
    assert pair.current.report.aggregate.recall.value == pair.baseline.report.aggregate.recall.value
    assert pair.current.report.aggregate.ranking_mrr.value == pair.baseline.report.aggregate.ranking_mrr.value


def test_browse_returns_sets_and_filters_restricted_inactive_evidence():
    run = _run(
        ("C-B01", "C-B02", "C-B03", "C-B04"),
        current_product_backend, current_backend_identity(), "browse-current",
    )
    assert all(len(prediction.ranked_evidence_ids) > 1 for prediction in run.predictions)
    assert all("E-SEC-001" not in prediction.ranked_evidence_ids for prediction in run.predictions)
    assert all("E-OLD-001" not in prediction.ranked_evidence_ids for prediction in run.predictions)
    assert run.report.aggregate.safety_pass_rate.value == 1.0


def test_apply_never_emits_a_final_business_decision_and_safety_is_clean():
    run = _run(
        ("C-A01", "C-A02", "C-A03", "C-A04", "C-A05", "C-A06"),
        current_product_backend, current_backend_identity(), "apply-current",
    )
    assert all(not prediction.final_business_decision for prediction in run.predictions)
    assert run.report.aggregate.safety_pass_rate.value == 1.0


def test_external_legal_requests_stay_out_of_scope_without_catalog_evidence():
    run = _run(
        ("C-O04",),
        current_product_backend, current_backend_identity(), "external-law-current",
    )
    assert all(prediction.request_type.value == "OUT_OF_SCOPE" for prediction in run.predictions)
    assert all(not prediction.ranked_evidence_ids for prediction in run.predictions)
