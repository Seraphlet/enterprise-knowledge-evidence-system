from dataclasses import FrozenInstanceError, replace
import json
from pathlib import Path

import pytest

from knowledge_system import (
    BoundaryTruth,
    ClarificationTruth,
    EvaluationComparison,
    EvaluationReport,
    EvaluationValidationError,
    ForbiddenBehavior,
    KnowledgeRequestType,
    OutputMode,
    Prediction,
    RankingTruth,
    RecallTruth,
    RunMetadata,
    VerificationTruth,
    compare_reports,
    evaluate_predictions,
    load_golden_dataset,
)


FIXTURE = Path(__file__).parents[1] / "data" / "golden_dataset_v1.json"


def _dataset():
    return load_golden_dataset(FIXTURE)


def _case(dataset, case_id):
    return next(item for item in dataset.cases if item.case_id == case_id)


def _expression(case, index=0):
    return case.expressions[index].expression_id


def _metadata(dataset, expression_ids, run_id="run-1", system_id="system-a"):
    return RunMetadata(
        run_id,
        system_id,
        dataset.dataset_version,
        dataset.corpus_version,
        tuple(sorted(expression_ids)),
    )


def _perfect(case, expression_id, *, ranked=None):
    if ranked is None:
        ranked = case.truth.recall.relevant_evidence_ids
    comparison = case.truth.comparison
    return Prediction(
        expression_id=expression_id,
        request_type=case.request_type,
        ranked_evidence_ids=tuple(ranked),
        output_mode=case.truth.allowed_output_modes[0],
        boundary=case.truth.boundary,
        clarification=case.truth.clarification,
        verification=case.truth.verification,
        comparison_outcome=(
            comparison.acceptable_outcomes[0] if comparison is not None else None
        ),
        missing_information=(
            comparison.required_missing_information if comparison is not None else ()
        ),
        trace_fields=case.truth.provenance.required_trace_fields,
        source_evidence_ids=tuple(ranked),
        lineage_evidence_ids=tuple(ranked),
    )


def test_perfect_metrics_and_empty_truth_are_na_not_fake_scores():
    dataset = _dataset()
    cases = [_case(dataset, item) for item in ("C-L01", "C-B01", "C-A01", "C-O01")]
    ids = [_expression(case) for case in cases]
    report = evaluate_predictions(
        dataset, _metadata(dataset, ids),
        [_perfect(case, expression_id) for case, expression_id in zip(cases, ids)],
    )
    by_id = {item.expression_id: item for item in report.expressions}
    assert by_id[ids[0]].recall.value == 1.0
    assert by_id[ids[0]].ranking.mrr == 1.0
    assert by_id[ids[1]].browse.coverage == 1.0
    assert by_id[ids[1]].browse.precision == 1.0
    assert by_id[ids[2]].behavior.passed and by_id[ids[2]].safety.passed
    out = by_id[ids[3]]
    assert not out.recall.applicable and out.recall.value is None
    assert not out.ranking.applicable and out.ranking.mrr is None
    assert not out.browse.applicable and out.browse.coverage is None
    assert report.aggregate.recall.applicable_count == 3


def test_recall_and_ranking_are_independent_in_both_directions():
    dataset = _dataset()
    case = _case(dataset, "C-B01")
    expression_id = _expression(case)
    high_rank_low_recall = _perfect(case, expression_id, ranked=("E-PAY-001",))
    report = evaluate_predictions(dataset, _metadata(dataset, (expression_id,)), (high_rank_low_recall,))
    result = report.expressions[0]
    assert abs(result.recall.value - (1 / 3)) < 1e-12
    assert result.ranking.mrr == 1.0
    assert result.browse.top1_only and not result.browse.passed

    full_recall_bad_rank = _perfect(
        case, expression_id,
        ranked=("E-HR-001", "E-PAY-003", "E-PAY-002", "E-PAY-001"),
    )
    report = evaluate_predictions(dataset, _metadata(dataset, (expression_id,)), (full_recall_bad_rank,))
    result = report.expressions[0]
    assert result.recall.value == 1.0
    assert result.ranking.first_relevant_rank == 2
    assert result.ranking.mrr == 0.5
    assert result.ranking.ndcg < 1.0


def test_graded_ndcg_and_required_order_within_tier():
    dataset = _dataset()
    original = _case(dataset, "C-B01")
    ranking = RankingTruth(
        (("E-PAY-001",), ("E-PAY-002", "E-PAY-003")), True
    )
    truth = replace(original.truth, ranking=ranking)
    changed = replace(original, truth=truth)
    dataset = replace(
        dataset,
        cases=tuple(changed if item.case_id == changed.case_id else item for item in dataset.cases),
    )
    expression_id = _expression(changed)
    prediction = _perfect(
        changed, expression_id,
        ranked=("E-PAY-003", "E-PAY-002", "E-PAY-001"),
    )
    result = evaluate_predictions(dataset, _metadata(dataset, (expression_id,)), (prediction,)).expressions[0]
    assert result.recall.value == 1.0
    assert result.ranking.ndcg < 1.0
    assert not result.ranking.order_compliant


def test_missing_prediction_is_explicit_failure_and_slice_counts_are_stable():
    dataset = _dataset()
    locate = _case(dataset, "C-L01")
    discover = _case(dataset, "C-D01")
    ids = (_expression(locate), _expression(discover))
    report = evaluate_predictions(
        dataset, _metadata(dataset, ids), (_perfect(locate, ids[0]),)
    )
    missing = next(item for item in report.expressions if item.expression_id == ids[1])
    assert missing.missing_prediction
    assert missing.recall.value == 0.0
    assert not missing.behavior.passed
    assert missing.safety.violations == ("MISSING_PREDICTION",)
    assert report.aggregate.count == 2 and report.aggregate.missing == 1
    assert tuple(item.slice_value for item in report.request_type_slices) == (
        "DISCOVER", "LOCATE"
    )
    assert {item.slice_value: item.count for item in report.tag_slices}["fuzzy"] == 1
    assert all(item.summary.count == 1 for item in report.cases)


def test_unknown_forbidden_and_hard_filter_evidence_are_safety_not_ranking_only():
    dataset = _dataset()
    case = _case(dataset, "C-L01")
    expression_id = _expression(case)
    ranked = ("E-UNKNOWN", "E-SEC-001", "E-OLD-001", "E-PAY-001")
    prediction = _perfect(case, expression_id, ranked=ranked)
    result = evaluate_predictions(dataset, _metadata(dataset, (expression_id,)), (prediction,)).expressions[0]
    assert result.recall.value == 1.0
    assert result.ranking.first_relevant_rank == 4
    assert not result.safety.passed
    assert "UNKNOWN_EVIDENCE:E-UNKNOWN" in result.safety.violations
    assert "HARD_FILTER_VIOLATION:E-SEC-001" in result.safety.violations
    assert "HARD_FILTER_VIOLATION:E-OLD-001" in result.safety.violations


def test_behavior_trace_provenance_comparison_and_final_decision_violations():
    dataset = _dataset()
    case = _case(dataset, "C-A01")
    expression_id = _expression(case)
    prediction = Prediction(
        expression_id,
        KnowledgeRequestType.LOCATE,
        ("E-PAY-001",),
        OutputMode.CLARIFICATION,
        BoundaryTruth.CASE_INFORMATION_MISSING,
        ClarificationTruth.REQUIRED,
        VerificationTruth.REQUIRED,
        "NOT_SATISFIED",
        ("invented",),
        ("raw_query",),
        (),
        (),
        (ForbiddenBehavior.FINAL_BUSINESS_DECISION,),
        True,
    )
    result = evaluate_predictions(dataset, _metadata(dataset, (expression_id,)), (prediction,)).expressions[0]
    assert not result.behavior.passed
    assert {"request_type", "output_mode", "boundary", "clarification", "verification", "trace_fields", "source_reference_coverage", "lineage_coverage", "comparison_outcome"} <= set(result.behavior.failures)
    assert "FINAL_BUSINESS_DECISION" in result.safety.violations
    assert "FORBIDDEN_BEHAVIOR:FINAL_BUSINESS_DECISION" in result.safety.violations


def test_prediction_and_run_structure_reject_duplicates_extra_and_identity_conflicts():
    dataset = _dataset()
    case = _case(dataset, "C-L01")
    expression_id = _expression(case)
    with pytest.raises(EvaluationValidationError, match="duplicates"):
        replace(_perfect(case, expression_id), ranked_evidence_ids=("E-PAY-001", "E-PAY-001"))
    with pytest.raises(EvaluationValidationError, match="sorted"):
        RunMetadata(
            "run-x", "system-a", dataset.dataset_version, dataset.corpus_version,
            (case.expressions[1].expression_id, expression_id),
        )
    metadata = _metadata(dataset, (expression_id,))
    prediction = _perfect(case, expression_id)
    with pytest.raises(EvaluationValidationError, match="duplicate prediction"):
        evaluate_predictions(dataset, metadata, (prediction, prediction))
    extra = replace(prediction, expression_id=case.expressions[1].expression_id)
    with pytest.raises(EvaluationValidationError, match="outside selected"):
        evaluate_predictions(dataset, metadata, (extra,))
    with pytest.raises(EvaluationValidationError, match="identity mismatch"):
        evaluate_predictions(dataset, replace(metadata, corpus_version="other-v1"), ())


def test_reports_are_frozen_deterministic_and_strict_json_round_trip():
    dataset = _dataset()
    case = _case(dataset, "C-B01")
    ids = tuple(item.expression_id for item in case.expressions[:2])
    predictions = tuple(_perfect(case, item) for item in reversed(ids))
    report = evaluate_predictions(dataset, _metadata(dataset, ids), predictions)
    assert report.to_json() == report.to_json()
    assert EvaluationReport.from_json(report.to_json()) == report
    with pytest.raises(FrozenInstanceError):
        report.schema = "changed"
    value = report.to_dict()
    value["extra"] = True
    with pytest.raises(EvaluationValidationError, match="fields do not match"):
        EvaluationReport.from_json(json.dumps(value))
    value = report.to_dict()
    value["aggregate"]["count"] = "two"
    with pytest.raises(EvaluationValidationError, match="non-negative integer"):
        EvaluationReport.from_json(json.dumps(value))
    with pytest.raises(EvaluationValidationError, match="duplicate JSON key"):
        EvaluationReport.from_json('{"schema":"a","schema":"b"}')


def test_compare_reports_has_metric_deltas_and_local_expression_regressions():
    dataset = _dataset()
    case = _case(dataset, "C-B01")
    ids = tuple(item.expression_id for item in case.expressions[:2])
    baseline = evaluate_predictions(
        dataset, _metadata(dataset, ids, "baseline"),
        tuple(_perfect(case, item) for item in ids),
    )
    current = evaluate_predictions(
        dataset, _metadata(dataset, ids, "current"),
        (_perfect(case, ids[0], ranked=("E-PAY-001",)), _perfect(case, ids[1])),
    )
    comparison = compare_reports(baseline, current)
    assert comparison.regressed_expression_ids == (ids[0],)
    recall = next(item for item in comparison.aggregate_deltas if item.metric == "recall")
    assert recall.delta < 0 and recall.regression
    assert not hasattr(comparison, "overall_improved")
    assert EvaluationComparison.from_json(comparison.to_json()) == comparison


def test_compare_rejects_dataset_corpus_or_selection_mismatch():
    dataset = _dataset()
    case = _case(dataset, "C-L01")
    first, second = case.expressions[0].expression_id, case.expressions[1].expression_id
    left = evaluate_predictions(dataset, _metadata(dataset, (first,), "left"), (_perfect(case, first),))
    right_selection = evaluate_predictions(dataset, _metadata(dataset, (second,), "right"), (_perfect(case, second),))
    with pytest.raises(EvaluationValidationError, match="incompatible"):
        compare_reports(left, right_selection)
    incompatible_metadata = replace(left.metadata, run_id="right", dataset_version="other-v1")
    with pytest.raises(EvaluationValidationError, match="incompatible"):
        compare_reports(left, replace(left, metadata=incompatible_metadata))
    with pytest.raises(EvaluationValidationError, match="run identity conflict"):
        compare_reports(left, left)


def test_loader_has_no_execution_or_composite_score_surface():
    dataset = _dataset()
    case = _case(dataset, "C-L01")
    expression_id = _expression(case)
    report = evaluate_predictions(dataset, _metadata(dataset, (expression_id,)), (_perfect(case, expression_id),))
    assert not hasattr(report.aggregate, "score")
    assert not hasattr(report, "runner")
    assert not hasattr(report, "retrieval")
