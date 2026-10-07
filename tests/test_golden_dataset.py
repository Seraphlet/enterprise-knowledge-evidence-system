from dataclasses import FrozenInstanceError
import json
from pathlib import Path

import pytest

from knowledge_system import (
    BoundaryTruth,
    ForbiddenBehavior,
    GoldenDatasetValidationError,
    KnowledgeRequestType,
    OutputMode,
    VerificationTruth,
    load_golden_dataset,
)


FIXTURE = Path(__file__).parents[1] / "data" / "golden_dataset_v1.json"


def _raw():
    return json.loads(FIXTURE.read_text(encoding="utf-8"))


def _write(tmp_path, value):
    path = tmp_path / "mutated.json"
    path.write_text(
        json.dumps(value, ensure_ascii=False, sort_keys=True), encoding="utf-8"
    )
    return path


def _case(value, case_id):
    return next(item for item in value["cases"] if item["case_id"] == case_id)


def test_fixture_size_distribution_versions_and_synthetic_authority():
    dataset = load_golden_dataset(FIXTURE)
    assert dataset.schema_version == 1
    assert dataset.dataset_version == "golden-v1"
    assert dataset.corpus_version == "synthetic-corpus-v1"
    assert dataset.authority == "SYNTHETIC_EVALUATION_ONLY_NOT_FORMAL_POLICY"
    assert "not formal" in dataset.fixture_notice.casefold()
    assert len(dataset.cases) == 24
    assert dataset.expression_count == 72
    assert dict(dataset.request_type_counts) == {
        "APPLY": 6,
        "BROWSE": 4,
        "DISCOVER": 5,
        "LOCATE": 4,
        "OUT_OF_SCOPE": 5,
    }
    assert all(len(case.expressions) >= 3 for case in dataset.cases)


def test_catalog_is_closed_traceable_and_never_claims_real_policy_authority():
    dataset = load_golden_dataset(FIXTURE)
    catalog = {item.evidence.evidence_id: item for item in dataset.evidence_catalog}
    assert len(catalog) == 10
    assert all(item.evidence.original_content.startswith("合成") for item in catalog.values())
    assert all(item.evidence.source_reference.name for item in catalog.values())
    assert all(item.evidence.source_reference.url is None for item in catalog.values())
    assert all(item.evidence.source_reference.page is None for item in catalog.values())
    assert all(item.evidence.lineage.processing_version for item in catalog.values())
    for case in dataset.cases:
        references = (
            set(case.truth.recall.relevant_evidence_ids)
            | set(case.truth.forbidden_evidence_ids)
            | {item for tier in case.truth.ranking.preferred_tiers for item in tier}
        )
        if case.truth.browse is not None:
            references |= set(case.truth.browse.expected_evidence_ids)
        assert references <= set(catalog)


def test_truth_is_behavioral_with_recall_ranking_and_browse_set_separated():
    dataset = load_golden_dataset(FIXTURE)
    browse_cases = [case for case in dataset.cases if case.request_type is KnowledgeRequestType.BROWSE]
    assert browse_cases
    for case in browse_cases:
        assert case.truth.browse is not None
        assert not case.truth.browse.top1_sufficient
        assert set(case.truth.browse.expected_evidence_ids) == set(case.truth.recall.relevant_evidence_ids)
        assert ForbiddenBehavior.BROWSE_TOP1_ONLY in case.truth.forbidden_behaviors
    assert not any(hasattr(case.truth, name) for case in dataset.cases for name in ("answer", "expected_text", "final_answer"))


def test_boundaries_failures_multi_turn_and_apply_human_responsibility_are_covered():
    dataset = load_golden_dataset(FIXTURE)
    boundaries = {case.truth.boundary for case in dataset.cases}
    assert {
        BoundaryTruth.CASE_INFORMATION_MISSING,
        BoundaryTruth.KNOWLEDGE_MISSING,
        BoundaryTruth.KNOWLEDGE_AMBIGUOUS,
        BoundaryTruth.SOURCE_FAILURE,
        BoundaryTruth.RETRIEVAL_FAILURE,
        BoundaryTruth.PROCESSING_FAILURE,
        BoundaryTruth.RECOVERY_FAILURE,
        BoundaryTruth.QUERY_INSUFFICIENT,
        BoundaryTruth.OUT_OF_SCOPE,
    } <= boundaries
    assert any(len(expression.turns) > 1 for case in dataset.cases for expression in case.expressions)
    multi_turn = next(case for case in dataset.cases if case.case_id == "C-A06")
    assert multi_turn.truth.comparison.acceptable_outcomes == ("SATISFIED",)
    assert all(expression.turns[-1].relation == "UPDATE" for expression in multi_turn.expressions)
    clarification = next(case for case in dataset.cases if case.case_id == "C-D05")
    assert all(len(expression.turns) == 2 for expression in clarification.expressions)
    assert all(expression.turns[-1].relation == "AMBIGUOUS" for expression in clarification.expressions)
    for case in dataset.cases:
        if case.request_type is KnowledgeRequestType.APPLY:
            assert case.truth.comparison is not None
            assert case.truth.comparison.final_decision_forbidden
            assert ForbiddenBehavior.FINAL_BUSINESS_DECISION in case.truth.forbidden_behaviors
    assert any(case.truth.verification is VerificationTruth.REQUIRED for case in dataset.cases)


def test_loaded_models_are_frozen_and_stably_sorted():
    dataset = load_golden_dataset(FIXTURE)
    assert tuple(case.case_id for case in dataset.cases) == tuple(sorted(case.case_id for case in dataset.cases))
    assert tuple(item.evidence.evidence_id for item in dataset.evidence_catalog) == tuple(sorted(item.evidence.evidence_id for item in dataset.evidence_catalog))
    with pytest.raises(FrozenInstanceError):
        dataset.dataset_version = "changed"
    with pytest.raises(TypeError):
        dataset.request_type_counts["LOCATE"] = 0


@pytest.mark.parametrize(
    "mutation,match",
    [
        (lambda value: value.update({"extra": True}), "fields do not match"),
        (lambda value: value.pop("corpus_version"), "fields do not match"),
        (lambda value: value.update({"schema_version": 99}), "schema version"),
        (lambda value: value.update({"authority": "FORMAL_POLICY"}), "disclaim"),
    ],
)
def test_rejects_unknown_missing_version_and_false_authority(tmp_path, mutation, match):
    value = _raw()
    mutation(value)
    with pytest.raises(GoldenDatasetValidationError, match=match):
        load_golden_dataset(_write(tmp_path, value))


def test_rejects_duplicate_case_expression_query_and_coverage_gap(tmp_path):
    value = _raw()
    value["cases"][1]["case_id"] = value["cases"][0]["case_id"]
    with pytest.raises(GoldenDatasetValidationError, match="duplicate case_id"):
        load_golden_dataset(_write(tmp_path, value))

    value = _raw()
    value["cases"][1]["expressions"][0]["expression_id"] = value["cases"][0]["expressions"][0]["expression_id"]
    with pytest.raises(GoldenDatasetValidationError, match="duplicate expression_id"):
        load_golden_dataset(_write(tmp_path, value))

    value = _raw()
    value["cases"][1]["expressions"][0]["query"] = value["cases"][0]["expressions"][0]["query"]
    with pytest.raises(GoldenDatasetValidationError, match="duplicate expression query"):
        load_golden_dataset(_write(tmp_path, value))

    value = _raw()
    value["cases"][0]["expressions"][0]["query"] = "Find Policy"
    value["cases"][0]["expressions"][0]["turns"][-1]["text"] = "Find Policy"
    value["cases"][1]["expressions"][0]["query"] = "ＦＩＮＤ　ＰＯＬＩＣＹ"
    value["cases"][1]["expressions"][0]["turns"][-1]["text"] = "ＦＩＮＤ　ＰＯＬＩＣＹ"
    with pytest.raises(GoldenDatasetValidationError, match="duplicate expression query"):
        load_golden_dataset(_write(tmp_path, value))

    value = _raw()
    value["cases"] = [item for item in value["cases"] if item["case_id"] != "C-L01"]
    with pytest.raises(GoldenDatasetValidationError, match="coverage"):
        load_golden_dataset(_write(tmp_path, value))


def test_rejects_dangling_evidence_and_recall_ranking_conflation(tmp_path):
    value = _raw()
    _case(value, "C-L01")["truth"]["recall"]["relevant_evidence_ids"] = ["E-MISSING"]
    with pytest.raises(GoldenDatasetValidationError, match="dangling"):
        load_golden_dataset(_write(tmp_path, value))

    value = _raw()
    _case(value, "C-L01")["truth"]["ranking"]["preferred_tiers"] = [["E-HR-001"]]
    with pytest.raises(GoldenDatasetValidationError, match="ranking preference"):
        load_golden_dataset(_write(tmp_path, value))


def test_rejects_fabricated_page_and_internal_path_locator(tmp_path):
    value = _raw()
    value["evidence_catalog"][0]["source_reference"]["page"] = 7
    with pytest.raises(GoldenDatasetValidationError, match="cannot be verified"):
        load_golden_dataset(_write(tmp_path, value))

    value = _raw()
    value["evidence_catalog"][0]["source_reference"]["page"] = 7
    value["evidence_catalog"][0]["page_is_verified"] = True
    with pytest.raises(GoldenDatasetValidationError, match="cannot be verified"):
        load_golden_dataset(_write(tmp_path, value))

    value = _raw()
    value["evidence_catalog"][0]["source_reference"]["name"] = "C:\\internal\\policy.md"
    with pytest.raises(GoldenDatasetValidationError, match="internal path"):
        load_golden_dataset(_write(tmp_path, value))

    value = _raw()
    value["evidence_catalog"][0]["lineage"]["source_position"] = "C:\\private\\source.md"
    with pytest.raises(GoldenDatasetValidationError, match="internal path"):
        load_golden_dataset(_write(tmp_path, value))


def test_rejects_duplicate_unit_inaccessible_recall_and_unverified_unreliable_evidence(tmp_path):
    value = _raw()
    value["evidence_catalog"][1]["unit_id"] = value["evidence_catalog"][0]["unit_id"]
    with pytest.raises(GoldenDatasetValidationError, match="duplicate unit_id"):
        load_golden_dataset(_write(tmp_path, value))

    value = _raw()
    _case(value, "C-L01")["truth"]["recall"]["relevant_evidence_ids"] = ["E-SEC-001"]
    _case(value, "C-L01")["truth"]["ranking"]["preferred_tiers"] = [["E-SEC-001"]]
    _case(value, "C-L01")["truth"]["forbidden_evidence_ids"].remove("E-SEC-001")
    with pytest.raises(GoldenDatasetValidationError, match="permission filters"):
        load_golden_dataset(_write(tmp_path, value))

    value = _raw()
    _case(value, "C-L03")["truth"]["verification"] = "FORBIDDEN"
    _case(value, "C-L03")["truth"]["allowed_output_modes"] = ["EVIDENCE"]
    with pytest.raises(GoldenDatasetValidationError, match="requires Verification"):
        load_golden_dataset(_write(tmp_path, value))


def test_rejects_query_turn_mismatch_and_missing_hard_filter_forbidden_set(tmp_path):
    value = _raw()
    _case(value, "C-L01")["expressions"][0]["turns"][-1]["text"] = "different"
    with pytest.raises(GoldenDatasetValidationError, match="final turn"):
        load_golden_dataset(_write(tmp_path, value))

    value = _raw()
    _case(value, "C-L01")["truth"]["forbidden_evidence_ids"].remove("E-SEC-001")
    with pytest.raises(GoldenDatasetValidationError, match="inaccessible or inactive"):
        load_golden_dataset(_write(tmp_path, value))


def test_rejects_browse_top1_fixed_answer_truth_and_illegal_apply_decision(tmp_path):
    value = _raw()
    _case(value, "C-B01")["truth"]["browse"]["top1_sufficient"] = True
    with pytest.raises(GoldenDatasetValidationError, match="never Top-1"):
        load_golden_dataset(_write(tmp_path, value))

    value = _raw()
    _case(value, "C-L01")["truth"]["expected_text"] = "唯一固定答案"
    with pytest.raises(GoldenDatasetValidationError, match="fields do not match"):
        load_golden_dataset(_write(tmp_path, value))

    value = _raw()
    _case(value, "C-A01")["truth"]["comparison"]["final_decision_forbidden"] = False
    with pytest.raises(GoldenDatasetValidationError, match="human final responsibility"):
        load_golden_dataset(_write(tmp_path, value))


def test_rejects_duplicate_json_key_invalid_utf8_and_non_file(tmp_path):
    duplicate = tmp_path / "duplicate.json"
    duplicate.write_text('{"schema":"a","schema":"b"}', encoding="utf-8")
    with pytest.raises(GoldenDatasetValidationError, match="duplicate JSON key"):
        load_golden_dataset(duplicate)
    invalid = tmp_path / "invalid.json"
    invalid.write_bytes(b"\xff\xfe")
    with pytest.raises(GoldenDatasetValidationError, match="strict UTF-8"):
        load_golden_dataset(invalid)
    with pytest.raises(GoldenDatasetValidationError, match="regular file"):
        load_golden_dataset(tmp_path)


def test_loader_has_no_retrieval_metric_runner_llm_or_network_surface():
    dataset = load_golden_dataset(FIXTURE)
    forbidden = {"run", "retrieve", "score", "metrics", "report", "llm", "network"}
    assert forbidden.isdisjoint(dataset.__dataclass_fields__)
    assert all(mode in OutputMode for case in dataset.cases for mode in case.truth.allowed_output_modes)
