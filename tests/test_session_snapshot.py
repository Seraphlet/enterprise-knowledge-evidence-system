from dataclasses import FrozenInstanceError
from datetime import datetime, timezone
from decimal import Decimal
import hashlib
import json
from pathlib import Path

import pytest

from knowledge_system import (
    AgentState,
    AgentStateError,
    ApplyCompareResult,
    CaseContext,
    ClarificationAction,
    ClarificationDecision,
    ContextRelation,
    ContextTransition,
    Evidence,
    EvidenceMap,
    EvidenceMapEntry,
    EvidenceMapGroup,
    EvidenceMapPlan,
    EvidenceMapSection,
    EvidenceQualitySignal,
    EvidenceQualitySignalType,
    HydratedEvidenceGroup,
    HydratedEvidenceSection,
    KnowledgeBoundary,
    KnowledgeLineage,
    KnowledgeRequestType,
    LogicalOutcome,
    QueryContext,
    RecoveryAction,
    RecoveryResult,
    RecoveryRoundTrace,
    RecoveryStatus,
    RecoveryTarget,
    RequestClassification,
    RetrievalOutcome,
    SessionSnapshotError,
    SourceReference,
    TaskContext,
    resume_session_snapshot,
    save_session_snapshot,
)
from knowledge_system.rule_structure import RuleExtractionResult, RuleExtractionStatus
import knowledge_system.session_snapshot as snapshot_module


NOW = "2026-10-06T07:30:00+08:00"


def _source():
    return SourceReference("规则.md", url="https://example.test/rule", section="S1", page=2)


def _lineage():
    return KnowledgeLineage("doc-1", "parser-1", "chunker-1", "process-1", "p:4", "parent-1")


def _evidence():
    return Evidence(
        "e-1",
        "u-1",
        "正式规则原文",
        _source(),
        _lineage(),
        {"threshold": Decimal("52.00"), "labels": ("正式", "规则")},
    )


def _full_state():
    raw = "  是否适用规则  "
    query = QueryContext(raw, "支付", KnowledgeRequestType.APPLY, ())
    case = CaseContext(("当前有52个赞",), ())
    transition = ContextTransition(
        ContextRelation.CONTINUE,
        raw,
        TaskContext(query, case),
        False,
        "continued",
    )
    evidence = _evidence()
    signal = EvidenceQualitySignal(
        EvidenceQualitySignalType.MISSING_CONTEXT,
        "parent_missing",
        evidence.evidence_id,
        evidence.source_reference,
        evidence.lineage,
    )
    target = RecoveryTarget("parent-1", evidence.source_reference, evidence.lineage)
    recovery_trace = RecoveryRoundTrace(
        1,
        RecoveryAction.PARENT_UNIT,
        target,
        (signal,),
        (),
        "not_found",
        (signal,),
    )
    recovery = RecoveryResult(
        evidence,
        RecoveryStatus.UNRESOLVED,
        (),
        (signal,),
        (recovery_trace,),
        1,
        True,
        "round_budget_exhausted",
        ((RecoveryAction.PARENT_UNIT, target),),
    )
    extraction = RuleExtractionResult(evidence, RuleExtractionStatus.EMPTY, None)
    comparison = ApplyCompareResult(
        extraction,
        (),
        (),
        (),
        (),
        (),
        logical_outcome=LogicalOutcome.INDETERMINATE,
    )
    plan = EvidenceMapPlan(
        "plan-1",
        (EvidenceMapSection("section-1", 0, (EvidenceMapGroup("group-1", 0, ("e-1",)),)),),
    )
    entry = EvidenceMapEntry(
        evidence.evidence_id,
        evidence.unit_id,
        evidence.original_content,
        evidence.source_reference,
        evidence.lineage,
        '{"safe":true}',
    )
    evidence_map = EvidenceMap(
        "plan-1",
        "request-1",
        "candidate-1",
        "fingerprint-1",
        (HydratedEvidenceSection("section-1", 0, (HydratedEvidenceGroup("group-1", 0, (entry,)),)),),
    )
    return AgentState(
        "session-1",
        raw,
        query_context=query,
        case_context=case,
        knowledge_request=RequestClassification(raw, KnowledgeRequestType.APPLY, ()),
        context_resolution=transition,
        clarification=ClarificationDecision(
            raw,
            RetrievalOutcome.USEFUL,
            ClarificationAction.PROCEED_WITH_EVIDENCE,
            KnowledgeBoundary.CASE_INFORMATION_MISSING,
            ("e-1",),
            ("merchant_level",),
            ("case_information_missing_does_not_block",),
        ),
        candidate_evidence=(evidence,),
        evidence_quality=(signal,),
        recovery_state=recovery,
        comparison_result=comparison,
        evidence_map_plan=plan,
        evidence_map=evidence_map,
        error=AgentStateError("UPSTREAM", "explicit failure"),
    )


def _read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def _write_resigned(path, payload):
    signed = {key: value for key, value in payload.items() if key != "digest"}
    canonical = json.dumps(signed, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)
    payload["digest"] = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    path.write_text(json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n", encoding="utf-8")


def test_minimal_and_full_state_round_trip_as_new_immutable_values(tmp_path):
    for index, state in enumerate((AgentState("session-1", "问题"), _full_state())):
        path = tmp_path / f"state-{index}.json"
        metadata = save_session_snapshot(state, path, revision=index, saved_at=NOW)
        resumed = resume_session_snapshot(path, "session-1")
        assert resumed.state == state
        assert resumed.state is not state
        assert resumed.metadata == metadata
        with pytest.raises(FrozenInstanceError):
            resumed.state.raw_query = "changed"


def test_exact_types_tuple_mapping_decimal_and_provenance_survive(tmp_path):
    path = tmp_path / "state.json"
    state = _full_state()
    save_session_snapshot(state, path, saved_at=NOW)
    restored = resume_session_snapshot(path, "session-1").state
    structure = restored.candidate_evidence[0].derived_structure
    assert type(restored.candidate_evidence) is tuple
    assert type(structure["labels"]) is tuple
    assert structure["threshold"] == Decimal("52.00")
    assert type(structure["threshold"]) is Decimal
    assert restored.candidate_evidence[0].source_reference.page == 2
    assert restored.recovery_state.trace[0].target.lineage.source_position == "p:4"


def test_output_is_stable_utf8_and_contains_only_minimal_envelope(tmp_path):
    first = tmp_path / "one.json"
    second = tmp_path / "two.json"
    state = _full_state()
    save_session_snapshot(state, first, revision=7, saved_at=NOW)
    save_session_snapshot(state, second, revision=7, clock=lambda: datetime.fromisoformat(NOW))
    assert first.read_bytes() == second.read_bytes()
    assert "正式规则原文".encode("utf-8") in first.read_bytes()
    root = _read(first)
    assert set(root) == {"schema", "version", "session_id", "saved_at", "revision", "state", "digest"}
    text = first.read_text(encoding="utf-8")
    for forbidden in ("trace_history", "visited_nodes", "route_audit", "permissions", "llm", "config", "chat_history", "user_profile"):
        assert forbidden not in text


def test_atomic_overwrite_and_replace_failure_preserves_original(tmp_path, monkeypatch):
    path = tmp_path / "state.json"
    save_session_snapshot(AgentState("session-1", "old"), path, saved_at=NOW)
    save_session_snapshot(AgentState("session-1", "new"), path, revision=2, saved_at=NOW)
    assert resume_session_snapshot(path, "session-1").state.raw_query == "new"
    old = path.read_bytes()
    monkeypatch.setattr(snapshot_module.os, "replace", lambda *args: (_ for _ in ()).throw(OSError("replace failed")))
    with pytest.raises(SessionSnapshotError, match="SNAPSHOT_WRITE_FAILED"):
        save_session_snapshot(AgentState("session-1", "new"), path, saved_at=NOW)
    assert path.read_bytes() == old
    assert list(tmp_path.glob("*.tmp")) == []


def test_save_does_not_mutate_input(tmp_path):
    state = _full_state()
    before = repr(state)
    save_session_snapshot(state, tmp_path / "state.json", saved_at=NOW)
    assert repr(state) == before


def test_resume_requires_explicit_existing_file_and_same_session(tmp_path):
    path = tmp_path / "state.json"
    save_session_snapshot(AgentState("session-1", "query"), path, saved_at=NOW)
    with pytest.raises(SessionSnapshotError, match="CROSS_SESSION"):
        resume_session_snapshot(path, "session-2")
    with pytest.raises(SessionSnapshotError, match="INVALID_PATH"):
        resume_session_snapshot(tmp_path, "session-1")
    with pytest.raises(SessionSnapshotError, match="INVALID_PATH"):
        resume_session_snapshot(tmp_path / "missing.json", "session-1")


@pytest.mark.parametrize("field,value,code", [
    ("schema", "unknown", "UNKNOWN_SCHEMA"),
    ("version", 999, "UNKNOWN_VERSION"),
    ("session_id", "different", "SESSION_MISMATCH"),
])
def test_rejects_unknown_schema_version_and_state_identity(tmp_path, field, value, code):
    path = tmp_path / "state.json"
    save_session_snapshot(AgentState("session-1", "query"), path, saved_at=NOW)
    root = _read(path)
    if field == "session_id":
        root["state"]["fields"]["session_id"] = value
    else:
        root[field] = value
    _write_resigned(path, root)
    with pytest.raises(SessionSnapshotError, match=code):
        resume_session_snapshot(path, "session-1")


def test_rejects_extra_missing_unknown_type_and_duplicate_json_keys(tmp_path):
    path = tmp_path / "state.json"
    save_session_snapshot(AgentState("session-1", "query"), path, saved_at=NOW)
    root = _read(path)
    root["extra"] = True
    _write_resigned(path, root)
    with pytest.raises(SessionSnapshotError, match="INVALID_FIELDS"):
        resume_session_snapshot(path, "session-1")

    save_session_snapshot(AgentState("session-1", "query"), path, saved_at=NOW)
    root = _read(path)
    del root["state"]["fields"]["error"]
    _write_resigned(path, root)
    with pytest.raises(SessionSnapshotError, match="INVALID_FIELDS"):
        resume_session_snapshot(path, "session-1")

    save_session_snapshot(AgentState("session-1", "query"), path, saved_at=NOW)
    root = _read(path)
    root["state"]["name"] = "os.system"
    _write_resigned(path, root)
    with pytest.raises(SessionSnapshotError, match="UNKNOWN_TYPE"):
        resume_session_snapshot(path, "session-1")

    path.write_text('{"schema":"x","schema":"y"}', encoding="utf-8")
    with pytest.raises(SessionSnapshotError, match="DUPLICATE_KEY"):
        resume_session_snapshot(path, "session-1")


def test_rejects_invalid_utf8_corruption_and_tampering(tmp_path):
    path = tmp_path / "state.json"
    path.write_bytes(b"\xff\xfe")
    with pytest.raises(SessionSnapshotError, match="INVALID_UTF8"):
        resume_session_snapshot(path, "session-1")
    path.write_text("{broken", encoding="utf-8")
    with pytest.raises(SessionSnapshotError, match="INVALID_JSON"):
        resume_session_snapshot(path, "session-1")
    save_session_snapshot(AgentState("session-1", "query"), path, saved_at=NOW)
    root = _read(path)
    root["state"]["fields"]["raw_query"] = "tampered"
    path.write_text(json.dumps(root), encoding="utf-8")
    with pytest.raises(SessionSnapshotError, match="TAMPERED"):
        resume_session_snapshot(path, "session-1")


def test_unsupported_values_are_rejected_before_original_is_replaced(tmp_path):
    path = tmp_path / "state.json"
    save_session_snapshot(AgentState("session-1", "old"), path, saved_at=NOW)
    old = path.read_bytes()
    state = AgentState("session-1", "query", candidate_evidence=(Evidence("e", "u", "x", _source(), _lineage(), {"bad": object()}),))
    with pytest.raises(SessionSnapshotError, match="UNSUPPORTED_TYPE"):
        save_session_snapshot(state, path, saved_at=NOW)
    assert path.read_bytes() == old


def test_resume_has_no_graph_trace_or_service_side_effect_surface(tmp_path):
    path = tmp_path / "state.json"
    save_session_snapshot(AgentState("session-1", "query"), path, saved_at=NOW)
    result = resume_session_snapshot(path, "session-1")
    assert set(result.__dataclass_fields__) == {"state", "metadata"}
    assert "trace" not in result.state.__dataclass_fields__
    assert not hasattr(result, "graph")
    assert not hasattr(result, "retrieval_service")
