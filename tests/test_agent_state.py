from dataclasses import FrozenInstanceError, fields

import pytest

from knowledge_system import (
    AgentState,
    AgentStateError,
    AgentStateValidationError,
    ApplyCompareResult,
    CaseContext,
    ClarificationAction,
    ClarificationDecision,
    ContextRelation,
    ContextTransition,
    Evidence,
    EvidenceMap,
    EvidenceMapGroup,
    EvidenceMapPlan,
    EvidenceMapSection,
    EvidenceQualitySignal,
    EvidenceQualitySignalType,
    KnowledgeBoundary,
    KnowledgeLineage,
    KnowledgeRequestType,
    LogicalOutcome,
    QueryContext,
    RecoveryResult,
    RecoveryStatus,
    RequestClassification,
    RetrievalOutcome,
    SourceReference,
    TaskContext,
    update_agent_state,
)
from knowledge_system.rule_structure import RuleExtractionResult, RuleExtractionStatus


def source():
    return SourceReference(name="规则.md", section="S1")


def lineage():
    return KnowledgeLineage("doc-1", "p1", "c1", "v1")


def evidence(structure=None):
    return Evidence("e-1", "u-1", "原文", source(), lineage(), structure)


def query_context(raw="  原始问题  "):
    return QueryContext(raw, "scope", KnowledgeRequestType.DISCOVER)


def request(raw="  原始问题  "):
    return RequestClassification(raw, KnowledgeRequestType.DISCOVER, ())


def test_minimal_state_preserves_raw_query_and_is_frozen():
    state = AgentState("session-1", "  原始问题  ")
    assert state.raw_query == "  原始问题  "
    assert state.candidate_evidence == ()
    with pytest.raises(FrozenInstanceError):
        state.raw_query = "changed"
    for value in ("", "   "):
        with pytest.raises(AgentStateValidationError):
            AgentState(value, "query")
        with pytest.raises(AgentStateValidationError):
            AgentState("session-1", value)


def test_all_slots_use_existing_domain_types():
    raw = "  原始问题  "
    query = query_context(raw)
    case = CaseContext(("事实描述",), ())
    transition = ContextTransition(
        ContextRelation.CONTINUE, raw, TaskContext(query, case), False, "continued"
    )
    clarification = ClarificationDecision(
        raw,
        RetrievalOutcome.NOT_FOUND,
        ClarificationAction.RETURN_BOUNDARY,
        KnowledgeBoundary.KNOWLEDGE_MISSING,
    )
    candidate = evidence()
    signal = EvidenceQualitySignal(
        EvidenceQualitySignalType.MISSING_CONTEXT,
        "missing_parent_context",
        candidate.evidence_id,
        candidate.source_reference,
        candidate.lineage,
    )
    recovery = RecoveryResult(
        candidate, RecoveryStatus.UNRESOLVED, (), (signal,), (), 1, True, "budget"
    )
    extraction = RuleExtractionResult(RuleExtractionStatus.EMPTY, None, ())
    comparison = ApplyCompareResult(
        extraction, (), (), (), (), (), logical_outcome=LogicalOutcome.INDETERMINATE
    )
    plan = EvidenceMapPlan(
        "plan-1", (EvidenceMapSection("section-1", 0, (EvidenceMapGroup("group-1", 0, ("e-1",)),)),)
    )
    evidence_map = EvidenceMap("plan-1", "request-1", "snapshot-1", "fingerprint", ())
    state = AgentState(
        "session-1",
        raw,
        query_context=query,
        case_context=case,
        knowledge_request=request(raw),
        context_resolution=transition,
        clarification=clarification,
        candidate_evidence=(candidate,),
        evidence_quality=(signal,),
        recovery_state=recovery,
        comparison_result=comparison,
        evidence_map_plan=plan,
        evidence_map=evidence_map,
        error=AgentStateError("UPSTREAM", "explicit failure"),
    )
    assert state.query_context == query
    assert state.recovery_state == recovery
    assert state.evidence_map_plan == plan


@pytest.mark.parametrize("slot", ["query_context", "knowledge_request"])
def test_original_query_conflict_is_rejected(slot):
    value = query_context("different") if slot == "query_context" else request("different")
    with pytest.raises(AgentStateValidationError, match="exactly match"):
        AgentState("session-1", "original", **{slot: value})


def test_explicit_update_transition_allows_task_query_to_differ_from_turn_query():
    task_query = query_context("原始任务问题")
    old_case = CaseContext(("已有事实",), ())
    transition = ContextTransition(
        ContextRelation.UPDATE,
        "现在52个赞",
        TaskContext(task_query, old_case.append("现在52个赞")),
        True,
        "case_updated",
    )
    state = AgentState(
        "session-1",
        transition.turn_query,
        query_context=transition.context.query,
        case_context=transition.context.case,
        context_resolution=transition,
        knowledge_request=request(transition.turn_query),
    )
    assert state.query_context.original_query == "原始任务问题"
    assert state.raw_query == "现在52个赞"


@pytest.mark.parametrize("mismatch", ["raw_query", "query_context", "case_context"])
def test_explicit_transition_binding_rejects_mismatched_state(mismatch):
    query = query_context("原始任务问题")
    case = CaseContext(("现在52个赞",), ())
    transition = ContextTransition(
        ContextRelation.UPDATE,
        "现在52个赞",
        TaskContext(query, case),
        True,
        "case_updated",
    )
    values = {
        "raw_query": transition.turn_query,
        "query_context": query,
        "case_context": case,
    }
    if mismatch == "raw_query":
        values[mismatch] = "different"
    elif mismatch == "query_context":
        values[mismatch] = query_context("different")
    else:
        values[mismatch] = CaseContext(("different",), ())
    with pytest.raises(AgentStateValidationError, match="exactly match"):
        AgentState(
            "session-1",
            values["raw_query"],
            query_context=values["query_context"],
            case_context=values["case_context"],
            context_resolution=transition,
        )


def test_new_transition_rejects_inherited_case_and_wrong_original_query():
    raw = "新的任务"
    wrong_query = query_context("旧任务")
    inherited_case = CaseContext(("旧事实",), ())
    with pytest.raises(AgentStateValidationError, match="NEW query_context"):
        transition = ContextTransition(
            ContextRelation.NEW,
            raw,
            TaskContext(wrong_query, CaseContext()),
            True,
            "new_query_context",
        )
        AgentState(
            "session-1",
            raw,
            query_context=wrong_query,
            case_context=CaseContext(),
            context_resolution=transition,
        )
    with pytest.raises(AgentStateValidationError, match="inherit prior case"):
        query = query_context(raw)
        transition = ContextTransition(
            ContextRelation.NEW,
            raw,
            TaskContext(query, inherited_case),
            True,
            "new_query_context",
        )
        AgentState(
            "session-1",
            raw,
            query_context=query,
            case_context=inherited_case,
            context_resolution=transition,
        )


def test_collections_and_nested_mappings_are_immutable_snapshots():
    structure = {"items": ["first"]}
    candidates = [evidence(structure)]
    state = AgentState("session-1", "query", candidate_evidence=candidates)
    candidates.append(Evidence("e-2", "u-2", "other", source(), lineage()))
    structure["items"].append("later")
    assert len(state.candidate_evidence) == 1
    assert state.candidate_evidence[0].derived_structure == {"items": ("first",)}
    with pytest.raises(TypeError):
        state.candidate_evidence[0].derived_structure["new"] = "value"


def test_pure_allowlisted_update_leaves_old_state_unchanged():
    old = AgentState("session-1", "query")
    candidates = [evidence()]
    new = update_agent_state(
        old,
        candidate_evidence=candidates,
        error=AgentStateError("FAILED", "reason"),
    )
    candidates.append(Evidence("e-2", "u-2", "other", source(), lineage()))
    assert old.error is None
    assert old.candidate_evidence == ()
    assert new.error == AgentStateError("FAILED", "reason")
    assert new.candidate_evidence == (evidence(),)
    assert new is not old
    with pytest.raises(AgentStateValidationError, match="unknown"):
        update_agent_state(old, trace_history=())
    with pytest.raises(AgentStateValidationError, match="non-updatable"):
        update_agent_state(old, session_id="session-2")


def test_wrong_slot_and_error_types_are_rejected():
    with pytest.raises(TypeError, match="query_context"):
        AgentState("session-1", "query", query_context="query")
    with pytest.raises(TypeError, match="error"):
        AgentState("session-1", "query", error={"code": "x"})
    with pytest.raises(AgentStateValidationError):
        AgentStateError("", "reason")


def test_state_surface_excludes_trace_runtime_and_future_phase_mechanisms():
    names = {item.name for item in fields(AgentState)}
    forbidden = {
        "trace",
        "trace_history",
        "retrieval_telemetry",
        "permissions",
        "knowledge_commit",
        "retrieval_service",
        "vector_store",
        "llm",
        "config",
    }
    assert names.isdisjoint(forbidden)
    assert not hasattr(AgentState, "save")
    assert not hasattr(AgentState, "resume")
    assert not hasattr(AgentState, "next_node")
