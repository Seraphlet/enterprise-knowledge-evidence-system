from dataclasses import fields, replace

import pytest

from knowledge_system import (
    AgentState,
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
    GraphExecutionError,
    GraphNode,
    GraphPipelines,
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
    Trace,
    VerificationMode,
    run_product_graph,
    update_agent_state,
)
from knowledge_system.product_graph import _RunGuard
from knowledge_system.rule_structure import RuleExtractionResult, RuleExtractionStatus


RAW = "  是否适用规则  "


def evidence():
    return Evidence(
        "e-1",
        "u-1",
        "正式规则原文",
        SourceReference("规则.md", section="S1"),
        KnowledgeLineage("doc-1", "p1", "c1", "v1"),
    )


def signal():
    item = evidence()
    return EvidenceQualitySignal(
        EvidenceQualitySignalType.MISSING_CONTEXT,
        "parent_missing",
        item.evidence_id,
        item.source_reference,
        item.lineage,
    )


def comparison():
    extraction = RuleExtractionResult(RuleExtractionStatus.EMPTY, None, ())
    return ApplyCompareResult(
        extraction,
        (),
        (),
        (),
        (),
        (),
        logical_outcome=LogicalOutcome.INDETERMINATE,
    )


def organized(state):
    plan = EvidenceMapPlan(
        "plan-1",
        (
            EvidenceMapSection(
                "section-1",
                0,
                (EvidenceMapGroup("group-1", 0, ("e-1",)),),
            ),
        ),
    )
    result = EvidenceMap("plan-1", "request-1", "snapshot-1", "fingerprint", ())
    return update_agent_state(state, evidence_map_plan=plan, evidence_map=result)


def trace_for(state):
    return Trace(
        "trace-1",
        state.raw_query,
        state.raw_query,
        candidate_evidence_ids=[item.evidence_id for item in state.candidate_evidence],
        final_evidence_ids=[item.evidence_id for item in state.candidate_evidence],
    )


def pipelines(
    *,
    request_type=KnowledgeRequestType.DISCOVER,
    clarification=None,
    quality=(),
    recovery=None,
    counters=None,
    retrieve_override=None,
):
    counts = counters if counters is not None else {}

    def counted(name, function):
        def call(state):
            counts[name] = counts.get(name, 0) + 1
            return function(state)

        return call

    def resolve(state):
        query = QueryContext(state.raw_query, None, request_type)
        case = CaseContext()
        transition = ContextTransition(
            ContextRelation.CONTINUE,
            state.raw_query,
            TaskContext(query, case),
            False,
            "resolved",
        )
        return update_agent_state(
            state,
            query_context=query,
            case_context=case,
            context_resolution=transition,
        )

    def understand(state):
        return update_agent_state(
            state,
            knowledge_request=RequestClassification(
                state.raw_query, request_type, ()
            ),
        )

    def clarify(state):
        return update_agent_state(state, clarification=clarification)

    def retrieve(state):
        if retrieve_override is not None:
            return retrieve_override(state)
        return update_agent_state(
            state,
            candidate_evidence=(evidence(),),
            evidence_quality=quality,
        )

    def recover(state):
        selected = recovery or RecoveryResult(
            evidence(), RecoveryStatus.RESOLVED, (), (), (), 1, False
        )
        return update_agent_state(state, recovery_state=selected)

    def analyze(state):
        return update_agent_state(state, comparison_result=comparison())

    def verify(state):
        counts["verification"] = counts.get("verification", 0) + 1
        return VerificationMode(True, ())

    def finalize(state):
        counts["trace"] = counts.get("trace", 0) + 1
        return trace_for(state)

    return GraphPipelines(
        counted("resolve", resolve),
        counted("understand", understand),
        counted("clarify", clarify),
        counted("retrieve", retrieve),
        counted("recover", recover),
        counted("compare", analyze),
        counted("organize", organized),
        verify,
        finalize,
    )


def test_reliable_evidence_runs_straight_through_without_recovery_or_verification():
    counts = {}
    result = run_product_graph(AgentState("s-1", RAW), pipelines(counters=counts))
    assert result.state.evidence_map is not None
    assert counts.get("recover", 0) == 0
    assert counts.get("verification", 0) == 0
    assert counts.get("compare", 0) == 0
    assert "evidence:reliable" in result.route_audit
    assert result.verification_mode is None


def test_clarification_block_short_circuits_retrieval():
    decision = ClarificationDecision(
        RAW,
        RetrievalOutcome.QUERY_INSUFFICIENT,
        ClarificationAction.BLOCK_FOR_CLARIFICATION,
        KnowledgeBoundary.RETRIEVAL_QUERY_INSUFFICIENT,
    )
    counts = {}
    result = run_product_graph(
        AgentState("s-1", RAW),
        pipelines(clarification=decision, counters=counts),
    )
    assert counts.get("retrieve", 0) == 0
    assert counts["trace"] == 1
    assert result.route_audit == ("clarification:block",)


def test_ambiguous_context_requires_blocking_clarification():
    query = QueryContext(RAW, None, KnowledgeRequestType.DISCOVER)
    case = CaseContext()
    transition = ContextTransition(
        ContextRelation.AMBIGUOUS,
        RAW,
        TaskContext(query, case),
        False,
        "ambiguous_relation_not_applied",
    )

    def resolve(state):
        return update_agent_state(
            state,
            query_context=query,
            case_context=case,
            context_resolution=transition,
        )

    ambiguous = replace(pipelines(), resolve_context=resolve)
    with pytest.raises(GraphExecutionError, match="ambiguous context"):
        run_product_graph(AgentState("s-1", RAW), ambiguous)

    decision = ClarificationDecision(
        RAW,
        RetrievalOutcome.QUERY_INSUFFICIENT,
        ClarificationAction.BLOCK_FOR_CLARIFICATION,
        KnowledgeBoundary.RETRIEVAL_QUERY_INSUFFICIENT,
    )
    blocked = replace(
        pipelines(clarification=decision),
        resolve_context=resolve,
    )
    result = run_product_graph(AgentState("s-1", RAW), blocked)
    assert result.route_audit == ("clarification:block",)


def test_out_of_scope_short_circuits_retrieval():
    counts = {}
    result = run_product_graph(
        AgentState("s-1", RAW),
        pipelines(request_type=KnowledgeRequestType.OUT_OF_SCOPE, counters=counts),
    )
    assert counts.get("retrieve", 0) == 0
    assert result.route_audit == ("request:out_of_scope",)


def test_successful_recovery_continues_without_verification():
    counts = {}
    result = run_product_graph(
        AgentState("s-1", RAW), pipelines(quality=(signal(),), counters=counts)
    )
    assert counts["recover"] == 1
    assert counts.get("verification", 0) == 0
    assert result.state.evidence_map is not None
    assert "recovery:resolved" in result.route_audit


def test_failed_recovery_enters_verification_and_stops_grounded_output():
    unresolved = RecoveryResult(
        evidence(),
        RecoveryStatus.UNRESOLVED,
        (),
        (signal(),),
        (),
        1,
        True,
        "budget_exhausted",
    )
    counts = {}
    result = run_product_graph(
        AgentState("s-1", RAW),
        pipelines(quality=(signal(),), recovery=unresolved, counters=counts),
    )
    assert result.verification_mode is not None
    assert result.verification_mode.active
    assert counts["verification"] == 1
    assert counts.get("organize", 0) == 0
    assert counts.get("compare", 0) == 0


def test_apply_compares_once_while_non_apply_skips_comparison():
    apply_counts = {}
    apply_result = run_product_graph(
        AgentState("s-1", RAW),
        pipelines(request_type=KnowledgeRequestType.APPLY, counters=apply_counts),
    )
    other_counts = {}
    other_result = run_product_graph(
        AgentState("s-2", RAW), pipelines(counters=other_counts)
    )
    assert apply_counts["compare"] == 1
    assert apply_result.state.comparison_result is not None
    assert other_counts.get("compare", 0) == 0
    assert "comparison:skipped" in other_result.route_audit


def test_every_routed_service_is_called_at_most_once():
    counts = {}
    run_product_graph(
        AgentState("s-1", RAW),
        pipelines(
            request_type=KnowledgeRequestType.APPLY,
            quality=(signal(),),
            counters=counts,
        ),
    )
    assert counts
    assert set(counts.values()) == {1}


def test_state_identity_raw_query_and_old_state_are_preserved():
    initial = AgentState("s-1", RAW)
    result = run_product_graph(initial, pipelines())
    assert initial.query_context is None
    assert initial.candidate_evidence == ()
    assert result.state is not initial
    assert result.state.session_id == initial.session_id
    assert result.state.raw_query == RAW


def test_invalid_service_return_and_identity_drift_fail_explicitly():
    invalid = pipelines()
    invalid = replace(invalid, resolve_context=lambda state: object())
    with pytest.raises(GraphExecutionError, match="INVALID_STATE_RETURN"):
        run_product_graph(AgentState("s-1", RAW), invalid)

    drift = pipelines()
    drift = replace(
        drift,
        resolve_context=lambda state: replace(state, session_id="different"),
    )
    with pytest.raises(GraphExecutionError, match="STATE_IDENTITY_DRIFT"):
        run_product_graph(AgentState("s-1", RAW), drift)


def test_pipeline_must_return_a_new_state_instance():
    unchanged = pipelines()
    unchanged = replace(unchanged, resolve_context=lambda state: state)
    with pytest.raises(GraphExecutionError, match="STATE_NOT_REPLACED"):
        run_product_graph(AgentState("s-1", RAW), unchanged)


def test_missing_required_artifact_fails_explicitly():
    missing = pipelines()
    missing = replace(
        missing, resolve_context=lambda state: update_agent_state(state)
    )
    with pytest.raises(GraphExecutionError, match="MISSING_ARTIFACT"):
        run_product_graph(AgentState("s-1", RAW), missing)


def test_cycle_and_step_guards_are_explicit():
    guard = _RunGuard(2)
    guard.visit(GraphNode.RESOLVE_CONTEXT)
    with pytest.raises(GraphExecutionError, match="GRAPH_CYCLE"):
        guard.visit(GraphNode.RESOLVE_CONTEXT)
    with pytest.raises(GraphExecutionError, match="GRAPH_STEP_LIMIT"):
        run_product_graph(AgentState("s-1", RAW), pipelines(), max_steps=1)


def test_invalid_verification_and_trace_returns_fail_explicitly():
    unresolved = RecoveryResult(
        evidence(),
        RecoveryStatus.UNRESOLVED,
        (),
        (signal(),),
        (),
        1,
        True,
        "budget_exhausted",
    )
    invalid_verification = replace(
        pipelines(quality=(signal(),), recovery=unresolved),
        build_verification=lambda state: object(),
    )
    with pytest.raises(GraphExecutionError, match="INVALID_VERIFICATION_RETURN"):
        run_product_graph(AgentState("s-1", RAW), invalid_verification)

    invalid_trace = replace(pipelines(), finalize_trace=lambda state: object())
    with pytest.raises(GraphExecutionError, match="INVALID_TRACE_RETURN"):
        run_product_graph(AgentState("s-1", RAW), invalid_trace)

    drifted_trace = replace(
        pipelines(),
        finalize_trace=lambda state: replace(trace_for(state), raw_query="different"),
    )
    with pytest.raises(GraphExecutionError, match="TRACE_IDENTITY_DRIFT"):
        run_product_graph(AgentState("s-1", RAW), drifted_trace)


def test_trace_and_control_audit_stay_outside_state_without_persistence_surface():
    result = run_product_graph(AgentState("s-1", RAW), pipelines())
    state_fields = {item.name for item in fields(AgentState)}
    assert isinstance(result.trace, Trace)
    assert result.visited_nodes[-1] is GraphNode.END
    assert "trace" not in state_fields
    assert "visited_nodes" not in state_fields
    assert "route_audit" not in state_fields
    assert not hasattr(run_product_graph, "save")
    assert not hasattr(run_product_graph, "resume")
