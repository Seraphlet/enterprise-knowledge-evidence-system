"""Exact P10-F001 multi-turn regression across Graph, context, and snapshot."""

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
    ExtractedFact,
    GraphPipelines,
    KnowledgeBoundary,
    KnowledgeLineage,
    KnowledgeRequestType,
    LogicalOutcome,
    QueryContext,
    RequestClassification,
    RetrievalOutcome,
    SourceReference,
    TaskContext,
    Trace,
    VerificationMode,
    apply_context_turn,
    resume_session_snapshot,
    run_product_graph,
    save_session_snapshot,
    update_agent_state,
)
from knowledge_system.rule_structure import RuleExtractionResult, RuleExtractionStatus


ORIGINAL = "这个情况是否符合支付升级规则？"
FOLLOW_UP = "现在52个赞"
NOW = "2026-10-06T08:30:00+08:00"


def _evidence():
    return Evidence(
        "e-1",
        "u-1",
        "正式规则原文",
        SourceReference("规则.md", section="S1"),
        KnowledgeLineage("doc-1", "parser-1", "chunker-1", "process-1"),
    )


def _comparison():
    extraction = RuleExtractionResult(
        _evidence(),
        RuleExtractionStatus.EMPTY,
        None,
    )
    return ApplyCompareResult(
        extraction,
        (),
        (),
        (),
        (),
        (),
        logical_outcome=LogicalOutcome.INDETERMINATE,
    )


def _organize(state):
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
    evidence_map = EvidenceMap("plan-1", "request-1", "candidate-1", "fp", ())
    return update_agent_state(
        state,
        evidence_map_plan=plan,
        evidence_map=evidence_map,
    )


def _pipelines(trace_id, *, block=False):
    def resolve(state):
        if state.context_resolution is None:
            query = QueryContext(
                state.raw_query,
                "支付",
                KnowledgeRequestType.APPLY,
            )
            case = CaseContext()
            transition = ContextTransition(
                ContextRelation.CONTINUE,
                state.raw_query,
                TaskContext(query, case),
                False,
                "initial_context",
            )
        else:
            transition = state.context_resolution
            query = transition.context.query
            case = transition.context.case
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
                state.raw_query,
                state.query_context.intent,
                (),
            ),
        )

    def clarify(state):
        decision = None
        if block:
            decision = ClarificationDecision(
                state.raw_query,
                RetrievalOutcome.QUERY_INSUFFICIENT,
                ClarificationAction.BLOCK_FOR_CLARIFICATION,
                KnowledgeBoundary.RETRIEVAL_QUERY_INSUFFICIENT,
            )
        return update_agent_state(state, clarification=decision)

    def retrieve(state):
        return update_agent_state(state, candidate_evidence=(_evidence(),))

    def compare(state):
        return update_agent_state(state, comparison_result=_comparison())

    def finalize(state):
        return Trace(
            trace_id,
            state.raw_query,
            state.raw_query,
            candidate_evidence_ids=["e-1"] if state.candidate_evidence else [],
            final_evidence_ids=["e-1"] if state.candidate_evidence else [],
        )

    return GraphPipelines(
        resolve,
        understand,
        clarify,
        retrieve,
        lambda state: update_agent_state(state),
        compare,
        _organize,
        lambda state: VerificationMode(True, ()),
        finalize,
    )


def _next_state(resumed, turn_query, transition):
    return AgentState(
        resumed.session_id,
        turn_query,
        query_context=transition.context.query,
        case_context=transition.context.case,
        context_resolution=transition,
    )


def test_completed_turn_resume_update_runs_second_graph_with_independent_trace(tmp_path):
    initial = AgentState("session-1", ORIGINAL)
    turn1 = run_product_graph(initial, _pipelines("trace-turn-1"))
    snapshot = tmp_path / "turn-1.json"
    save_session_snapshot(turn1.state, snapshot, revision=1, saved_at=NOW)
    resumed = resume_session_snapshot(snapshot, "session-1").state
    old_task = TaskContext(resumed.query_context, resumed.case_context)
    transition = apply_context_turn(
        old_task,
        FOLLOW_UP,
        ContextRelation.UPDATE,
        extracted_facts=(ExtractedFact("like_count", 52, 0),),
    )
    second_initial = _next_state(resumed, FOLLOW_UP, transition)
    turn2 = run_product_graph(second_initial, _pipelines("trace-turn-2"))

    assert turn1.trace.trace_id != turn2.trace.trace_id
    assert turn1.trace.raw_query == ORIGINAL
    assert turn2.trace.raw_query == FOLLOW_UP == turn2.state.raw_query
    assert turn2.state.query_context.original_query == ORIGINAL
    assert turn2.state.case_context.raw_descriptions == (FOLLOW_UP,)
    assert turn2.state.case_context.extracted_facts[0].value == 52
    assert initial.query_context is None
    assert turn1.state.case_context == CaseContext()
    assert resumed.case_context == CaseContext()
    assert old_task.case == CaseContext()

    second_snapshot = tmp_path / "turn-2.json"
    save_session_snapshot(turn2.state, second_snapshot, revision=2, saved_at=NOW)
    round_trip = resume_session_snapshot(second_snapshot, "session-1").state
    assert round_trip == turn2.state
    assert round_trip is not turn2.state
    snapshot_text = second_snapshot.read_text(encoding="utf-8")
    assert "trace-turn-2" not in snapshot_text
    assert "visited_nodes" not in snapshot_text
    assert "route_audit" not in snapshot_text


def test_clarification_resume_update_supplement_runs_second_graph(tmp_path):
    turn1 = run_product_graph(
        AgentState("session-clarify", ORIGINAL),
        _pipelines("trace-clarify-1", block=True),
    )
    assert turn1.route_audit == ("clarification:block",)
    assert turn1.state.candidate_evidence == ()
    snapshot = tmp_path / "clarification.json"
    save_session_snapshot(turn1.state, snapshot, saved_at=NOW)
    resumed = resume_session_snapshot(snapshot, "session-clarify").state
    old_task = TaskContext(resumed.query_context, resumed.case_context)
    supplement = "范围是支付"
    transition = apply_context_turn(
        old_task,
        supplement,
        ContextRelation.UPDATE,
    )
    turn2 = run_product_graph(
        _next_state(resumed, supplement, transition),
        _pipelines("trace-clarify-2"),
    )

    assert turn2.trace.raw_query == supplement
    assert turn2.trace.trace_id != turn1.trace.trace_id
    assert turn2.state.query_context.original_query == ORIGINAL
    assert turn2.state.case_context.raw_descriptions == (supplement,)
    assert turn2.state.evidence_map is not None
    assert resumed.clarification is not None
    assert resumed.case_context == CaseContext()
    assert old_task.case == CaseContext()


def test_continue_ambiguous_and_new_context_relations_remain_safe():
    current = TaskContext(
        QueryContext(ORIGINAL, "old-scope", KnowledgeRequestType.APPLY),
        CaseContext(("旧事实",), ()),
    )

    continued = apply_context_turn(current, "继续", ContextRelation.CONTINUE)
    continued_state = _next_state(
        AgentState("session-relations", ORIGINAL),
        "继续",
        continued,
    )
    assert continued_state.query_context.original_query == ORIGINAL
    assert continued_state.case_context == current.case
    assert continued.context is current
    assert not continued.changed

    ambiguous = apply_context_turn(current, "那这个呢？", ContextRelation.AMBIGUOUS)
    ambiguous_state = _next_state(
        AgentState("session-relations", ORIGINAL),
        "那这个呢？",
        ambiguous,
    )
    assert ambiguous_state.case_context == current.case
    assert "那这个呢？" not in ambiguous_state.case_context.raw_descriptions
    assert ambiguous.context is current
    assert not ambiguous.changed

    new = apply_context_turn(current, "列出全部差旅制度", ContextRelation.NEW)
    new_state = _next_state(
        AgentState("session-relations", ORIGINAL),
        "列出全部差旅制度",
        new,
    )
    assert new_state.query_context.original_query == new_state.raw_query
    assert new_state.query_context.scope is None
    assert new_state.case_context == CaseContext()
    assert current.query.scope == "old-scope"
    assert current.case.raw_descriptions == ("旧事实",)
