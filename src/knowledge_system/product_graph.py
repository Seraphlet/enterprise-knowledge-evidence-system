"""Lightweight deterministic orchestration over immutable product State."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum

from .agent_state import AgentState, update_agent_state
from .clarification import ClarificationAction
from .context import ContextRelation
from .contracts import Trace
from .limited_recovery import RecoveryStatus
from .request_classification import KnowledgeRequestType
from .verification_mode import VerificationMode


StatePipeline = Callable[[AgentState], AgentState]
VerificationPipeline = Callable[[AgentState], VerificationMode]
TraceFinalizer = Callable[[AgentState], Trace]


class GraphExecutionError(RuntimeError):
    """An explicit graph boundary or routing invariant failed."""

    def __init__(self, code: str, reason: str) -> None:
        self.code = code
        self.reason = reason
        super().__init__(f"{code}: {reason}")


class GraphNode(str, Enum):
    RESOLVE_CONTEXT = "resolve_context"
    UNDERSTAND_REQUEST = "understand_request"
    CLARIFY_CONTROL = "clarify_control"
    RETRIEVE_CHECK = "retrieve_check_evidence"
    RECOVER_DIAGNOSE = "recover_diagnose_control"
    ANALYZE_COMPARE = "rule_analysis_compare"
    ORGANIZE_HYDRATE = "organize_validate_hydrate"
    VERIFICATION = "verification_control"
    TRACE_FINALIZE = "trace_finalize"
    END = "end"


@dataclass(frozen=True)
class GraphPipelines:
    """Runtime dependencies kept outside AgentState and called at most once."""

    resolve_context: StatePipeline
    understand_request: StatePipeline
    clarify: StatePipeline
    retrieve_and_check: StatePipeline
    recover_and_diagnose: StatePipeline
    analyze_and_compare: StatePipeline
    organize_validate_hydrate: StatePipeline
    build_verification: VerificationPipeline
    finalize_trace: TraceFinalizer

    def __post_init__(self) -> None:
        for name in self.__dataclass_fields__:
            if not callable(getattr(self, name)):
                raise TypeError(f"{name} must be callable")


@dataclass(frozen=True)
class GraphRunResult:
    """Final State plus run artifacts that must never be stored in State."""

    state: AgentState
    trace: Trace
    verification_mode: VerificationMode | None
    visited_nodes: tuple[GraphNode, ...]
    route_audit: tuple[str, ...]


class _RunGuard:
    def __init__(self, max_steps: int) -> None:
        if type(max_steps) is not int or max_steps < 1:
            raise ValueError("max_steps must be a positive integer")
        self.max_steps = max_steps
        self.visited: list[GraphNode] = []
        self.audit: list[str] = []

    def visit(self, node: GraphNode) -> None:
        if node in self.visited:
            raise GraphExecutionError(
                "GRAPH_CYCLE", f"node visited more than once: {node.value}"
            )
        if len(self.visited) >= self.max_steps:
            raise GraphExecutionError(
                "GRAPH_STEP_LIMIT",
                f"graph exceeded max_steps={self.max_steps} before {node.value}",
            )
        self.visited.append(node)

    def route(self, decision: str) -> None:
        self.audit.append(decision)


def _call_state_pipeline(
    node: GraphNode,
    pipeline: StatePipeline,
    state: AgentState,
) -> AgentState:
    try:
        result = pipeline(state)
    except Exception as error:
        raise GraphExecutionError(
            "PIPELINE_FAILURE", f"{node.value}: {type(error).__name__}: {error}"
        ) from error
    if not isinstance(result, AgentState):
        raise GraphExecutionError(
            "INVALID_STATE_RETURN", f"{node.value} did not return AgentState"
        )
    if result is state:
        raise GraphExecutionError(
            "STATE_NOT_REPLACED", f"{node.value} returned the input State"
        )
    if result.session_id != state.session_id or result.raw_query != state.raw_query:
        raise GraphExecutionError(
            "STATE_IDENTITY_DRIFT",
            f"{node.value} changed session_id or raw_query",
        )
    return result


def _required(value: object, node: GraphNode, artifact: str) -> None:
    if value is None:
        raise GraphExecutionError(
            "MISSING_ARTIFACT", f"{node.value} did not produce {artifact}"
        )


def _recovery_resolved(state: AgentState) -> bool:
    recovery = state.recovery_state
    assert recovery is not None
    return (
        recovery.status in {RecoveryStatus.NOT_NEEDED, RecoveryStatus.RESOLVED}
        and not recovery.remaining_signals
        and not recovery.verification_required
    )


def run_product_graph(
    initial_state: AgentState,
    pipelines: GraphPipelines,
    *,
    max_steps: int = 10,
) -> GraphRunResult:
    """Run one deterministic product turn without persistence or hidden state."""

    if not isinstance(initial_state, AgentState):
        raise TypeError("initial_state must be an AgentState")
    if not isinstance(pipelines, GraphPipelines):
        raise TypeError("pipelines must be GraphPipelines")

    guard = _RunGuard(max_steps)
    state = initial_state
    verification: VerificationMode | None = None

    guard.visit(GraphNode.RESOLVE_CONTEXT)
    state = _call_state_pipeline(
        GraphNode.RESOLVE_CONTEXT, pipelines.resolve_context, state
    )
    _required(state.context_resolution, GraphNode.RESOLVE_CONTEXT, "context_resolution")
    _required(state.query_context, GraphNode.RESOLVE_CONTEXT, "query_context")
    _required(state.case_context, GraphNode.RESOLVE_CONTEXT, "case_context")

    guard.visit(GraphNode.UNDERSTAND_REQUEST)
    state = _call_state_pipeline(
        GraphNode.UNDERSTAND_REQUEST, pipelines.understand_request, state
    )
    _required(
        state.knowledge_request, GraphNode.UNDERSTAND_REQUEST, "knowledge_request"
    )

    guard.visit(GraphNode.CLARIFY_CONTROL)
    state = _call_state_pipeline(GraphNode.CLARIFY_CONTROL, pipelines.clarify, state)
    clarification = state.clarification
    blocked = (
        clarification is not None
        and clarification.action is ClarificationAction.BLOCK_FOR_CLARIFICATION
    )
    ambiguous = (
        state.context_resolution.relation is ContextRelation.AMBIGUOUS
    )
    if ambiguous and not blocked:
        raise GraphExecutionError(
            "MISSING_ARTIFACT",
            "ambiguous context requires blocking clarification",
        )
    if blocked:
        guard.route("clarification:block")
    elif state.knowledge_request.request_type is KnowledgeRequestType.OUT_OF_SCOPE:
        guard.route("request:out_of_scope")
    else:
        guard.route("clarification:continue")
        guard.visit(GraphNode.RETRIEVE_CHECK)
        state = _call_state_pipeline(
            GraphNode.RETRIEVE_CHECK, pipelines.retrieve_and_check, state
        )
        if not state.candidate_evidence:
            raise GraphExecutionError(
                "MISSING_ARTIFACT",
                "retrieve_check_evidence did not produce candidate_evidence",
            )

        unreliable = bool(state.evidence_quality)
        if unreliable:
            guard.route("evidence:unreliable")
            guard.visit(GraphNode.RECOVER_DIAGNOSE)
            state = _call_state_pipeline(
                GraphNode.RECOVER_DIAGNOSE,
                pipelines.recover_and_diagnose,
                state,
            )
            _required(
                state.recovery_state,
                GraphNode.RECOVER_DIAGNOSE,
                "recovery_state",
            )
            unreliable = not _recovery_resolved(state)
            guard.route(
                "recovery:unresolved" if unreliable else "recovery:resolved"
            )
        else:
            guard.route("evidence:reliable")

        if unreliable:
            guard.visit(GraphNode.VERIFICATION)
            try:
                verification = pipelines.build_verification(state)
            except Exception as error:
                raise GraphExecutionError(
                    "PIPELINE_FAILURE",
                    "verification_control: "
                    f"{type(error).__name__}: {error}",
                ) from error
            if not isinstance(verification, VerificationMode):
                raise GraphExecutionError(
                    "INVALID_VERIFICATION_RETURN",
                    "verification_control did not return VerificationMode",
                )
            if not verification.active:
                raise GraphExecutionError(
                    "MISSING_ARTIFACT",
                    "verification_control did not produce active VerificationMode",
                )
            state = update_agent_state(state)
            guard.route("verification:entered")
        else:
            request_type = state.knowledge_request.request_type
            if request_type is KnowledgeRequestType.APPLY:
                guard.visit(GraphNode.ANALYZE_COMPARE)
                state = _call_state_pipeline(
                    GraphNode.ANALYZE_COMPARE,
                    pipelines.analyze_and_compare,
                    state,
                )
                _required(
                    state.comparison_result,
                    GraphNode.ANALYZE_COMPARE,
                    "comparison_result",
                )
                guard.route("comparison:apply")
            else:
                guard.route("comparison:skipped")

            guard.visit(GraphNode.ORGANIZE_HYDRATE)
            state = _call_state_pipeline(
                GraphNode.ORGANIZE_HYDRATE,
                pipelines.organize_validate_hydrate,
                state,
            )
            _required(
                state.evidence_map_plan,
                GraphNode.ORGANIZE_HYDRATE,
                "evidence_map_plan",
            )
            _required(
                state.evidence_map,
                GraphNode.ORGANIZE_HYDRATE,
                "evidence_map",
            )
            guard.route("verification:skipped")

    guard.visit(GraphNode.TRACE_FINALIZE)
    try:
        trace = pipelines.finalize_trace(state)
    except Exception as error:
        raise GraphExecutionError(
            "PIPELINE_FAILURE",
            f"trace_finalize: {type(error).__name__}: {error}",
        ) from error
    if not isinstance(trace, Trace):
        raise GraphExecutionError(
            "INVALID_TRACE_RETURN", "trace_finalize did not return Trace"
        )
    if trace.raw_query != state.raw_query:
        raise GraphExecutionError(
            "TRACE_IDENTITY_DRIFT", "Trace raw_query does not match State"
        )
    state = update_agent_state(state)

    guard.visit(GraphNode.END)
    return GraphRunResult(
        state=state,
        trace=trace,
        verification_mode=verification,
        visited_nodes=tuple(guard.visited),
        route_audit=tuple(guard.audit),
    )
