"""Immutable product work context, deliberately separate from runtime and Trace."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, fields, is_dataclass, replace
from typing import Any

from .apply_compare import ApplyCompareResult
from .clarification import ClarificationDecision
from .context import CaseContext, ContextRelation, ContextTransition, QueryContext
from .contracts import Evidence
from .evidence_map import EvidenceMap
from .evidence_map_plan import EvidenceMapPlan
from .evidence_quality import EvidenceQualitySignal
from .limited_recovery import RecoveryResult
from .request_classification import RequestClassification


class AgentStateValidationError(ValueError):
    """The proposed product state violates its explicit data boundary."""


class _FrozenDict(dict[Any, Any]):
    """A dict-compatible snapshot that cannot be changed through the State."""

    def _immutable(self, *args: object, **kwargs: object) -> None:
        raise TypeError("AgentState mappings are immutable")

    __setitem__ = _immutable
    __delitem__ = _immutable
    clear = _immutable
    pop = _immutable
    popitem = _immutable
    setdefault = _immutable
    update = _immutable

    def __deepcopy__(self, memo: dict[int, object]) -> "_FrozenDict":
        return self


def _freeze(value: Any) -> Any:
    """Take an immutable recursive snapshot without changing domain types."""

    if isinstance(value, _FrozenDict):
        return value
    if isinstance(value, dict):
        return _FrozenDict(
            {deepcopy(key): _freeze(item) for key, item in value.items()}
        )
    if isinstance(value, list):
        return tuple(_freeze(item) for item in value)
    if isinstance(value, tuple):
        return tuple(_freeze(item) for item in value)
    if is_dataclass(value) and not isinstance(value, type):
        replacements = {
            item.name: _freeze(getattr(value, item.name)) for item in fields(value)
        }
        return replace(value, **replacements)
    return deepcopy(value)


@dataclass(frozen=True)
class AgentStateError:
    """Minimal structured product error; it carries no runtime dependency."""

    code: str
    reason: str

    def __post_init__(self) -> None:
        if not isinstance(self.code, str) or not self.code.strip():
            raise AgentStateValidationError("error.code must be non-empty text")
        if not isinstance(self.reason, str) or not self.reason.strip():
            raise AgentStateValidationError("error.reason must be non-empty text")


@dataclass(frozen=True)
class AgentState:
    """One session's structured work context, not control flow or telemetry."""

    session_id: str
    raw_query: str
    query_context: QueryContext | None = None
    case_context: CaseContext | None = None
    knowledge_request: RequestClassification | None = None
    context_resolution: ContextTransition | None = None
    clarification: ClarificationDecision | None = None
    candidate_evidence: tuple[Evidence, ...] = ()
    evidence_quality: tuple[EvidenceQualitySignal, ...] = ()
    recovery_state: RecoveryResult | None = None
    comparison_result: ApplyCompareResult | None = None
    evidence_map_plan: EvidenceMapPlan | None = None
    evidence_map: EvidenceMap | None = None
    error: AgentStateError | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.session_id, str) or not self.session_id.strip():
            raise AgentStateValidationError("session_id must be non-empty text")
        if not isinstance(self.raw_query, str) or not self.raw_query.strip():
            raise AgentStateValidationError("raw_query must be non-empty text")

        expected_types = {
            "query_context": QueryContext,
            "case_context": CaseContext,
            "knowledge_request": RequestClassification,
            "context_resolution": ContextTransition,
            "clarification": ClarificationDecision,
            "recovery_state": RecoveryResult,
            "comparison_result": ApplyCompareResult,
            "evidence_map_plan": EvidenceMapPlan,
            "evidence_map": EvidenceMap,
            "error": AgentStateError,
        }
        for name, expected in expected_types.items():
            value = getattr(self, name)
            if value is not None and not isinstance(value, expected):
                raise TypeError(f"{name} must be {expected.__name__} or None")

        candidate_evidence = self._typed_tuple(
            self.candidate_evidence, Evidence, "candidate_evidence"
        )
        evidence_quality = self._typed_tuple(
            self.evidence_quality, EvidenceQualitySignal, "evidence_quality"
        )

        transition = self.context_resolution
        if transition is None:
            if (
                self.query_context is not None
                and self.query_context.original_query != self.raw_query
            ):
                raise AgentStateValidationError(
                    "query_context.original_query must exactly match raw_query "
                    "when context_resolution is absent"
                )
        else:
            self._validate_context_transition(transition)
        if (
            self.knowledge_request is not None
            and self.knowledge_request.original_query != self.raw_query
        ):
            raise AgentStateValidationError(
                "knowledge_request.original_query must exactly match raw_query"
            )

        object.__setattr__(self, "candidate_evidence", _freeze(candidate_evidence))
        object.__setattr__(self, "evidence_quality", _freeze(evidence_quality))
        for name in expected_types:
            value = getattr(self, name)
            if value is not None:
                object.__setattr__(self, name, _freeze(value))

    @staticmethod
    def _typed_tuple(value: object, item_type: type, name: str) -> tuple[Any, ...]:
        if not isinstance(value, (tuple, list)):
            raise TypeError(f"{name} must be a tuple or list")
        result = tuple(value)
        if any(not isinstance(item, item_type) for item in result):
            raise TypeError(f"{name} must contain only {item_type.__name__}")
        return result

    def _validate_context_transition(self, transition: ContextTransition) -> None:
        if transition.turn_query != self.raw_query:
            raise AgentStateValidationError(
                "context_resolution.turn_query must exactly match raw_query"
            )
        if self.query_context != transition.context.query:
            raise AgentStateValidationError(
                "query_context must exactly match context_resolution.context.query"
            )
        if self.case_context != transition.context.case:
            raise AgentStateValidationError(
                "case_context must exactly match context_resolution.context.case"
            )

        relation = transition.relation
        if relation in {ContextRelation.NEW, ContextRelation.UPDATE}:
            if not transition.changed:
                raise AgentStateValidationError(
                    f"{relation.value} context transition must be marked changed"
                )
        elif transition.changed:
            raise AgentStateValidationError(
                f"{relation.value} context transition cannot be marked changed"
            )

        if relation is ContextRelation.NEW:
            if transition.context.query.original_query != self.raw_query:
                raise AgentStateValidationError(
                    "NEW query_context.original_query must exactly match raw_query"
                )
            if transition.context.case != CaseContext():
                raise AgentStateValidationError(
                    "NEW context transition cannot inherit prior case context"
                )


_UPDATABLE_FIELDS = frozenset(
    {
        "raw_query",
        "query_context",
        "case_context",
        "knowledge_request",
        "context_resolution",
        "clarification",
        "candidate_evidence",
        "evidence_quality",
        "recovery_state",
        "comparison_result",
        "evidence_map_plan",
        "evidence_map",
        "error",
    }
)


def update_agent_state(state: AgentState, **changes: object) -> AgentState:
    """Return a validated new State using only declared work-context slots."""

    if not isinstance(state, AgentState):
        raise TypeError("state must be an AgentState")
    unknown = set(changes) - _UPDATABLE_FIELDS
    if unknown:
        names = ", ".join(sorted(unknown))
        raise AgentStateValidationError(f"unknown or non-updatable fields: {names}")
    if not changes:
        return replace(state)
    return replace(state, **changes)
