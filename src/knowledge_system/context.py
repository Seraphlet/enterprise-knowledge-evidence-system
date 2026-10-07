"""Lightweight query/case context with explicit, safe update boundaries."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

from .request_classification import (
    ClassificationSignal,
    KnowledgeRequestType,
    classify_request,
)


FactValue = None | bool | int | float | str


class ContextRelation(str, Enum):
    """Relationship of a new turn to the current task context."""

    CONTINUE = "CONTINUE"
    UPDATE = "UPDATE"
    NEW = "NEW"
    AMBIGUOUS = "AMBIGUOUS"


@dataclass(frozen=True)
class QueryContext:
    """What knowledge is requested, kept separate from case observations."""

    original_query: str
    scope: str | None
    intent: KnowledgeRequestType
    classification_signals: tuple[ClassificationSignal, ...] = field(
        default_factory=tuple
    )


@dataclass(frozen=True)
class ExtractedFact:
    """A caller-supplied fact without a global or domain-specific schema."""

    name: str
    value: FactValue
    source_description_index: int | None = None


@dataclass(frozen=True)
class CaseContext:
    """Raw case observations and optional facts derived elsewhere."""

    raw_descriptions: tuple[str, ...] = field(default_factory=tuple)
    extracted_facts: tuple[ExtractedFact, ...] = field(default_factory=tuple)

    def append(
        self,
        raw_description: str,
        *,
        extracted_facts: tuple[ExtractedFact, ...] = (),
    ) -> "CaseContext":
        if not isinstance(raw_description, str):
            raise TypeError("raw_description must be a string")
        return CaseContext(
            raw_descriptions=(*self.raw_descriptions, raw_description),
            extracted_facts=(*self.extracted_facts, *extracted_facts),
        )


@dataclass(frozen=True)
class TaskContext:
    """Current query and case, joined without merging their responsibilities."""

    query: QueryContext
    case: CaseContext = field(default_factory=CaseContext)


@dataclass(frozen=True)
class ContextTransition:
    """Auditable outcome of applying one explicit context relation."""

    relation: ContextRelation
    turn_query: str
    context: TaskContext
    changed: bool
    reason: str


def create_task_context(
    original_query: str,
    *,
    scope: str | None = None,
    scope_covered: bool | None = None,
    initial_case_description: str | None = None,
) -> TaskContext:
    """Create context without inferring case facts from the query text."""

    classification = classify_request(
        original_query, scope_covered=scope_covered
    )
    query = QueryContext(
        original_query=original_query,
        scope=scope,
        intent=classification.request_type,
        classification_signals=classification.signals,
    )
    case = CaseContext()
    if initial_case_description is not None:
        case = case.append(initial_case_description)
    return TaskContext(query=query, case=case)


def apply_context_turn(
    current: TaskContext,
    turn_query: str,
    relation: ContextRelation,
    *,
    scope: str | None = None,
    scope_covered: bool | None = None,
    extracted_facts: tuple[ExtractedFact, ...] = (),
) -> ContextTransition:
    """Apply an explicit relation without unsafe implicit inheritance.

    Fact extraction and relation detection are deliberately outside this T-401
    boundary. Callers must supply both the relation and any extracted facts.
    """

    if not isinstance(turn_query, str):
        raise TypeError("turn_query must be a string")
    if not isinstance(relation, ContextRelation):
        raise TypeError("relation must be a ContextRelation")

    if relation is ContextRelation.UPDATE:
        updated = TaskContext(
            query=current.query,
            case=current.case.append(
                turn_query, extracted_facts=extracted_facts
            ),
        )
        return ContextTransition(
            relation, turn_query, updated, True, "case_updated"
        )

    if relation is ContextRelation.NEW:
        updated = create_task_context(
            turn_query,
            scope=scope,
            scope_covered=scope_covered,
        )
        return ContextTransition(
            relation, turn_query, updated, True, "new_query_context"
        )

    if relation is ContextRelation.CONTINUE:
        return ContextTransition(
            relation,
            turn_query,
            current,
            False,
            "continued_without_context_mutation",
        )

    return ContextTransition(
        relation,
        turn_query,
        current,
        False,
        "ambiguous_relation_not_applied",
    )
