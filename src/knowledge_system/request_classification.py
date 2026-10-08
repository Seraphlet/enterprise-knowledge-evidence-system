"""Deterministic, auditable Knowledge Request classification."""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum


class KnowledgeRequestType(str, Enum):
    """The five request types supported by the V0 product contract."""

    LOCATE = "LOCATE"
    BROWSE = "BROWSE"
    DISCOVER = "DISCOVER"
    APPLY = "APPLY"
    OUT_OF_SCOPE = "OUT_OF_SCOPE"


@dataclass(frozen=True)
class ClassificationSignal:
    """A deterministic rule match retained for audit."""

    rule: str
    matched_text: str


@dataclass(frozen=True)
class RequestClassification:
    """Classification result that preserves the user's input verbatim."""

    original_query: str
    request_type: KnowledgeRequestType
    signals: tuple[ClassificationSignal, ...]


_SPACE_RE = re.compile(r"\s+")

# Ordering within each group is stable so classification evidence is repeatable.
_APPLY_PATTERNS = (
    re.compile(r"(?:是否|能否|可不可以|符不符合|是否符合|是否适用|怎么处理|如何处理)"),
    re.compile(r"\b(?:does|do|is|are|can)\b.+\b(?:qualify|eligible|apply|comply)\b", re.I),
    re.compile(r"\b(?:my|our|this)\s+(?:case|situation|request)\b", re.I),
)
_BROWSE_PATTERNS = (
    # Browse requires an explicit collection/navigation intent.  Topic questions
    # such as "包含哪些目录" still ask for knowledge and must not be
    # routed to scope-dependent collection browsing merely because they contain
    # a quantifier or a domain noun.
    re.compile(r"(?:列出|浏览|汇总)"),
    re.compile(r"\b(?:list|browse|catalog|overview)\b", re.I),
)
_LOCATE_PATTERNS = (
    re.compile(r"(?:在哪里|在哪儿|位置|打开|定位|找到|查找|原文|链接)"),
    re.compile(r"\b(?:where is|locate|find|open|link to|source of)\b", re.I),
)
_DISCOVER_PATTERNS = (
    re.compile(r"(?:有没有关于|是否有关于|相关规定|相关规则|相关制度|怎么办|怎么做|如何)"),
    re.compile(r"\b(?:policy|rule|guidance|how to|what should)\b", re.I),
)
_EXPLICIT_OUT_OF_SCOPE_PATTERNS = (
    re.compile(r"(?:与|跟)(?:公司|企业知识库|知识库|内部知识)(?:无关|没有关系)"),
    re.compile(r"\b(?:outside|unrelated to)\s+(?:the\s+)?(?:company|enterprise|knowledge base)\b", re.I),
)


def _normalize_for_matching(query: str) -> str:
    """Normalize only the private matching view, never the returned query."""

    return _SPACE_RE.sub(" ", query).strip()


def _matched_signals(
    query: str, rule: str, patterns: tuple[re.Pattern[str], ...]
) -> tuple[ClassificationSignal, ...]:
    matches: list[ClassificationSignal] = []
    for pattern in patterns:
        match = pattern.search(query)
        if match is not None:
            matches.append(ClassificationSignal(rule=rule, matched_text=match.group(0)))
    return tuple(matches)


def classify_request(
    original_query: str, *, scope_covered: bool | None = None
) -> RequestClassification:
    """Classify a request using fixed rules and an optional explicit scope decision.

    ``scope_covered=False`` is authoritative routing information from the caller.
    ``None`` does not guess that an unfamiliar request is out of scope. This keeps
    classification from inventing the enterprise knowledge boundary.
    """

    if not isinstance(original_query, str):
        raise TypeError("original_query must be a string")

    query = _normalize_for_matching(original_query)
    if scope_covered is False:
        signals = (ClassificationSignal("scope_covered_false", "false"),)
        return RequestClassification(
            original_query, KnowledgeRequestType.OUT_OF_SCOPE, signals
        )

    signals = _matched_signals(
        query, "explicit_out_of_scope", _EXPLICIT_OUT_OF_SCOPE_PATTERNS
    )
    if signals:
        return RequestClassification(
            original_query, KnowledgeRequestType.OUT_OF_SCOPE, signals
        )

    for request_type, rule, patterns in (
        (KnowledgeRequestType.APPLY, "case_comparison", _APPLY_PATTERNS),
        (KnowledgeRequestType.BROWSE, "collection_request", _BROWSE_PATTERNS),
        (KnowledgeRequestType.LOCATE, "source_location", _LOCATE_PATTERNS),
        (KnowledgeRequestType.DISCOVER, "knowledge_discovery", _DISCOVER_PATTERNS),
    ):
        signals = _matched_signals(query, rule, patterns)
        if signals:
            return RequestClassification(original_query, request_type, signals)

    # Unknown wording remains an in-scope discovery attempt. T-402, rather than
    # this classifier, decides whether useful retrieval requires clarification.
    return RequestClassification(
        original_query,
        KnowledgeRequestType.DISCOVER,
        (ClassificationSignal("discover_fallback", ""),),
    )
