"""Hard candidate eligibility filters, deliberately separate from ranking."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from enum import Enum

from knowledge_system.contracts import KnowledgeUnit


class FilterReason(str, Enum):
    PERMISSION_DENIED = "PERMISSION_DENIED"
    INVALID_PERMISSION_METADATA = "INVALID_PERMISSION_METADATA"
    INACTIVE_STATUS = "INACTIVE_STATUS"


@dataclass(frozen=True)
class RejectedCandidate:
    unit_id: str
    reason: FilterReason


@dataclass(frozen=True)
class CandidateFilterResult:
    """Eligible units plus auditable exclusions; contains no ranking data."""

    candidates: tuple[KnowledgeUnit, ...]
    rejected: tuple[RejectedCandidate, ...]


_INACTIVE_STATUSES = frozenset(
    {
        "inactive",
        "expired",
        "revoked",
        "disabled",
        "obsolete",
        "失效",
        "已失效",
        "过期",
        "已过期",
        "停用",
        "禁用",
        "已撤销",
        "作废",
    }
)


def _required_permissions(unit: KnowledgeUnit) -> frozenset[str] | None:
    """Read the explicit required_permissions metadata; malformed means deny."""

    raw = unit.metadata.get("required_permissions", [])
    if isinstance(raw, str):
        return frozenset({raw}) if raw else frozenset()
    if not isinstance(raw, list) or not all(
        isinstance(value, str) and value for value in raw
    ):
        return None
    return frozenset(raw)


def filter_candidates(
    units: Iterable[KnowledgeUnit],
    *,
    granted_permissions: Iterable[str] = (),
) -> CandidateFilterResult:
    """Apply status and permission hard boundaries without changing order."""

    granted = frozenset(granted_permissions)
    candidates: list[KnowledgeUnit] = []
    rejected: list[RejectedCandidate] = []
    for unit in units:
        status = unit.metadata.get("status")
        if isinstance(status, str) and status.strip().casefold() in _INACTIVE_STATUSES:
            rejected.append(
                RejectedCandidate(unit.unit_id, FilterReason.INACTIVE_STATUS)
            )
            continue
        required = _required_permissions(unit)
        if required is None:
            rejected.append(
                RejectedCandidate(
                    unit.unit_id, FilterReason.INVALID_PERMISSION_METADATA
                )
            )
            continue
        if not required.issubset(granted):
            rejected.append(
                RejectedCandidate(unit.unit_id, FilterReason.PERMISSION_DENIED)
            )
            continue
        candidates.append(unit)
    return CandidateFilterResult(tuple(candidates), tuple(rejected))
