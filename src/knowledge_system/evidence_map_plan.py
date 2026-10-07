"""Strict organization-only schema for an Evidence Map plan."""

from __future__ import annotations

import json
import re
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any


class EvidenceMapPlanValidationError(ValueError):
    """The proposed plan is not valid organization-only data."""


_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:/-]{0,127}$")


def _identity(value: object, path: str) -> str:
    if not isinstance(value, str) or not _ID_RE.fullmatch(value):
        raise EvidenceMapPlanValidationError(
            f"{path} must be a non-empty stable ID"
        )
    return value


def _order(value: object, path: str) -> int:
    if type(value) is not int or value < 0:
        raise EvidenceMapPlanValidationError(
            f"{path} must be a non-negative integer"
        )
    return value


def _object(
    value: object,
    *,
    allowed: frozenset[str],
    required: frozenset[str],
    path: str,
) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise EvidenceMapPlanValidationError(f"{path} must be an object")
    keys = set(value)
    if any(not isinstance(key, str) for key in keys):
        raise EvidenceMapPlanValidationError(f"{path} keys must be strings")
    unknown = keys - allowed
    if unknown:
        raise EvidenceMapPlanValidationError(
            f"{path} contains unknown fields: {sorted(unknown)}"
        )
    missing = required - keys
    if missing:
        raise EvidenceMapPlanValidationError(
            f"{path} is missing required fields: {sorted(missing)}"
        )
    return value


def _array(value: object, path: str) -> list[Any]:
    if type(value) is not list:
        raise EvidenceMapPlanValidationError(f"{path} must be an array")
    return value


@dataclass(frozen=True)
class EvidenceMapGroup:
    group_id: str
    order: int
    evidence_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        _identity(self.group_id, "group_id")
        _order(self.order, "group.order")
        if type(self.evidence_ids) is not tuple or not self.evidence_ids:
            raise EvidenceMapPlanValidationError(
                "group.evidence_ids must be a non-empty tuple"
            )
        checked = tuple(
            _identity(item, f"group.evidence_ids[{index}]")
            for index, item in enumerate(self.evidence_ids)
        )
        if len(set(checked)) != len(checked):
            raise EvidenceMapPlanValidationError(
                "duplicate evidence_id within group"
            )
        object.__setattr__(self, "evidence_ids", tuple(sorted(checked)))

    @classmethod
    def from_dict(cls, value: object, path: str = "group") -> "EvidenceMapGroup":
        item = _object(
            value,
            allowed=frozenset({"group_id", "order", "evidence_ids"}),
            required=frozenset({"group_id", "order", "evidence_ids"}),
            path=path,
        )
        evidence_ids = _array(item["evidence_ids"], f"{path}.evidence_ids")
        return cls(
            group_id=_identity(item["group_id"], f"{path}.group_id"),
            order=_order(item["order"], f"{path}.order"),
            evidence_ids=tuple(evidence_ids),
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "group_id": self.group_id,
            "order": self.order,
            "evidence_ids": list(self.evidence_ids),
        }


@dataclass(frozen=True)
class EvidenceMapSection:
    section_id: str
    order: int
    groups: tuple[EvidenceMapGroup, ...]

    def __post_init__(self) -> None:
        _identity(self.section_id, "section_id")
        _order(self.order, "section.order")
        if type(self.groups) is not tuple or not self.groups:
            raise EvidenceMapPlanValidationError(
                "section.groups must be a non-empty tuple"
            )
        if any(not isinstance(item, EvidenceMapGroup) for item in self.groups):
            raise EvidenceMapPlanValidationError(
                "section.groups must contain EvidenceMapGroup items"
            )
        identities = tuple(item.group_id for item in self.groups)
        orders = tuple(item.order for item in self.groups)
        if len(set(identities)) != len(identities):
            raise EvidenceMapPlanValidationError(
                "duplicate group_id within section"
            )
        if len(set(orders)) != len(orders):
            raise EvidenceMapPlanValidationError(
                "duplicate group order within section"
            )
        object.__setattr__(
            self, "groups", tuple(sorted(self.groups, key=lambda item: item.order))
        )

    @classmethod
    def from_dict(
        cls, value: object, path: str = "section"
    ) -> "EvidenceMapSection":
        item = _object(
            value,
            allowed=frozenset({"section_id", "order", "groups"}),
            required=frozenset({"section_id", "order", "groups"}),
            path=path,
        )
        groups = _array(item["groups"], f"{path}.groups")
        return cls(
            section_id=_identity(item["section_id"], f"{path}.section_id"),
            order=_order(item["order"], f"{path}.order"),
            groups=tuple(
                EvidenceMapGroup.from_dict(group, f"{path}.groups[{index}]")
                for index, group in enumerate(groups)
            ),
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "section_id": self.section_id,
            "order": self.order,
            "groups": [item.to_dict() for item in self.groups],
        }


@dataclass(frozen=True)
class EvidenceMapPlan:
    plan_id: str
    sections: tuple[EvidenceMapSection, ...]

    def __post_init__(self) -> None:
        _identity(self.plan_id, "plan_id")
        if type(self.sections) is not tuple or not self.sections:
            raise EvidenceMapPlanValidationError(
                "plan.sections must be a non-empty tuple"
            )
        if any(not isinstance(item, EvidenceMapSection) for item in self.sections):
            raise EvidenceMapPlanValidationError(
                "plan.sections must contain EvidenceMapSection items"
            )
        section_ids = tuple(item.section_id for item in self.sections)
        section_orders = tuple(item.order for item in self.sections)
        if len(set(section_ids)) != len(section_ids):
            raise EvidenceMapPlanValidationError("duplicate section_id")
        if len(set(section_orders)) != len(section_orders):
            raise EvidenceMapPlanValidationError("duplicate section order")

        group_ids: set[str] = set()
        evidence_ids: set[str] = set()
        for section in self.sections:
            for group in section.groups:
                if group.group_id in group_ids:
                    raise EvidenceMapPlanValidationError(
                        "duplicate group_id across plan"
                    )
                group_ids.add(group.group_id)
                for evidence_id in group.evidence_ids:
                    if evidence_id in evidence_ids:
                        raise EvidenceMapPlanValidationError(
                            "duplicate evidence_id across plan"
                        )
                    evidence_ids.add(evidence_id)
        object.__setattr__(
            self,
            "sections",
            tuple(sorted(self.sections, key=lambda item: item.order)),
        )

    @classmethod
    def from_dict(cls, value: object) -> "EvidenceMapPlan":
        item = _object(
            value,
            allowed=frozenset({"plan_id", "sections"}),
            required=frozenset({"plan_id", "sections"}),
            path="plan",
        )
        sections = _array(item["sections"], "plan.sections")
        return cls(
            plan_id=_identity(item["plan_id"], "plan.plan_id"),
            sections=tuple(
                EvidenceMapSection.from_dict(
                    section, f"plan.sections[{index}]"
                )
                for index, section in enumerate(sections)
            ),
        )

    @classmethod
    def from_json(cls, payload: str) -> "EvidenceMapPlan":
        if not isinstance(payload, str):
            raise TypeError("payload must be a JSON string")

        def strict_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
            value: dict[str, object] = {}
            for key, item in pairs:
                if key in value:
                    raise EvidenceMapPlanValidationError(
                        f"duplicate JSON field: {key}"
                    )
                value[key] = item
            return value

        try:
            value = json.loads(
                payload,
                object_pairs_hook=strict_object,
                parse_constant=lambda item: (_ for _ in ()).throw(
                    EvidenceMapPlanValidationError(
                        f"invalid JSON constant: {item}"
                    )
                ),
            )
        except EvidenceMapPlanValidationError:
            raise
        except (TypeError, ValueError, json.JSONDecodeError) as error:
            raise EvidenceMapPlanValidationError("invalid plan JSON") from error
        return cls.from_dict(value)

    def to_dict(self) -> dict[str, object]:
        return {
            "plan_id": self.plan_id,
            "sections": [item.to_dict() for item in self.sections],
        }

    def to_json(self) -> str:
        return json.dumps(
            self.to_dict(),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
