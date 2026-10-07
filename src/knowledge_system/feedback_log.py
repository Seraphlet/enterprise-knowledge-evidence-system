"""Append-only employee feedback observations, isolated from Formal Knowledge."""

from __future__ import annotations

import json
import os
import re
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any
from uuid import uuid4

from .contracts import Evidence, Trace
from .verification_mode import VerificationMode, VerificationReason


class FeedbackAuthority(str, Enum):
    OBSERVATION = "OBSERVATION"


class FeedbackPurpose(str, Enum):
    KNOWLEDGE_MAINTENANCE_SIGNAL = "KNOWLEDGE_MAINTENANCE_SIGNAL"


class FeedbackValidationError(ValueError):
    """Feedback content or target binding is invalid."""


class FeedbackLogError(RuntimeError):
    """Append-only persistence could not be safely completed."""


_IDENTITY_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:/-]{0,127}$")


def _identity(value: object, path: str) -> str:
    if not isinstance(value, str) or not _IDENTITY_RE.fullmatch(value):
        raise FeedbackValidationError(f"{path} must be a stable identity")
    return value


@dataclass(frozen=True)
class FeedbackRecord:
    """Employee words preserved as a non-authoritative maintenance observation."""

    feedback_id: str
    created_at: str
    feedback_text: str
    target_evidence_ids: tuple[str, ...]
    target_trace_id: str | None
    verification_reasons: tuple[VerificationReason, ...]
    verification_context_codes: tuple[str, ...]
    authority: FeedbackAuthority = FeedbackAuthority.OBSERVATION
    purpose: FeedbackPurpose = FeedbackPurpose.KNOWLEDGE_MAINTENANCE_SIGNAL

    def __post_init__(self) -> None:
        _identity(self.feedback_id, "feedback_id")
        if not isinstance(self.created_at, str):
            raise FeedbackValidationError("created_at must be text")
        try:
            created = datetime.fromisoformat(self.created_at)
        except ValueError as error:
            raise FeedbackValidationError("created_at must be ISO-8601") from error
        if created.tzinfo is None or created.utcoffset() is None:
            raise FeedbackValidationError("created_at must include timezone")
        if not isinstance(self.feedback_text, str) or not self.feedback_text.strip():
            raise FeedbackValidationError("feedback_text must not be empty")
        if type(self.target_evidence_ids) is not tuple:
            raise FeedbackValidationError("target_evidence_ids must be a tuple")
        checked_ids = tuple(
            _identity(item, f"target_evidence_ids[{index}]")
            for index, item in enumerate(self.target_evidence_ids)
        )
        if len(set(checked_ids)) != len(checked_ids):
            raise FeedbackValidationError("target_evidence_ids contain duplicates")
        if self.target_trace_id is not None:
            _identity(self.target_trace_id, "target_trace_id")
        if not checked_ids and self.target_trace_id is None:
            raise FeedbackValidationError(
                "feedback must target Evidence or Trace"
            )
        if type(self.verification_reasons) is not tuple or any(
            not isinstance(item, VerificationReason)
            for item in self.verification_reasons
        ):
            raise FeedbackValidationError(
                "verification_reasons must contain VerificationReason items"
            )
        if type(self.verification_context_codes) is not tuple or any(
            not isinstance(item, str) or not item
            for item in self.verification_context_codes
        ):
            raise FeedbackValidationError(
                "verification_context_codes must contain non-empty codes"
            )
        if self.authority is not FeedbackAuthority.OBSERVATION:
            raise FeedbackValidationError("feedback authority must be OBSERVATION")
        if self.purpose is not FeedbackPurpose.KNOWLEDGE_MAINTENANCE_SIGNAL:
            raise FeedbackValidationError(
                "feedback purpose must be KNOWLEDGE_MAINTENANCE_SIGNAL"
            )

    def to_dict(self) -> dict[str, object]:
        return {
            "feedback_id": self.feedback_id,
            "created_at": self.created_at,
            "feedback_text": self.feedback_text,
            "target_evidence_ids": list(self.target_evidence_ids),
            "target_trace_id": self.target_trace_id,
            "verification_reasons": [
                item.value for item in self.verification_reasons
            ],
            "verification_context_codes": list(
                self.verification_context_codes
            ),
            "authority": self.authority.value,
            "purpose": self.purpose.value,
        }

    def to_json(self) -> str:
        return json.dumps(
            self.to_dict(),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )

    @classmethod
    def from_dict(cls, value: object) -> "FeedbackRecord":
        expected = frozenset(
            {
                "feedback_id",
                "created_at",
                "feedback_text",
                "target_evidence_ids",
                "target_trace_id",
                "verification_reasons",
                "verification_context_codes",
                "authority",
                "purpose",
            }
        )
        if not isinstance(value, Mapping) or set(value) != expected:
            raise FeedbackValidationError(
                "FeedbackRecord fields do not match schema"
            )
        evidence_ids = value["target_evidence_ids"]
        reasons = value["verification_reasons"]
        context = value["verification_context_codes"]
        if type(evidence_ids) is not list or type(reasons) is not list or type(context) is not list:
            raise FeedbackValidationError("FeedbackRecord arrays are invalid")
        try:
            return cls(
                feedback_id=value["feedback_id"],
                created_at=value["created_at"],
                feedback_text=value["feedback_text"],
                target_evidence_ids=tuple(evidence_ids),
                target_trace_id=value["target_trace_id"],
                verification_reasons=tuple(
                    VerificationReason(item) for item in reasons
                ),
                verification_context_codes=tuple(context),
                authority=FeedbackAuthority(value["authority"]),
                purpose=FeedbackPurpose(value["purpose"]),
            )
        except (TypeError, ValueError) as error:
            if isinstance(error, FeedbackValidationError):
                raise
            raise FeedbackValidationError("FeedbackRecord value is invalid") from error

    @classmethod
    def from_json(cls, payload: str) -> "FeedbackRecord":
        if not isinstance(payload, str):
            raise FeedbackValidationError("FeedbackRecord JSON must be text")

        def strict_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
            value: dict[str, object] = {}
            for key, item in pairs:
                if key in value:
                    raise FeedbackValidationError(
                        f"duplicate FeedbackRecord field: {key}"
                    )
                value[key] = item
            return value

        try:
            value = json.loads(payload, object_pairs_hook=strict_object)
        except FeedbackValidationError:
            raise
        except (TypeError, ValueError, json.JSONDecodeError) as error:
            raise FeedbackValidationError("invalid FeedbackRecord JSON") from error
        return cls.from_dict(value)


FeedbackIdFactory = Callable[[], str]
FeedbackClock = Callable[[], datetime]


def _default_feedback_id() -> str:
    return f"feedback-{uuid4().hex}"


def _default_clock() -> datetime:
    return datetime.now(timezone.utc)


def _evidence_ids(items: Iterable[Evidence]) -> dict[str, Evidence]:
    values: dict[str, Evidence] = {}
    for item in items:
        if not isinstance(item, Evidence):
            raise TypeError("authoritative_evidence must contain Evidence items")
        _identity(item.evidence_id, "Evidence.evidence_id")
        existing = values.get(item.evidence_id)
        if existing is not None and existing != item:
            raise FeedbackValidationError(
                "authoritative Evidence ID has conflicting content"
            )
        values.setdefault(item.evidence_id, item)
    return values


def _verification_bindings(
    verification_mode: VerificationMode | None,
) -> tuple[dict[str, tuple[object, ...]], set[str]]:
    if verification_mode is None:
        return {}, set()
    if not isinstance(verification_mode, VerificationMode):
        raise TypeError("verification_mode must be VerificationMode or None")
    bindings: dict[str, tuple[object, ...]] = {}
    allowed: set[str] = set()
    for record in verification_mode.records:
        identity = (
            record.original_content,
            record.source_reference,
            record.lineage,
        )
        existing = bindings.get(record.evidence_id)
        if existing is not None and existing != identity:
            raise FeedbackValidationError(
                "Verification Mode has conflicting Evidence identity"
            )
        bindings.setdefault(record.evidence_id, identity)
        allowed.add(record.evidence_id)
    return bindings, allowed


def _trace_ids(items: Iterable[Trace]) -> dict[str, Trace]:
    values: dict[str, Trace] = {}
    for item in items:
        if not isinstance(item, Trace):
            raise TypeError("trace_context must contain Trace items")
        _identity(item.trace_id, "Trace.trace_id")
        existing = values.get(item.trace_id)
        if existing is not None and existing != item:
            raise FeedbackValidationError("Trace ID has conflicting context")
        values.setdefault(item.trace_id, item)
    return values


def create_feedback_record(
    feedback_text: str,
    *,
    target_evidence_ids: Iterable[str] = (),
    target_trace_id: str | None = None,
    verification_mode: VerificationMode | None = None,
    authoritative_evidence: Iterable[Evidence] = (),
    trace_context: Iterable[Trace] = (),
    id_factory: FeedbackIdFactory = _default_feedback_id,
    clock: FeedbackClock = _default_clock,
) -> FeedbackRecord:
    """Create a grounded observation without interpreting employee words."""

    if not callable(id_factory) or not callable(clock):
        raise TypeError("id_factory and clock must be callable")
    if not isinstance(feedback_text, str) or not feedback_text.strip():
        raise FeedbackValidationError("feedback_text must not be empty")
    candidates = _evidence_ids(authoritative_evidence)
    verification_bindings, verification_ids = _verification_bindings(
        verification_mode
    )
    for evidence_id, identity in verification_bindings.items():
        candidate = candidates.get(evidence_id)
        if candidate is not None and identity != (
            candidate.original_content,
            candidate.source_reference,
            candidate.lineage,
        ):
            raise FeedbackValidationError(
                "Verification Mode does not match authoritative Evidence"
            )
    allowed_evidence_ids = set(candidates) | verification_ids
    requested_ids = tuple(target_evidence_ids)
    checked_ids = tuple(
        _identity(item, f"target_evidence_ids[{index}]")
        for index, item in enumerate(requested_ids)
    )
    if len(set(checked_ids)) != len(checked_ids):
        raise FeedbackValidationError("target_evidence_ids contain duplicates")
    unknown_ids = tuple(
        item for item in checked_ids if item not in allowed_evidence_ids
    )
    if unknown_ids:
        raise FeedbackValidationError(
            f"unknown target Evidence IDs: {list(unknown_ids)}"
        )

    traces = _trace_ids(trace_context)
    checked_trace_id: str | None = None
    if target_trace_id is not None:
        checked_trace_id = _identity(target_trace_id, "target_trace_id")
        if checked_trace_id not in traces:
            raise FeedbackValidationError("unknown target Trace identity")
    if checked_ids and checked_trace_id is not None:
        target_trace = traces[checked_trace_id]
        trace_evidence_ids = (
            *target_trace.candidate_evidence_ids,
            *target_trace.final_evidence_ids,
        )
        if any(not isinstance(item, str) for item in trace_evidence_ids):
            raise FeedbackValidationError(
                "target Trace contains invalid Evidence identity"
            )
        unrelated_ids = tuple(
            evidence_id
            for evidence_id in checked_ids
            if evidence_id not in trace_evidence_ids
        )
        if unrelated_ids:
            raise FeedbackValidationError(
                "target Evidence does not belong to target Trace"
            )
    if not checked_ids and checked_trace_id is None:
        raise FeedbackValidationError("feedback must target Evidence or Trace")

    reasons: set[VerificationReason] = set()
    contexts: set[str] = set()
    if verification_mode is not None:
        targets = set(checked_ids)
        for record in verification_mode.records:
            if record.evidence_id in targets:
                reasons.add(record.reason)
                contexts.update(record.problem_codes)
                contexts.update(record.uncertainty_codes)

    feedback_id = _identity(id_factory(), "feedback_id")
    created_at = clock()
    if not isinstance(created_at, datetime):
        raise FeedbackValidationError("clock must return datetime")
    if created_at.tzinfo is None or created_at.utcoffset() is None:
        raise FeedbackValidationError("clock must return timezone-aware datetime")
    return FeedbackRecord(
        feedback_id=feedback_id,
        created_at=created_at.isoformat(),
        feedback_text=feedback_text,
        target_evidence_ids=tuple(sorted(checked_ids)),
        target_trace_id=checked_trace_id,
        verification_reasons=tuple(
            sorted(reasons, key=lambda item: item.value)
        ),
        verification_context_codes=tuple(sorted(contexts)),
    )


class FeedbackLog:
    """Small append-only JSONL store; it has no Formal Knowledge dependency."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        if self.path.suffix.casefold() != ".jsonl":
            raise FeedbackLogError("Feedback Log path must use .jsonl")

    def read_all(self) -> tuple[FeedbackRecord, ...]:
        if not self.path.exists():
            return ()
        if not self.path.is_file():
            raise FeedbackLogError("Feedback Log path is not a file")
        try:
            lines = self.path.read_text(encoding="utf-8").splitlines()
        except (OSError, UnicodeError) as error:
            raise FeedbackLogError("Feedback Log cannot be read") from error
        records: list[FeedbackRecord] = []
        seen: set[str] = set()
        for line_number, line in enumerate(lines, start=1):
            if not line:
                raise FeedbackLogError(
                    f"Feedback Log contains blank line {line_number}"
                )
            try:
                record = FeedbackRecord.from_json(line)
            except FeedbackValidationError as error:
                raise FeedbackLogError(
                    f"Feedback Log contains invalid record at line {line_number}"
                ) from error
            if record.feedback_id in seen:
                raise FeedbackLogError("Feedback Log contains duplicate identity")
            seen.add(record.feedback_id)
            records.append(record)
        return tuple(records)

    def append(self, record: FeedbackRecord) -> None:
        if not isinstance(record, FeedbackRecord):
            raise TypeError("record must be a FeedbackRecord")
        if not self.path.parent.exists() or not self.path.parent.is_dir():
            raise FeedbackLogError("Feedback Log parent directory is unavailable")
        existing = self.read_all()
        if any(item.feedback_id == record.feedback_id for item in existing):
            raise FeedbackLogError("duplicate feedback identity")
        prefix_newline = False
        if self.path.exists():
            try:
                with self.path.open("rb") as stream:
                    stream.seek(0, os.SEEK_END)
                    if stream.tell() > 0:
                        stream.seek(-1, os.SEEK_END)
                        prefix_newline = stream.read(1) not in {b"\n", b"\r"}
            except OSError as error:
                raise FeedbackLogError("Feedback Log cannot be inspected") from error
        payload = ("\n" if prefix_newline else "") + record.to_json() + "\n"
        try:
            with self.path.open("a", encoding="utf-8", newline="") as stream:
                stream.write(payload)
                stream.flush()
                os.fsync(stream.fileno())
        except OSError as error:
            raise FeedbackLogError("Feedback Log append failed") from error
