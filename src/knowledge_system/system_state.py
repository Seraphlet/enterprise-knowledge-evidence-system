"""Persistent knowledge-source and derived-index synchronization state.

This module deliberately models only system version state.  It does not contain
session/agent state or trace data, and loading/checking state never builds an
index or advances the indexed commit.
"""

from __future__ import annotations

import json
import os
import re
import stat
import tempfile
from collections.abc import Callable
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any


SYSTEM_STATE_SCHEMA = "knowledge-system.system-state"
SYSTEM_STATE_SCHEMA_VERSION = 1
_STATE_FIELDS = frozenset(
    {
        "schema",
        "schema_version",
        "source_commit",
        "indexed_commit",
        "index_status",
        "last_check_at",
    }
)
_COMMIT_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9._/@:+-]{0,254}\Z")


class SystemStateError(ValueError):
    """A system-state value or persisted artifact is invalid."""

    def __init__(self, code: str, reason: str) -> None:
        self.code = code
        self.reason = reason
        super().__init__(f"{code}: {reason}")


class IndexStatus(str, Enum):
    """Relationship between authoritative source and derived index."""

    NOT_BUILT = "NOT_BUILT"
    SYNCED = "SYNCED"
    OUT_OF_SYNC = "OUT_OF_SYNC"


def _validate_commit(value: object, field: str, *, optional: bool) -> str | None:
    if value is None and optional:
        return None
    if type(value) is not str or not _COMMIT_PATTERN.fullmatch(value):
        raise SystemStateError(
            "INVALID_COMMIT",
            f"{field} must be a non-blank commit identifier without whitespace",
        )
    return value


def _validate_timestamp(value: object, field: str = "last_check_at") -> datetime:
    if not isinstance(value, datetime):
        raise SystemStateError("INVALID_TIME", f"{field} must be a datetime")
    if value.tzinfo is None or value.utcoffset() is None:
        raise SystemStateError("INVALID_TIME", f"{field} must include a timezone")
    return value


def status_for(source_commit: str, indexed_commit: str | None) -> IndexStatus:
    """Return the deterministic relationship without mutating either version."""

    source = _validate_commit(source_commit, "source_commit", optional=False)
    indexed = _validate_commit(indexed_commit, "indexed_commit", optional=True)
    if indexed is None:
        return IndexStatus.NOT_BUILT
    if source == indexed:
        return IndexStatus.SYNCED
    return IndexStatus.OUT_OF_SYNC


@dataclass(frozen=True)
class SystemState:
    """Small immutable record separating source and derived-index commits."""

    source_commit: str
    indexed_commit: str | None
    index_status: IndexStatus
    last_check_at: datetime
    schema: str = SYSTEM_STATE_SCHEMA
    schema_version: int = SYSTEM_STATE_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if type(self.schema) is not str or self.schema != SYSTEM_STATE_SCHEMA:
            raise SystemStateError("UNKNOWN_SCHEMA", "unknown system-state schema")
        if (
            type(self.schema_version) is not int
            or self.schema_version != SYSTEM_STATE_SCHEMA_VERSION
        ):
            raise SystemStateError("UNKNOWN_VERSION", "unknown system-state schema version")
        _validate_commit(self.source_commit, "source_commit", optional=False)
        _validate_commit(self.indexed_commit, "indexed_commit", optional=True)
        if not isinstance(self.index_status, IndexStatus):
            raise SystemStateError("INVALID_STATUS", "index_status must be an IndexStatus")
        _validate_timestamp(self.last_check_at)
        expected = status_for(self.source_commit, self.indexed_commit)
        if self.index_status is not expected:
            raise SystemStateError(
                "INCONSISTENT_STATE",
                f"index_status must be {expected.value} for the recorded commits",
            )


def _checked_at(
    checked_at: datetime | None,
    clock: Callable[[], datetime] | None,
) -> datetime:
    if checked_at is not None and clock is not None:
        raise TypeError("checked_at and clock are mutually exclusive")
    selected = checked_at if checked_at is not None else (clock or _utc_now)()
    return _validate_timestamp(selected, "checked_at")


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def create_system_state(
    source_commit: str,
    *,
    indexed_commit: str | None = None,
    checked_at: datetime | None = None,
    clock: Callable[[], datetime] | None = None,
) -> SystemState:
    """Create a checked state without performing index work."""

    return SystemState(
        source_commit=source_commit,
        indexed_commit=indexed_commit,
        index_status=status_for(source_commit, indexed_commit),
        last_check_at=_checked_at(checked_at, clock),
    )


def check_source_commit(
    state: SystemState,
    source_commit: str,
    *,
    checked_at: datetime | None = None,
    clock: Callable[[], datetime] | None = None,
) -> SystemState:
    """Record a source check while preserving the last successful index commit.

    In particular, this operation cannot mark an index build successful.  A
    source change therefore produces ``OUT_OF_SYNC`` until a later update
    workflow explicitly and completely builds that source version.
    """

    if not isinstance(state, SystemState):
        raise TypeError("state must be a SystemState")
    checked_source = _validate_commit(source_commit, "source_commit", optional=False)
    return replace(
        state,
        source_commit=checked_source,
        index_status=status_for(checked_source, state.indexed_commit),
        last_check_at=_checked_at(checked_at, clock),
    )


def system_state_to_json(state: SystemState) -> str:
    """Serialize a state to stable, strict UTF-8-compatible JSON text."""

    if not isinstance(state, SystemState):
        raise TypeError("state must be a SystemState")
    payload = {
        "schema": state.schema,
        "schema_version": state.schema_version,
        "source_commit": state.source_commit,
        "indexed_commit": state.indexed_commit,
        "index_status": state.index_status.value,
        "last_check_at": state.last_check_at.isoformat(),
    }
    return json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ) + "\n"


def _strict_json(payload: str) -> object:
    def object_pairs(pairs: list[tuple[str, object]]) -> dict[str, object]:
        result: dict[str, object] = {}
        for key, value in pairs:
            if key in result:
                raise SystemStateError("DUPLICATE_KEY", f"duplicate JSON key: {key}")
            result[key] = value
        return result

    try:
        return json.loads(
            payload,
            object_pairs_hook=object_pairs,
            parse_constant=lambda value: (_ for _ in ()).throw(
                SystemStateError("INVALID_JSON", f"invalid JSON constant: {value}")
            ),
        )
    except SystemStateError:
        raise
    except (TypeError, ValueError, json.JSONDecodeError) as error:
        raise SystemStateError("INVALID_JSON", "system state is not valid JSON") from error


def _exact_root(value: object) -> dict[str, Any]:
    if type(value) is not dict or set(value) != _STATE_FIELDS:
        raise SystemStateError("INVALID_FIELDS", "system-state fields do not match schema")
    return value


def system_state_from_json(payload: str) -> SystemState:
    """Deserialize strict JSON without triggering any update or external work."""

    if type(payload) is not str:
        raise TypeError("payload must be text")
    root = _exact_root(_strict_json(payload))
    if type(root["schema"]) is not str or root["schema"] != SYSTEM_STATE_SCHEMA:
        raise SystemStateError("UNKNOWN_SCHEMA", "unknown system-state schema")
    if (
        type(root["schema_version"]) is not int
        or root["schema_version"] != SYSTEM_STATE_SCHEMA_VERSION
    ):
        raise SystemStateError("UNKNOWN_VERSION", "unknown system-state schema version")
    if type(root["index_status"]) is not str:
        raise SystemStateError("INVALID_STATUS", "index_status must be text")
    try:
        status = IndexStatus(root["index_status"])
    except ValueError as error:
        raise SystemStateError("INVALID_STATUS", "unknown index_status") from error
    if type(root["last_check_at"]) is not str:
        raise SystemStateError("INVALID_TIME", "last_check_at must be ISO-8601 text")
    try:
        timestamp = datetime.fromisoformat(root["last_check_at"].replace("Z", "+00:00"))
    except ValueError as error:
        raise SystemStateError("INVALID_TIME", "last_check_at is not ISO-8601") from error
    return SystemState(
        source_commit=root["source_commit"],
        indexed_commit=root["indexed_commit"],
        index_status=status,
        last_check_at=timestamp,
        schema=root["schema"],
        schema_version=root["schema_version"],
    )


def _state_path(path: os.PathLike[str] | str, *, for_write: bool) -> Path:
    if not isinstance(path, (str, os.PathLike)):
        raise TypeError("path must be a string or path-like value")
    target = Path(path)
    if not target.name or target.name in {".", ".."}:
        raise SystemStateError("INVALID_PATH", "state path must name a file")
    if not target.parent.is_dir():
        raise SystemStateError("INVALID_PATH", "state parent directory does not exist")
    if target.exists() or target.is_symlink():
        try:
            mode = target.lstat().st_mode
        except OSError as error:
            raise SystemStateError("INVALID_PATH", str(error)) from error
        if stat.S_ISLNK(mode) or not stat.S_ISREG(mode):
            raise SystemStateError("INVALID_PATH", "state path must be a regular file")
    elif not for_write:
        raise SystemStateError("INVALID_PATH", "state file does not exist")
    return target


def save_system_state(state: SystemState, path: os.PathLike[str] | str) -> None:
    """Atomically persist an already-computed state."""

    target = _state_path(path, for_write=True)
    payload = system_state_to_json(state).encode("utf-8")
    descriptor: int | None = None
    temporary: str | None = None
    try:
        descriptor, temporary = tempfile.mkstemp(
            prefix=f".{target.name}.", suffix=".tmp", dir=str(target.parent)
        )
        with os.fdopen(descriptor, "wb") as stream:
            descriptor = None
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, target)
        temporary = None
    except Exception as error:
        raise SystemStateError("STATE_WRITE_FAILED", str(error)) from error
    finally:
        if descriptor is not None:
            os.close(descriptor)
        if temporary is not None:
            try:
                os.unlink(temporary)
            except FileNotFoundError:
                pass


def load_system_state(path: os.PathLike[str] | str) -> SystemState:
    """Read state only; this function never checks sources or builds indexes."""

    source = _state_path(path, for_write=False)
    try:
        payload = source.read_bytes().decode("utf-8", errors="strict")
    except UnicodeDecodeError as error:
        raise SystemStateError("INVALID_UTF8", "system state is not strict UTF-8") from error
    except OSError as error:
        raise SystemStateError("STATE_READ_FAILED", str(error)) from error
    return system_state_from_json(payload)


__all__ = [
    "SYSTEM_STATE_SCHEMA",
    "SYSTEM_STATE_SCHEMA_VERSION",
    "IndexStatus",
    "SystemState",
    "SystemStateError",
    "check_source_commit",
    "create_system_state",
    "load_system_state",
    "save_system_state",
    "status_for",
    "system_state_from_json",
    "system_state_to_json",
]
