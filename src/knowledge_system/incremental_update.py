"""Explicit startup coordination for derived-index version updates.

Importing this module performs no I/O.  Callers provide the source-version
reader, state store, clock, and incremental updater.  There is deliberately no
full-rebuild fallback: a failed incremental update leaves the last successful
``indexed_commit`` unchanged.
"""

from __future__ import annotations

import json
import re
from collections.abc import Callable
from dataclasses import dataclass, replace
from datetime import datetime
from enum import Enum
from typing import Protocol, runtime_checkable

from .system_state import (
    IndexStatus,
    SystemState,
    SystemStateError,
    check_source_commit,
    status_for,
)


UPDATE_RESULT_SCHEMA = "knowledge-system.incremental-update-result"
UPDATE_RESULT_SCHEMA_VERSION = 1
_STEP_PATTERN = re.compile(r"[A-Za-z][A-Za-z0-9_.:-]{0,127}\Z")
_COMMIT_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9._/@:+-]{0,254}\Z")
_RESULT_FIELDS = frozenset(
    {
        "action",
        "error_code",
        "error_type",
        "indexed_commit_after",
        "indexed_commit_before",
        "schema",
        "schema_version",
        "source_commit_after",
        "source_commit_before",
        "stage",
        "status",
        "updater_invoked",
    }
)


class UpdateAction(str, Enum):
    SKIPPED = "SKIPPED"
    UPDATED = "UPDATED"
    FAILED = "FAILED"


class UpdateStage(str, Enum):
    LOAD_STATE = "LOAD_STATE"
    CHECK_SOURCE = "CHECK_SOURCE"
    SAVE_CHECK = "SAVE_CHECK"
    UPDATE_INDEX = "UPDATE_INDEX"
    SAVE_SYNC = "SAVE_SYNC"
    COMPLETE = "COMPLETE"


class UpdateErrorType(str, Enum):
    STATE_LOAD_FAILED = "STATE_LOAD_FAILED"
    SOURCE_READ_FAILED = "SOURCE_READ_FAILED"
    CHECK_PERSIST_FAILED = "CHECK_PERSIST_FAILED"
    UPDATER_EXCEPTION = "UPDATER_EXCEPTION"
    INVALID_UPDATER_RESULT = "INVALID_UPDATER_RESULT"
    UPDATE_FAILED = "UPDATE_FAILED"
    UPDATE_INCOMPLETE = "UPDATE_INCOMPLETE"
    TARGET_MISMATCH = "TARGET_MISMATCH"
    FINAL_PERSIST_FAILED = "FINAL_PERSIST_FAILED"


class IncrementalUpdateResultError(ValueError):
    """A serialized incremental-update result is invalid."""

    def __init__(self, code: str, reason: str) -> None:
        self.code = code
        self.reason = reason
        super().__init__(f"{code}: {reason}")


@runtime_checkable
class SystemStateStore(Protocol):
    """The T-1200 state persistence boundary used by the coordinator."""

    def load(self) -> SystemState: ...

    def save(self, state: SystemState) -> None: ...


@dataclass(frozen=True)
class IncrementalUpdaterResult:
    """Strict acknowledgement returned by an injected incremental updater."""

    target_commit: str
    succeeded: bool
    required_steps: tuple[str, ...]
    completed_steps: tuple[str, ...]
    failure_code: str | None = None

    def __post_init__(self) -> None:
        if type(self.target_commit) is not str or not self.target_commit:
            raise ValueError("target_commit must be non-blank text")
        if type(self.succeeded) is not bool:
            raise TypeError("succeeded must be bool")
        for field_name, steps in (
            ("required_steps", self.required_steps),
            ("completed_steps", self.completed_steps),
        ):
            if type(steps) is not tuple or not steps:
                raise ValueError(f"{field_name} must be a non-empty tuple")
            if len(set(steps)) != len(steps) or any(
                type(step) is not str or not _STEP_PATTERN.fullmatch(step)
                for step in steps
            ):
                raise ValueError(f"{field_name} contains invalid or duplicate steps")
        if not set(self.completed_steps).issubset(self.required_steps):
            raise ValueError("completed_steps must be a subset of required_steps")
        if self.failure_code is not None and (
            type(self.failure_code) is not str
            or not _STEP_PATTERN.fullmatch(self.failure_code)
        ):
            raise ValueError("failure_code must be a stable code")
        if self.succeeded and self.failure_code is not None:
            raise ValueError("successful result cannot contain failure_code")

    @property
    def complete(self) -> bool:
        return self.succeeded and set(self.completed_steps) == set(self.required_steps)


@dataclass(frozen=True)
class IncrementalUpdateResult:
    """Serializable audit result; it contains no index body or session trace."""

    action: UpdateAction
    status: IndexStatus | None
    source_commit_before: str | None
    source_commit_after: str | None
    indexed_commit_before: str | None
    indexed_commit_after: str | None
    updater_invoked: bool
    stage: UpdateStage
    error_type: UpdateErrorType | None = None
    error_code: str | None = None
    schema: str = UPDATE_RESULT_SCHEMA
    schema_version: int = UPDATE_RESULT_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema != UPDATE_RESULT_SCHEMA:
            raise ValueError("unknown incremental-update-result schema")
        if type(self.schema_version) is not int or self.schema_version != 1:
            raise ValueError("unknown incremental-update-result version")
        if not isinstance(self.action, UpdateAction):
            raise TypeError("action must be UpdateAction")
        if self.status is not None and not isinstance(self.status, IndexStatus):
            raise TypeError("status must be IndexStatus or None")
        if type(self.updater_invoked) is not bool:
            raise TypeError("updater_invoked must be bool")
        if not isinstance(self.stage, UpdateStage):
            raise TypeError("stage must be UpdateStage")
        if self.error_type is not None and not isinstance(
            self.error_type, UpdateErrorType
        ):
            raise TypeError("error_type must be UpdateErrorType or None")
        if self.error_code is not None and (
            type(self.error_code) is not str
            or not _STEP_PATTERN.fullmatch(self.error_code)
        ):
            raise ValueError("error_code must be a stable code")
        for field_name, commit in (
            ("source_commit_before", self.source_commit_before),
            ("source_commit_after", self.source_commit_after),
            ("indexed_commit_before", self.indexed_commit_before),
            ("indexed_commit_after", self.indexed_commit_after),
        ):
            if commit is not None and (
                type(commit) is not str or not _COMMIT_PATTERN.fullmatch(commit)
            ):
                raise ValueError(f"{field_name} must be a valid commit identifier")
        if self.action is UpdateAction.FAILED:
            if (
                self.error_type is None
                or self.error_code is None
                or self.stage is UpdateStage.COMPLETE
            ):
                raise ValueError("failed result requires failure metadata")
        elif self.error_type is not None or self.error_code is not None:
            raise ValueError("non-failed result cannot contain failure metadata")
        if self.action is UpdateAction.SKIPPED:
            commits = (
                self.source_commit_before,
                self.source_commit_after,
                self.indexed_commit_before,
                self.indexed_commit_after,
            )
            if (
                self.status is not IndexStatus.SYNCED
                or self.updater_invoked
                or self.stage is not UpdateStage.COMPLETE
                or commits[0] is None
                or len(set(commits)) != 1
            ):
                raise ValueError("SKIPPED result is inconsistent with a synced check")
        elif self.action is UpdateAction.UPDATED:
            if (
                self.status is not IndexStatus.SYNCED
                or not self.updater_invoked
                or self.stage is not UpdateStage.COMPLETE
                or self.source_commit_before is None
                or self.source_commit_after is None
                or self.indexed_commit_after != self.source_commit_after
                or self.indexed_commit_before == self.source_commit_after
            ):
                raise ValueError("UPDATED result is inconsistent with a completed update")
        else:
            expected_errors = {
                UpdateStage.LOAD_STATE: {UpdateErrorType.STATE_LOAD_FAILED},
                UpdateStage.CHECK_SOURCE: {UpdateErrorType.SOURCE_READ_FAILED},
                UpdateStage.SAVE_CHECK: {UpdateErrorType.CHECK_PERSIST_FAILED},
                UpdateStage.UPDATE_INDEX: {
                    UpdateErrorType.UPDATER_EXCEPTION,
                    UpdateErrorType.INVALID_UPDATER_RESULT,
                    UpdateErrorType.UPDATE_FAILED,
                    UpdateErrorType.UPDATE_INCOMPLETE,
                    UpdateErrorType.TARGET_MISMATCH,
                },
                UpdateStage.SAVE_SYNC: {UpdateErrorType.FINAL_PERSIST_FAILED},
            }
            if self.error_type not in expected_errors.get(self.stage, set()):
                raise ValueError("failure stage and error_type are inconsistent")
            expected_invocation = self.stage in {
                UpdateStage.UPDATE_INDEX,
                UpdateStage.SAVE_SYNC,
            }
            if self.updater_invoked is not expected_invocation:
                raise ValueError("failure stage and updater_invoked are inconsistent")
            if self.stage is UpdateStage.LOAD_STATE:
                if self.status is not None or any(
                    commit is not None
                    for commit in (
                        self.source_commit_before,
                        self.source_commit_after,
                        self.indexed_commit_before,
                        self.indexed_commit_after,
                    )
                ):
                    raise ValueError("LOAD_STATE failure cannot report loaded state")
            else:
                if (
                    self.status is None
                    or self.source_commit_before is None
                    or self.source_commit_after is None
                    or self.indexed_commit_after != self.indexed_commit_before
                ):
                    raise ValueError("failed update result has inconsistent commits")
                if self.status is not status_for(
                    self.source_commit_after, self.indexed_commit_after
                ):
                    raise ValueError("failed update status does not match commits")
                if self.stage in {UpdateStage.CHECK_SOURCE, UpdateStage.SAVE_CHECK}:
                    if self.source_commit_after != self.source_commit_before:
                        raise ValueError("pre-update failure cannot change source commit")
                elif self.status is IndexStatus.SYNCED:
                    raise ValueError("invoked update failure cannot report SYNCED")


def incremental_update_result_to_json(result: IncrementalUpdateResult) -> str:
    """Return deterministic JSON for an update audit event."""

    if not isinstance(result, IncrementalUpdateResult):
        raise TypeError("result must be IncrementalUpdateResult")
    payload = {
        "action": result.action.value,
        "error_code": result.error_code,
        "error_type": result.error_type.value if result.error_type else None,
        "indexed_commit_after": result.indexed_commit_after,
        "indexed_commit_before": result.indexed_commit_before,
        "schema": result.schema,
        "schema_version": result.schema_version,
        "source_commit_after": result.source_commit_after,
        "source_commit_before": result.source_commit_before,
        "stage": result.stage.value,
        "status": result.status.value if result.status else None,
        "updater_invoked": result.updater_invoked,
    }
    return json.dumps(
        payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False
    ) + "\n"


def _strict_result_json(payload: str | bytes) -> object:
    if type(payload) is bytes:
        try:
            payload = payload.decode("utf-8", errors="strict")
        except UnicodeDecodeError as error:
            raise IncrementalUpdateResultError(
                "INVALID_UTF8", "incremental-update result is not strict UTF-8"
            ) from error
    elif type(payload) is not str:
        raise TypeError("payload must be text or bytes")

    def object_pairs(pairs: list[tuple[str, object]]) -> dict[str, object]:
        result: dict[str, object] = {}
        for key, value in pairs:
            if key in result:
                raise IncrementalUpdateResultError(
                    "DUPLICATE_KEY", f"duplicate JSON key: {key}"
                )
            result[key] = value
        return result

    try:
        return json.loads(
            payload,
            object_pairs_hook=object_pairs,
            parse_constant=lambda value: (_ for _ in ()).throw(
                IncrementalUpdateResultError(
                    "INVALID_JSON", f"invalid JSON constant: {value}"
                )
            ),
        )
    except IncrementalUpdateResultError:
        raise
    except (TypeError, ValueError, json.JSONDecodeError) as error:
        raise IncrementalUpdateResultError(
            "INVALID_JSON", "incremental-update result is not valid JSON"
        ) from error


def _result_enum(
    value: object,
    enum_type: type[Enum],
    field: str,
    *,
    optional: bool = False,
) -> Enum | None:
    if value is None and optional:
        return None
    if type(value) is not str:
        raise IncrementalUpdateResultError(
            f"INVALID_{field.upper()}", f"{field} must be text"
        )
    try:
        return enum_type(value)
    except ValueError as error:
        raise IncrementalUpdateResultError(
            f"INVALID_{field.upper()}", f"unknown {field}"
        ) from error


def incremental_update_result_from_json(
    payload: str | bytes,
) -> IncrementalUpdateResult:
    """Deserialize one strict audit result without performing external work."""

    root = _strict_result_json(payload)
    if type(root) is not dict or set(root) != _RESULT_FIELDS:
        raise IncrementalUpdateResultError(
            "INVALID_FIELDS", "incremental-update-result fields do not match schema"
        )
    if type(root["schema"]) is not str or root["schema"] != UPDATE_RESULT_SCHEMA:
        raise IncrementalUpdateResultError(
            "UNKNOWN_SCHEMA", "unknown incremental-update-result schema"
        )
    if (
        type(root["schema_version"]) is not int
        or root["schema_version"] != UPDATE_RESULT_SCHEMA_VERSION
    ):
        raise IncrementalUpdateResultError(
            "UNKNOWN_VERSION", "unknown incremental-update-result version"
        )
    if type(root["updater_invoked"]) is not bool:
        raise IncrementalUpdateResultError(
            "INVALID_UPDATER_INVOKED", "updater_invoked must be bool"
        )
    for field in (
        "source_commit_before",
        "source_commit_after",
        "indexed_commit_before",
        "indexed_commit_after",
    ):
        value = root[field]
        if value is not None and (
            type(value) is not str or not _COMMIT_PATTERN.fullmatch(value)
        ):
            raise IncrementalUpdateResultError(
                "INVALID_COMMIT", f"{field} must be a valid commit identifier or null"
            )
    if root["error_code"] is not None and (
        type(root["error_code"]) is not str
        or not _STEP_PATTERN.fullmatch(root["error_code"])
    ):
        raise IncrementalUpdateResultError(
            "INVALID_ERROR_CODE", "error_code must be a stable code or null"
        )
    try:
        return IncrementalUpdateResult(
            action=_result_enum(root["action"], UpdateAction, "action"),  # type: ignore[arg-type]
            status=_result_enum(
                root["status"], IndexStatus, "status", optional=True
            ),  # type: ignore[arg-type]
            source_commit_before=root["source_commit_before"],  # type: ignore[arg-type]
            source_commit_after=root["source_commit_after"],  # type: ignore[arg-type]
            indexed_commit_before=root["indexed_commit_before"],  # type: ignore[arg-type]
            indexed_commit_after=root["indexed_commit_after"],  # type: ignore[arg-type]
            updater_invoked=root["updater_invoked"],
            stage=_result_enum(root["stage"], UpdateStage, "stage"),  # type: ignore[arg-type]
            error_type=_result_enum(
                root["error_type"], UpdateErrorType, "error_type", optional=True
            ),  # type: ignore[arg-type]
            error_code=root["error_code"],  # type: ignore[arg-type]
            schema=root["schema"],
            schema_version=root["schema_version"],
        )
    except IncrementalUpdateResultError:
        raise
    except (TypeError, ValueError, SystemStateError) as error:
        raise IncrementalUpdateResultError(
            "INCONSISTENT_RESULT", str(error)
        ) from error


def _code(error: BaseException) -> str:
    if isinstance(error, SystemStateError):
        return error.code
    return type(error).__name__


def _failure(
    *,
    state: SystemState | None,
    original: SystemState | None,
    stage: UpdateStage,
    error_type: UpdateErrorType,
    error_code: str,
    updater_invoked: bool,
) -> IncrementalUpdateResult:
    visible = state or original
    before = original
    return IncrementalUpdateResult(
        action=UpdateAction.FAILED,
        status=visible.index_status if visible else None,
        source_commit_before=before.source_commit if before else None,
        source_commit_after=visible.source_commit if visible else None,
        indexed_commit_before=before.indexed_commit if before else None,
        indexed_commit_after=before.indexed_commit if before else None,
        updater_invoked=updater_invoked,
        stage=stage,
        error_type=error_type,
        error_code=error_code,
    )


def synchronize_index_on_startup(
    *,
    state_store: SystemStateStore,
    source_commit_reader: Callable[[], str],
    incremental_updater: Callable[[str | None, str], IncrementalUpdaterResult],
    clock: Callable[[], datetime],
) -> IncrementalUpdateResult:
    """Check the source version and perform at most one incremental update.

    The checked state is persisted before an updater is considered.  A synced
    version is therefore an idempotent skip.  On every update failure the
    original successful ``indexed_commit`` remains the reported and persisted
    index version.  The updater receives ``(indexed_commit, source_commit)``;
    no full-rebuild callback exists in this API.
    """

    try:
        original = state_store.load()
        if not isinstance(original, SystemState):
            raise TypeError("state store returned a non-SystemState value")
    except Exception as error:
        return _failure(
            state=None,
            original=None,
            stage=UpdateStage.LOAD_STATE,
            error_type=UpdateErrorType.STATE_LOAD_FAILED,
            error_code=_code(error),
            updater_invoked=False,
        )

    try:
        source_commit = source_commit_reader()
        checked = check_source_commit(original, source_commit, clock=clock)
    except Exception as error:
        return _failure(
            state=original,
            original=original,
            stage=UpdateStage.CHECK_SOURCE,
            error_type=UpdateErrorType.SOURCE_READ_FAILED,
            error_code=_code(error),
            updater_invoked=False,
        )

    try:
        state_store.save(checked)
    except Exception as error:
        return _failure(
            state=original,
            original=original,
            stage=UpdateStage.SAVE_CHECK,
            error_type=UpdateErrorType.CHECK_PERSIST_FAILED,
            error_code=_code(error),
            updater_invoked=False,
        )

    if checked.index_status is IndexStatus.SYNCED:
        return IncrementalUpdateResult(
            action=UpdateAction.SKIPPED,
            status=IndexStatus.SYNCED,
            source_commit_before=original.source_commit,
            source_commit_after=checked.source_commit,
            indexed_commit_before=original.indexed_commit,
            indexed_commit_after=original.indexed_commit,
            updater_invoked=False,
            stage=UpdateStage.COMPLETE,
        )

    try:
        update = incremental_updater(checked.indexed_commit, checked.source_commit)
    except Exception as error:
        return _failure(
            state=checked,
            original=original,
            stage=UpdateStage.UPDATE_INDEX,
            error_type=UpdateErrorType.UPDATER_EXCEPTION,
            error_code=_code(error),
            updater_invoked=True,
        )
    if not isinstance(update, IncrementalUpdaterResult):
        return _failure(
            state=checked,
            original=original,
            stage=UpdateStage.UPDATE_INDEX,
            error_type=UpdateErrorType.INVALID_UPDATER_RESULT,
            error_code="INVALID_RESULT_TYPE",
            updater_invoked=True,
        )
    if update.target_commit != checked.source_commit:
        return _failure(
            state=checked,
            original=original,
            stage=UpdateStage.UPDATE_INDEX,
            error_type=UpdateErrorType.TARGET_MISMATCH,
            error_code="TARGET_MISMATCH",
            updater_invoked=True,
        )
    if not update.succeeded:
        return _failure(
            state=checked,
            original=original,
            stage=UpdateStage.UPDATE_INDEX,
            error_type=UpdateErrorType.UPDATE_FAILED,
            error_code=update.failure_code or "UPDATER_REPORTED_FAILURE",
            updater_invoked=True,
        )
    if not update.complete:
        return _failure(
            state=checked,
            original=original,
            stage=UpdateStage.UPDATE_INDEX,
            error_type=UpdateErrorType.UPDATE_INCOMPLETE,
            error_code="REQUIRED_STEPS_INCOMPLETE",
            updater_invoked=True,
        )

    synced = replace(
        checked,
        indexed_commit=checked.source_commit,
        index_status=IndexStatus.SYNCED,
    )
    try:
        state_store.save(synced)
    except Exception as error:
        return _failure(
            state=checked,
            original=original,
            stage=UpdateStage.SAVE_SYNC,
            error_type=UpdateErrorType.FINAL_PERSIST_FAILED,
            error_code=_code(error),
            updater_invoked=True,
        )
    return IncrementalUpdateResult(
        action=UpdateAction.UPDATED,
        status=IndexStatus.SYNCED,
        source_commit_before=original.source_commit,
        source_commit_after=synced.source_commit,
        indexed_commit_before=original.indexed_commit,
        indexed_commit_after=synced.indexed_commit,
        updater_invoked=True,
        stage=UpdateStage.COMPLETE,
    )


__all__ = [
    "UPDATE_RESULT_SCHEMA",
    "UPDATE_RESULT_SCHEMA_VERSION",
    "IncrementalUpdateResult",
    "IncrementalUpdateResultError",
    "IncrementalUpdaterResult",
    "SystemStateStore",
    "UpdateAction",
    "UpdateErrorType",
    "UpdateStage",
    "incremental_update_result_from_json",
    "incremental_update_result_to_json",
    "synchronize_index_on_startup",
]
