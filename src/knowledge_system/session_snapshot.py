"""Strict, explicit persistence for completed-turn :class:`AgentState` values.

The format is intentionally private to this package.  It is a versioned JSON
envelope whose value codec has a closed allowlist; it is not a general object
serializer and never imports or executes a type named by the input.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import os
import stat
import tempfile
import types
from collections.abc import Callable, Mapping
from dataclasses import dataclass, fields, is_dataclass
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation
from enum import Enum
from pathlib import Path
from typing import Any, Union, get_args, get_origin, get_type_hints

from .agent_state import AgentState, AgentStateError
from .apply_compare import ApplyCompareResult, DuplicateComparisonGroup, LogicalOutcome
from .clarification import (
    ClarificationAction,
    ClarificationDecision,
    KnowledgeBoundary,
    RetrievalOutcome,
)
from .context import (
    CaseContext,
    ContextRelation,
    ContextTransition,
    ExtractedFact,
    QueryContext,
    TaskContext,
)
from .contracts import Evidence, KnowledgeLineage, SourceReference
from .deterministic_comparison import (
    ComparisonOutcome,
    ConditionComparison,
    ExactCondition,
)
from .evidence_map import (
    EvidenceMap,
    EvidenceMapEntry,
    HydratedEvidenceGroup,
    HydratedEvidenceSection,
)
from .evidence_map_plan import EvidenceMapGroup, EvidenceMapPlan, EvidenceMapSection
from .evidence_quality import EvidenceQualitySignal, EvidenceQualitySignalType
from .fact_extraction import (
    CaseSourceSpan,
    ExtractedRequirementFact,
    FactExtractionStatus,
    FactObservation,
)
from .information_requirements import InformationRequirement, RequirementCertainty
from .limited_recovery import (
    RecoveryAction,
    RecoveryResult,
    RecoveryRoundTrace,
    RecoveryStatus,
    RecoveryTarget,
)
from .request_classification import (
    ClassificationSignal,
    KnowledgeRequestType,
    RequestClassification,
)
from .rule_structure import (
    LogicalRelation,
    RuleExtractionResult,
    RuleExtractionStatus,
    RuleStructure,
    SourceFragment,
    Threshold,
)


SNAPSHOT_SCHEMA = "knowledge-system.agent-state-snapshot"
SNAPSHOT_VERSION = 1
_ROOT_FIELDS = frozenset(
    {"schema", "version", "session_id", "saved_at", "revision", "state", "digest"}
)
_SIGNED_FIELDS = frozenset(_ROOT_FIELDS - {"digest"})


class SessionSnapshotError(ValueError):
    """A snapshot cannot be safely saved or resumed."""

    def __init__(self, code: str, reason: str) -> None:
        self.code = code
        self.reason = reason
        super().__init__(f"{code}: {reason}")


@dataclass(frozen=True)
class SessionSnapshotMetadata:
    schema: str
    version: int
    session_id: str
    saved_at: str
    revision: int


@dataclass(frozen=True)
class SessionResumeResult:
    state: AgentState
    metadata: SessionSnapshotMetadata


_DATACLASS_TYPES = (
    AgentState,
    AgentStateError,
    ApplyCompareResult,
    CaseContext,
    CaseSourceSpan,
    ClassificationSignal,
    ClarificationDecision,
    ConditionComparison,
    ContextTransition,
    DuplicateComparisonGroup,
    Evidence,
    EvidenceMap,
    EvidenceMapEntry,
    EvidenceMapGroup,
    EvidenceMapPlan,
    EvidenceMapSection,
    EvidenceQualitySignal,
    ExactCondition,
    ExtractedFact,
    ExtractedRequirementFact,
    FactObservation,
    HydratedEvidenceGroup,
    HydratedEvidenceSection,
    InformationRequirement,
    KnowledgeLineage,
    LogicalRelation,
    QueryContext,
    RecoveryResult,
    RecoveryRoundTrace,
    RecoveryTarget,
    RequestClassification,
    RuleExtractionResult,
    RuleStructure,
    SourceFragment,
    SourceReference,
    TaskContext,
    Threshold,
)
_ENUM_TYPES = (
    ClarificationAction,
    ComparisonOutcome,
    ContextRelation,
    EvidenceQualitySignalType,
    FactExtractionStatus,
    KnowledgeBoundary,
    KnowledgeRequestType,
    LogicalOutcome,
    RecoveryAction,
    RecoveryStatus,
    RequirementCertainty,
    RetrievalOutcome,
    RuleExtractionStatus,
)
_DATACLASS_BY_NAME = {item.__name__: item for item in _DATACLASS_TYPES}
_ENUM_BY_NAME = {item.__name__: item for item in _ENUM_TYPES}


def _canonical(value: object) -> bytes:
    try:
        return json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as error:
        raise SessionSnapshotError("UNSUPPORTED_VALUE", str(error)) from error


def _encode(value: object, path: str = "state") -> object:
    if value is None or type(value) in {bool, int, str}:
        return value
    if type(value) is float:
        if not (float("-inf") < value < float("inf")):
            raise SessionSnapshotError("UNSUPPORTED_VALUE", f"{path} has non-finite float")
        return {"$type": "float", "value": repr(value)}
    if isinstance(value, Decimal):
        if not value.is_finite():
            raise SessionSnapshotError("UNSUPPORTED_VALUE", f"{path} has non-finite Decimal")
        return {"$type": "decimal", "value": str(value)}
    if isinstance(value, Enum):
        enum_type = type(value)
        if enum_type not in _ENUM_TYPES:
            raise SessionSnapshotError("UNSUPPORTED_TYPE", f"{path} has unsupported Enum")
        return {"$type": "enum", "name": enum_type.__name__, "value": value.value}
    if type(value) is tuple:
        return {
            "$type": "tuple",
            "items": [_encode(item, f"{path}[{index}]") for index, item in enumerate(value)],
        }
    if isinstance(value, Mapping):
        encoded_items: list[tuple[str, object, object]] = []
        for key, item in value.items():
            if type(key) is not str:
                raise SessionSnapshotError(
                    "UNSUPPORTED_VALUE", f"{path} mapping keys must be strings"
                )
            encoded_key = _encode(key, f"{path}.key")
            encoded_value = _encode(item, f"{path}.{key}")
            encoded_items.append((_canonical(encoded_key).decode("utf-8"), encoded_key, encoded_value))
        encoded_items.sort(key=lambda item: item[0])
        return {
            "$type": "mapping",
            "items": [[key, item] for _, key, item in encoded_items],
        }
    if is_dataclass(value) and not isinstance(value, type):
        data_type = type(value)
        if data_type not in _DATACLASS_TYPES:
            raise SessionSnapshotError(
                "UNSUPPORTED_TYPE", f"{path} has unsupported dataclass {data_type.__name__}"
            )
        return {
            "$type": "dataclass",
            "name": data_type.__name__,
            "fields": {
                item.name: _encode(getattr(value, item.name), f"{path}.{item.name}")
                for item in fields(value)
            },
        }
    raise SessionSnapshotError(
        "UNSUPPORTED_TYPE", f"{path} has unsupported type {type(value).__name__}"
    )


def _exact_object(value: object, expected: frozenset[str], path: str) -> dict[str, Any]:
    if type(value) is not dict or set(value) != expected:
        raise SessionSnapshotError("INVALID_FIELDS", f"{path} fields do not match schema")
    return value


def _matches_annotation(value: object, annotation: object) -> bool:
    """Check decoded fields without invoking coercive or user-defined logic."""

    if annotation is Any:
        return True
    origin = get_origin(annotation)
    arguments = get_args(annotation)
    if origin in {Union, types.UnionType}:
        if isinstance(value, Decimal) and int in arguments and float in arguments:
            return True
        return any(_matches_annotation(value, item) for item in arguments)
    if origin is tuple:
        if type(value) is not tuple:
            return False
        if len(arguments) == 2 and arguments[1] is Ellipsis:
            return all(_matches_annotation(item, arguments[0]) for item in value)
        return len(value) == len(arguments) and all(
            _matches_annotation(item, expected)
            for item, expected in zip(value, arguments)
        )
    if origin is list:
        # AgentState recursively freezes list-shaped JSON values to tuples.
        return type(value) is tuple and all(
            _matches_annotation(item, arguments[0]) for item in value
        )
    if origin is dict:
        return type(value) is dict and all(
            _matches_annotation(key, arguments[0])
            and _matches_annotation(item, arguments[1])
            for key, item in value.items()
        )
    if annotation is type(None):
        return value is None
    if annotation in {bool, int, float, str}:
        return type(value) is annotation
    if isinstance(annotation, type):
        return isinstance(value, annotation)
    return False


def _validate_dataclass_fields(value: object, path: str) -> None:
    try:
        annotations = get_type_hints(type(value))
    except (NameError, TypeError) as error:
        raise SessionSnapshotError(
            "INVALID_TYPE", f"{path} has unresolved field types"
        ) from error
    for item in fields(value):
        annotation = annotations.get(item.name)
        if annotation is None or not _matches_annotation(
            getattr(value, item.name), annotation
        ):
            raise SessionSnapshotError(
                "INVALID_TYPE",
                f"{path}.{item.name} does not match {type(value).__name__} schema",
            )


def _decode(value: object, path: str = "state") -> object:
    if value is None or type(value) in {bool, int, str}:
        return value
    if type(value) is not dict:
        raise SessionSnapshotError("INVALID_TYPE", f"{path} has invalid JSON value type")
    tag = value.get("$type")
    if tag == "float":
        item = _exact_object(value, frozenset({"$type", "value"}), path)
        if type(item["value"]) is not str:
            raise SessionSnapshotError("INVALID_TYPE", f"{path}.value must be text")
        try:
            result = float(item["value"])
        except ValueError as error:
            raise SessionSnapshotError("INVALID_VALUE", f"{path} has invalid float") from error
        if not (float("-inf") < result < float("inf")):
            raise SessionSnapshotError("INVALID_VALUE", f"{path} has non-finite float")
        return result
    if tag == "decimal":
        item = _exact_object(value, frozenset({"$type", "value"}), path)
        if type(item["value"]) is not str:
            raise SessionSnapshotError("INVALID_TYPE", f"{path}.value must be text")
        try:
            result = Decimal(item["value"])
        except InvalidOperation as error:
            raise SessionSnapshotError("INVALID_VALUE", f"{path} has invalid Decimal") from error
        if not result.is_finite():
            raise SessionSnapshotError("INVALID_VALUE", f"{path} has non-finite Decimal")
        return result
    if tag == "enum":
        item = _exact_object(value, frozenset({"$type", "name", "value"}), path)
        enum_type = _ENUM_BY_NAME.get(item["name"]) if type(item["name"]) is str else None
        if enum_type is None:
            raise SessionSnapshotError("UNKNOWN_TYPE", f"{path} has unknown Enum type")
        try:
            return enum_type(item["value"])
        except (TypeError, ValueError) as error:
            raise SessionSnapshotError("INVALID_VALUE", f"{path} has invalid Enum value") from error
    if tag == "tuple":
        item = _exact_object(value, frozenset({"$type", "items"}), path)
        if type(item["items"]) is not list:
            raise SessionSnapshotError("INVALID_TYPE", f"{path}.items must be an array")
        return tuple(_decode(child, f"{path}[{index}]") for index, child in enumerate(item["items"]))
    if tag == "mapping":
        item = _exact_object(value, frozenset({"$type", "items"}), path)
        if type(item["items"]) is not list:
            raise SessionSnapshotError("INVALID_TYPE", f"{path}.items must be an array")
        result: dict[str, object] = {}
        for index, pair in enumerate(item["items"]):
            if type(pair) is not list or len(pair) != 2:
                raise SessionSnapshotError("INVALID_TYPE", f"{path}.items[{index}] must be a pair")
            key = _decode(pair[0], f"{path}.items[{index}].key")
            if type(key) is not str:
                raise SessionSnapshotError("INVALID_TYPE", f"{path} mapping key must be text")
            if key in result:
                raise SessionSnapshotError("DUPLICATE_KEY", f"{path} has duplicate mapping key")
            result[key] = _decode(pair[1], f"{path}.{key}")
        return result
    if tag == "dataclass":
        item = _exact_object(value, frozenset({"$type", "name", "fields"}), path)
        data_type = _DATACLASS_BY_NAME.get(item["name"]) if type(item["name"]) is str else None
        if data_type is None:
            raise SessionSnapshotError("UNKNOWN_TYPE", f"{path} has unknown dataclass type")
        field_values = item["fields"]
        expected = frozenset(field.name for field in fields(data_type))
        decoded_fields = _exact_object(field_values, expected, f"{path}.fields")
        try:
            result = data_type(
                **{
                    name: _decode(field_value, f"{path}.{name}")
                    for name, field_value in decoded_fields.items()
                }
            )
            _validate_dataclass_fields(result, path)
            return result
        except SessionSnapshotError:
            raise
        except Exception as error:
            raise SessionSnapshotError(
                "INVALID_VALUE", f"{path} could not construct {data_type.__name__}: {error}"
            ) from error
    raise SessionSnapshotError("UNKNOWN_TYPE", f"{path} has unknown type tag")


def _strict_json(payload: str) -> object:
    def object_pairs(pairs: list[tuple[str, object]]) -> dict[str, object]:
        result: dict[str, object] = {}
        for key, value in pairs:
            if key in result:
                raise SessionSnapshotError("DUPLICATE_KEY", f"duplicate JSON key: {key}")
            result[key] = value
        return result

    try:
        return json.loads(
            payload,
            object_pairs_hook=object_pairs,
            parse_constant=lambda value: (_ for _ in ()).throw(
                SessionSnapshotError("INVALID_JSON", f"invalid JSON constant: {value}")
            ),
            parse_float=lambda value: (_ for _ in ()).throw(
                SessionSnapshotError("INVALID_JSON", "untagged JSON floats are forbidden")
            ),
        )
    except SessionSnapshotError:
        raise
    except (TypeError, ValueError, json.JSONDecodeError) as error:
        raise SessionSnapshotError("INVALID_JSON", "snapshot is not valid JSON") from error


def _snapshot_path(path: os.PathLike[str] | str, *, for_write: bool) -> Path:
    if not isinstance(path, (str, os.PathLike)):
        raise TypeError("path must be a string or path-like value")
    candidate = Path(path)
    if not candidate.name or candidate.name in {".", ".."}:
        raise SessionSnapshotError("INVALID_PATH", "snapshot path must name a file")
    parent = candidate.parent
    if not parent.is_dir():
        raise SessionSnapshotError("INVALID_PATH", "snapshot parent directory does not exist")
    if candidate.exists() or candidate.is_symlink():
        try:
            mode = candidate.lstat().st_mode
        except OSError as error:
            raise SessionSnapshotError("INVALID_PATH", str(error)) from error
        if stat.S_ISLNK(mode) or not stat.S_ISREG(mode):
            raise SessionSnapshotError("INVALID_PATH", "snapshot path must be a regular file")
    elif not for_write:
        raise SessionSnapshotError("INVALID_PATH", "snapshot file does not exist")
    return candidate


def _saved_at(value: str | datetime | None, clock: Callable[[], str | datetime] | None) -> str:
    if value is not None and clock is not None:
        raise TypeError("saved_at and clock are mutually exclusive")
    selected: str | datetime = value if value is not None else (clock or (lambda: datetime.now(timezone.utc)))()
    if isinstance(selected, datetime):
        if selected.tzinfo is None or selected.utcoffset() is None:
            raise SessionSnapshotError("INVALID_METADATA", "saved_at must include a timezone")
        text = selected.isoformat()
    elif isinstance(selected, str):
        text = selected
    else:
        raise SessionSnapshotError("INVALID_METADATA", "clock must return datetime or text")
    try:
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError as error:
        raise SessionSnapshotError("INVALID_METADATA", "saved_at is not ISO-8601") from error
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise SessionSnapshotError("INVALID_METADATA", "saved_at must include a timezone")
    return text


def save_session_snapshot(
    state: AgentState,
    path: os.PathLike[str] | str,
    *,
    revision: int = 1,
    saved_at: str | datetime | None = None,
    clock: Callable[[], str | datetime] | None = None,
) -> SessionSnapshotMetadata:
    """Atomically save one completed turn's immutable work context."""

    if not isinstance(state, AgentState):
        raise TypeError("state must be an AgentState")
    if type(revision) is not int or revision < 0:
        raise SessionSnapshotError("INVALID_METADATA", "revision must be a non-negative integer")
    target = _snapshot_path(path, for_write=True)
    timestamp = _saved_at(saved_at, clock)
    signed: dict[str, object] = {
        "schema": SNAPSHOT_SCHEMA,
        "version": SNAPSHOT_VERSION,
        "session_id": state.session_id,
        "saved_at": timestamp,
        "revision": revision,
        "state": _encode(state),
    }
    digest = hashlib.sha256(_canonical(signed)).hexdigest()
    payload = _canonical({**signed, "digest": digest}) + b"\n"

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
        raise SessionSnapshotError("SNAPSHOT_WRITE_FAILED", str(error)) from error
    finally:
        if descriptor is not None:
            os.close(descriptor)
        if temporary is not None:
            try:
                os.unlink(temporary)
            except FileNotFoundError:
                pass

    return SessionSnapshotMetadata(
        SNAPSHOT_SCHEMA, SNAPSHOT_VERSION, state.session_id, timestamp, revision
    )


def resume_session_snapshot(
    path: os.PathLike[str] | str,
    expected_session_id: str,
) -> SessionResumeResult:
    """Restore a new State from one explicitly selected same-session snapshot."""

    if not isinstance(expected_session_id, str) or not expected_session_id.strip():
        raise SessionSnapshotError("INVALID_SESSION", "expected_session_id must be non-empty text")
    source = _snapshot_path(path, for_write=False)
    try:
        payload = source.read_bytes().decode("utf-8", errors="strict")
    except UnicodeDecodeError as error:
        raise SessionSnapshotError("INVALID_UTF8", "snapshot is not strict UTF-8") from error
    except OSError as error:
        raise SessionSnapshotError("SNAPSHOT_READ_FAILED", str(error)) from error
    root = _exact_object(_strict_json(payload), _ROOT_FIELDS, "snapshot")
    signed = {key: root[key] for key in _SIGNED_FIELDS}
    digest = root["digest"]
    if type(digest) is not str or len(digest) != 64:
        raise SessionSnapshotError("INVALID_DIGEST", "snapshot digest is malformed")
    expected_digest = hashlib.sha256(_canonical(signed)).hexdigest()
    if not hmac.compare_digest(digest, expected_digest):
        raise SessionSnapshotError("TAMPERED", "snapshot digest does not match payload")
    if root["schema"] != SNAPSHOT_SCHEMA:
        raise SessionSnapshotError("UNKNOWN_SCHEMA", "snapshot schema is not supported")
    if type(root["version"]) is not int or root["version"] != SNAPSHOT_VERSION:
        raise SessionSnapshotError("UNKNOWN_VERSION", "snapshot version is not supported")
    if type(root["session_id"]) is not str or not root["session_id"].strip():
        raise SessionSnapshotError("INVALID_SESSION", "snapshot session_id is invalid")
    if root["session_id"] != expected_session_id:
        raise SessionSnapshotError("CROSS_SESSION", "snapshot belongs to another session")
    if type(root["revision"]) is not int or root["revision"] < 0:
        raise SessionSnapshotError("INVALID_METADATA", "snapshot revision is invalid")
    timestamp = _saved_at(root["saved_at"], None)
    state = _decode(root["state"])
    if not isinstance(state, AgentState):
        raise SessionSnapshotError("INVALID_STATE", "snapshot root is not AgentState")
    if state.session_id != root["session_id"]:
        raise SessionSnapshotError("SESSION_MISMATCH", "State and snapshot session identities differ")
    # Re-construct once more so callers never receive an object shared with codec internals.
    restored = AgentState(**{item.name: getattr(state, item.name) for item in fields(AgentState)})
    metadata = SessionSnapshotMetadata(
        SNAPSHOT_SCHEMA,
        SNAPSHOT_VERSION,
        root["session_id"],
        timestamp,
        root["revision"],
    )
    return SessionResumeResult(restored, metadata)
