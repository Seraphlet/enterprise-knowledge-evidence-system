import json
import tempfile
import unittest
from dataclasses import FrozenInstanceError
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import patch

import knowledge_system.system_state as system_state_module
from knowledge_system.system_state import (
    SYSTEM_STATE_SCHEMA,
    IndexStatus,
    SystemState,
    SystemStateError,
    check_source_commit,
    create_system_state,
    load_system_state,
    save_system_state,
    status_for,
    system_state_from_json,
    system_state_to_json,
)


NOW = datetime(2026, 10, 7, 10, 30, tzinfo=timezone(timedelta(hours=8)))
LATER = datetime(2026, 10, 7, 11, 0, tzinfo=timezone.utc)


class SystemStateTests(unittest.TestCase):
    def test_initial_unbuilt_state_is_immutable_and_versions_are_separate(self):
        state = create_system_state("source-A", clock=lambda: NOW)
        self.assertEqual(state.source_commit, "source-A")
        self.assertIsNone(state.indexed_commit)
        self.assertIs(state.index_status, IndexStatus.NOT_BUILT)
        self.assertEqual(state.last_check_at, NOW)
        self.assertEqual(state.schema, SYSTEM_STATE_SCHEMA)
        with self.assertRaises(FrozenInstanceError):
            state.source_commit = "source-B"

    def test_status_is_deterministic_for_unbuilt_synced_and_out_of_sync(self):
        self.assertIs(status_for("A", None), IndexStatus.NOT_BUILT)
        self.assertIs(status_for("A", "A"), IndexStatus.SYNCED)
        self.assertIs(status_for("B", "A"), IndexStatus.OUT_OF_SYNC)

        synced = create_system_state("A", indexed_commit="A", checked_at=NOW)
        changed = check_source_commit(synced, "B", checked_at=LATER)
        unchanged = check_source_commit(synced, "A", checked_at=LATER)
        self.assertIs(changed.index_status, IndexStatus.OUT_OF_SYNC)
        self.assertEqual(changed.indexed_commit, "A")
        self.assertIs(unchanged.index_status, IndexStatus.SYNCED)
        self.assertEqual(synced.last_check_at, NOW)

    def test_check_does_not_advance_an_unbuilt_or_old_index(self):
        unbuilt = create_system_state("A", checked_at=NOW)
        checked = check_source_commit(unbuilt, "B", checked_at=LATER)
        self.assertIsNone(checked.indexed_commit)
        self.assertIs(checked.index_status, IndexStatus.NOT_BUILT)

        old = create_system_state("A", indexed_commit="old", checked_at=NOW)
        checked_old = check_source_commit(old, "B", checked_at=LATER)
        self.assertEqual(checked_old.indexed_commit, "old")
        self.assertIs(checked_old.index_status, IndexStatus.OUT_OF_SYNC)

    def test_clock_is_injected_and_timezone_is_required(self):
        self.assertEqual(create_system_state("A", clock=lambda: NOW).last_check_at, NOW)
        with self.assertRaisesRegex(TypeError, "mutually exclusive"):
            create_system_state("A", checked_at=NOW, clock=lambda: NOW)
        with self.assertRaisesRegex(SystemStateError, "INVALID_TIME"):
            create_system_state("A", checked_at=datetime(2026, 1, 1))
        with self.assertRaisesRegex(SystemStateError, "INVALID_TIME"):
            create_system_state("A", clock=lambda: "2026-01-01Z")

    def test_invalid_commits_are_rejected(self):
        for value in ("", " ", "A B", "\n", 1, True):
            with self.subTest(value=value), self.assertRaisesRegex(
                SystemStateError, "INVALID_COMMIT"
            ):
                create_system_state(value, checked_at=NOW)

    def test_inconsistent_or_illegal_direct_state_is_rejected(self):
        with self.assertRaisesRegex(SystemStateError, "INCONSISTENT_STATE"):
            SystemState("A", "A", IndexStatus.OUT_OF_SYNC, NOW)
        with self.assertRaisesRegex(SystemStateError, "INVALID_STATUS"):
            SystemState("A", None, "NOT_BUILT", NOW)
        with self.assertRaisesRegex(SystemStateError, "UNKNOWN_VERSION"):
            SystemState("A", None, IndexStatus.NOT_BUILT, NOW, schema_version=2)

    def test_strict_stable_json_round_trip(self):
        state = create_system_state("source-B", indexed_commit="source-A", checked_at=NOW)
        first = system_state_to_json(state)
        self.assertEqual(first, system_state_to_json(state))
        self.assertTrue(first.endswith("\n"))
        self.assertEqual(system_state_from_json(first), state)
        self.assertEqual(
            set(json.loads(first)),
            {
                "schema",
                "schema_version",
                "source_commit",
                "indexed_commit",
                "index_status",
                "last_check_at",
            },
        )

    def test_json_rejects_unknown_missing_illegal_and_naive_fields(self):
        changes = (
            (lambda item: item.update({"extra": True}), "INVALID_FIELDS"),
            (lambda item: item.pop("source_commit"), "INVALID_FIELDS"),
            (lambda item: item.update({"schema": "unknown"}), "UNKNOWN_SCHEMA"),
            (lambda item: item.update({"schema_version": 99}), "UNKNOWN_VERSION"),
            (lambda item: item.update({"index_status": "UNKNOWN"}), "INVALID_STATUS"),
            (lambda item: item.update({"last_check_at": "2026-01-01T00:00:00"}), "INVALID_TIME"),
        )
        for change, code in changes:
            with self.subTest(code=code):
                value = json.loads(
                    system_state_to_json(create_system_state("A", checked_at=NOW))
                )
                change(value)
                with self.assertRaisesRegex(SystemStateError, code):
                    system_state_from_json(json.dumps(value))

    def test_json_rejects_wrong_types_duplicate_keys_and_corruption(self):
        with self.assertRaisesRegex(SystemStateError, "DUPLICATE_KEY"):
            system_state_from_json('{"schema":"a","schema":"b"}')
        with self.assertRaisesRegex(SystemStateError, "INVALID_JSON"):
            system_state_from_json("{broken")
        value = json.loads(system_state_to_json(create_system_state("A", checked_at=NOW)))
        value["schema_version"] = True
        with self.assertRaisesRegex(SystemStateError, "UNKNOWN_VERSION"):
            system_state_from_json(json.dumps(value))

    def test_atomic_save_overwrite_and_replace_failure_preserves_original(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "system-state.json"
            first = create_system_state("A", checked_at=NOW)
            second = create_system_state("B", indexed_commit="A", checked_at=LATER)
            save_system_state(first, path)
            save_system_state(second, path)
            self.assertEqual(load_system_state(path), second)
            original = path.read_bytes()

            with patch.object(
                system_state_module.os,
                "replace",
                side_effect=OSError("replace failed"),
            ), self.assertRaisesRegex(SystemStateError, "STATE_WRITE_FAILED"):
                save_system_state(first, path)
            self.assertEqual(path.read_bytes(), original)
            self.assertEqual(list(Path(folder).glob("*.tmp")), [])

    def test_load_is_read_only_and_rejects_bad_files(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "system-state.json"
            state = create_system_state("A", indexed_commit="A", checked_at=NOW)
            save_system_state(state, path)
            before = path.read_bytes()
            with patch.object(
                system_state_module,
                "check_source_commit",
                side_effect=AssertionError("load performed check"),
            ):
                self.assertEqual(load_system_state(path), state)
            self.assertEqual(path.read_bytes(), before)

            path.write_bytes(b"\xff")
            with self.assertRaisesRegex(SystemStateError, "INVALID_UTF8"):
                load_system_state(path)
            path.write_text("[]", encoding="utf-8")
            with self.assertRaisesRegex(SystemStateError, "INVALID_FIELDS"):
                load_system_state(path)
            with self.assertRaisesRegex(SystemStateError, "INVALID_PATH"):
                load_system_state(Path(folder) / "missing.json")


if __name__ == "__main__":
    unittest.main()
