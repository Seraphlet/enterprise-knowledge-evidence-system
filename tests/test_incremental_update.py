import json
import unittest
from datetime import datetime, timedelta, timezone

from knowledge_system.incremental_update import (
    UPDATE_RESULT_SCHEMA,
    IncrementalUpdateResultError,
    IncrementalUpdaterResult,
    UpdateAction,
    UpdateErrorType,
    UpdateStage,
    incremental_update_result_from_json,
    incremental_update_result_to_json,
    synchronize_index_on_startup,
)
from knowledge_system.system_state import IndexStatus, create_system_state


NOW = datetime(2026, 10, 7, 15, 0, tzinfo=timezone(timedelta(hours=8)))
STEPS = ("keyword", "vector", "manifest")


class MemoryStore:
    def __init__(self, state, *, fail_on_save=None):
        self.state = state
        self.saved = []
        self.fail_on_save = fail_on_save

    def load(self):
        return self.state

    def save(self, state):
        call = len(self.saved) + 1
        if self.fail_on_save == call:
            raise OSError(f"save {call} failed")
        self.saved.append(state)
        self.state = state


def completed(target):
    return IncrementalUpdaterResult(target, True, STEPS, STEPS)


class IncrementalUpdateTests(unittest.TestCase):
    def run_sync(self, store, source, updater):
        return synchronize_index_on_startup(
            state_store=store,
            source_commit_reader=lambda: source,
            incremental_updater=updater,
            clock=lambda: NOW,
        )

    def test_synced_startup_persists_check_and_skips_updater(self):
        store = MemoryStore(create_system_state("A", indexed_commit="A", checked_at=NOW))
        calls = []
        result = self.run_sync(store, "A", lambda old, target: calls.append((old, target)))
        self.assertIs(result.action, UpdateAction.SKIPPED)
        self.assertIs(result.status, IndexStatus.SYNCED)
        self.assertFalse(result.updater_invoked)
        self.assertEqual(calls, [])
        self.assertEqual(len(store.saved), 1)

    def test_first_build_and_a_to_b_success_call_updater_once(self):
        for initial, source, expected_old in (
            (create_system_state("A", checked_at=NOW), "A", None),
            (create_system_state("A", indexed_commit="A", checked_at=NOW), "B", "A"),
        ):
            with self.subTest(source=source, old=expected_old):
                store = MemoryStore(initial)
                calls = []

                def updater(old, target):
                    calls.append((old, target))
                    return completed(target)

                result = self.run_sync(store, source, updater)
                self.assertIs(result.action, UpdateAction.UPDATED)
                self.assertEqual(calls, [(expected_old, source)])
                self.assertEqual(store.state.indexed_commit, source)
                self.assertIs(store.state.index_status, IndexStatus.SYNCED)
                self.assertEqual(len(store.saved), 2)

    def test_exception_and_explicit_failure_preserve_indexed_commit(self):
        cases = (
            (lambda old, target: (_ for _ in ()).throw(RuntimeError("boom")), UpdateErrorType.UPDATER_EXCEPTION),
            (
                lambda old, target: IncrementalUpdaterResult(
                    target, False, STEPS, ("keyword",), "VECTOR_FAILED"
                ),
                UpdateErrorType.UPDATE_FAILED,
            ),
        )
        for updater, expected in cases:
            with self.subTest(expected=expected):
                store = MemoryStore(create_system_state("A", indexed_commit="A", checked_at=NOW))
                result = self.run_sync(store, "B", updater)
                self.assertIs(result.error_type, expected)
                self.assertEqual(result.indexed_commit_after, "A")
                self.assertEqual(store.state.indexed_commit, "A")
                self.assertIs(store.state.index_status, IndexStatus.OUT_OF_SYNC)

    def test_incomplete_and_wrong_target_never_advance(self):
        cases = (
            (
                lambda old, target: IncrementalUpdaterResult(
                    target, True, STEPS, ("keyword", "vector")
                ),
                UpdateErrorType.UPDATE_INCOMPLETE,
            ),
            (lambda old, target: completed("C"), UpdateErrorType.TARGET_MISMATCH),
            (lambda old, target: {"success": True}, UpdateErrorType.INVALID_UPDATER_RESULT),
        )
        for updater, expected in cases:
            with self.subTest(expected=expected):
                store = MemoryStore(create_system_state("A", indexed_commit="A", checked_at=NOW))
                result = self.run_sync(store, "B", updater)
                self.assertIs(result.action, UpdateAction.FAILED)
                self.assertIs(result.error_type, expected)
                self.assertEqual(store.state.indexed_commit, "A")

    def test_final_save_failure_does_not_claim_synced(self):
        store = MemoryStore(
            create_system_state("A", indexed_commit="A", checked_at=NOW), fail_on_save=2
        )
        result = self.run_sync(store, "B", lambda old, target: completed(target))
        self.assertIs(result.action, UpdateAction.FAILED)
        self.assertIs(result.error_type, UpdateErrorType.FINAL_PERSIST_FAILED)
        self.assertIs(result.stage, UpdateStage.SAVE_SYNC)
        self.assertIs(result.status, IndexStatus.OUT_OF_SYNC)
        self.assertEqual(result.indexed_commit_after, "A")
        self.assertEqual(store.state.indexed_commit, "A")

    def test_check_save_failure_never_calls_updater(self):
        store = MemoryStore(
            create_system_state("A", indexed_commit="A", checked_at=NOW), fail_on_save=1
        )
        calls = []
        result = self.run_sync(store, "B", lambda old, target: calls.append((old, target)))
        self.assertIs(result.error_type, UpdateErrorType.CHECK_PERSIST_FAILED)
        self.assertFalse(result.updater_invoked)
        self.assertEqual(calls, [])
        self.assertEqual(store.state.source_commit, "A")

    def test_failed_update_can_retry_then_becomes_idempotent(self):
        store = MemoryStore(create_system_state("A", indexed_commit="A", checked_at=NOW))
        calls = []

        def fail(old, target):
            calls.append((old, target, "fail"))
            return IncrementalUpdaterResult(target, False, STEPS, ("keyword",), "FAILED")

        first = self.run_sync(store, "B", fail)

        def pass_update(old, target):
            calls.append((old, target, "pass"))
            return completed(target)

        second = self.run_sync(store, "B", pass_update)
        third = self.run_sync(store, "B", lambda *_: self.fail("must skip"))
        self.assertIs(first.action, UpdateAction.FAILED)
        self.assertIs(second.action, UpdateAction.UPDATED)
        self.assertIs(third.action, UpdateAction.SKIPPED)
        self.assertEqual(calls, [("A", "B", "fail"), ("A", "B", "pass")])

    def test_source_failure_is_deterministic_and_serialization_is_stable(self):
        store = MemoryStore(create_system_state("A", indexed_commit="A", checked_at=NOW))
        result = self.run_sync(store, "bad commit", lambda *_: self.fail("must not call"))
        self.assertIs(result.error_type, UpdateErrorType.SOURCE_READ_FAILED)
        self.assertEqual(result.error_code, "INVALID_COMMIT")
        first = incremental_update_result_to_json(result)
        self.assertEqual(first, incremental_update_result_to_json(result))
        payload = json.loads(first)
        self.assertEqual(payload["schema"], UPDATE_RESULT_SCHEMA)
        self.assertNotIn("trace", payload)
        self.assertNotIn("session", payload)
        self.assertNotIn("index_body", payload)

    def test_updater_result_rejects_ambiguous_steps(self):
        with self.assertRaisesRegex(ValueError, "subset"):
            IncrementalUpdaterResult("A", True, ("keyword",), ("vector",))
        with self.assertRaisesRegex(ValueError, "duplicate"):
            IncrementalUpdaterResult("A", True, ("keyword", "keyword"), ("keyword",))

    def test_result_json_round_trip_is_equal_and_byte_stable(self):
        scenarios = (
            self.run_sync(
                MemoryStore(create_system_state("A", indexed_commit="A", checked_at=NOW)),
                "A",
                lambda *_: self.fail("must skip"),
            ),
            self.run_sync(
                MemoryStore(create_system_state("A", indexed_commit="A", checked_at=NOW)),
                "B",
                lambda old, target: completed(target),
            ),
            self.run_sync(
                MemoryStore(create_system_state("A", indexed_commit="A", checked_at=NOW)),
                "B",
                lambda *_: (_ for _ in ()).throw(RuntimeError("boom")),
            ),
        )
        for result in scenarios:
            with self.subTest(action=result.action):
                payload = incremental_update_result_to_json(result)
                rebuilt = incremental_update_result_from_json(payload.encode("utf-8"))
                self.assertEqual(rebuilt, result)
                self.assertEqual(incremental_update_result_to_json(rebuilt), payload)

    def test_result_json_rejects_malformed_shape_types_and_enums(self):
        result = self.run_sync(
            MemoryStore(create_system_state("A", indexed_commit="A", checked_at=NOW)),
            "A",
            lambda *_: self.fail("must skip"),
        )
        valid = json.loads(incremental_update_result_to_json(result))
        cases = {
            "invalid json": ("{", "INVALID_JSON"),
            "invalid utf8": (b"\xff", "INVALID_UTF8"),
            "duplicate": (
                incremental_update_result_to_json(result).replace(
                    '"action":"SKIPPED"', '"action":"SKIPPED","action":"SKIPPED"'
                ),
                "DUPLICATE_KEY",
            ),
            "missing": ({key: value for key, value in valid.items() if key != "stage"}, "INVALID_FIELDS"),
            "unknown": ({**valid, "extra": 1}, "INVALID_FIELDS"),
            "schema": ({**valid, "schema": "other"}, "UNKNOWN_SCHEMA"),
            "version bool": ({**valid, "schema_version": True}, "UNKNOWN_VERSION"),
            "action": ({**valid, "action": "OTHER"}, "INVALID_ACTION"),
            "status": ({**valid, "status": "OTHER"}, "INVALID_STATUS"),
            "stage": ({**valid, "stage": 1}, "INVALID_STAGE"),
            "error type": ({**valid, "error_type": "OTHER"}, "INVALID_ERROR_TYPE"),
            "updater bool": ({**valid, "updater_invoked": 1}, "INVALID_UPDATER_INVOKED"),
            "commit": ({**valid, "source_commit_after": "bad commit"}, "INVALID_COMMIT"),
        }
        for name, (payload, code) in cases.items():
            with self.subTest(name=name):
                if isinstance(payload, dict):
                    payload = json.dumps(payload)
                with self.assertRaises(IncrementalUpdateResultError) as caught:
                    incremental_update_result_from_json(payload)
                self.assertEqual(caught.exception.code, code)

    def test_result_json_rejects_impossible_cross_field_combinations(self):
        skipped = self.run_sync(
            MemoryStore(create_system_state("A", indexed_commit="A", checked_at=NOW)),
            "A",
            lambda *_: self.fail("must skip"),
        )
        failed = self.run_sync(
            MemoryStore(create_system_state("A", indexed_commit="A", checked_at=NOW)),
            "B",
            lambda *_: (_ for _ in ()).throw(RuntimeError("boom")),
        )
        skipped_payload = json.loads(incremental_update_result_to_json(skipped))
        failed_payload = json.loads(incremental_update_result_to_json(failed))
        mutations = (
            {**skipped_payload, "updater_invoked": True},
            {**skipped_payload, "indexed_commit_after": "B"},
            {**skipped_payload, "error_type": "UPDATE_FAILED", "error_code": "FAILED"},
            {**failed_payload, "stage": "CHECK_SOURCE"},
            {**failed_payload, "status": "SYNCED"},
            {**failed_payload, "indexed_commit_after": "B"},
            {**failed_payload, "error_code": None},
        )
        for payload in mutations:
            with self.subTest(payload=payload):
                with self.assertRaises(IncrementalUpdateResultError) as caught:
                    incremental_update_result_from_json(json.dumps(payload))
                self.assertEqual(caught.exception.code, "INCONSISTENT_RESULT")


if __name__ == "__main__":
    unittest.main()
