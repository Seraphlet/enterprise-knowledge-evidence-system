"""Focused tests for append-only, non-authoritative employee feedback."""

import copy
import tempfile
import unittest
from dataclasses import FrozenInstanceError, fields, replace
from datetime import datetime, timezone
from pathlib import Path

from knowledge_system import (
    Evidence,
    EvidenceQualitySignal,
    EvidenceQualitySignalType,
    FeedbackAuthority,
    FeedbackLog,
    FeedbackLogError,
    FeedbackPurpose,
    FeedbackRecord,
    FeedbackValidationError,
    KnowledgeLineage,
    SourceReference,
    Trace,
    VerificationReason,
    build_verification_mode,
    create_feedback_record,
)


FIXED_TIME = datetime(2026, 10, 5, 8, 30, 0, tzinfo=timezone.utc)


def _evidence(evidence_id="evidence-1"):
    return Evidence(
        evidence_id=evidence_id,
        unit_id=f"unit-{evidence_id}",
        original_content="正式规则原文",
        source_reference=SourceReference(
            name="审核规则.html",
            url="https://kb.example/rules",
            section="审核规则",
        ),
        lineage=KnowledgeLineage(
            document_id="doc-rules",
            parser_version="p1",
            chunker_version="c1",
            processing_version="v1",
            source_position="blocks:2-3",
            parent_section_id="section-rules",
        ),
    )


def _verification(evidence):
    signal = EvidenceQualitySignal(
        signal_type=EvidenceQualitySignalType.POSSIBLY_AMBIGUOUS,
        reason="explicit_rule_ambiguity",
        evidence_id=evidence.evidence_id,
        source_reference=evidence.source_reference,
        lineage=evidence.lineage,
    )
    return build_verification_mode((evidence,), quality_signals=(signal,))


def _trace(trace_id="trace-1"):
    return Trace(
        trace_id=trace_id,
        raw_query="原始问题",
        retrieval_query="检索问题",
        candidate_evidence_ids=["evidence-1"],
        final_evidence_ids=["evidence-1"],
    )


def _record(
    feedback_id="feedback-1",
    feedback_text="  员工原话：这里可能不准确。\n请核验。  ",
):
    evidence = _evidence()
    trace = _trace()
    return create_feedback_record(
        feedback_text,
        target_evidence_ids=(evidence.evidence_id,),
        target_trace_id=trace.trace_id,
        verification_mode=_verification(evidence),
        authoritative_evidence=(evidence,),
        trace_context=(trace,),
        id_factory=lambda: feedback_id,
        clock=lambda: FIXED_TIME,
    )


class FeedbackLogTest(unittest.TestCase):
    def test_feedback_is_verbatim_observation_and_maintenance_signal(self) -> None:
        text = "  员工原话：阈值看起来不一致。\n不要改写。  "

        record = _record(feedback_text=text)

        self.assertEqual(record.feedback_text, text)
        self.assertEqual(record.authority, FeedbackAuthority.OBSERVATION)
        self.assertEqual(
            record.purpose, FeedbackPurpose.KNOWLEDGE_MAINTENANCE_SIGNAL
        )
        self.assertEqual(record.feedback_id, "feedback-1")
        self.assertEqual(record.created_at, FIXED_TIME.isoformat())

    def test_evidence_trace_and_verification_context_are_grounded(self) -> None:
        evidence = _evidence()
        verification = _verification(evidence)
        trace = _trace()

        record = create_feedback_record(
            "需要核验",
            target_evidence_ids=(evidence.evidence_id,),
            target_trace_id=trace.trace_id,
            verification_mode=verification,
            authoritative_evidence=(evidence,),
            trace_context=(trace,),
            id_factory=lambda: "feedback-grounded",
            clock=lambda: FIXED_TIME,
        )

        self.assertEqual(record.target_evidence_ids, ("evidence-1",))
        self.assertEqual(record.target_trace_id, "trace-1")
        self.assertEqual(
            record.verification_reasons,
            (VerificationReason.SOURCE_AMBIGUOUS,),
        )
        self.assertIn(
            "quality:POSSIBLY_AMBIGUOUS",
            record.verification_context_codes,
        )
        self.assertIn(
            "signal_reason:explicit_rule_ambiguity",
            record.verification_context_codes,
        )

    def test_unknown_empty_and_missing_targets_are_rejected(self) -> None:
        evidence = _evidence()
        trace = _trace()
        base = {
            "feedback_text": "核验",
            "authoritative_evidence": (evidence,),
            "trace_context": (trace,),
            "id_factory": lambda: "feedback-invalid",
            "clock": lambda: FIXED_TIME,
        }
        cases = (
            {"target_evidence_ids": ("unknown-evidence",)},
            {"target_evidence_ids": ("",)},
            {"target_trace_id": "unknown-trace"},
            {},
        )
        for kwargs in cases:
            with self.subTest(kwargs=kwargs):
                with self.assertRaises(FeedbackValidationError):
                    create_feedback_record(**base, **kwargs)

        with self.assertRaises(FeedbackValidationError):
            create_feedback_record(
                "   ",
                target_evidence_ids=(evidence.evidence_id,),
                authoritative_evidence=(evidence,),
                id_factory=lambda: "feedback-empty",
                clock=lambda: FIXED_TIME,
            )

    def test_trace_only_target_is_allowed_when_context_is_explicit(self) -> None:
        trace = _trace()

        record = create_feedback_record(
            "本次检索路径需要复查",
            target_trace_id=trace.trace_id,
            trace_context=(trace,),
            id_factory=lambda: "feedback-trace-only",
            clock=lambda: FIXED_TIME,
        )

        self.assertEqual(record.target_evidence_ids, ())
        self.assertEqual(record.target_trace_id, trace.trace_id)

    def test_unrelated_evidence_and_trace_are_rejected_without_record_or_log(self) -> None:
        evidence = _evidence("e-2")
        trace = Trace(
            trace_id="trace-1",
            raw_query="原始问题",
            retrieval_query="检索问题",
            candidate_evidence_ids=["e-1"],
            final_evidence_ids=["e-1"],
        )
        factory_calls = []
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "feedback.jsonl"

            with self.assertRaisesRegex(
                FeedbackValidationError, "does not belong"
            ):
                create_feedback_record(
                    "check",
                    target_evidence_ids=("e-2",),
                    target_trace_id="trace-1",
                    authoritative_evidence=(evidence,),
                    trace_context=(trace,),
                    id_factory=lambda: factory_calls.append(True) or "f-1",
                    clock=lambda: FIXED_TIME,
                )

            self.assertEqual(factory_calls, [])
            self.assertFalse(path.exists())

    def test_candidate_and_final_evidence_bindings_both_succeed(self) -> None:
        candidate = _evidence("e-candidate")
        final = _evidence("e-final")
        trace = Trace(
            trace_id="trace-binding",
            raw_query="原始问题",
            retrieval_query="检索问题",
            candidate_evidence_ids=[candidate.evidence_id],
            final_evidence_ids=[final.evidence_id],
        )

        for index, evidence in enumerate((candidate, final), start=1):
            with self.subTest(evidence=evidence.evidence_id):
                record = create_feedback_record(
                    "check",
                    target_evidence_ids=(evidence.evidence_id,),
                    target_trace_id=trace.trace_id,
                    authoritative_evidence=(evidence,),
                    trace_context=(trace,),
                    id_factory=lambda index=index: f"feedback-binding-{index}",
                    clock=lambda: FIXED_TIME,
                )
                self.assertEqual(
                    record.target_evidence_ids, (evidence.evidence_id,)
                )
                self.assertEqual(record.target_trace_id, trace.trace_id)

    def test_append_two_records_never_overwrites_and_preserves_read_order(self) -> None:
        first = _record("feedback-1", "第一条反馈")
        second = _record("feedback-2", "第二条反馈")
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "feedback.jsonl"
            log = FeedbackLog(path)

            log.append(first)
            first_bytes = path.read_bytes()
            log.append(second)
            final_bytes = path.read_bytes()

            self.assertTrue(final_bytes.startswith(first_bytes))
            self.assertEqual(log.read_all(), (first, second))
            self.assertEqual(len(final_bytes.splitlines()), 2)

    def test_duplicate_identity_is_rejected_without_file_change(self) -> None:
        first = _record("feedback-duplicate", "第一条")
        conflict = _record("feedback-duplicate", "冲突内容")
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "feedback.jsonl"
            log = FeedbackLog(path)
            log.append(first)
            before = path.read_bytes()

            for duplicate in (first, conflict):
                with self.subTest(text=duplicate.feedback_text):
                    with self.assertRaisesRegex(
                        FeedbackLogError, "duplicate"
                    ):
                        log.append(duplicate)
                    self.assertEqual(path.read_bytes(), before)

    def test_canonical_round_trip_is_stable(self) -> None:
        record = _record()

        encoded = record.to_json()
        decoded = FeedbackRecord.from_json(encoded)

        self.assertEqual(decoded, record)
        self.assertEqual(decoded.to_json(), encoded)
        self.assertEqual(
            tuple(sorted(record.target_evidence_ids)),
            record.target_evidence_ids,
        )

    def test_inputs_and_record_are_immutable(self) -> None:
        evidence = _evidence()
        verification = _verification(evidence)
        trace = _trace()
        snapshots = copy.deepcopy((evidence, verification, trace))

        record = create_feedback_record(
            "逐字反馈",
            target_evidence_ids=(evidence.evidence_id,),
            target_trace_id=trace.trace_id,
            verification_mode=verification,
            authoritative_evidence=(evidence,),
            trace_context=(trace,),
            id_factory=lambda: "feedback-immutable",
            clock=lambda: FIXED_TIME,
        )

        self.assertEqual((evidence, verification, trace), snapshots)
        with self.assertRaises(FrozenInstanceError):
            record.feedback_text = "改写"

    def test_no_formal_knowledge_or_answer_mutation_surface(self) -> None:
        record = _record()
        record_fields = {item.name for item in fields(type(record))}
        log_members = set(dir(FeedbackLog))
        for prohibited in (
            "knowledge_unit",
            "formal_knowledge",
            "commit",
            "update",
            "index",
            "answer",
            "decision",
            "llm_output",
            "session",
        ):
            self.assertNotIn(prohibited, record_fields)
            self.assertNotIn(prohibited, log_members)

    def test_bad_files_and_unavailable_parent_fail_explicitly(self) -> None:
        record = _record()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with self.assertRaises(FeedbackLogError):
                FeedbackLog(root / "feedback.txt")

            missing_parent = FeedbackLog(root / "missing" / "feedback.jsonl")
            with self.assertRaisesRegex(FeedbackLogError, "parent"):
                missing_parent.append(record)

            malformed_path = root / "malformed.jsonl"
            malformed_path.write_text("{not-json}\n", encoding="utf-8")
            malformed = FeedbackLog(malformed_path)
            with self.assertRaisesRegex(FeedbackLogError, "invalid record"):
                malformed.read_all()
            before = malformed_path.read_bytes()
            with self.assertRaises(FeedbackLogError):
                malformed.append(record)
            self.assertEqual(malformed_path.read_bytes(), before)

            invalid_utf8_path = root / "invalid-utf8.jsonl"
            invalid_utf8_path.write_bytes(b"\xff\xfe")
            with self.assertRaisesRegex(FeedbackLogError, "cannot be read"):
                FeedbackLog(invalid_utf8_path).read_all()

            duplicate_key_path = root / "duplicate-key.jsonl"
            duplicate_key_path.write_text(
                record.to_json().replace(
                    '"feedback_id":"feedback-1"',
                    '"feedback_id":"feedback-1","feedback_id":"feedback-2"',
                )
                + "\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(FeedbackLogError, "invalid record"):
                FeedbackLog(duplicate_key_path).read_all()


if __name__ == "__main__":
    unittest.main()
