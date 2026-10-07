"""Focused tests for bounded deterministic Evidence recovery orchestration."""

import unittest
from dataclasses import fields

from knowledge_system import (
    Evidence,
    EvidenceQualitySignal,
    EvidenceQualitySignalType,
    KnowledgeLineage,
    RecoveryAction,
    RecoveryExecutionResult,
    RecoveryResult,
    RecoveryRoundTrace,
    RecoveryStatus,
    SourceReference,
    run_limited_recovery,
)


def _evidence(
    evidence_id: str = "evidence-original",
    *,
    section: str | None = "审核规则",
    source_position: str | None = "blocks:4-5",
    parent_section_id: str | None = "section-audit",
    url: str | None = "https://kb.example/rules",
) -> Evidence:
    return Evidence(
        evidence_id=evidence_id,
        unit_id=f"unit-{evidence_id}",
        original_content="正式规则原文",
        source_reference=SourceReference(
            name="审核规则.html",
            url=url,
            section=section,
        ),
        lineage=KnowledgeLineage(
            document_id="doc-rules",
            parser_version="p1",
            chunker_version="c1",
            processing_version="v1",
            source_position=source_position,
            parent_section_id=parent_section_id,
        ),
    )


def _signal(
    evidence: Evidence, signal_type: EvidenceQualitySignalType
) -> EvidenceQualitySignal:
    return EvidenceQualitySignal(
        signal_type=signal_type,
        reason=f"test_{signal_type.value.casefold()}",
        evidence_id=evidence.evidence_id,
        source_reference=evidence.source_reference,
        lineage=evidence.lineage,
    )


class LimitedRecoveryTest(unittest.TestCase):
    def test_no_signal_is_no_op_without_executor(self) -> None:
        evidence = _evidence()

        result = run_limited_recovery(evidence, ())

        self.assertEqual(result.status, RecoveryStatus.NOT_NEEDED)
        self.assertEqual(result.trace, ())
        self.assertEqual(result.recovered_evidence, ())
        self.assertFalse(result.verification_required)

    def test_each_signal_routes_only_to_source_grounded_allowed_action(self) -> None:
        cases = (
            (
                _evidence(),
                EvidenceQualitySignalType.POSSIBLY_INCOMPLETE,
                RecoveryAction.NEIGHBOR_EXPANSION,
                "doc-rules:blocks:4-5",
            ),
            (
                _evidence(source_position=None),
                EvidenceQualitySignalType.MISSING_CONTEXT,
                RecoveryAction.PARENT_UNIT,
                "section-audit",
            ),
            (
                _evidence(source_position=None, parent_section_id=None),
                EvidenceQualitySignalType.POSSIBLY_AMBIGUOUS,
                RecoveryAction.SECTION_EXPANSION,
                "审核规则.html:审核规则",
            ),
            (
                _evidence(
                    section=None,
                    source_position=None,
                    parent_section_id=None,
                ),
                EvidenceQualitySignalType.BROKEN_CONTEXT,
                RecoveryAction.RE_RETRIEVAL,
                "doc-rules",
            ),
            (
                _evidence(
                    section=None,
                    source_position=None,
                    parent_section_id=None,
                ),
                EvidenceQualitySignalType.POSSIBLY_AMBIGUOUS,
                RecoveryAction.ORIGINAL_SOURCE_CONTEXT,
                "https://kb.example/rules",
            ),
        )
        for evidence, signal_type, action, target_id in cases:
            with self.subTest(action=action):
                seen = []

                def executor(request):
                    seen.append(request)
                    return RecoveryExecutionResult((), ())

                result = run_limited_recovery(
                    evidence, (_signal(evidence, signal_type),), executor
                )

                self.assertEqual(result.status, RecoveryStatus.RESOLVED)
                self.assertEqual(len(seen), 1)
                self.assertEqual(seen[0].action, action)
                self.assertEqual(seen[0].target.target_id, target_id)
                self.assertIn(action, tuple(RecoveryAction))

    def test_one_and_two_round_caps_use_unique_ordered_actions(self) -> None:
        evidence = _evidence()
        signal = _signal(
            evidence, EvidenceQualitySignalType.POSSIBLY_INCOMPLETE
        )

        def unresolved(request):
            return RecoveryExecutionResult((), (signal,))

        one = run_limited_recovery(
            evidence, (signal,), unresolved, max_rounds=1
        )
        two = run_limited_recovery(
            evidence, (signal,), unresolved, max_rounds=2
        )

        self.assertEqual(len(one.trace), 1)
        self.assertEqual(len(two.trace), 2)
        self.assertEqual(
            tuple(item.action for item in two.trace),
            (
                RecoveryAction.NEIGHBOR_EXPANSION,
                RecoveryAction.PARENT_UNIT,
            ),
        )
        self.assertEqual(len(set(two.attempted_actions)), 2)
        self.assertEqual(two.status, RecoveryStatus.UNRESOLVED)
        self.assertTrue(two.verification_required)
        self.assertEqual(two.unresolved_reason, "round_budget_exhausted")

    def test_resolved_signals_stop_before_remaining_budget(self) -> None:
        evidence = _evidence()
        signal = _signal(
            evidence, EvidenceQualitySignalType.POSSIBLY_INCOMPLETE
        )
        calls = []

        def resolved(request):
            calls.append(request)
            return RecoveryExecutionResult((_evidence("evidence-recovered"),), ())

        result = run_limited_recovery(
            evidence, (signal,), resolved, max_rounds=2
        )

        self.assertEqual(len(calls), 1)
        self.assertEqual(len(result.trace), 1)
        self.assertEqual(result.status, RecoveryStatus.RESOLVED)
        self.assertFalse(result.verification_required)

    def test_duplicate_signals_actions_and_evidence_are_deduplicated(self) -> None:
        evidence = _evidence()
        signal = _signal(
            evidence, EvidenceQualitySignalType.POSSIBLY_INCOMPLETE
        )
        recovered = _evidence("evidence-recovered")
        calls = []

        def executor(request):
            calls.append(request)
            if request.round_number == 1:
                return RecoveryExecutionResult(
                    (recovered, recovered), (signal, signal)
                )
            return RecoveryExecutionResult((recovered,), ())

        result = run_limited_recovery(
            evidence, (signal, signal), executor, max_rounds=2
        )

        self.assertEqual(len(calls[0].trigger_signals), 1)
        self.assertEqual(len(result.recovered_evidence), 1)
        self.assertIs(result.recovered_evidence[0], recovered)
        self.assertEqual(result.trace[0].recovered_evidence, (recovered,))
        self.assertEqual(result.trace[1].recovered_evidence, ())
        self.assertEqual(len(set(result.attempted_actions)), 2)
        self.assertEqual(result.status, RecoveryStatus.RESOLVED)

    def test_failed_recovery_is_unresolved_and_requires_verification(self) -> None:
        evidence = _evidence()
        signal = _signal(evidence, EvidenceQualitySignalType.BROKEN_CONTEXT)

        def failed(request):
            return RecoveryExecutionResult(
                (), request.trigger_signals, "target_not_available"
            )

        result = run_limited_recovery(
            evidence, (signal,), failed, max_rounds=1
        )

        self.assertEqual(result.status, RecoveryStatus.UNRESOLVED)
        self.assertTrue(result.verification_required)
        self.assertEqual(result.unresolved_reason, "round_budget_exhausted")
        self.assertEqual(result.trace[0].failure_reason, "target_not_available")
        self.assertEqual(result.remaining_signals, (signal,))

    def test_trace_preserves_provenance_inputs_and_has_no_diagnosis(self) -> None:
        evidence = _evidence()
        signal = _signal(evidence, EvidenceQualitySignalType.MISSING_CONTEXT)
        recovered = _evidence("evidence-recovered")
        snapshots = (evidence, signal, recovered)

        def executor(request):
            self.assertIs(request.target.source_reference, evidence.source_reference)
            self.assertIs(request.target.lineage, evidence.lineage)
            return RecoveryExecutionResult((recovered,), ())

        result = run_limited_recovery(evidence, (signal,), executor)

        self.assertEqual((evidence, signal, recovered), snapshots)
        self.assertIs(result.original_evidence, evidence)
        self.assertIs(result.trace[0].recovered_evidence[0], recovered)
        result_fields = {item.name for item in fields(RecoveryResult)}
        trace_fields = {item.name for item in fields(RecoveryRoundTrace)}
        for prohibited in (
            "diagnosis",
            "problem_type",
            "knowledge_gap",
            "retrieval_problem",
            "processing_problem",
        ):
            self.assertNotIn(prohibited, result_fields)
            self.assertNotIn(prohibited, trace_fields)

    def test_round_limit_rejects_unbounded_values(self) -> None:
        evidence = _evidence()
        for invalid in (0, 3, True):
            with self.subTest(max_rounds=invalid):
                with self.assertRaisesRegex(ValueError, "1 or 2"):
                    run_limited_recovery(evidence, (), max_rounds=invalid)


if __name__ == "__main__":
    unittest.main()
