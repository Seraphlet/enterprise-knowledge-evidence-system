"""Focused tests for deterministic recovery-based failure diagnosis."""

import unittest
from dataclasses import fields, replace

from knowledge_system import (
    DiagnosisBoundary,
    Evidence,
    EvidenceQualitySignal,
    EvidenceQualitySignalType,
    FailureDiagnosis,
    FailureDiagnosisType,
    KnowledgeLineage,
    RecoveryAction,
    RecoveryExecutionResult,
    RecoveryStatus,
    SourceFindingType,
    SourceFragment,
    SourceInspection,
    SourceReference,
    diagnose_failure,
    run_limited_recovery,
)


def _evidence(
    evidence_id: str,
    content: str,
    *,
    section: str | None = "审核规则",
) -> Evidence:
    return Evidence(
        evidence_id=evidence_id,
        unit_id=f"unit-{evidence_id}",
        original_content=content,
        source_reference=SourceReference(
            name="审核规则.html",
            url="https://kb.example/rules",
            section=section,
        ),
        lineage=KnowledgeLineage(
            document_id="doc-rules",
            parser_version="p1",
            chunker_version="c1",
            processing_version="v1",
            source_position="blocks:2-3",
            parent_section_id="section-audit",
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


def _retrieval_recovery():
    original = _evidence("e-original", "上下文不足")
    signal = _signal(original, EvidenceQualitySignalType.MISSING_CONTEXT)
    recovered = _evidence("e-parent", "父级中的完整规则")

    def executor(request):
        return RecoveryExecutionResult((recovered,), ())

    return (
        original,
        signal,
        recovered,
        run_limited_recovery(original, (signal,), executor),
    )


def _original_source_recovery(
    signal_type: EvidenceQualitySignalType,
    source_content: str,
):
    original = _evidence("e-original", "派生片段", section=None)
    signal = _signal(original, signal_type)
    source = _evidence("e-source", source_content, section=None)

    def executor(request):
        if request.action is RecoveryAction.ORIGINAL_SOURCE_CONTEXT:
            return RecoveryExecutionResult((source,), ())
        return RecoveryExecutionResult((), (signal,))

    recovery = run_limited_recovery(
        original, (signal,), executor, max_rounds=2
    )
    return original, signal, source, recovery


def _inspection(
    evidence: Evidence,
    finding_type: SourceFindingType,
    *,
    fragments: tuple[SourceFragment, ...] = (),
    complete: bool = True,
) -> SourceInspection:
    return SourceInspection(
        finding_type=finding_type,
        evidence_id=evidence.evidence_id,
        source_reference=evidence.source_reference,
        lineage=evidence.lineage,
        reason=f"checked_{finding_type.value.casefold()}",
        complete_scope_checked=complete,
        source_fragments=fragments,
    )


class FailureDiagnosisTest(unittest.TestCase):
    def test_retrieval_problem_requires_successful_context_recovery(self) -> None:
        original, signal, recovered, recovery = _retrieval_recovery()

        result = diagnose_failure(recovery)

        self.assertEqual(
            result.diagnosis_type, FailureDiagnosisType.RETRIEVAL_PROBLEM
        )
        self.assertFalse(result.verification_required)
        self.assertFalse(result.knowledge_gap_allowed)
        self.assertEqual(result.signals, (signal,))
        self.assertIn(recovered, result.evidence)

        broken_original = _evidence("e-broken", "错误候选", section=None)
        broken_signal = _signal(
            broken_original, EvidenceQualitySignalType.BROKEN_CONTEXT
        )
        correct = _evidence("e-correct", "重新检索的正式规则")

        def reretrieve(request):
            self.assertEqual(request.action, RecoveryAction.RE_RETRIEVAL)
            return RecoveryExecutionResult((correct,), ())

        reretrieved = run_limited_recovery(
            broken_original, (broken_signal,), reretrieve
        )
        reretrieval_result = diagnose_failure(reretrieved)
        self.assertEqual(
            reretrieval_result.diagnosis_type,
            FailureDiagnosisType.RETRIEVAL_PROBLEM,
        )
        self.assertFalse(reretrieval_result.knowledge_gap_allowed)

    def test_processing_problem_requires_complete_original_source_content(self) -> None:
        original, signal, source, recovery = _original_source_recovery(
            EvidenceQualitySignalType.BROKEN_CONTEXT, "完整规则内容"
        )
        fragment = SourceFragment("完整规则内容", 0, len("完整规则内容"))
        inspection = _inspection(
            source,
            SourceFindingType.REQUIRED_CONTENT_PRESENT,
            fragments=(fragment,),
        )

        result = diagnose_failure(
            recovery, source_inspections=(inspection,)
        )

        self.assertEqual(
            result.diagnosis_type, FailureDiagnosisType.PROCESSING_PROBLEM
        )
        self.assertFalse(result.verification_required)
        self.assertFalse(result.knowledge_gap_allowed)
        self.assertEqual(result.source_inspections, (inspection,))

        section_original = _evidence("e-section", "不完整片段")
        section_original = replace(
            section_original,
            lineage=replace(
                section_original.lineage,
                source_position=None,
                parent_section_id=None,
            ),
        )
        section_signal = _signal(
            section_original,
            EvidenceQualitySignalType.POSSIBLY_INCOMPLETE,
        )
        expanded = _evidence("e-expanded", "章节中的完整内容")

        def section_executor(request):
            self.assertEqual(request.action, RecoveryAction.SECTION_EXPANSION)
            return RecoveryExecutionResult((expanded,), ())

        section_recovery = run_limited_recovery(
            section_original, (section_signal,), section_executor
        )
        section_inspection = _inspection(
            expanded,
            SourceFindingType.REQUIRED_CONTENT_PRESENT,
            fragments=(SourceFragment("章节中的完整内容", 0, 8),),
        )
        section_result = diagnose_failure(
            section_recovery, source_inspections=(section_inspection,)
        )
        self.assertEqual(
            section_result.diagnosis_type,
            FailureDiagnosisType.PROCESSING_PROBLEM,
        )
        self.assertFalse(section_result.knowledge_gap_allowed)

    def test_source_ambiguity_and_missing_require_strict_original_source_gate(self) -> None:
        _, _, ambiguous_source, ambiguous_recovery = _original_source_recovery(
            EvidenceQualitySignalType.POSSIBLY_AMBIGUOUS,
            "规则可能适用",
        )
        ambiguity = _inspection(
            ambiguous_source,
            SourceFindingType.SOURCE_AMBIGUITY,
            fragments=(SourceFragment("可能", 2, 4),),
        )
        ambiguous_result = diagnose_failure(
            ambiguous_recovery, source_inspections=(ambiguity,)
        )

        _, _, missing_source, missing_recovery = _original_source_recovery(
            EvidenceQualitySignalType.POSSIBLY_AMBIGUOUS,
            "完整来源范围",
        )
        missing = _inspection(
            missing_source, SourceFindingType.REQUIRED_CONTENT_MISSING
        )
        missing_result = diagnose_failure(
            missing_recovery, source_inspections=(missing,)
        )

        _, _, conflict_source, conflict_recovery = _original_source_recovery(
            EvidenceQualitySignalType.POSSIBLY_AMBIGUOUS,
            "规则甲；规则乙",
        )
        conflict = _inspection(
            conflict_source,
            SourceFindingType.SOURCE_CONFLICT,
            fragments=(
                SourceFragment("规则甲", 0, 3),
                SourceFragment("规则乙", 4, 7),
            ),
        )
        conflict_result = diagnose_failure(
            conflict_recovery, source_inspections=(conflict,)
        )

        for result in (ambiguous_result, missing_result, conflict_result):
            self.assertEqual(
                result.diagnosis_type,
                FailureDiagnosisType.SOURCE_KNOWLEDGE_PROBLEM,
            )
            self.assertTrue(result.verification_required)
            self.assertTrue(result.knowledge_gap_allowed)

        incomplete = replace(missing, complete_scope_checked=False)
        gated = diagnose_failure(
            missing_recovery, source_inspections=(incomplete,)
        )
        self.assertEqual(
            gated.diagnosis_type, FailureDiagnosisType.UNDETERMINED
        )
        self.assertFalse(gated.knowledge_gap_allowed)

    def test_conflicting_source_inspections_are_undetermined(self) -> None:
        _, _, source, recovery = _original_source_recovery(
            EvidenceQualitySignalType.BROKEN_CONTEXT, "完整规则内容"
        )
        present = _inspection(
            source,
            SourceFindingType.REQUIRED_CONTENT_PRESENT,
            fragments=(SourceFragment("完整规则内容", 0, 6),),
        )
        missing = _inspection(
            source, SourceFindingType.REQUIRED_CONTENT_MISSING
        )

        result = diagnose_failure(
            recovery, source_inspections=(present, missing)
        )

        self.assertEqual(
            result.diagnosis_type, FailureDiagnosisType.UNDETERMINED
        )
        self.assertTrue(result.verification_required)
        self.assertFalse(result.knowledge_gap_allowed)
        self.assertEqual(result.reason, "source_inspections_conflict")

    def test_no_recovery_budget_exhaustion_and_incomplete_trace_are_undetermined(self) -> None:
        original = _evidence("e-original", "片段")
        no_recovery = run_limited_recovery(original, ())
        self.assertEqual(
            diagnose_failure(no_recovery).diagnosis_type,
            FailureDiagnosisType.UNDETERMINED,
        )

        signal = _signal(
            original, EvidenceQualitySignalType.POSSIBLY_INCOMPLETE
        )

        def unresolved(request):
            return RecoveryExecutionResult((), (signal,))

        exhausted = run_limited_recovery(
            original, (signal,), unresolved, max_rounds=1
        )
        exhausted_result = diagnose_failure(exhausted)
        self.assertEqual(
            exhausted_result.diagnosis_type,
            FailureDiagnosisType.UNDETERMINED,
        )
        self.assertTrue(exhausted_result.verification_required)

        _, _, _, resolved = _retrieval_recovery()
        incomplete = replace(resolved, attempted_actions=())
        incomplete_result = diagnose_failure(incomplete)
        self.assertEqual(
            incomplete_result.diagnosis_type,
            FailureDiagnosisType.UNDETERMINED,
        )
        self.assertEqual(
            incomplete_result.reason, "recovery_attempt_map_inconsistent"
        )

    def test_case_missing_not_found_and_retrieval_failure_never_allow_gap(self) -> None:
        _, _, _, recovery = _retrieval_recovery()

        for boundary in DiagnosisBoundary:
            with self.subTest(boundary=boundary):
                result = diagnose_failure(recovery, boundary=boundary)
                self.assertEqual(
                    result.diagnosis_type,
                    FailureDiagnosisType.UNDETERMINED,
                )
                self.assertTrue(result.verification_required)
                self.assertFalse(result.knowledge_gap_allowed)
                self.assertEqual(result.boundary, boundary)

    def test_source_inspection_without_original_source_trace_is_rejected(self) -> None:
        _, _, recovered, recovery = _retrieval_recovery()
        inspection = _inspection(
            recovered,
            SourceFindingType.SOURCE_AMBIGUITY,
            fragments=(SourceFragment("完整", 4, 6),),
        )

        result = diagnose_failure(
            recovery, source_inspections=(inspection,)
        )

        self.assertEqual(
            result.diagnosis_type, FailureDiagnosisType.UNDETERMINED
        )
        self.assertFalse(result.knowledge_gap_allowed)
        self.assertEqual(
            result.reason,
            "source_inspection_not_linked_to_original_source_trace",
        )

    def test_diagnosis_is_read_only_traceable_and_performs_no_recovery(self) -> None:
        original, signal, recovered, recovery = _retrieval_recovery()
        snapshots = (original, signal, recovered, recovery)

        first = diagnose_failure(recovery)
        second = diagnose_failure(recovery)

        self.assertEqual(first, second)
        self.assertEqual((original, signal, recovered, recovery), snapshots)
        self.assertEqual(first.recovery_trace, recovery.trace)
        self.assertIn(original.source_reference, first.source_references)
        self.assertIn(original.lineage, first.lineages)
        diagnosis_fields = {item.name for item in fields(FailureDiagnosis)}
        self.assertNotIn("new_recovery", diagnosis_fields)
        self.assertNotIn("final_decision", diagnosis_fields)
        self.assertNotIn("llm_result", diagnosis_fields)


if __name__ == "__main__":
    unittest.main()
