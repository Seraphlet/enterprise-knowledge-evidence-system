"""Focused tests for deterministic, source-grounded Verification Mode."""

import copy
import unittest
from dataclasses import FrozenInstanceError, fields

from knowledge_system import (
    Evidence,
    EvidenceQualitySignal,
    EvidenceQualitySignalType,
    FailureDiagnosis,
    FailureDiagnosisType,
    KnowledgeLineage,
    LocatorLevel,
    RecoveryResult,
    RecoveryStatus,
    SourceFindingType,
    SourceFragment,
    SourceInspection,
    SourceReference,
    VerificationReason,
    build_verification_mode,
)


def _evidence(
    evidence_id="evidence-1",
    *,
    content="规则可能不完整",
    name="审核规则.html",
    url="https://kb.example/rules",
    section="审核规则",
    page=3,
    sheet=None,
):
    return Evidence(
        evidence_id=evidence_id,
        unit_id=f"unit-{evidence_id}",
        original_content=content,
        source_reference=SourceReference(
            name=name,
            url=url,
            section=section,
            page=page,
            sheet=sheet,
        ),
        lineage=KnowledgeLineage(
            document_id=f"doc-{evidence_id}",
            parser_version="p1",
            chunker_version="c1",
            processing_version="v1",
            source_position="blocks:2-3",
            parent_section_id="parent-rules",
        ),
    )


def _signal(evidence, signal_type, *, fragment=None, reason=None):
    return EvidenceQualitySignal(
        signal_type=signal_type,
        reason=reason or f"observed_{signal_type.value.casefold()}",
        evidence_id=evidence.evidence_id,
        source_reference=evidence.source_reference,
        lineage=evidence.lineage,
        source_fragment=fragment,
    )


def _source_diagnosis(evidence, finding_type, *, fragments=()):
    inspection = SourceInspection(
        finding_type=finding_type,
        evidence_id=evidence.evidence_id,
        source_reference=evidence.source_reference,
        lineage=evidence.lineage,
        reason=f"checked_{finding_type.value.casefold()}",
        complete_scope_checked=True,
        source_fragments=fragments,
    )
    return FailureDiagnosis(
        diagnosis_type=FailureDiagnosisType.SOURCE_KNOWLEDGE_PROBLEM,
        verification_required=True,
        knowledge_gap_allowed=True,
        reason="original_source_contains_confirmed_defect",
        signals=(),
        recovery_trace=(),
        evidence=(evidence,),
        source_references=(evidence.source_reference,),
        lineages=(evidence.lineage,),
        source_inspections=(inspection,),
    )


def _unresolved_recovery(evidence, signal):
    return RecoveryResult(
        original_evidence=evidence,
        status=RecoveryStatus.UNRESOLVED,
        recovered_evidence=(),
        remaining_signals=(signal,),
        trace=(),
        max_rounds=1,
        verification_required=True,
        unresolved_reason="round_budget_exhausted",
    )


class VerificationModeTest(unittest.TestCase):
    def test_reliable_evidence_is_a_no_op(self) -> None:
        evidence = _evidence()

        result = build_verification_mode((evidence,))

        self.assertFalse(result.active)
        self.assertEqual(result.records, ())

    def test_five_verification_reasons_remain_distinct(self) -> None:
        knowledge = _evidence("e-knowledge")
        ambiguous = _evidence("e-ambiguous", content="规则可能适用")
        recovery_evidence = _evidence("e-recovery")
        retrieval = _evidence("e-retrieval")
        processing = _evidence("e-processing")
        ambiguous_fragment = SourceFragment("可能", 2, 4)
        recovery_signal = _signal(
            recovery_evidence, EvidenceQualitySignalType.BROKEN_CONTEXT
        )
        cases = (
            (
                (knowledge,),
                {},
                _source_diagnosis(
                    knowledge, SourceFindingType.REQUIRED_CONTENT_MISSING
                ),
                VerificationReason.KNOWLEDGE_MISSING,
            ),
            (
                (ambiguous,),
                {},
                _source_diagnosis(
                    ambiguous,
                    SourceFindingType.SOURCE_AMBIGUITY,
                    fragments=(ambiguous_fragment,),
                ),
                VerificationReason.SOURCE_AMBIGUOUS,
            ),
            (
                (recovery_evidence,),
                {"recovery": _unresolved_recovery(recovery_evidence, recovery_signal)},
                None,
                VerificationReason.RECOVERY_FAILED,
            ),
            (
                (retrieval,),
                {
                    "quality_signals": (
                        _signal(retrieval, EvidenceQualitySignalType.MISSING_CONTEXT),
                    )
                },
                None,
                VerificationReason.RETRIEVAL_UNCERTAINTY,
            ),
            (
                (processing,),
                {
                    "quality_signals": (
                        _signal(processing, EvidenceQualitySignalType.BROKEN_CONTEXT),
                    )
                },
                None,
                VerificationReason.PROCESSING_UNCERTAINTY,
            ),
        )
        for evidence_items, kwargs, diagnosis, expected in cases:
            if diagnosis is not None:
                kwargs = {**kwargs, "diagnosis": diagnosis}
            with self.subTest(reason=expected):
                result = build_verification_mode(evidence_items, **kwargs)
                self.assertTrue(result.active)
                self.assertIn(expected, tuple(item.reason for item in result.records))

    def test_original_text_provenance_and_offsets_are_exact(self) -> None:
        evidence = _evidence(content="规则可能不完整")
        fragment = SourceFragment("可能", 2, 4)
        signal = _signal(
            evidence,
            EvidenceQualitySignalType.POSSIBLY_AMBIGUOUS,
            fragment=fragment,
            reason="explicit_rule_ambiguity",
        )

        result = build_verification_mode(
            (evidence,), quality_signals=(signal,)
        )
        record = result.records[0]

        self.assertIs(record.original_content, evidence.original_content)
        self.assertIs(record.source_reference, evidence.source_reference)
        self.assertIs(record.lineage, evidence.lineage)
        self.assertIs(record.source_fragments[0], fragment)
        self.assertEqual(
            record.original_content[fragment.start:fragment.end], fragment.text
        )
        self.assertIn(
            "quality:POSSIBLY_AMBIGUOUS", record.problem_codes
        )
        self.assertIn(
            "signal_reason:explicit_rule_ambiguity",
            record.uncertainty_codes,
        )

    def test_locator_uses_only_actual_fields_with_conservative_fallback(self) -> None:
        cases = (
            (
                _evidence("e-url"),
                LocatorLevel.STABLE_URL,
                "https://kb.example/rules",
            ),
            (
                _evidence(
                    "e-position",
                    url=None,
                    section="S1",
                    page=7,
                    sheet="Sheet1",
                ),
                LocatorLevel.SOURCE_POSITION,
                None,
            ),
            (
                _evidence(
                    "e-name", url=None, section=None, page=None, sheet=None
                ),
                LocatorLevel.SOURCE_NAME,
                None,
            ),
        )
        for evidence, level, url in cases:
            signal = _signal(evidence, EvidenceQualitySignalType.MISSING_CONTEXT)
            with self.subTest(level=level):
                locator = build_verification_mode(
                    (evidence,), quality_signals=(signal,)
                ).records[0].verification_locator
                self.assertEqual(locator.level, level)
                self.assertEqual(locator.url, url)
                self.assertEqual(locator.source_name, evidence.source_reference.name)
                if level is LocatorLevel.SOURCE_POSITION:
                    self.assertEqual(locator.section, "S1")
                    self.assertEqual(locator.page, 7)
                    self.assertEqual(locator.sheet, "Sheet1")

    def test_missing_or_internal_path_locator_is_not_invented(self) -> None:
        cases = (
            _evidence(
                "e-missing",
                name="",
                url=None,
                section=None,
                page=None,
            ),
            _evidence(
                "e-internal",
                name=r"\\server\share\rules.html",
                url="file://server/share/rules.html",
                section=None,
                page=None,
            ),
        )
        for evidence in cases:
            with self.subTest(evidence=evidence.evidence_id):
                signal = _signal(
                    evidence, EvidenceQualitySignalType.MISSING_CONTEXT
                )
                record = build_verification_mode(
                    (evidence,), quality_signals=(signal,)
                ).records[0]
                self.assertIsNone(record.verification_locator)
                self.assertIs(record.source_reference, evidence.source_reference)

    def test_duplicate_inputs_are_deduplicated_with_stable_order(self) -> None:
        second = _evidence("evidence-b")
        first = _evidence("evidence-a")
        first_signal = _signal(
            first, EvidenceQualitySignalType.POSSIBLY_AMBIGUOUS
        )
        second_signal = _signal(
            second, EvidenceQualitySignalType.MISSING_CONTEXT
        )

        one = build_verification_mode(
            (second, first, first),
            quality_signals=(second_signal, first_signal, first_signal),
        )
        two = build_verification_mode(
            (first, second), quality_signals=(first_signal, second_signal)
        )

        self.assertEqual(one, two)
        self.assertEqual(
            tuple(item.evidence_id for item in one.records),
            ("evidence-a", "evidence-b"),
        )

    def test_inputs_and_output_are_immutable(self) -> None:
        evidence = _evidence()
        signal = _signal(evidence, EvidenceQualitySignalType.BROKEN_CONTEXT)
        evidence_before = copy.deepcopy(evidence)
        signal_before = copy.deepcopy(signal)

        result = build_verification_mode(
            (evidence,), quality_signals=(signal,)
        )

        self.assertEqual(evidence, evidence_before)
        self.assertEqual(signal, signal_before)
        with self.assertRaises(FrozenInstanceError):
            result.active = False
        with self.assertRaises(FrozenInstanceError):
            result.records[0].original_content = "rewritten"

    def test_output_has_no_answer_feedback_or_business_decision_surface(self) -> None:
        evidence = _evidence()
        signal = _signal(evidence, EvidenceQualitySignalType.BROKEN_CONTEXT)
        result = build_verification_mode(
            (evidence,), quality_signals=(signal,)
        )

        mode_fields = {item.name for item in fields(type(result))}
        record_fields = {item.name for item in fields(type(result.records[0]))}
        for prohibited in (
            "answer",
            "conclusion",
            "approval",
            "decision",
            "feedback",
            "llm_output",
            "session",
        ):
            self.assertNotIn(prohibited, mode_fields)
            self.assertNotIn(prohibited, record_fields)


if __name__ == "__main__":
    unittest.main()
