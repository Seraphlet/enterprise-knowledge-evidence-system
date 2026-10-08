"""Focused tests for deterministic, non-diagnostic Evidence quality signals."""

import unittest
from dataclasses import fields, replace

from knowledge_system import (
    Evidence,
    EvidenceQualitySignal,
    EvidenceQualitySignalType,
    KnowledgeLineage,
    KnowledgeUnit,
    ParseStatus,
    SourceReference,
    detect_evidence_quality,
    extract_rule_structure,
)


def _unit(
    content: str = "条件：点赞数不少于50个\n结果：继续处理",
    *,
    parse_status: ParseStatus = ParseStatus.SUCCESS,
    section: str | None = "审核规则",
    parent_section_id: str | None = "section-audit",
    structure_kind: str = "section",
) -> KnowledgeUnit:
    return KnowledgeUnit(
        unit_id="unit-quality",
        original_content=content,
        semantic_content=f"审核规则\n{content}",
        metadata={"structure_kind": structure_kind},
        source_reference=SourceReference(
            name="审核规则.html",
            url="https://kb.example/rules",
            section=section,
        ),
        lineage=KnowledgeLineage(
            document_id="doc-quality",
            parser_version="p1",
            chunker_version="c1",
            processing_version="v1",
            source_position="blocks:1-2",
            parent_section_id=parent_section_id,
        ),
        parse_status=parse_status,
    )


def _evidence(unit: KnowledgeUnit) -> Evidence:
    return Evidence(
        evidence_id="evidence-quality",
        unit_id=unit.unit_id,
        original_content=unit.original_content,
        source_reference=unit.source_reference,
        lineage=unit.lineage,
        derived_structure=unit.rule_structure,
    )


class EvidenceQualityTest(unittest.TestCase):
    def test_context_dependent_markdown_fragments_are_flagged(self) -> None:
        for content in (
            "然后问：",
            "而是：",
            "完全一致。",
            "初学时很容易理解成：",
        ):
            with self.subTest(content=content):
                unit = _unit(content, structure_kind="paragraph")

                signals = detect_evidence_quality(
                    _evidence(unit), knowledge_unit=unit
                )

                self.assertEqual(len(signals), 1)
                self.assertEqual(
                    signals[0].signal_type,
                    EvidenceQualitySignalType.POSSIBLY_INCOMPLETE,
                )
                self.assertEqual(
                    signals[0].reason, "context_dependent_fragment"
                )

    def test_short_structural_blocks_and_independent_fact_are_not_flagged(self) -> None:
        cases = (
            ("# Python 工程最小结构", "heading"),
            ("`main.py`", "code"),
            ("默认端口是 8080。", "paragraph"),
        )
        for content, structure_kind in cases:
            with self.subTest(content=content, structure_kind=structure_kind):
                unit = _unit(content, structure_kind=structure_kind)

                signals = detect_evidence_quality(
                    _evidence(unit), knowledge_unit=unit
                )

                self.assertEqual(signals, ())

    def test_empty_source_content_is_incomplete_and_missing_context(self) -> None:
        unit = _unit(" ")
        signals = detect_evidence_quality(
            _evidence(unit), knowledge_unit=unit
        )

        self.assertEqual(
            tuple(signal.signal_type for signal in signals),
            (
                EvidenceQualitySignalType.POSSIBLY_INCOMPLETE,
                EvidenceQualitySignalType.MISSING_CONTEXT,
            ),
        )

    def test_partial_or_failed_parse_is_possibly_incomplete(self) -> None:
        for status, reason in (
            (ParseStatus.PARTIAL, "parse_status_partial"),
            (ParseStatus.FAILED, "parse_status_failed"),
        ):
            with self.subTest(status=status):
                unit = _unit(parse_status=status)
                signals = detect_evidence_quality(
                    _evidence(unit), knowledge_unit=unit
                )

                self.assertEqual(len(signals), 1)
                self.assertEqual(
                    signals[0].signal_type,
                    EvidenceQualitySignalType.POSSIBLY_INCOMPLETE,
                )
                self.assertEqual(signals[0].reason, reason)

    def test_evidence_unit_mismatch_is_broken_context(self) -> None:
        unit = _unit()
        evidence = _evidence(unit)
        mismatched = replace(unit, original_content="另一段原文")

        signals = detect_evidence_quality(
            evidence, knowledge_unit=mismatched
        )

        self.assertEqual(len(signals), 1)
        self.assertEqual(
            signals[0].signal_type,
            EvidenceQualitySignalType.BROKEN_CONTEXT,
        )
        self.assertEqual(
            signals[0].reason, "evidence_unit_content_mismatch"
        )

    def test_declared_parent_without_section_locator_is_missing_context(self) -> None:
        unit = _unit(section=None, parent_section_id="section-audit")
        evidence = _evidence(unit)

        signals = detect_evidence_quality(evidence, knowledge_unit=unit)

        self.assertEqual(len(signals), 1)
        self.assertEqual(
            signals[0].signal_type,
            EvidenceQualitySignalType.MISSING_CONTEXT,
        )
        self.assertEqual(
            signals[0].reason, "parent_section_locator_missing"
        )

    def test_explicit_rule_ambiguity_retains_verbatim_provenance(self) -> None:
        unit = _unit("条件：原则上由负责人确认\n歧义：负责人范围不明确")
        evidence = _evidence(unit)
        extraction = extract_rule_structure(evidence)

        signals = detect_evidence_quality(
            evidence, knowledge_unit=unit, rule_extraction=extraction
        )

        ambiguous = tuple(
            signal
            for signal in signals
            if signal.signal_type
            is EvidenceQualitySignalType.POSSIBLY_AMBIGUOUS
        )
        self.assertTrue(ambiguous)
        for signal in ambiguous:
            fragment = signal.source_fragment
            assert fragment is not None
            self.assertEqual(
                evidence.original_content[fragment.start:fragment.end],
                fragment.text,
            )
            self.assertIs(signal.source_reference, evidence.source_reference)
            self.assertIs(signal.lineage, evidence.lineage)

    def test_evidence_derived_ambiguity_requires_exact_source_slice(self) -> None:
        unit = _unit("范围不明确")
        base = _evidence(unit)
        valid = replace(
            base,
            derived_structure={
                "ambiguity": [{"text": "不明确", "start": 2, "end": 5}]
            },
        )
        invalid = replace(
            base,
            derived_structure={
                "ambiguity": [{"text": "不明确", "start": 0, "end": 3}]
            },
        )

        valid_signals = detect_evidence_quality(valid)
        invalid_signals = detect_evidence_quality(invalid)

        self.assertEqual(
            valid_signals[0].signal_type,
            EvidenceQualitySignalType.POSSIBLY_AMBIGUOUS,
        )
        self.assertEqual(valid_signals[0].source_fragment.text, "不明确")
        self.assertEqual(
            invalid_signals[0].signal_type,
            EvidenceQualitySignalType.BROKEN_CONTEXT,
        )
        self.assertEqual(
            invalid_signals[0].reason, "derived_ambiguity_untraceable:0"
        )

    def test_multiple_signals_coexist_without_diagnosis(self) -> None:
        unit = _unit(
            "条件：原则上由负责人确认\n歧义：负责人范围不明确",
            parse_status=ParseStatus.PARTIAL,
            section=None,
        )
        evidence = _evidence(unit)
        mismatched = replace(unit, original_content="不一致的 Unit 原文")
        extraction = extract_rule_structure(evidence)
        snapshots = (evidence, mismatched, extraction)

        signals = detect_evidence_quality(
            evidence,
            knowledge_unit=mismatched,
            rule_extraction=extraction,
        )

        self.assertEqual(
            {signal.signal_type for signal in signals},
            set(EvidenceQualitySignalType),
        )
        self.assertEqual((evidence, mismatched, extraction), snapshots)
        signal_fields = {item.name for item in fields(EvidenceQualitySignal)}
        self.assertEqual(
            signal_fields,
            {
                "signal_type",
                "reason",
                "evidence_id",
                "source_reference",
                "lineage",
                "source_fragment",
            },
        )
        for signal in signals:
            serialized = signal.to_dict()
            self.assertNotIn("diagnosis", serialized)
            self.assertNotIn("problem_type", serialized)
            self.assertNotIn("knowledge_gap", serialized)
            self.assertNotIn("recovery", serialized)

    def test_duplicate_signal_identity_is_deduplicated_and_ordered(self) -> None:
        unit = _unit("条件：原则上由负责人确认\n歧义：负责人范围不明确")
        evidence = _evidence(unit)
        extraction = extract_rule_structure(evidence)
        assert extraction.structure is not None
        fragments = extraction.structure.ambiguity
        self.assertGreaterEqual(len(fragments), 2)
        duplicated = replace(
            extraction,
            structure=replace(
                extraction.structure,
                ambiguity=(fragments[-1], fragments[0], fragments[-1]),
            ),
        )

        first = detect_evidence_quality(
            evidence, rule_extraction=duplicated
        )
        second = detect_evidence_quality(
            evidence, rule_extraction=duplicated
        )

        self.assertEqual(first, second)
        self.assertEqual(len(first), 2)
        self.assertEqual(
            tuple(item.source_fragment.start for item in first),
            tuple(sorted({fragments[-1].start, fragments[0].start})),
        )

    def test_complete_consistent_evidence_has_no_signal(self) -> None:
        unit = _unit()
        evidence = _evidence(unit)
        extraction = extract_rule_structure(evidence)

        signals = detect_evidence_quality(
            evidence, knowledge_unit=unit, rule_extraction=extraction
        )

        self.assertEqual(signals, ())


if __name__ == "__main__":
    unittest.main()
