"""Targeted tests for the core data contracts."""

import unittest

from knowledge_system import (
    Evidence,
    KnowledgeLineage,
    KnowledgeUnit,
    ParseStatus,
    SourceReference,
    Trace,
)


class ContractRoundTripTest(unittest.TestCase):
    def setUp(self) -> None:
        self.source = SourceReference(
            name="审核规则.html",
            url="https://kb.example/rules",
            section="二级审核",
            page=3,
            sheet="规则表",
        )
        self.lineage = KnowledgeLineage(
            document_id="doc-1",
            source_position="section:2/table:1",
            parent_section_id="section-2",
            parser_version="parser-v1",
            chunker_version="chunker-v1",
            processing_version="processing-v1",
        )

    def test_source_reference_and_lineage_round_trip(self) -> None:
        self.assertEqual(
            SourceReference.from_json(self.source.to_json()), self.source
        )
        self.assertEqual(
            KnowledgeLineage.from_json(self.lineage.to_json()), self.lineage
        )

    def test_knowledge_unit_round_trip_keeps_source_and_derived_separate(self) -> None:
        unit = KnowledgeUnit(
            unit_id="unit-1",
            original_content="点赞数达到 50。",
            semantic_content="点赞阈值规则",
            metadata={"status": "active", "scope": "demo"},
            source_reference=self.source,
            lineage=self.lineage,
            parse_status=ParseStatus.SUCCESS,
            rule_structure={"threshold": 50},
        )

        encoded = unit.to_dict()
        self.assertEqual(encoded["original_content"], "点赞数达到 50。")
        self.assertEqual(encoded["rule_structure"], {"threshold": 50})
        self.assertNotIn("rule_structure", encoded["source_reference"])
        self.assertEqual(KnowledgeUnit.from_json(unit.to_json()), unit)

    def test_evidence_and_trace_round_trip(self) -> None:
        evidence = Evidence(
            evidence_id="evidence-1",
            unit_id="unit-1",
            original_content="正式资料原文",
            source_reference=self.source,
            lineage=self.lineage,
            derived_structure={"ambiguity": None},
        )
        trace = Trace(
            trace_id="trace-1",
            raw_query="原始问题",
            retrieval_query="检索问题",
            candidate_evidence_ids=["evidence-1", "evidence-2"],
            final_evidence_ids=["evidence-1"],
            applied_filters=["permission"],
            knowledge_version="commit-1",
            rerank_summary={"enabled": False, "applied": False},
            discover_summary={"request": "discover"},
        )

        self.assertEqual(Evidence.from_json(evidence.to_json()), evidence)
        self.assertEqual(Trace.from_json(trace.to_json()), trace)

    def test_all_parse_status_values_round_trip(self) -> None:
        for status in ParseStatus:
            unit = KnowledgeUnit(
                unit_id=f"unit-{status.value}",
                original_content="source",
                semantic_content="semantic",
                metadata={},
                source_reference=self.source,
                lineage=self.lineage,
                parse_status=status,
            )
            self.assertEqual(KnowledgeUnit.from_json(unit.to_json()), unit)


if __name__ == "__main__":
    unittest.main()
