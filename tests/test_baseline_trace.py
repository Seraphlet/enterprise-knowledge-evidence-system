"""Targeted tests for baseline retrieval Trace."""

import unittest

from knowledge_system import (
    BaselineRetrieval,
    KnowledgeLineage,
    KnowledgeUnit,
    ParseStatus,
    SourceReference,
)


def _unit(
    unit_id: str, *, permissions: list[str] | None = None
) -> KnowledgeUnit:
    metadata: dict[str, object] = {"scope": "demo", "status": "active"}
    if permissions is not None:
        metadata["required_permissions"] = permissions
    return KnowledgeUnit(
        unit_id=unit_id,
        original_content="Trace Keyword",
        semantic_content="Trace Keyword",
        metadata=metadata,
        source_reference=SourceReference(
            name="trace.html",
            url="https://kb.example/trace",
            section="Trace Section",
        ),
        lineage=KnowledgeLineage(
            document_id="trace-document",
            parser_version="parser-v1",
            chunker_version="chunker-v1",
            processing_version="processing-v1",
        ),
        parse_status=ParseStatus.SUCCESS,
    )


class BaselineTraceTest(unittest.TestCase):
    def setUp(self) -> None:
        self.allowed = _unit("allowed")
        self.denied = _unit("denied", permissions=["secret.read"])
        self.service = BaselineRetrieval(
            [self.allowed, self.denied],
            covered_scopes=["demo"],
            knowledge_version="knowledge-v1",
        )

    def test_locate_trace_records_query_ids_filters_source_and_versions(self) -> None:
        result = self.service.locate("  Trace Keyword  ", scope="demo")
        trace = result.trace

        self.assertIsNotNone(trace)
        assert trace is not None
        self.assertEqual(trace.raw_query, "  Trace Keyword  ")
        self.assertEqual(trace.retrieval_query, "trace keyword")
        self.assertEqual(trace.candidate_evidence_ids, ["evidence-allowed"])
        self.assertEqual(trace.final_evidence_ids, ["evidence-allowed"])
        self.assertEqual(trace.applied_filters, ["permission", "status", "scope"])
        self.assertEqual(trace.knowledge_version, "knowledge-v1")
        self.assertEqual(trace.source_references[0]["name"], "trace.html")
        self.assertEqual(
            trace.processing_versions[0]["processing_version"], "processing-v1"
        )

    def test_permission_denied_identity_never_leaks_to_candidate_or_final_ids(self) -> None:
        result = self.service.locate("Trace Keyword", scope="demo")
        trace = result.trace

        assert trace is not None
        self.assertNotIn("evidence-denied", trace.candidate_evidence_ids)
        self.assertNotIn("evidence-denied", trace.final_evidence_ids)
        self.assertNotIn("denied", trace.to_json())
        self.assertEqual(trace.filter_summary["rejected_count"], 1)

    def test_trace_is_separate_from_evidence_and_round_trips(self) -> None:
        result = self.service.browse("demo", raw_query="浏览 demo 范围")
        trace = result.trace

        assert trace is not None
        self.assertFalse(hasattr(result.evidence[0], "trace"))
        self.assertEqual(type(trace).from_json(trace.to_json()), trace)
        self.assertEqual(trace.raw_query, "浏览 demo 范围")

    def test_not_found_still_has_boundary_trace(self) -> None:
        result = self.service.locate("missing", scope="demo")
        trace = result.trace

        assert trace is not None
        self.assertEqual(trace.candidate_evidence_ids, [])
        self.assertEqual(trace.final_evidence_ids, [])
        self.assertEqual(trace.source_references, [])


if __name__ == "__main__":
    unittest.main()

