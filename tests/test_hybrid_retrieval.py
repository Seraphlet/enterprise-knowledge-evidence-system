"""Targeted tests for deterministic hybrid recall/fusion."""

import unittest

from knowledge_system import (
    HybridRetrieval,
    KnowledgeLineage,
    KnowledgeUnit,
    ParseStatus,
    RerankConfig,
    SourceReference,
)


def _unit(
    unit_id: str,
    semantic: str,
    *,
    permissions: list[str] | None = None,
) -> KnowledgeUnit:
    metadata: dict[str, object] = {"scope": "demo", "status": "active"}
    if permissions is not None:
        metadata["required_permissions"] = permissions
    return KnowledgeUnit(
        unit_id=unit_id,
        original_content=semantic,
        semantic_content=semantic,
        metadata=metadata,
        source_reference=SourceReference(
            name=f"{unit_id}.html",
            url=f"https://kb.example/{unit_id}",
            section=f"Section {unit_id}",
        ),
        lineage=KnowledgeLineage(
            document_id=f"doc-{unit_id}",
            parser_version="p1",
            chunker_version="c1",
            processing_version="v1",
        ),
        parse_status=ParseStatus.SUCCESS,
    )


class HybridRetrievalTest(unittest.TestCase):
    def setUp(self) -> None:
        self.keyword = _unit("keyword", "payment rule")
        self.vector = _unit("vector", "payment review rule")
        self.denied = _unit(
            "denied", "payment rule secret", permissions=["secret.read"]
        )
        self.service = HybridRetrieval(
            [self.keyword, self.vector, self.denied],
            knowledge_version="knowledge-v1",
            index_version="index-v1",
            covered_scopes=["demo"],
        )

    def test_keyword_and_vector_results_are_fused_and_deduplicated(self) -> None:
        result = self.service.search("payment rule", scope="demo")
        ids = [item.unit_id for item in result.evidence]

        self.assertIn("keyword", ids)
        self.assertIn("vector", ids)
        self.assertEqual(len(ids), len(set(ids)))

    def test_trace_records_rank_and_score_for_each_path(self) -> None:
        trace = self.service.search("payment rule", scope="demo").trace
        keyword_records = [
            item for item in trace.retrieval_paths if item["path"] == "keyword"
        ]
        vector_records = [
            item for item in trace.retrieval_paths if item["path"] == "vector"
        ]

        self.assertTrue(keyword_records)
        self.assertTrue(vector_records)
        for record in trace.retrieval_paths:
            self.assertIsInstance(record["rank"], int)
            self.assertIsInstance(record["score"], float)
        self.assertEqual(trace.knowledge_version, "knowledge-v1")
        self.assertEqual(trace.index_version, "index-v1")
        self.assertEqual(set(trace.final_evidence_ids), set(trace.fusion_scores))

    def test_fusion_is_deterministic_and_source_traceable(self) -> None:
        first = self.service.search("payment rule", scope="demo")
        second = self.service.search("payment rule", scope="demo")

        self.assertEqual(
            [item.evidence_id for item in first.evidence],
            [item.evidence_id for item in second.evidence],
        )
        self.assertEqual(first.trace.fusion_scores, second.trace.fusion_scores)
        self.assertTrue(all(item.source_reference.name for item in first.evidence))

    def test_permission_filter_precedes_both_recall_paths(self) -> None:
        result = self.service.search("payment rule", scope="demo")
        trace_text = result.trace.to_json()

        self.assertNotIn("evidence-denied", result.trace.candidate_evidence_ids)
        self.assertNotIn("evidence-denied", result.trace.final_evidence_ids)
        self.assertNotIn("evidence-denied", trace_text)
        self.assertNotIn("denied", [item.unit_id for item in result.evidence])

    def test_one_shot_permissions_feed_the_same_unit_to_both_paths(self) -> None:
        protected = _unit(
            "protected", "payment rule protected", permissions=["rules.read"]
        )
        granted_permissions = (permission for permission in ["rules.read"])
        service = HybridRetrieval(
            [protected],
            knowledge_version="knowledge-v1",
            index_version="index-v1",
            covered_scopes=["demo"],
            granted_permissions=granted_permissions,
        )

        result = service.search("payment rule protected", scope="demo")
        paths = {
            record["path"]
            for record in result.trace.retrieval_paths
            if record["evidence_id"] == "evidence-protected"
        }

        self.assertEqual(paths, {"keyword", "vector"})
        self.assertEqual(
            [evidence.unit_id for evidence in result.evidence], ["protected"]
        )

    def test_scope_boundary_produces_no_recall_candidates(self) -> None:
        result = self.service.search("payment rule", scope="outside")

        self.assertEqual(result.evidence, ())
        self.assertEqual(result.trace.retrieval_paths, [])
        self.assertEqual(result.trace.final_evidence_ids, [])

    def test_rerank_is_disabled_by_default_and_preserves_fusion_order(self) -> None:
        baseline = self.service.search("payment rule", scope="demo")
        ignored_calls: list[str] = []

        configured = HybridRetrieval(
            [self.keyword, self.vector],
            knowledge_version="knowledge-v1",
            index_version="index-v1",
            covered_scopes=["demo"],
            reranker=lambda query, candidates: ignored_calls.append(query) or (),
        ).search("payment rule", scope="demo")

        self.assertEqual(configured.trace.final_evidence_ids, baseline.trace.final_evidence_ids)
        self.assertEqual(ignored_calls, [])
        self.assertEqual(
            configured.trace.rerank_summary,
            {"enabled": False, "applied": False, "reason": "DISABLED"},
        )

    def test_rerank_requires_a_measured_ranking_gap(self) -> None:
        with self.assertRaisesRegex(ValueError, "evaluation_id"):
            RerankConfig(enabled=True)
        with self.assertRaisesRegex(ValueError, "observed ranking gap"):
            RerankConfig(
                enabled=True,
                evaluation_id="evaluation-1",
                observed_ranking_metric=0.9,
                minimum_acceptable_metric=0.8,
            )

    def test_enabled_rerank_reorders_only_existing_evidence_ids(self) -> None:
        config = RerankConfig(
            enabled=True,
            evaluation_id="evaluation-1",
            observed_ranking_metric=0.4,
            minimum_acceptable_metric=0.8,
        )
        service = HybridRetrieval(
            [self.keyword, self.vector],
            knowledge_version="knowledge-v1",
            index_version="index-v1",
            covered_scopes=["demo"],
            rerank_config=config,
            reranker=lambda query, candidates: reversed(
                [candidate.evidence_id for candidate in candidates]
            ),
        )
        baseline_ids = self.service.search(
            "payment rule", scope="demo"
        ).trace.final_evidence_ids

        result = service.search("payment rule", scope="demo")

        self.assertEqual(result.trace.final_evidence_ids, list(reversed(baseline_ids)))
        self.assertEqual(result.trace.rerank_summary["applied"], True)
        self.assertEqual(
            result.trace.rerank_summary["evaluation_id"], "evaluation-1"
        )

        invalid_service = HybridRetrieval(
            [self.keyword, self.vector],
            knowledge_version="knowledge-v1",
            index_version="index-v1",
            covered_scopes=["demo"],
            rerank_config=config,
            reranker=lambda query, candidates: ["evidence-not-a-candidate"],
        )
        with self.assertRaisesRegex(ValueError, "each candidate evidence_id once"):
            invalid_service.search("payment rule", scope="demo")


if __name__ == "__main__":
    unittest.main()
