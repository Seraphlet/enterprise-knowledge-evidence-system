"""Targeted tests for the rebuildable local VectorIndex."""

import unittest

from knowledge_system import (
    KnowledgeLineage,
    KnowledgeUnit,
    ParseStatus,
    SourceReference,
    VectorIndex,
)


def _unit(
    unit_id: str,
    semantic: str,
    *,
    status: str = "active",
    permissions: list[str] | None = None,
) -> KnowledgeUnit:
    metadata: dict[str, object] = {"status": status}
    if permissions is not None:
        metadata["required_permissions"] = permissions
    return KnowledgeUnit(
        unit_id=unit_id,
        original_content=f"source:{semantic}",
        semantic_content=semantic,
        metadata=metadata,
        source_reference=SourceReference(name=f"{unit_id}.html"),
        lineage=KnowledgeLineage(
            document_id=f"doc-{unit_id}",
            parser_version="p1",
            chunker_version="c1",
            processing_version="v1",
        ),
        parse_status=ParseStatus.SUCCESS,
    )


class VectorIndexTest(unittest.TestCase):
    def setUp(self) -> None:
        self.first = _unit("u1", "payment review rule")
        self.second = _unit("u2", "employee travel policy")

    def test_rebuild_is_deterministic_from_knowledge_units(self) -> None:
        first = VectorIndex.rebuild(
            [self.first, self.second],
            knowledge_version="knowledge-a",
            index_version="index-1",
        )
        second = VectorIndex.rebuild(
            [self.first, self.second],
            knowledge_version="knowledge-a",
            index_version="index-1",
        )

        self.assertEqual(first.snapshot(), second.snapshot())
        self.assertEqual(len(first), 2)

    def test_index_and_knowledge_versions_are_separate(self) -> None:
        index = VectorIndex.rebuild(
            [self.first],
            knowledge_version="knowledge-commit-7",
            index_version="vector-build-3",
        )

        self.assertEqual(index.info.knowledge_version, "knowledge-commit-7")
        self.assertEqual(index.info.index_version, "vector-build-3")
        self.assertNotEqual(index.info.knowledge_version, index.info.index_version)

    def test_rebuild_can_advance_index_version_without_changing_knowledge(self) -> None:
        before = VectorIndex.rebuild(
            [self.first], knowledge_version="k1", index_version="i1"
        )
        after = VectorIndex.rebuild(
            [self.first], knowledge_version="k1", index_version="i2"
        )

        self.assertEqual(before.info.knowledge_version, after.info.knowledge_version)
        self.assertNotEqual(before.info.index_version, after.info.index_version)
        self.assertEqual(before.snapshot(), after.snapshot())

    def test_vector_lookup_returns_evidence_without_hybrid_fusion(self) -> None:
        index = VectorIndex.rebuild(
            [self.first, self.second], knowledge_version="k1", index_version="i1"
        )

        hits = index.search("payment review rule", top_k=1)

        self.assertEqual(len(hits), 1)
        self.assertEqual(hits[0].evidence.unit_id, "u1")
        self.assertAlmostEqual(hits[0].score, 1.0)

    def test_permission_and_status_filter_before_vectorization(self) -> None:
        denied = _unit("denied", "secret vector phrase", permissions=["secret"])
        expired = _unit("expired", "expired vector phrase", status="expired")
        index = VectorIndex.rebuild(
            [self.first, denied, expired],
            knowledge_version="k1",
            index_version="i1",
        )

        self.assertEqual([unit_id for unit_id, _ in index.snapshot()], ["u1"])
        secret_ids = {
            hit.evidence.unit_id for hit in index.search("secret vector phrase")
        }
        expired_ids = {
            hit.evidence.unit_id for hit in index.search("expired vector phrase")
        }
        self.assertNotIn("denied", secret_ids)
        self.assertNotIn("expired", expired_ids)


if __name__ == "__main__":
    unittest.main()
