"""Targeted tests for evidence-grounded Discover requests."""

import unittest

from knowledge_system import (
    DiscoverRetrieval,
    HybridRetrieval,
    KeywordIndex,
    KnowledgeLineage,
    KnowledgeUnit,
    ParseStatus,
    SourceReference,
)


def _unit(
    unit_id: str,
    semantic: str,
    *,
    section: str | None,
    scope: str = "demo",
    status: str = "active",
    permissions: list[str] | None = None,
) -> KnowledgeUnit:
    metadata: dict[str, object] = {"scope": scope, "status": status}
    if permissions is not None:
        metadata["required_permissions"] = permissions
    return KnowledgeUnit(
        unit_id=unit_id,
        original_content=semantic,
        semantic_content=semantic,
        metadata=metadata,
        source_reference=SourceReference(
            name=f"Official {unit_id} Handbook",
            url=f"https://kb.example/{unit_id}",
            section=section,
        ),
        lineage=KnowledgeLineage(
            document_id=f"doc-{unit_id}",
            parser_version="p1",
            chunker_version="c1",
            processing_version="v1",
        ),
        parse_status=ParseStatus.SUCCESS,
    )


def _service(
    units: list[KnowledgeUnit], *, permissions: list[str] | None = None
) -> DiscoverRetrieval:
    hybrid = HybridRetrieval(
        units,
        knowledge_version="knowledge-v1",
        index_version="index-v1",
        covered_scopes=["demo", "other"],
        granted_permissions=permissions or [],
    )
    return DiscoverRetrieval(hybrid)


class DiscoverRetrievalTest(unittest.TestCase):
    def test_fuzzy_expression_discovers_formal_kb_unit_via_vector_path(self) -> None:
        travel = _unit(
            "travel",
            "travel reimbursement policy for employees",
            section="Travel Expense Policy",
        )
        query = "how to submit a travel reimbursement request"

        self.assertEqual(KeywordIndex([travel]).search(query), ())
        result = _service([travel]).discover(query, scope="demo")

        self.assertEqual([match.evidence.unit_id for match in result.matches], ["travel"])
        self.assertEqual(result.matches[0].formal_name, travel.source_reference.section)
        self.assertEqual(
            result.matches[0].formal_name_source, "source_reference.section"
        )
        self.assertTrue(
            any(path["path"] == "vector" for path in result.trace.retrieval_paths)
        )

    def test_formal_name_falls_back_to_unmodified_source_document_name(self) -> None:
        unit = _unit("benefits", "employee benefits enrollment", section=None)

        result = _service([unit]).discover("employee benefits", scope="demo")

        self.assertEqual(result.matches[0].formal_name, unit.source_reference.name)
        self.assertEqual(
            result.matches[0].formal_name_source, "source_reference.name"
        )

    def test_discover_keeps_permission_status_and_scope_hard_boundaries(self) -> None:
        allowed = _unit("allowed", "benefit enrollment guidance", section="Benefits")
        denied = _unit(
            "denied",
            "benefit enrollment guidance",
            section="Secret Benefits",
            permissions=["benefits.secret"],
        )
        inactive = _unit(
            "inactive",
            "benefit enrollment guidance",
            section="Old Benefits",
            status="revoked",
        )
        outside = _unit(
            "outside",
            "benefit enrollment guidance",
            section="Other Benefits",
            scope="other",
        )

        result = _service([allowed, denied, inactive, outside]).discover(
            "benefit enrollment guidance", scope="demo"
        )
        serialized = result.trace.to_json()

        self.assertEqual([match.evidence.unit_id for match in result.matches], ["allowed"])
        for rejected_id in ("denied", "inactive", "outside"):
            self.assertNotIn(rejected_id, serialized)

    def test_exact_formal_name_keeps_keyword_identity_priority_and_trace(self) -> None:
        exact = _unit(
            "exact",
            "official payment escalation approval rules",
            section="Payment Escalation Policy",
        )
        related = _unit(
            "related",
            "payment escalation review guidance",
            section="Payment Review Guide",
        )

        result = _service([related, exact]).discover(
            "Payment Escalation Policy", scope="demo"
        )

        self.assertEqual(result.matches[0].evidence.unit_id, "exact")
        self.assertEqual(result.matches[0].formal_name, "Payment Escalation Policy")
        self.assertEqual(result.trace.raw_query, "Payment Escalation Policy")
        self.assertEqual(result.trace.discover_summary["request"], "discover")
        self.assertEqual(
            result.trace.discover_summary["formal_names"][0]["formal_name"],
            "Payment Escalation Policy",
        )


if __name__ == "__main__":
    unittest.main()
