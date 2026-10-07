"""Targeted tests for permission and status candidate filtering."""

import unittest

from knowledge_system import (
    FilterReason,
    KeywordIndex,
    KnowledgeLineage,
    KnowledgeUnit,
    ParseStatus,
    SourceReference,
    filter_candidates,
)


def _unit(unit_id: str, **metadata: object) -> KnowledgeUnit:
    return KnowledgeUnit(
        unit_id=unit_id,
        original_content="shared keyword",
        semantic_content="shared keyword",
        metadata=dict(metadata),
        source_reference=SourceReference(name=f"{unit_id}.html"),
        lineage=KnowledgeLineage(
            document_id=f"doc-{unit_id}",
            parser_version="p1",
            chunker_version="c1",
            processing_version="v1",
        ),
        parse_status=ParseStatus.SUCCESS,
    )


class CandidateFilterTest(unittest.TestCase):
    def test_unauthorized_content_never_enters_candidates(self) -> None:
        public = _unit("public")
        restricted = _unit("restricted", required_permissions=["finance.read"])

        result = filter_candidates([public, restricted])

        self.assertEqual([unit.unit_id for unit in result.candidates], ["public"])
        self.assertEqual(result.rejected[0].reason, FilterReason.PERMISSION_DENIED)

    def test_all_required_permissions_must_be_granted(self) -> None:
        restricted = _unit(
            "restricted", required_permissions=["finance.read", "audit.read"]
        )

        denied = filter_candidates(
            [restricted], granted_permissions=["finance.read"]
        )
        allowed = filter_candidates(
            [restricted], granted_permissions=["finance.read", "audit.read"]
        )

        self.assertEqual(denied.candidates, ())
        self.assertEqual(allowed.candidates, (restricted,))

    def test_explicitly_inactive_content_never_enters_candidates(self) -> None:
        units = [
            _unit("active", status="active"),
            _unit("expired", status="expired"),
            _unit("disabled", status="已失效"),
        ]

        result = filter_candidates(units)

        self.assertEqual([unit.unit_id for unit in result.candidates], ["active"])
        self.assertEqual(
            [item.reason for item in result.rejected],
            [FilterReason.INACTIVE_STATUS, FilterReason.INACTIVE_STATUS],
        )

    def test_chinese_inactive_status_regression_matrix(self) -> None:
        for status in ("禁用", "已撤销", "作废"):
            with self.subTest(status=status):
                result = filter_candidates([_unit(status, status=status)])

                self.assertEqual(result.candidates, ())
                self.assertEqual(len(result.rejected), 1)
                self.assertEqual(
                    result.rejected[0].reason, FilterReason.INACTIVE_STATUS
                )

    def test_malformed_permission_metadata_fails_closed(self) -> None:
        malformed = _unit("malformed", required_permissions={"role": "admin"})

        result = filter_candidates([malformed], granted_permissions=["admin"])

        self.assertEqual(result.candidates, ())
        self.assertEqual(
            result.rejected[0].reason, FilterReason.INVALID_PERMISSION_METADATA
        )

    def test_filter_preserves_order_and_is_separate_from_index(self) -> None:
        first = _unit("first")
        denied = _unit("denied", required_permissions=["secret"])
        last = _unit("last")

        filtered = filter_candidates([first, denied, last])
        evidence = KeywordIndex(filtered.candidates).search("shared keyword")

        self.assertEqual([unit.unit_id for unit in filtered.candidates], ["first", "last"])
        self.assertEqual([item.unit_id for item in evidence], ["first", "last"])
        self.assertNotIn("denied", [item.unit_id for item in evidence])


if __name__ == "__main__":
    unittest.main()
