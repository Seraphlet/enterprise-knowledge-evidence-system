"""Targeted tests for the deterministic clarification policy."""

import unittest

from knowledge_system import (
    ClarificationAction,
    KnowledgeBoundary,
    RetrievalOutcome,
    decide_clarification,
)


class ClarificationPolicyTest(unittest.TestCase):
    def test_only_unformable_retrieval_blocks_for_clarification(self) -> None:
        outcomes = tuple(RetrievalOutcome)

        decisions = {
            outcome: decide_clarification(
                "原始问题",
                outcome,
                evidence_ids=("e-1",)
                if outcome
                in {
                    RetrievalOutcome.USEFUL,
                    RetrievalOutcome.KNOWLEDGE_AMBIGUOUS,
                    RetrievalOutcome.EVIDENCE_INSUFFICIENT,
                }
                else (),
            )
            for outcome in outcomes
        }

        self.assertTrue(
            decisions[RetrievalOutcome.QUERY_INSUFFICIENT].blocks_for_clarification
        )
        for outcome, decision in decisions.items():
            if outcome is not RetrievalOutcome.QUERY_INSUFFICIENT:
                self.assertFalse(decision.blocks_for_clarification)

    def test_useful_evidence_proceeds_while_showing_missing_case_information(self) -> None:
        original = "  我是否符合升级规则？\r\n"
        missing = ("当前点赞数", "账号状态")

        decision = decide_clarification(
            original,
            RetrievalOutcome.USEFUL,
            evidence_ids=("e-rule-1",),
            missing_case_information=missing,
        )

        self.assertEqual(decision.original_query, original)
        self.assertEqual(
            decision.action, ClarificationAction.PROCEED_WITH_EVIDENCE
        )
        self.assertEqual(
            decision.boundary, KnowledgeBoundary.CASE_INFORMATION_MISSING
        )
        self.assertEqual(decision.evidence_ids, ("e-rule-1",))
        self.assertEqual(decision.missing_case_information, missing)
        self.assertIn(
            "case_information_missing_does_not_block", decision.audit_reasons
        )

    def test_useful_evidence_without_case_gap_has_no_boundary(self) -> None:
        decision = decide_clarification(
            "找到规则", RetrievalOutcome.USEFUL, evidence_ids=("e-1",)
        )

        self.assertEqual(decision.boundary, KnowledgeBoundary.NONE)
        self.assertEqual(
            decision.action, ClarificationAction.PROCEED_WITH_EVIDENCE
        )

    def test_query_insufficient_preserves_supplied_missing_items_exactly(self) -> None:
        original = "\ufeff  original\r\n"
        missing = ("case.metric", "account state")

        decision = decide_clarification(
            original,
            RetrievalOutcome.QUERY_INSUFFICIENT,
            missing_case_information=missing,
        )

        self.assertEqual(decision.original_query, original)
        self.assertEqual(decision.missing_case_information, missing)
        self.assertEqual(
            decision.action, ClarificationAction.BLOCK_FOR_CLARIFICATION
        )
        self.assertEqual(
            decision.boundary,
            KnowledgeBoundary.RETRIEVAL_QUERY_INSUFFICIENT,
        )

    def test_not_found_is_knowledge_missing_not_case_missing(self) -> None:
        decision = decide_clarification(
            "未知规则", RetrievalOutcome.NOT_FOUND,
            missing_case_information=("审批金额",),
        )

        self.assertEqual(decision.action, ClarificationAction.RETURN_BOUNDARY)
        self.assertEqual(decision.boundary, KnowledgeBoundary.KNOWLEDGE_MISSING)
        self.assertFalse(decision.blocks_for_clarification)

    def test_out_of_scope_and_knowledge_ambiguity_are_distinct(self) -> None:
        out_of_scope = decide_clarification(
            "查天气", RetrievalOutcome.OUT_OF_SCOPE
        )
        ambiguous = decide_clarification(
            "这个阈值是多少？",
            RetrievalOutcome.KNOWLEDGE_AMBIGUOUS,
            evidence_ids=("e-conflict-a", "e-conflict-b"),
        )

        self.assertEqual(out_of_scope.boundary, KnowledgeBoundary.OUT_OF_SCOPE)
        self.assertEqual(
            ambiguous.boundary, KnowledgeBoundary.KNOWLEDGE_AMBIGUOUS
        )
        self.assertEqual(
            ambiguous.evidence_ids, ("e-conflict-a", "e-conflict-b")
        )
        self.assertFalse(out_of_scope.blocks_for_clarification)
        self.assertFalse(ambiguous.blocks_for_clarification)

    def test_retrieved_but_insufficient_is_not_claimed_as_knowledge_missing(self) -> None:
        decision = decide_clarification(
            "规则细节",
            RetrievalOutcome.EVIDENCE_INSUFFICIENT,
            evidence_ids=("e-heading-only",),
        )

        self.assertEqual(
            decision.boundary, KnowledgeBoundary.EVIDENCE_INSUFFICIENT
        )
        self.assertNotEqual(decision.boundary, KnowledgeBoundary.KNOWLEDGE_MISSING)
        self.assertEqual(decision.action, ClarificationAction.RETURN_BOUNDARY)

    def test_invalid_outcome_evidence_combinations_are_rejected(self) -> None:
        with self.assertRaises(ValueError):
            decide_clarification("问题", RetrievalOutcome.USEFUL)
        with self.assertRaises(ValueError):
            decide_clarification(
                "问题",
                RetrievalOutcome.NOT_FOUND,
                evidence_ids=("impossible",),
            )
        with self.assertRaises(TypeError):
            decide_clarification("问题", "USEFUL")  # type: ignore[arg-type]


if __name__ == "__main__":
    unittest.main()
