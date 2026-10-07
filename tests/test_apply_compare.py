"""Targeted tests for evidence-backed Apply/Compare organization."""

import unittest
from dataclasses import replace

from knowledge_system import (
    CaseContext,
    CaseSourceSpan,
    ComparisonOutcome,
    ExtractedRequirementFact,
    FactExtractionStatus,
    FactObservation,
    LogicalOutcome,
    compare_condition,
    extract_required_facts,
    extract_rule_structure,
    generate_information_requirements,
    organize_apply_comparisons,
)

from tests.test_information_requirements import _evidence


def _single(condition: str, description: str):
    evidence = _evidence(f"条件：{condition}\n结果：继续处理")
    extraction = extract_rule_structure(evidence)
    assert extraction.structure is not None
    requirement = generate_information_requirements(extraction)[0]
    case = CaseContext(raw_descriptions=(description,))
    fact = extract_required_facts((requirement,), case)[0]
    comparison = compare_condition(
        extraction.structure.thresholds[0],
        requirement,
        fact,
        evidence,
        case,
    )
    return extraction, comparison, case


def _fact(requirement, description_index, text, value, unit):
    return ExtractedRequirementFact(
        requirement_id=requirement.requirement_id,
        evidence_id=requirement.evidence_id,
        fact_type=requirement.fact_type,
        status=FactExtractionStatus.FOUND,
        value=value,
        observations=(
            FactObservation(
                value=value,
                source=CaseSourceSpan(
                    description_index, 0, len(text), text
                ),
                unit=unit,
            ),
        ),
    )


def _pair(relation: str, likes: int, amount: int | None):
    marker = "且" if relation == "AND" else "或"
    evidence = _evidence(
        f"条件：点赞数不少于50个 {marker} 金额不超过100元\n"
        "结果：继续处理"
    )
    extraction = extract_rule_structure(evidence)
    assert extraction.structure is not None
    like_requirement, amount_requirement = generate_information_requirements(
        extraction
    )
    descriptions = (f"点赞数是{likes}个",)
    if amount is not None:
        descriptions = (*descriptions, f"金额是{amount}元")
    case = CaseContext(raw_descriptions=descriptions)
    like_fact = _fact(like_requirement, 0, descriptions[0], likes, "个")
    if amount is None:
        amount_fact = ExtractedRequirementFact(
            requirement_id=amount_requirement.requirement_id,
            evidence_id=amount_requirement.evidence_id,
            fact_type=amount_requirement.fact_type,
            status=FactExtractionStatus.MISSING,
            reason="required_fact_not_found",
        )
    else:
        amount_fact = _fact(
            amount_requirement, 1, descriptions[1], amount, "元"
        )
    comparisons = (
        compare_condition(
            extraction.structure.thresholds[0],
            like_requirement,
            like_fact,
            evidence,
            case,
        ),
        compare_condition(
            extraction.structure.thresholds[1],
            amount_requirement,
            amount_fact,
            evidence,
            case,
        ),
    )
    return extraction, comparisons, case


class ApplyCompareTest(unittest.TestCase):
    def test_classifies_all_single_condition_outcomes(self) -> None:
        satisfied = _single("点赞数不少于50个", "点赞数是52个")
        not_satisfied = _single("点赞数不少于50个", "点赞数是49个")
        missing = _single("点赞数不少于50个", "无相关事实")
        extraction, comparison, case = _single(
            "点赞数不少于50个", "点赞数是52"
        )
        unknown = (extraction, comparison, case)

        expected = (
            (satisfied, "matched", ComparisonOutcome.SATISFIED),
            (not_satisfied, "not_satisfied", ComparisonOutcome.NOT_SATISFIED),
            (missing, "missing", ComparisonOutcome.MISSING),
            (unknown, "unknown", ComparisonOutcome.UNKNOWN),
        )
        for inputs, bucket, outcome in expected:
            with self.subTest(bucket=bucket):
                extraction, comparison, case = inputs
                result = organize_apply_comparisons(
                    extraction, (comparison,), case
                )
                self.assertEqual(getattr(result, bucket), (comparison,))
                self.assertEqual(comparison.outcome, outcome)
                other = {
                    "matched",
                    "missing",
                    "unknown",
                    "not_satisfied",
                } - {bucket}
                self.assertTrue(all(not getattr(result, name) for name in other))
                self.assertFalse(hasattr(result, "eligibility"))
                self.assertFalse(hasattr(result, "approval"))

    def test_and_truth_is_conservative_and_retains_uncertainty(self) -> None:
        extraction, comparisons, case = _pair("AND", 52, None)
        uncertain = organize_apply_comparisons(extraction, comparisons, case)
        self.assertEqual(uncertain.logical_relation, "AND")
        self.assertEqual(
            uncertain.logical_outcome, LogicalOutcome.INDETERMINATE
        )
        self.assertEqual(uncertain.matched, (comparisons[0],))
        self.assertEqual(uncertain.missing, (comparisons[1],))

        extraction, comparisons, case = _pair("AND", 49, None)
        definite_false = organize_apply_comparisons(
            extraction, comparisons, case
        )
        self.assertEqual(
            definite_false.logical_outcome, LogicalOutcome.NOT_SATISFIED
        )
        self.assertEqual(definite_false.not_satisfied, (comparisons[0],))
        self.assertEqual(definite_false.missing, (comparisons[1],))

        extraction, comparisons, case = _pair("AND", 52, 90)
        all_true = organize_apply_comparisons(extraction, comparisons, case)
        self.assertEqual(all_true.logical_outcome, LogicalOutcome.SATISFIED)

    def test_or_truth_is_conservative_and_retains_uncertainty(self) -> None:
        extraction, comparisons, case = _pair("OR", 49, None)
        uncertain = organize_apply_comparisons(extraction, comparisons, case)
        self.assertEqual(uncertain.logical_relation, "OR")
        self.assertEqual(
            uncertain.logical_outcome, LogicalOutcome.INDETERMINATE
        )
        self.assertEqual(uncertain.not_satisfied, (comparisons[0],))
        self.assertEqual(uncertain.missing, (comparisons[1],))

        extraction, comparisons, case = _pair("OR", 52, None)
        definite_true = organize_apply_comparisons(extraction, comparisons, case)
        self.assertEqual(
            definite_true.logical_outcome, LogicalOutcome.SATISFIED
        )
        self.assertEqual(definite_true.matched, (comparisons[0],))
        self.assertEqual(definite_true.missing, (comparisons[1],))

        extraction, comparisons, case = _pair("OR", 49, 101)
        all_false = organize_apply_comparisons(extraction, comparisons, case)
        self.assertEqual(
            all_false.logical_outcome, LogicalOutcome.NOT_SATISFIED
        )

    def test_short_circuit_keeps_unevaluated_threshold_provenance(self) -> None:
        extraction, comparisons, case = _pair("AND", 49, 90)
        and_result = organize_apply_comparisons(
            extraction, (comparisons[0],), case
        )
        self.assertEqual(
            and_result.logical_outcome, LogicalOutcome.NOT_SATISFIED
        )
        self.assertEqual(
            and_result.unevaluated_conditions,
            (extraction.structure.thresholds[1].source,),
        )

        extraction, comparisons, case = _pair("OR", 52, 101)
        or_result = organize_apply_comparisons(
            extraction, (comparisons[0],), case
        )
        self.assertEqual(or_result.logical_outcome, LogicalOutcome.SATISFIED)
        self.assertEqual(
            or_result.unevaluated_conditions,
            (extraction.structure.thresholds[1].source,),
        )

    def test_duplicate_identity_is_counted_once_with_mapping(self) -> None:
        extraction, comparison, case = _single(
            "点赞数不少于50个", "点赞数是52个"
        )

        result = organize_apply_comparisons(
            extraction, (comparison, comparison), case
        )

        self.assertEqual(result.comparisons, (comparison,))
        self.assertEqual(result.matched, (comparison,))
        self.assertEqual(len(result.duplicate_groups), 1)
        self.assertIs(result.duplicate_groups[0].canonical, comparison)
        self.assertEqual(result.duplicate_groups[0].duplicates, (comparison,))

    def test_same_text_at_different_offsets_is_not_deduplicated(self) -> None:
        evidence = _evidence(
            "条件：点赞数不少于50个\n"
            "条件：点赞数不少于50个\n"
            "逻辑关系：AND\n结果：继续处理"
        )
        extraction = extract_rule_structure(evidence)
        assert extraction.structure is not None
        requirements = generate_information_requirements(extraction)
        case = CaseContext(raw_descriptions=("点赞数是52个",))
        comparisons = []
        for threshold, requirement in zip(
            extraction.structure.thresholds, requirements, strict=True
        ):
            fact = _fact(requirement, 0, case.raw_descriptions[0], 52, "个")
            comparisons.append(
                compare_condition(
                    threshold, requirement, fact, evidence, case
                )
            )

        result = organize_apply_comparisons(extraction, comparisons, case)

        self.assertEqual(len(result.comparisons), 2)
        self.assertEqual(result.duplicate_groups, ())
        self.assertNotEqual(
            result.comparisons[0].requirement.condition.start,
            result.comparisons[1].requirement.condition.start,
        )

    def test_provenance_is_revalidated_and_inputs_are_immutable(self) -> None:
        extraction, comparison, case = _single(
            "点赞数不少于50个", "点赞数是52个"
        )
        snapshots = (extraction, comparison, case)

        result = organize_apply_comparisons(
            extraction, (comparison,), case
        )

        self.assertIs(result.extraction, extraction)
        self.assertIs(result.comparisons[0], comparison)
        self.assertEqual((extraction, comparison, case), snapshots)
        self.assertEqual(
            result.comparisons[0].fact.observations[0].source.text,
            "点赞数是52个",
        )
        with self.assertRaisesRegex(ValueError, "outcome"):
            organize_apply_comparisons(
                extraction,
                (
                    replace(
                        comparison,
                        outcome=ComparisonOutcome.NOT_SATISFIED,
                    ),
                ),
                case,
            )


if __name__ == "__main__":
    unittest.main()
