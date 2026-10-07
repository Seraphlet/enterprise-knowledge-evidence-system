"""Targeted tests for the single-condition deterministic comparator."""

import unittest
from dataclasses import replace
from decimal import Decimal

from knowledge_system import (
    CaseContext,
    CaseSourceSpan,
    ComparisonOutcome,
    ExactCondition,
    ExtractedRequirementFact,
    FactExtractionStatus,
    FactObservation,
    RequirementCertainty,
    SourceFragment,
    compare_condition,
    extract_required_facts,
    extract_rule_structure,
    generate_information_requirements,
)

from tests.test_information_requirements import _evidence


def _numeric_inputs(condition: str, description: str):
    evidence = _evidence(f"条件：{condition}\n结果：继续处理")
    extraction = extract_rule_structure(evidence)
    assert extraction.structure is not None
    requirement = generate_information_requirements(extraction)[0]
    case = CaseContext(raw_descriptions=(description,))
    fact = extract_required_facts(
        (requirement,), case
    )[0]
    return extraction.structure.thresholds[0], requirement, fact, evidence, case


def _exact_inputs(expected: bool | str, actual: bool | str):
    condition_text = f"状态等于{expected}"
    evidence = _evidence(f"条件：{condition_text}\n结果：继续处理")
    requirement = generate_information_requirements(
        extract_rule_structure(evidence)
    )[0]
    source = requirement.condition
    case_text = f"状态是{actual}"
    fact = ExtractedRequirementFact(
        requirement_id=requirement.requirement_id,
        evidence_id=requirement.evidence_id,
        fact_type=requirement.fact_type,
        status=FactExtractionStatus.FOUND,
        value=actual,
        observations=(
            FactObservation(
                value=actual,
                source=CaseSourceSpan(0, 0, len(case_text), case_text),
            ),
        ),
    )
    case = CaseContext(raw_descriptions=(case_text,))
    return ExactCondition(source, expected), requirement, fact, evidence, case


class DeterministicComparisonTest(unittest.TestCase):
    def test_numeric_operators_handle_equal_boundaries(self) -> None:
        cases = (
            ("点赞数不少于50个", "点赞数是50个", ComparisonOutcome.SATISFIED),
            ("点赞数超过50个", "点赞数是50个", ComparisonOutcome.NOT_SATISFIED),
            ("点赞数不超过50个", "点赞数是50个", ComparisonOutcome.SATISFIED),
            ("点赞数小于50个", "点赞数是50个", ComparisonOutcome.NOT_SATISFIED),
            ("点赞数等于50个", "点赞数是50个", ComparisonOutcome.SATISFIED),
        )

        for condition, description, expected in cases:
            with self.subTest(condition=condition):
                threshold, requirement, fact, evidence, case = _numeric_inputs(
                    condition, description
                )
                result = compare_condition(
                    threshold, requirement, fact, evidence, case
                )
                self.assertEqual(result.outcome, expected)

    def test_numeric_strict_sides_are_deterministic(self) -> None:
        cases = (
            ("点赞数不少于50个", "点赞数是49个", ComparisonOutcome.NOT_SATISFIED),
            ("点赞数超过50个", "点赞数是51个", ComparisonOutcome.SATISFIED),
            ("点赞数不超过50个", "点赞数是51个", ComparisonOutcome.NOT_SATISFIED),
            ("点赞数小于50个", "点赞数是49个", ComparisonOutcome.SATISFIED),
        )
        for condition, description, expected in cases:
            with self.subTest(condition=condition):
                threshold, requirement, fact, evidence, case = _numeric_inputs(
                    condition, description
                )
                self.assertEqual(
                    compare_condition(
                        threshold, requirement, fact, evidence, case
                    ).outcome,
                    expected,
                )

    def test_unreliable_numeric_type_or_unit_is_unknown(self) -> None:
        threshold, requirement, no_unit, evidence, case = _numeric_inputs(
            "点赞数不少于50个", "点赞数是50"
        )
        wrong_type = replace(
            no_unit,
            value="50",
            observations=tuple(
                replace(item, value="50") for item in no_unit.observations
            ),
        )

        unit_result = compare_condition(
            threshold, requirement, no_unit, evidence, case
        )
        type_result = compare_condition(
            threshold, requirement, wrong_type, evidence, case
        )

        self.assertEqual(unit_result.outcome, ComparisonOutcome.UNKNOWN)
        self.assertEqual(unit_result.reason, "numeric_unit_incompatible")
        self.assertEqual(type_result.outcome, ComparisonOutcome.UNKNOWN)
        self.assertEqual(type_result.reason, "numeric_type_incompatible")

    def test_non_finite_numeric_values_are_unknown_before_comparison(self) -> None:
        threshold, requirement, fact, evidence, case = _numeric_inputs(
            "点赞数不少于50个", "点赞数是50个"
        )
        non_finite = (
            float("nan"),
            float("inf"),
            float("-inf"),
            Decimal("NaN"),
            Decimal("Infinity"),
            Decimal("-Infinity"),
        )

        for value in non_finite:
            with self.subTest(location="fact", value=str(value)):
                invalid_fact = replace(
                    fact,
                    value=value,
                    observations=tuple(
                        replace(item, value=value) for item in fact.observations
                    ),
                )
                result = compare_condition(
                    threshold, requirement, invalid_fact, evidence, case
                )
                self.assertEqual(result.outcome, ComparisonOutcome.UNKNOWN)
                self.assertNotIn(
                    result.outcome,
                    {
                        ComparisonOutcome.SATISFIED,
                        ComparisonOutcome.NOT_SATISFIED,
                    },
                )

            with self.subTest(location="threshold", value=str(value)):
                result = compare_condition(
                    replace(threshold, value=value),
                    requirement,
                    fact,
                    evidence,
                    case,
                )
                self.assertEqual(result.outcome, ComparisonOutcome.UNKNOWN)

    def test_signaling_nan_is_classified_before_value_equality(self) -> None:
        threshold, requirement, fact, evidence, case = _numeric_inputs(
            "点赞数不少于50个", "点赞数是50个"
        )
        signaling_nan = Decimal("sNaN")
        quiet_nan = Decimal("NaN")

        cases = (
            replace(
                fact,
                value=signaling_nan,
                observations=tuple(
                    replace(item, value=signaling_nan)
                    for item in fact.observations
                ),
            ),
            replace(fact, value=signaling_nan),
            replace(
                fact,
                observations=tuple(
                    replace(item, value=signaling_nan)
                    for item in fact.observations
                ),
            ),
            replace(
                fact,
                value=quiet_nan,
                observations=tuple(
                    replace(item, value=quiet_nan)
                    for item in fact.observations
                ),
            ),
        )
        for invalid_fact in cases:
            with self.subTest(
                fact_value=str(invalid_fact.value),
                observation_value=str(invalid_fact.observations[0].value),
            ):
                result = compare_condition(
                    threshold, requirement, invalid_fact, evidence, case
                )
                self.assertEqual(result.outcome, ComparisonOutcome.UNKNOWN)
                self.assertEqual(
                    result.reason, "numeric_value_not_finite_real"
                )

        finite_decimal = Decimal("50")
        finite_fact = replace(
            fact,
            value=finite_decimal,
            observations=tuple(
                replace(item, value=finite_decimal)
                for item in fact.observations
            ),
        )
        finite_result = compare_condition(
            threshold, requirement, finite_fact, evidence, case
        )
        self.assertEqual(finite_result.outcome, ComparisonOutcome.UNKNOWN)
        self.assertEqual(finite_result.reason, "numeric_type_incompatible")

    def test_boolean_and_enum_are_strict_exact_comparisons(self) -> None:
        bool_condition, bool_requirement, bool_fact, bool_evidence, bool_case = (
            _exact_inputs(True, True)
        )
        enum_condition, enum_requirement, enum_fact, enum_evidence, enum_case = _exact_inputs(
            "APPROVED", "approved"
        )

        bool_result = compare_condition(
            bool_condition, bool_requirement, bool_fact, bool_evidence, bool_case
        )
        enum_result = compare_condition(
            enum_condition, enum_requirement, enum_fact, enum_evidence, enum_case
        )

        self.assertEqual(bool_result.outcome, ComparisonOutcome.SATISFIED)
        self.assertEqual(enum_result.outcome, ComparisonOutcome.NOT_SATISFIED)

    def test_boolean_does_not_compare_as_integer(self) -> None:
        condition, requirement, fact, evidence, case = _exact_inputs(True, True)
        integer_fact = replace(
            fact,
            value=1,
            observations=tuple(
                replace(item, value=1) for item in fact.observations
            ),
        )

        result = compare_condition(
            condition, requirement, integer_fact, evidence, case
        )

        self.assertEqual(result.outcome, ComparisonOutcome.UNKNOWN)
        self.assertEqual(result.reason, "exact_type_incompatible")

    def test_exact_condition_rejects_value_not_present_in_source(self) -> None:
        condition, _, _, _, _ = _exact_inputs("APPROVED", "APPROVED")

        with self.assertRaisesRegex(ValueError, "not explicit"):
            ExactCondition(condition.source, "PENDING")

    def test_missing_and_unknown_facts_stay_explicit(self) -> None:
        threshold, requirement, found, evidence, case = _numeric_inputs(
            "点赞数不少于50个", "点赞数是50个"
        )
        missing = replace(
            found,
            status=FactExtractionStatus.MISSING,
            value=None,
            observations=(),
            reason="required_fact_not_found",
        )
        unknown = replace(
            found,
            status=FactExtractionStatus.UNKNOWN,
            value=None,
            reason="conflicting_case_values",
        )

        self.assertEqual(
            compare_condition(
                threshold, requirement, missing, evidence, case
            ).outcome,
            ComparisonOutcome.MISSING,
        )
        unknown_result = compare_condition(
            threshold, requirement, unknown, evidence, case
        )
        self.assertEqual(unknown_result.outcome, ComparisonOutcome.UNKNOWN)
        self.assertEqual(unknown_result.reason, "conflicting_case_values")

    def test_semantic_or_ambiguous_condition_requires_human(self) -> None:
        condition, requirement, fact, evidence, case = _exact_inputs(
            "APPROVED", "APPROVED"
        )
        semantic = compare_condition(
            condition.source, requirement, fact, evidence, case
        )
        ambiguous = compare_condition(
            condition,
            replace(requirement, certainty=RequirementCertainty.HUMAN_REQUIRED),
            fact,
            evidence,
            case,
        )

        self.assertEqual(semantic.outcome, ComparisonOutcome.HUMAN_REQUIRED)
        self.assertEqual(semantic.reason, "non_mechanical_condition")
        self.assertEqual(ambiguous.outcome, ComparisonOutcome.HUMAN_REQUIRED)
        self.assertEqual(ambiguous.reason, "requirement_human_required")

    def test_result_retains_inputs_and_rejects_mismatched_links(self) -> None:
        threshold, requirement, fact, evidence, case = _numeric_inputs(
            "点赞数不少于50个", "点赞数是50个"
        )
        snapshots = (threshold, requirement, fact)

        result = compare_condition(
            threshold, requirement, fact, evidence, case
        )

        self.assertIs(result.condition, threshold)
        self.assertIs(result.requirement, requirement)
        self.assertIs(result.fact, fact)
        self.assertEqual((threshold, requirement, fact), snapshots)
        self.assertEqual(
            result.fact.observations[0].source.text, "点赞数是50个"
        )
        with self.assertRaisesRegex(ValueError, "requirement_id"):
            compare_condition(
                threshold,
                requirement,
                replace(fact, requirement_id="wrong"),
                evidence,
                case,
            )

    def test_case_spans_and_canonical_evidence_identity_are_validated(self) -> None:
        threshold, requirement, fact, evidence, case = _numeric_inputs(
            "点赞数不少于50个", "点赞数是50个"
        )
        source = fact.observations[0].source
        invalid_spans = (
            replace(source, description_index=-1),
            replace(source, start=-1),
            replace(source, end=99),
            replace(source, text="篡改文本"),
        )

        for invalid_span in invalid_spans:
            with self.subTest(span=invalid_span):
                invalid_fact = replace(
                    fact,
                    observations=(
                        replace(fact.observations[0], source=invalid_span),
                    ),
                )
                with self.assertRaisesRegex(ValueError, "fact observation"):
                    compare_condition(
                        threshold,
                        requirement,
                        invalid_fact,
                        evidence,
                        case,
                    )

        forged_requirement = replace(requirement, evidence_id="wrong-evidence")
        forged_fact = replace(fact, evidence_id="wrong-evidence")
        with self.assertRaisesRegex(ValueError, "does not match Evidence"):
            compare_condition(
                threshold,
                forged_requirement,
                forged_fact,
                evidence,
                case,
            )

        valid = compare_condition(
            threshold, requirement, fact, evidence, case
        )
        self.assertEqual(valid.outcome, ComparisonOutcome.SATISFIED)


if __name__ == "__main__":
    unittest.main()
