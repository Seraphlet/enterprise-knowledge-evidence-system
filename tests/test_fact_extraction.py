"""Targeted tests for on-demand, rule-driven Case fact extraction."""

import unittest
import unicodedata
from dataclasses import replace

from knowledge_system import (
    CaseContext,
    ExtractedFact,
    FactExtractionStatus,
    RequirementCertainty,
    extract_required_facts,
    extract_rule_structure,
    generate_information_requirements,
)

from tests.test_information_requirements import _evidence


def _like_requirement():
    evidence = _evidence("条件：点赞数不少于50个\n结果：可以升级")
    extraction = extract_rule_structure(evidence)
    return evidence, generate_information_requirements(extraction)[0]


def _amount_requirement():
    evidence = _evidence("条件：金额不超过1000元\n结果：需要复核")
    extraction = extract_rule_structure(evidence)
    return evidence, generate_information_requirements(extraction)[0]


class FactExtractionTest(unittest.TestCase):
    def test_extracts_only_required_numeric_fact_with_case_provenance(self) -> None:
        evidence, requirement = _like_requirement()
        case = CaseContext(
            raw_descriptions=("预算是1000元", "  现在52个赞\n")
        )

        results = extract_required_facts((requirement,), case)

        self.assertEqual(len(results), 1)
        result = results[0]
        self.assertEqual(result.status, FactExtractionStatus.FOUND)
        self.assertEqual(result.value, 52)
        self.assertEqual(result.requirement_id, requirement.requirement_id)
        self.assertEqual(result.evidence_id, evidence.evidence_id)
        self.assertEqual(result.fact_type, requirement.fact_type)
        self.assertEqual(len(result.observations), 1)
        source = result.observations[0].source
        self.assertEqual(source.description_index, 1)
        self.assertEqual(
            case.raw_descriptions[source.description_index][source.start : source.end],
            source.text,
        )
        self.assertIn("52", source.text)

    def test_unrelated_case_numbers_are_not_scanned_as_global_facts(self) -> None:
        _, requirement = _like_requirement()
        case = CaseContext(raw_descriptions=("预算是1000元", "账号编号是52"))

        result = extract_required_facts((requirement,), case)[0]

        self.assertEqual(result.status, FactExtractionStatus.MISSING)
        self.assertIsNone(result.value)
        self.assertEqual(result.observations, ())

    def test_missing_description_remains_explicit_missing(self) -> None:
        _, requirement = _like_requirement()

        result = extract_required_facts((requirement,), CaseContext())[0]

        self.assertEqual(result.status, FactExtractionStatus.MISSING)
        self.assertEqual(result.reason, "required_fact_not_found")

    def test_non_numeric_or_ambiguous_requirement_remains_unknown(self) -> None:
        non_numeric_evidence = _evidence("条件：账号状态正常\n结果：允许提交")
        non_numeric = generate_information_requirements(
            extract_rule_structure(non_numeric_evidence)
        )[0]
        _, numeric = _like_requirement()
        ambiguous = replace(
            numeric, certainty=RequirementCertainty.HUMAN_REQUIRED
        )
        case = CaseContext(raw_descriptions=("账号状态正常，当前52个赞",))

        results = extract_required_facts((non_numeric, ambiguous), case)

        self.assertEqual(
            tuple(item.status for item in results),
            (FactExtractionStatus.UNKNOWN, FactExtractionStatus.UNKNOWN),
        )
        self.assertEqual(results[0].reason, "no_single_numeric_metric")
        self.assertEqual(results[1].reason, "requirement_human_required")

    def test_duplicate_equal_values_are_deterministic_and_keep_all_sources(self) -> None:
        _, requirement = _like_requirement()
        case = CaseContext(
            raw_descriptions=("点赞数是52", "补充：现在52个赞")
        )

        first = extract_required_facts((requirement,), case)[0]
        second = extract_required_facts((requirement,), case)[0]

        self.assertEqual(first, second)
        self.assertEqual(first.status, FactExtractionStatus.FOUND)
        self.assertEqual(first.value, 52)
        self.assertEqual(len(first.observations), 2)
        self.assertEqual(
            tuple(item.source.description_index for item in first.observations),
            (0, 1),
        )

    def test_conflicting_values_are_unknown_and_not_silently_overwritten(self) -> None:
        _, requirement = _like_requirement()
        case = CaseContext(
            raw_descriptions=("现在52个赞", "更正：现在51个赞")
        )

        result = extract_required_facts((requirement,), case)[0]

        self.assertEqual(result.status, FactExtractionStatus.UNKNOWN)
        self.assertIsNone(result.value)
        self.assertEqual(result.reason, "conflicting_case_values")
        self.assertEqual(
            tuple(item.value for item in result.observations), (52, 51)
        )

    def test_same_description_cross_pattern_conflict_keeps_both_sources(self) -> None:
        _, requirement = _like_requirement()
        description = "首次记录点赞数是52，随后更正为53个赞"

        result = extract_required_facts(
            (requirement,), CaseContext(raw_descriptions=(description,))
        )[0]

        self.assertEqual(result.status, FactExtractionStatus.UNKNOWN)
        self.assertIsNone(result.value)
        self.assertEqual(result.reason, "conflicting_case_values")
        self.assertEqual(
            tuple(item.value for item in result.observations), (52, 53)
        )
        self.assertEqual(
            tuple(item.source.text for item in result.observations),
            ("点赞数是52", "53个赞"),
        )
        for observation in result.observations:
            source = observation.source
            self.assertEqual(description[source.start : source.end], source.text)

    def test_incompatible_adjacent_units_are_not_accepted(self) -> None:
        _, likes = _like_requirement()
        _, amount = _amount_requirement()

        like_result = extract_required_facts(
            (likes,), CaseContext(raw_descriptions=("点赞数是52元",))
        )[0]
        amount_result = extract_required_facts(
            (amount,), CaseContext(raw_descriptions=("金额是52个",))
        )[0]

        self.assertNotEqual(like_result.status, FactExtractionStatus.FOUND)
        self.assertNotEqual(amount_result.status, FactExtractionStatus.FOUND)
        self.assertEqual(like_result.status, FactExtractionStatus.MISSING)
        self.assertEqual(amount_result.status, FactExtractionStatus.MISSING)

    def test_numeric_continuation_equivalence_classes_are_not_truncated(self) -> None:
        _, requirement = _like_requirement()

        value_and_separator = (
            ("52.3", "."),
            ("52", ".."),
            ("52", "-"),
            ("52", "—"),
            ("52", "–"),
            ("52", "−"),
            ("52", "－"),
            ("52", "~"),
            ("52", "～"),
            ("52", "/"),
            ("52", ","),
            ("52", "，"),
            ("52", "、"),
        )
        whitespace = ("", " ", "\t")

        for value, separator in value_and_separator:
            for before in whitespace:
                for after in whitespace:
                    description = (
                        f"点赞数是{value}{before}{separator}{after}53个"
                    )
                    with self.subTest(description=description):
                        result = extract_required_facts(
                            (requirement,),
                            CaseContext(raw_descriptions=(description,)),
                        )[0]

                        self.assertEqual(result.status, FactExtractionStatus.MISSING)
                        self.assertIsNone(result.value)
                        self.assertEqual(result.observations, ())

    def test_unicode_category_numeric_continuations_are_not_truncated(self) -> None:
        _, requirement = _like_requirement()
        separators = ("／", "‒", "‑", "―", "﹣", "⁝", "∕", "÷", "↔")
        whitespace = ("", " ", "\t", "  ", "\u00a0", "\u3000", "\n")

        self.assertTrue(
            all(
                unicodedata.category(separator)[0] in {"P", "S"}
                for separator in separators
            )
        )
        for value in ("52", "52.3"):
            for separator in separators:
                for before in whitespace:
                    for after in whitespace:
                        for follower in ("53", "53.7"):
                            description = (
                                f"点赞数是{value}{before}{separator}"
                                f"{after}{follower}个"
                            )
                            with self.subTest(description=description):
                                result = extract_required_facts(
                                    (requirement,),
                                    CaseContext(raw_descriptions=(description,)),
                                )[0]

                                self.assertEqual(
                                    result.status, FactExtractionStatus.MISSING
                                )
                                self.assertIsNone(result.value)
                                self.assertEqual(result.observations, ())

    def test_decimal_and_sentence_punctuation_remain_valid(self) -> None:
        _, requirement = _like_requirement()

        for description, source_text in (
            ("点赞数是52", "点赞数是52"),
            ("点赞数是52.3个。", "点赞数是52.3个"),
            ("点赞数是52.3.", "点赞数是52.3"),
            ("点赞数是52，", "点赞数是52"),
            ("点赞数是52,", "点赞数是52"),
            ("点赞数是52。", "点赞数是52"),
            ("点赞数是52个!", "点赞数是52个"),
            ("点赞数是52个／说明", "点赞数是52个"),
        ):
            with self.subTest(description=description):
                result = extract_required_facts(
                    (requirement,),
                    CaseContext(raw_descriptions=(description,)),
                )[0]

                self.assertEqual(result.status, FactExtractionStatus.FOUND)
                self.assertEqual(result.value, 52.3 if ".3" in description else 52)
                self.assertEqual(
                    tuple(item.source.text for item in result.observations),
                    (source_text,),
                )

    def test_inputs_remain_unchanged_and_existing_extracted_facts_are_ignored(self) -> None:
        evidence, requirement = _like_requirement()
        case = CaseContext(
            raw_descriptions=("点赞数为52",),
            extracted_facts=(ExtractedFact("unrequested", 999, 0),),
        )
        original_evidence = evidence.to_dict()
        original_case = case
        original_requirement = requirement

        result = extract_required_facts((requirement,), case)[0]

        self.assertEqual(result.status, FactExtractionStatus.FOUND)
        self.assertIs(case, original_case)
        self.assertEqual(requirement, original_requirement)
        self.assertEqual(evidence.to_dict(), original_evidence)
        self.assertFalse(hasattr(result, "comparison_result"))

    def test_only_passed_candidate_requirements_are_processed_in_order(self) -> None:
        _, requirement = _like_requirement()

        none = extract_required_facts((), CaseContext(raw_descriptions=("52个赞",)))
        repeated = extract_required_facts(
            (requirement, requirement),
            CaseContext(raw_descriptions=("点赞数是52",)),
        )

        self.assertEqual(none, ())
        self.assertEqual(len(repeated), 2)
        self.assertEqual(
            tuple(item.requirement_id for item in repeated),
            (requirement.requirement_id, requirement.requirement_id),
        )


if __name__ == "__main__":
    unittest.main()
