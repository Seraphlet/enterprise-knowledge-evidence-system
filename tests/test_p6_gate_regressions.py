"""Exact regressions for the two P6 Gate integration failures."""

import unittest
from dataclasses import replace

from knowledge_system import (
    CaseContext,
    CaseSourceSpan,
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


class P6GateRegressionTest(unittest.TestCase):
    def test_inline_and_or_create_atomic_end_to_end_requirements(self) -> None:
        for marker, relation in (("且", "AND"), ("或", "OR")):
            with self.subTest(relation=relation):
                evidence = _evidence(
                    f"条件：点赞数不少于50个 {marker} 金额不超过100元\n"
                    "结果：继续处理"
                )
                extraction = extract_rule_structure(evidence)
                requirements = generate_information_requirements(extraction)
                case = CaseContext(
                    raw_descriptions=("点赞数是52个", "金额是90元")
                )
                snapshots = (evidence, extraction, requirements, case)

                facts = extract_required_facts(requirements, case)
                comparisons = tuple(
                    compare_condition(
                        requirement.threshold,
                        requirement,
                        fact,
                        evidence,
                        case,
                    )
                    for requirement, fact in zip(
                        requirements, facts, strict=True
                    )
                )
                result = organize_apply_comparisons(
                    extraction, comparisons, case
                )

                self.assertEqual(len(requirements), 2)
                self.assertEqual(
                    tuple(item.metric_hints for item in requirements),
                    (("点赞数",), ("金额",)),
                )
                self.assertEqual(
                    tuple(item.condition.text for item in requirements),
                    ("点赞数不少于50个", "金额不超过100元"),
                )
                self.assertEqual(
                    tuple(item.threshold.metric for item in requirements),
                    ("点赞数", "金额"),
                )
                self.assertEqual(
                    tuple(item.value for item in facts), (52, 90)
                )
                self.assertTrue(
                    all(item.status is FactExtractionStatus.FOUND for item in facts)
                )
                self.assertEqual(result.matched, comparisons)
                self.assertEqual(result.logical_relation, relation)
                self.assertEqual(
                    result.logical_outcome, LogicalOutcome.SATISFIED
                )
                for requirement in requirements:
                    source = requirement.condition
                    self.assertEqual(
                        evidence.original_content[source.start:source.end],
                        source.text,
                    )
                    self.assertEqual(
                        requirement.evidence_id, evidence.evidence_id
                    )
                    self.assertEqual(
                        requirement.source_reference, evidence.source_reference
                    )
                    self.assertEqual(requirement.lineage, evidence.lineage)
                self.assertEqual(
                    (evidence, extraction, requirements, case), snapshots
                )

    def test_equal_distinct_duplicate_instances_map_to_one_canonical(self) -> None:
        evidence = _evidence("条件：点赞数不少于50个\n结果：继续处理")
        extraction = extract_rule_structure(evidence)
        requirement = generate_information_requirements(extraction)[0]
        case = CaseContext(raw_descriptions=("点赞数是52个",))

        facts = extract_required_facts((requirement, requirement), case)
        comparisons = tuple(
            compare_condition(
                requirement.threshold,
                requirement,
                fact,
                evidence,
                case,
            )
            for fact in facts
        )
        result = organize_apply_comparisons(extraction, comparisons, case)

        self.assertIsNot(facts[0], facts[1])
        self.assertEqual(facts[0], facts[1])
        self.assertIsNot(comparisons[0], comparisons[1])
        self.assertEqual(result.comparisons, (comparisons[0],))
        self.assertEqual(len(result.duplicate_groups), 1)
        self.assertEqual(
            result.duplicate_groups[0].duplicates, (comparisons[1],)
        )

    def test_equivalent_distinct_atomic_threshold_instances_are_accepted(self) -> None:
        evidence = _evidence("条件：点赞数不少于50个\n结果：继续处理")
        extraction = extract_rule_structure(evidence)
        requirement = generate_information_requirements(extraction)[0]
        case = CaseContext(raw_descriptions=("点赞数是52个",))
        fact = extract_required_facts((requirement,), case)[0]
        assert requirement.threshold is not None
        condition_copy = replace(requirement.threshold)
        requirement_copy = replace(
            requirement, threshold=replace(requirement.threshold)
        )
        comparison = compare_condition(
            condition_copy, requirement_copy, fact, evidence, case
        )

        result = organize_apply_comparisons(
            extraction, (comparison,), case
        )

        self.assertIsNot(condition_copy, requirement.threshold)
        self.assertIsNot(requirement_copy.threshold, requirement.threshold)
        self.assertEqual(result.matched, (comparison,))

    def test_atomic_requirement_threshold_tampering_is_rejected(self) -> None:
        evidence = _evidence("条件：点赞数不少于50个\n结果：继续处理")
        extraction = extract_rule_structure(evidence)
        requirement = generate_information_requirements(extraction)[0]
        case = CaseContext(raw_descriptions=("点赞数是52个",))
        fact = extract_required_facts((requirement,), case)[0]
        assert requirement.threshold is not None
        comparison = compare_condition(
            requirement.threshold, requirement, fact, evidence, case
        )
        threshold = requirement.threshold
        tampered_requirements = {
            "value": replace(
                requirement, threshold=replace(threshold, value=999)
            ),
            "metric": replace(
                requirement, threshold=replace(threshold, metric="金额")
            ),
            "operator": replace(
                requirement, threshold=replace(threshold, operator="<")
            ),
            "unit": replace(
                requirement, threshold=replace(threshold, unit="元")
            ),
            "source": replace(
                requirement,
                threshold=replace(
                    threshold,
                    source=replace(threshold.source, text="tampered"),
                ),
            ),
            "metric_hints": replace(requirement, metric_hints=("金额",)),
        }

        for field, tampered in tampered_requirements.items():
            with self.subTest(field=field):
                tampered_comparison = replace(
                    comparison, requirement=tampered
                )
                with self.assertRaisesRegex(
                    ValueError, "canonical threshold|canonical metric"
                ):
                    organize_apply_comparisons(
                        extraction,
                        (comparison, tampered_comparison),
                        case,
                    )

    def test_same_identity_with_different_fact_provenance_is_rejected(self) -> None:
        evidence = _evidence("条件：点赞数不少于50个\n结果：继续处理")
        extraction = extract_rule_structure(evidence)
        requirement = generate_information_requirements(extraction)[0]
        case = CaseContext(
            raw_descriptions=("点赞数是52个", "点赞数是53个")
        )
        comparisons = []
        for index, value in enumerate((52, 53)):
            text = case.raw_descriptions[index]
            fact = ExtractedRequirementFact(
                requirement_id=requirement.requirement_id,
                evidence_id=requirement.evidence_id,
                fact_type=requirement.fact_type,
                status=FactExtractionStatus.FOUND,
                value=value,
                observations=(
                    FactObservation(
                        value=value,
                        source=CaseSourceSpan(index, 0, len(text), text),
                        unit="个",
                    ),
                ),
            )
            comparisons.append(
                compare_condition(
                    requirement.threshold,
                    requirement,
                    fact,
                    evidence,
                    case,
                )
            )

        with self.assertRaisesRegex(ValueError, "conflicting inputs"):
            organize_apply_comparisons(extraction, comparisons, case)


if __name__ == "__main__":
    unittest.main()
