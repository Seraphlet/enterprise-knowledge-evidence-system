"""Focused tests for the strict organization-only EvidenceMapPlan schema."""

import copy
import json
import unittest
from dataclasses import FrozenInstanceError, fields

from knowledge_system import (
    EvidenceMapGroup,
    EvidenceMapPlan,
    EvidenceMapPlanValidationError,
    EvidenceMapSection,
)


def _valid_mapping():
    return {
        "plan_id": "plan-1",
        "sections": [
            {
                "section_id": "section-b",
                "order": 10,
                "groups": [
                    {
                        "group_id": "group-b",
                        "order": 0,
                        "evidence_ids": ["evidence-4"],
                    }
                ],
            },
            {
                "section_id": "section-a",
                "order": 0,
                "groups": [
                    {
                        "group_id": "group-a2",
                        "order": 2,
                        "evidence_ids": ["evidence-3", "evidence-2"],
                    },
                    {
                        "group_id": "group-a1",
                        "order": 0,
                        "evidence_ids": ["evidence-1"],
                    },
                ],
            },
        ],
    }


class EvidenceMapPlanTest(unittest.TestCase):
    def test_valid_multi_section_group_plan_is_canonical(self) -> None:
        plan = EvidenceMapPlan.from_dict(_valid_mapping())

        self.assertEqual(
            tuple(item.section_id for item in plan.sections),
            ("section-a", "section-b"),
        )
        self.assertEqual(
            tuple(item.group_id for item in plan.sections[0].groups),
            ("group-a1", "group-a2"),
        )
        self.assertEqual(
            plan.sections[0].groups[1].evidence_ids,
            ("evidence-2", "evidence-3"),
        )
        self.assertEqual(
            set(plan.to_dict()), {"plan_id", "sections"}
        )

    def test_unknown_and_knowledge_fields_are_rejected_at_every_level(self) -> None:
        forbidden = (
            ((), "content"),
            ((), "comparison_result"),
            (("sections", 0), "source"),
            (("sections", 0), "page"),
            (("sections", 0, "groups", 0), "facts"),
            (("sections", 0, "groups", 0), "condition"),
            (("sections", 0, "groups", 0), "threshold"),
            (("sections", 0, "groups", 0), "consequence"),
            (("sections", 0, "groups", 0), "description"),
            (("sections", 0, "groups", 0), "url"),
            (("sections", 0, "groups", 0), "label"),
        )
        for path, field_name in forbidden:
            with self.subTest(path=path, field=field_name):
                value = _valid_mapping()
                target = value
                for part in path:
                    target = target[part]
                target[field_name] = "forbidden"
                with self.assertRaisesRegex(
                    EvidenceMapPlanValidationError, "unknown fields"
                ):
                    EvidenceMapPlan.from_dict(value)

    def test_type_identifier_and_order_edges_are_rejected(self) -> None:
        invalid_values = []
        bool_order = _valid_mapping()
        bool_order["sections"][0]["order"] = True
        invalid_values.append(bool_order)
        negative_order = _valid_mapping()
        negative_order["sections"][0]["groups"][0]["order"] = -1
        invalid_values.append(negative_order)
        float_order = _valid_mapping()
        float_order["sections"][0]["order"] = 1.0
        invalid_values.append(float_order)
        blank_id = _valid_mapping()
        blank_id["sections"][0]["section_id"] = " "
        invalid_values.append(blank_id)
        unstable_id = _valid_mapping()
        unstable_id["sections"][0]["groups"][0]["evidence_ids"] = [
            "evidence id"
        ]
        invalid_values.append(unstable_id)
        wrong_evidence_type = _valid_mapping()
        wrong_evidence_type["sections"][0]["groups"][0]["evidence_ids"] = [7]
        invalid_values.append(wrong_evidence_type)

        for index, value in enumerate(invalid_values):
            with self.subTest(index=index):
                with self.assertRaises(EvidenceMapPlanValidationError):
                    EvidenceMapPlan.from_dict(value)

    def test_duplicate_identity_order_and_evidence_rules_are_global_and_strict(self) -> None:
        variants = []
        section_id = _valid_mapping()
        section_id["sections"][1]["section_id"] = "section-b"
        variants.append(section_id)
        section_order = _valid_mapping()
        section_order["sections"][1]["order"] = 10
        variants.append(section_order)
        group_id = _valid_mapping()
        group_id["sections"][1]["groups"][0]["group_id"] = "group-b"
        variants.append(group_id)
        group_order = _valid_mapping()
        group_order["sections"][1]["groups"][1]["order"] = 2
        variants.append(group_order)
        within_group = _valid_mapping()
        within_group["sections"][1]["groups"][0]["evidence_ids"] = [
            "evidence-1",
            "evidence-1",
        ]
        variants.append(within_group)
        across_plan = _valid_mapping()
        across_plan["sections"][0]["groups"][0]["evidence_ids"] = [
            "evidence-1"
        ]
        variants.append(across_plan)

        for index, value in enumerate(variants):
            with self.subTest(index=index):
                with self.assertRaisesRegex(
                    EvidenceMapPlanValidationError, "duplicate"
                ):
                    EvidenceMapPlan.from_dict(value)

    def test_mapping_and_json_round_trip_are_deterministic_and_immutable(self) -> None:
        value = _valid_mapping()
        snapshot = copy.deepcopy(value)

        first = EvidenceMapPlan.from_dict(value)
        encoded = first.to_json()
        second = EvidenceMapPlan.from_json(encoded)

        self.assertEqual(value, snapshot)
        self.assertEqual(first, second)
        self.assertEqual(encoded, second.to_json())
        self.assertEqual(
            json.loads(encoded), first.to_dict()
        )
        with self.assertRaises(FrozenInstanceError):
            first.plan_id = "changed"

    def test_duplicate_json_keys_and_wrong_or_deep_nesting_are_rejected(self) -> None:
        with self.assertRaisesRegex(
            EvidenceMapPlanValidationError, "duplicate JSON field"
        ):
            EvidenceMapPlan.from_json(
                '{"plan_id":"plan-1","plan_id":"plan-2","sections":[]}'
            )

        wrong_sections = _valid_mapping()
        wrong_sections["sections"] = {"section_id": "section-a"}
        wrong_groups = _valid_mapping()
        wrong_groups["sections"][0]["groups"] = {"group_id": "group-a"}
        deep_evidence = _valid_mapping()
        deep_evidence["sections"][0]["groups"][0]["evidence_ids"] = [
            [[[["evidence-4"]]]]
        ]
        for value in (wrong_sections, wrong_groups, deep_evidence):
            with self.assertRaises(EvidenceMapPlanValidationError):
                EvidenceMapPlan.from_dict(value)

    def test_typed_constructors_require_nonempty_fixed_depth_tuples(self) -> None:
        group = EvidenceMapGroup("group-1", 0, ("evidence-1",))
        section = EvidenceMapSection("section-1", 0, (group,))
        plan = EvidenceMapPlan("plan-1", (section,))

        self.assertEqual(plan.sections[0].groups[0], group)
        with self.assertRaises(EvidenceMapPlanValidationError):
            EvidenceMapGroup("group-empty", 0, ())
        with self.assertRaises(EvidenceMapPlanValidationError):
            EvidenceMapSection("section-empty", 0, ())
        with self.assertRaises(EvidenceMapPlanValidationError):
            EvidenceMapPlan("plan-empty", ())

    def test_schema_checks_id_shape_but_not_candidate_membership(self) -> None:
        value = {
            "plan_id": "plan-unchecked",
            "sections": [
                {
                    "section_id": "section-1",
                    "order": 0,
                    "groups": [
                        {
                            "group_id": "group-1",
                            "order": 0,
                            "evidence_ids": ["evidence-not-in-candidate-set"],
                        }
                    ],
                }
            ],
        }

        plan = EvidenceMapPlan.from_dict(value)

        self.assertEqual(
            plan.sections[0].groups[0].evidence_ids,
            ("evidence-not-in-candidate-set",),
        )
        plan_fields = {item.name for item in fields(EvidenceMapPlan)}
        self.assertEqual(plan_fields, {"plan_id", "sections"})
        self.assertFalse(hasattr(plan, "hydrate"))
        self.assertFalse(hasattr(plan, "candidate_evidence"))


if __name__ == "__main__":
    unittest.main()
