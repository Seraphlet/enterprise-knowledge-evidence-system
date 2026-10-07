"""Targeted tests for lightweight QueryContext and CaseContext."""

import unittest

from knowledge_system import (
    ContextRelation,
    ExtractedFact,
    KnowledgeRequestType,
    apply_context_turn,
    create_task_context,
)


class ContextTest(unittest.TestCase):
    def test_query_scope_and_intent_are_separate_from_case_descriptions(self) -> None:
        original = "  这个情况是否符合升级规则？\n"
        context = create_task_context(
            original,
            scope="content-review",
            initial_case_description="初始有40个赞",
        )

        self.assertEqual(context.query.original_query, original)
        self.assertEqual(context.query.scope, "content-review")
        self.assertEqual(context.query.intent, KnowledgeRequestType.APPLY)
        self.assertEqual(context.case.raw_descriptions, ("初始有40个赞",))
        self.assertEqual(context.case.extracted_facts, ())

    def test_update_appends_case_without_rewriting_query_context(self) -> None:
        context = create_task_context(
            "这个情况是否符合升级规则？", scope="content-review"
        )
        fact = ExtractedFact("like_count", 52, source_description_index=0)

        transition = apply_context_turn(
            context,
            "  现在52个赞\n",
            ContextRelation.UPDATE,
            extracted_facts=(fact,),
        )

        self.assertIs(transition.context.query, context.query)
        self.assertEqual(
            transition.context.query.original_query,
            "这个情况是否符合升级规则？",
        )
        self.assertEqual(
            transition.context.case.raw_descriptions, ("  现在52个赞\n",)
        )
        self.assertEqual(transition.context.case.extracted_facts, (fact,))
        self.assertEqual(transition.turn_query, "  现在52个赞\n")
        self.assertTrue(transition.changed)

    def test_repeated_updates_preserve_raw_observations_and_fact_conflicts(self) -> None:
        context = create_task_context("这个情况是否符合升级规则？")
        first = apply_context_turn(
            context,
            "现在52个赞",
            ContextRelation.UPDATE,
            extracted_facts=(ExtractedFact("like_count", 52, 0),),
        ).context
        second = apply_context_turn(
            first,
            "更正：现在51个赞",
            ContextRelation.UPDATE,
            extracted_facts=(ExtractedFact("like_count", 51, 1),),
        ).context

        self.assertEqual(
            second.case.raw_descriptions,
            ("现在52个赞", "更正：现在51个赞"),
        )
        self.assertEqual(
            [fact.value for fact in second.case.extracted_facts], [52, 51]
        )

    def test_new_relation_replaces_query_and_does_not_inherit_case_or_scope(self) -> None:
        context = create_task_context(
            "是否符合升级规则？",
            scope="old-scope",
            initial_case_description="现在52个赞",
        )

        transition = apply_context_turn(
            context, "列出全部差旅制度", ContextRelation.NEW
        )

        self.assertEqual(
            transition.context.query.original_query, "列出全部差旅制度"
        )
        self.assertEqual(
            transition.context.query.intent, KnowledgeRequestType.BROWSE
        )
        self.assertIsNone(transition.context.query.scope)
        self.assertEqual(transition.context.case.raw_descriptions, ())
        self.assertTrue(transition.changed)

    def test_continue_and_ambiguous_do_not_mutate_or_inherit(self) -> None:
        context = create_task_context(
            "是否符合升级规则？", initial_case_description="现在52个赞"
        )

        for relation in (ContextRelation.CONTINUE, ContextRelation.AMBIGUOUS):
            with self.subTest(relation=relation):
                transition = apply_context_turn(
                    context, "那这个呢？", relation
                )
                self.assertIs(transition.context, context)
                self.assertFalse(transition.changed)
                self.assertEqual(transition.turn_query, "那这个呢？")

    def test_fact_names_are_open_and_not_a_global_case_schema(self) -> None:
        context = create_task_context("是否适用？")
        custom = ExtractedFact("campaign_specific_metric", "A-17", 0)

        updated = apply_context_turn(
            context,
            "活动指标是 A-17",
            ContextRelation.UPDATE,
            extracted_facts=(custom,),
        ).context

        self.assertEqual(updated.case.extracted_facts, (custom,))

    def test_relation_and_turn_types_are_explicit(self) -> None:
        context = create_task_context("是否适用？")
        with self.assertRaises(TypeError):
            apply_context_turn(context, None, ContextRelation.UPDATE)  # type: ignore[arg-type]
        with self.assertRaises(TypeError):
            apply_context_turn(context, "更新", "UPDATE")  # type: ignore[arg-type]


if __name__ == "__main__":
    unittest.main()
