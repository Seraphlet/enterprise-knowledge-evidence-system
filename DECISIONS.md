# DECISIONS.md — Frozen V0 Decisions

> Worker 不得在实现过程中“顺手优化”这些决策。若证据表明决策不可成立，按 BUILD 的 STRUCTURAL_BLOCK 流程暂停。

- D001 RAG core, not generic Agent.
- D002 Deterministic first, semantic fallback.
- D003 Keyword + Vector hybrid retrieval.
- D004 Query conditions ≠ Knowledge storage location.
- D005 Permission is Filter, not Ranking.
- D006 Chunk by knowledge structure, not fixed chars.
- D007 Source Evidence > Derived Structure.
- D008 Do not force all knowledge to structure.
- D009 Ambiguity is data.
- D010 No global Case Schema.
- D011 Rule-driven Fact Extraction.
- D012 No premature Concept Registry.
- D013 Rule Condition → Information Requirement.
- D014 Missing Information does not block by default.
- D015 Case Missing vs Knowledge Missing separate.
- D016 Deterministic comparison not delegated to LLM.
- D017 No full Semantic Rule Engine in V0.
- D018 Retrieved ≠ Answerable.
- D019 Evidence anomaly → Recovery before Knowledge Gap.
- D020 Distinguish Retrieval / Processing / Source Problem.
- D021 Knowledge Lineage required.
- D022 Organizer can see original text.
- D023 LLM cannot directly write final Evidence.
- D024 Organizer only has organization authority.
- D025 LLM writes directory, program fills body.
- D026 Evidence ID Reference Guard.
- D027 Source traceability for employee, not server.
- D028 Do not fabricate Page.
- D029 Grounding failure → Verification Mode.
- D030 Failure can still be valuable product result.
- D031 Employee Feedback → log, not Knowledge.
- D032 Feedback is knowledge maintenance signal.
- D033 Browse not Top-1.
- D034 Golden Dataset evaluates behavior, not fixed sentence.
- D035 Recall and Ranking independently evaluated.
- D036 Trace ≠ State.
- D037 Session saves work context, not full long-term personal memory.
- D038 Graph Node = state/control transitions, not every function.
- D039 Persistence stays lightweight.
- D040 Index is derived data.
- D041 Knowledge Version and Index Version separate.
- D042 No automatic full rebuild at every startup.
- D043 Alias discovery can be open, formalization conservative.
- D044 No mechanisms added merely because “advanced”.
- D045 Final responsibility remains human.

## Decision Gate

任何新增机制先回答：
1. 解决什么真实问题？
2. 问题已经发生了吗？
3. Evaluation / Trace 能证明吗？
4. 更简单机制为什么不够？
5. 新机制增加什么能力？
6. 增加什么风险/维护成本？
7. 会不会扩大 LLM 的事实权力？
8. 能否保持 Source / Evidence / Trace 可验证？

答不出来：不增加。

架构哲学：Knowledge First / Evidence First / Deterministic When Possible / Explicit Uncertainty / Complexity Must Be Earned。
