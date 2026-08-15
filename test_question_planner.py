from src.rag.question_planner import (
    TariffQuestionPlanner,
    TariffQuestionType
)
from src.rag.retriever import (
    RetrievalIntent
)


def print_plan(
    title: str,
    plan
) -> None:

    print()
    print(title)
    print("-" * 120)

    for key, value in plan.to_dict().items():

        print(
            f"{key:<28}: {value}"
        )


def main() -> None:

    print("=" * 120)
    print("TARIFF QUESTION PLANNER TEST")
    print("=" * 120)

    planner = TariffQuestionPlanner()

    # ------------------------------------------------------------------
    # 1. Specific rate lookup
    # ------------------------------------------------------------------

    rate_lookup_question = (
        "What is the Residential Service "
        "Customer Charge?"
    )

    rate_lookup_plan = planner.plan(
        question=rate_lookup_question,
        requested_top_k=8
    )

    print_plan(
        "SPECIFIC RATE LOOKUP",
        rate_lookup_plan
    )

    assert (
        rate_lookup_plan.question_type
        == TariffQuestionType.RATE_LOOKUP
    )

    assert (
        rate_lookup_plan.retrieval_intent
        == RetrievalIntent.RATE_LOOKUP
    )

    assert (
        rate_lookup_plan.retrieval_top_k
        == 8
    )

    assert (
        rate_lookup_plan.prompt_evidence_limit
        == 1
    )

    assert (
        rate_lookup_plan.use_deterministic_answer
        is True
    )

    assert (
        rate_lookup_plan.allow_llm
        is True
    )

    # ------------------------------------------------------------------
    # 2. Broad schedule charge listing
    # ------------------------------------------------------------------

    rate_list_question = (
        "What charges are available under "
        "Lighting Service?"
    )

    rate_list_plan = planner.plan(
        question=rate_list_question,
        requested_top_k=8
    )

    print_plan(
        "LIGHTING SERVICE RATE LIST",
        rate_list_plan
    )

    assert (
        rate_list_plan.question_type
        == TariffQuestionType.RATE_LIST
    )

    assert (
        rate_list_plan.retrieval_intent
        == RetrievalIntent.RATE_LOOKUP
    )

    assert (
        rate_list_plan.retrieval_top_k
        == 50
    )

    assert (
        rate_list_plan.prompt_evidence_limit
        == 20
    )

    assert (
        rate_list_plan.use_deterministic_answer
        is False
    )

    assert (
        rate_list_plan.allow_llm
        is True
    )

    # ------------------------------------------------------------------
    # 3. Historical rate comparison
    # ------------------------------------------------------------------

    comparison_question = (
        "Compare the Residential Transmission "
        "System Charge effective in 2018 and 2023."
    )

    comparison_plan = planner.plan(
        question=comparison_question,
        requested_top_k=8
    )

    print_plan(
        "RATE COMPARISON",
        comparison_plan
    )

    assert (
        comparison_plan.question_type
        == TariffQuestionType.RATE_COMPARISON
    )

    assert (
        comparison_plan.retrieval_intent
        == RetrievalIntent.COMPARISON
    )

    assert (
        comparison_plan.retrieval_top_k
        == 8
    )

    assert (
        comparison_plan.prompt_evidence_limit
        == 1
    )

    assert (
        comparison_plan.use_deterministic_answer
        is True
    )

    assert (
        comparison_plan.extracted_years
        == (
            2018,
            2023
        )
    )

    assert (
        comparison_plan.allow_llm
        is True
    )

    # ------------------------------------------------------------------
    # 4. Rider schedule change
    # ------------------------------------------------------------------

    rider_question = (
        "Which Riders were added in the "
        "new tariff?"
    )

    rider_plan = planner.plan(
        question=rider_question,
        requested_top_k=8
    )

    print_plan(
        "RIDER CHANGE",
        rider_plan
    )

    assert (
        rider_plan.question_type
        == TariffQuestionType.RIDER_CHANGE
    )

    assert (
        rider_plan.retrieval_intent
        == RetrievalIntent.COMPARISON
    )

    assert (
        rider_plan.retrieval_top_k
        == 20
    )

    assert (
        rider_plan.prompt_evidence_limit
        == 12
    )

    assert (
        rider_plan.use_deterministic_answer
        is True
    )

    # ------------------------------------------------------------------
    # 5. Section applicability
    # ------------------------------------------------------------------

    section_question = (
        "Which tariff sections are marked "
        "not applicable?"
    )

    section_plan = planner.plan(
        question=section_question,
        requested_top_k=8
    )

    print_plan(
        "SECTION COVERAGE",
        section_plan
    )

    assert (
        section_plan.question_type
        == TariffQuestionType.SECTION_COVERAGE
    )

    assert (
        section_plan.retrieval_intent
        == RetrievalIntent.SECTION_COVERAGE
    )

    assert (
        section_plan.retrieval_top_k
        == 25
    )

    assert (
        section_plan.prompt_evidence_limit
        == 20
    )

    assert (
        section_plan.use_deterministic_answer
        is True
    )

    # ------------------------------------------------------------------
    # 6. Out-of-scope user question
    # ------------------------------------------------------------------

    out_of_scope_question = (
        "What is tomorrow's weather?"
    )

    out_of_scope_plan = planner.plan(
        question=out_of_scope_question,
        requested_top_k=8
    )

    print_plan(
        "OUT-OF-SCOPE QUESTION",
        out_of_scope_plan
    )

    assert (
        out_of_scope_plan.question_type
        == TariffQuestionType.OUT_OF_SCOPE
    )

    assert (
        out_of_scope_plan.retrieval_intent
        == RetrievalIntent.GENERAL
    )

    assert (
        out_of_scope_plan.retrieval_top_k
        == 0
    )

    assert (
        out_of_scope_plan.prompt_evidence_limit
        == 0
    )

    assert (
        out_of_scope_plan.use_deterministic_answer
        is False
    )

    assert (
        out_of_scope_plan.allow_llm
        is False
    )

    # ------------------------------------------------------------------
    # 7. User-requested top_k should be respected for broad questions
    # ------------------------------------------------------------------

    large_list_plan = planner.plan(
        question=rate_list_question,
        requested_top_k=40
    )

    print_plan(
        "RATE LIST WITH USER TOP_K",
        large_list_plan
    )

    assert (
        large_list_plan.retrieval_top_k
        == 50
    )

    assert (
        large_list_plan.prompt_evidence_limit
        == 20
    )

    # ------------------------------------------------------------------
    # 8. Maximum retrieval limit
    # ------------------------------------------------------------------

    capped_plan = planner.plan(
        question=rate_list_question,
        requested_top_k=100
    )

    assert (
        capped_plan.retrieval_top_k
        == 50
    )

    # ------------------------------------------------------------------
    # 9. Validation
    # ------------------------------------------------------------------

    empty_question_failed = False

    try:

        planner.plan(
            question="   ",
            requested_top_k=8
        )

    except ValueError:

        empty_question_failed = True

    assert empty_question_failed

    invalid_top_k_failed = False

    try:

        planner.plan(
            question=rate_lookup_question,
            requested_top_k=0
        )

    except ValueError:

        invalid_top_k_failed = True

    assert invalid_top_k_failed

    print()
    print("=" * 120)
    print("ALL ASSERTIONS PASSED")
    print("=" * 120)


if __name__ == "__main__":
    main()
