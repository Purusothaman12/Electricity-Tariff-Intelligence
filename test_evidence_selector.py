from typing import Any

from src.rag.evidence_selector import (
    EvidenceSelection,
    TariffEvidenceSelector
)
from src.rag.question_planner import (
    TariffQuestionPlanner,
    TariffQuestionType
)
from src.rag.service import (
    TariffRAGService
)


PERSIST_DIRECTORY = (
    "output/chroma_test"
)

COLLECTION_NAME = (
    "electricity_tariff_test"
)

NEW_SOURCE_FILE = (
    "Oncor_May_1_2023.pdf"
)


def get_result_dictionary(
    result: Any
) -> dict[str, Any]:

    result_dictionary = result.to_dict()

    if not isinstance(
        result_dictionary,
        dict
    ):

        raise TypeError(
            "Retrieval result to_dict() must "
            "return a dictionary."
        )

    return result_dictionary


def get_result_content(
    result: Any
) -> str:

    result_dictionary = (
        get_result_dictionary(
            result
        )
    )

    content = result_dictionary.get(
        "content",
        ""
    )

    return " ".join(
        str(
            content
        ).split()
    )


def get_result_metadata(
    result: Any
) -> dict[str, Any]:

    result_dictionary = (
        get_result_dictionary(
            result
        )
    )

    metadata = result_dictionary.get(
        "metadata",
        {}
    )

    if isinstance(
        metadata,
        dict
    ):

        return metadata

    return {}


def print_selection(
    title: str,
    selection: EvidenceSelection
) -> None:

    print()
    print(title)
    print("-" * 120)

    print(
        "Question             :",
        selection.question
    )

    print(
        "Question type        :",
        selection.question_type.value
    )

    print(
        "Candidate count      :",
        selection.candidate_count
    )

    print(
        "Selected count       :",
        selection.selected_count
    )

    print(
        "Selected source file :",
        selection.selected_source_file
    )

    print(
        "Target terms         :",
        selection.target_terms
    )

    print(
        "Reason               :",
        selection.reason
    )

    print()
    print("SELECTED EVIDENCE")
    print("-" * 120)

    if not selection.results:

        print(
            "No evidence selected."
        )

        return

    for index, result in enumerate(
        selection.results,
        start=1
    ):

        result_dictionary = (
            get_result_dictionary(
                result
            )
        )

        print()
        print(
            f"Evidence {index}"
        )

        print(
            "Chunk ID :",
            result_dictionary.get(
                "chunk_id",
                ""
            )
        )

        print(
            "Score    :",
            result_dictionary.get(
                "score",
                result_dictionary.get(
                    "similarity",
                    ""
                )
            )
        )

        print(
            "Content  :",
            get_result_content(
                result
            )
        )


def retrieve_and_select(
    service: TariffRAGService,
    planner: TariffQuestionPlanner,
    selector: TariffEvidenceSelector,
    question: str,
    requested_top_k: int = 8
) -> EvidenceSelection:

    plan = planner.plan(
        question=question,
        requested_top_k=requested_top_k
    )

    if not plan.allow_llm:

        return selector.select(
            plan=plan,
            results=[]
        )

    retrieval_results = (
        service.retrieve(
            question=question,
            top_k=plan.retrieval_top_k,
            intent=plan.retrieval_intent
        )
    )

    return selector.select(
        plan=plan,
        results=retrieval_results
    )


def main() -> None:

    print("=" * 120)
    print("TARIFF EVIDENCE SELECTOR TEST")
    print("=" * 120)

    service = TariffRAGService(
        persist_directory=(
            PERSIST_DIRECTORY
        ),
        collection_name=(
            COLLECTION_NAME
        ),
        llm_enabled=False
    )

    planner = (
        TariffQuestionPlanner()
    )

    selector = (
        TariffEvidenceSelector()
    )

    service_status = (
        service.get_status()
    )

    assert service_status.ready

    assert (
        service_status.indexed_chunk_count
        == 1557
    )

    # ------------------------------------------------------------------
    # 1. Specific Residential Customer Charge
    # ------------------------------------------------------------------

    rate_lookup_question = (
        "What is the Residential Service "
        "Customer Charge?"
    )

    rate_lookup_selection = (
        retrieve_and_select(
            service=service,
            planner=planner,
            selector=selector,
            question=rate_lookup_question
        )
    )

    print_selection(
        "SPECIFIC RATE LOOKUP",
        rate_lookup_selection
    )

    assert (
        rate_lookup_selection.question_type
        == TariffQuestionType.RATE_LOOKUP
    )

    assert (
        rate_lookup_selection.selected_count
        == 1
    )

    assert (
        len(
            rate_lookup_selection.results
        )
        == 1
    )

    rate_lookup_content = (
        get_result_content(
            rate_lookup_selection.results[0]
        ).upper()
    )

    assert (
        "RESIDENTIAL SERVICE"
        in rate_lookup_content
    )

    assert (
        "CUSTOMER CHARGE"
        in rate_lookup_content
    )

    assert (
        NEW_SOURCE_FILE.upper()
        in rate_lookup_content
        or (
            rate_lookup_selection
            .selected_source_file
            == NEW_SOURCE_FILE
        )
    )

    # ------------------------------------------------------------------
    # 2. Multiple Lighting Service charges
    # ------------------------------------------------------------------

    lighting_question = (
        "What charges are available under "
        "Lighting Service?"
    )

    lighting_selection = (
        retrieve_and_select(
            service=service,
            planner=planner,
            selector=selector,
            question=lighting_question
        )
    )

    print_selection(
        "LIGHTING SERVICE RATE LIST",
        lighting_selection
    )

    assert (
        lighting_selection.question_type
        == TariffQuestionType.RATE_LIST
    )

    assert (
        lighting_selection.selected_count
        >= 2
    ), (
        "The Lighting Service question should "
        "select multiple charge records."
    )

    assert (
        lighting_selection.selected_count
        <= 20
    )

    assert (
        lighting_selection.selected_source_file
        == NEW_SOURCE_FILE
    ), (
        "A broad question without a requested "
        "year should prefer the latest tariff."
    )

    lighting_charge_names = set()

    for result in lighting_selection.results:

        content = (
            get_result_content(
                result
            ).upper()
        )

        metadata = (
            get_result_metadata(
                result
            )
        )

        searchable_text = " ".join(
            [
                content,
                " ".join(
                    str(
                        value
                    )
                    for value
                    in metadata.values()
                ).upper()
            ]
        )

        assert (
            "LIGHTING" in searchable_text
        ), (
            "The Lighting Service selection "
            "contains unrelated evidence."
        )

        charge_name = " ".join(
            str(
                metadata.get(
                    "charge_name",
                    ""
                )
            ).split()
        )

        if charge_name:

            lighting_charge_names.add(
                charge_name.upper()
            )

    assert (
        len(
            lighting_charge_names
        )
        >= 2
    ), (
        "The selected Lighting evidence should "
        "contain multiple unique charges."
    )

    # ------------------------------------------------------------------
    # 3. Residential Transmission comparison
    # ------------------------------------------------------------------

    comparison_question = (
        "Compare the Residential Transmission "
        "System Charge effective in 2018 and 2023."
    )

    comparison_selection = (
        retrieve_and_select(
            service=service,
            planner=planner,
            selector=selector,
            question=comparison_question
        )
    )

    print_selection(
        "RESIDENTIAL TRANSMISSION COMPARISON",
        comparison_selection
    )

    assert (
        comparison_selection.question_type
        == TariffQuestionType.RATE_COMPARISON
    )

    assert (
        comparison_selection.selected_count
        == 1
    )

    comparison_content = (
        get_result_content(
            comparison_selection.results[0]
        ).upper()
    )

    assert (
        "RESIDENTIAL SERVICE"
        in comparison_content
    )

    assert (
        "TRANSMISSION"
        in comparison_content
    )

    assert (
        "OLD RATE"
        in comparison_content
    )

    assert (
        "NEW RATE"
        in comparison_content
    )

    assert (
        "2018"
        in comparison_content
    )

    assert (
        "2023"
        in comparison_content
    )

    # ------------------------------------------------------------------
    # 4. Out-of-scope user question
    # ------------------------------------------------------------------

    out_of_scope_question = (
        "What is tomorrow's weather?"
    )

    out_of_scope_selection = (
        retrieve_and_select(
            service=service,
            planner=planner,
            selector=selector,
            question=out_of_scope_question
        )
    )

    print_selection(
        "OUT-OF-SCOPE QUESTION",
        out_of_scope_selection
    )

    assert (
        out_of_scope_selection.question_type
        == TariffQuestionType.OUT_OF_SCOPE
    )

    assert (
        out_of_scope_selection.candidate_count
        == 0
    )

    assert (
        out_of_scope_selection.selected_count
        == 0
    )

    assert (
        out_of_scope_selection.results
        == []
    )

    assert (
        out_of_scope_selection
        .selected_source_file
        is None
    )

    # ------------------------------------------------------------------
    # 5. Serialization
    # ------------------------------------------------------------------

    selection_dictionary = (
        lighting_selection.to_dict()
    )

    assert (
        selection_dictionary[
            "question_type"
        ]
        == "RATE_LIST"
    )

    assert (
        selection_dictionary[
            "selected_count"
        ]
        == lighting_selection.selected_count
    )

    assert (
        len(
            selection_dictionary[
                "results"
            ]
        )
        == lighting_selection.selected_count
    )

    print()
    print("=" * 120)
    print("ALL ASSERTIONS PASSED")
    print("=" * 120)


if __name__ == "__main__":
    main()