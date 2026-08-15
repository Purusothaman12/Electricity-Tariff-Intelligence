from src.rag.service import (
    TariffRAGService
)


PERSIST_DIRECTORY = (
    "output/chroma_test"
)

COLLECTION_NAME = (
    "electricity_tariff_test"
)


def print_answer(
    title: str,
    answer
) -> None:

    print()
    print(title)
    print("-" * 120)

    print(
        "Intent      :",
        answer.intent.value
    )

    print(
        "Answer type :",
        answer.answer_type
    )

    print(
        "Grounded    :",
        answer.is_grounded
    )

    print(
        "Evidence    :",
        len(answer.evidence)
    )

    print()
    print(
        answer.answer
    )


def main() -> None:

    print("=" * 120)
    print("TARIFF RAG SERVICE TEST")
    print("=" * 120)

    service = TariffRAGService(
        persist_directory=(
            PERSIST_DIRECTORY
        ),
        collection_name=(
            COLLECTION_NAME
        )
    )

    status = service.get_status()

    print()
    print("SERVICE STATUS")
    print("-" * 120)

    for key, value in (
        status.to_dict().items()
    ):

        print(
            f"{key:<25}: {value}"
        )

    assert status.ready

    assert (
        status.indexed_chunk_count
        == 1557
    )

    assert (
        status.old_rate_count
        == 536
    )

    assert (
        status.new_rate_count
        == 658
    )

    assert (
        status.comparison_count
        == 358
    )

    rate_answer = service.ask(
        (
            "What was the Oncor "
            "Residential Customer Charge "
            "in 2023?"
        )
    )

    assert rate_answer.is_grounded
    assert "$1.43" in rate_answer.answer

    comparison_answer = service.ask(
        (
            "How did the Residential "
            "Customer Charge change between "
            "the old and new tariffs?"
        )
    )

    assert comparison_answer.is_grounded

    assert (
        "$0.90"
        in comparison_answer.answer
    )

    assert (
        "$1.43"
        in comparison_answer.answer
    )

    added_riders_answer = service.ask(
        (
            "Which Riders were added "
            "in the new tariff?"
        )
    )

    assert (
        added_riders_answer.is_grounded
    )

    assert (
        "RIDER ISR"
        in added_riders_answer.answer
    )

    assert (
        "RIDER MG"
        in added_riders_answer.answer
    )

    section_answer = service.ask(
        (
            "Which tariff sections "
            "are not applicable?"
        ),
        top_k=10
    )

    assert section_answer.is_grounded

    assert (
        "SYSTEM BENEFIT FUND"
        in section_answer.answer.upper()
    )

    print_answer(
        title="RATE LOOKUP",
        answer=rate_answer
    )

    print_answer(
        title="RATE COMPARISON",
        answer=comparison_answer
    )

    print_answer(
        title="ADDED RIDERS",
        answer=added_riders_answer
    )

    print_answer(
        title="SECTION APPLICABILITY",
        answer=section_answer
    )

    print()
    print("=" * 120)
    print("ALL ASSERTIONS PASSED")
    print("=" * 120)


if __name__ == "__main__":
    main()