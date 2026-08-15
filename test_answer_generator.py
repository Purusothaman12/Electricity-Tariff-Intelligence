from src.comparison.rate_comparator import (
    RateComparator
)
from src.loaders.rate_json_loader import (
    RateJSONLoader
)
from src.rag.answer_generator import (
    TariffAnswerGenerator
)
from src.rag.embedding_model import (
    LocalEmbeddingModel
)
from src.rag.retriever import (
    TariffRetriever
)
from src.rag.vector_store import (
    TariffVectorStore
)


OLD_SOURCE_FILE = (
    "Oncor_November_27_2017.pdf"
)

NEW_SOURCE_FILE = (
    "Oncor_May_1_2023.pdf"
)

PERSIST_DIRECTORY = (
    "output/chroma_test"
)

COLLECTION_NAME = (
    "electricity_tariff_test"
)


def print_answer(
    title: str,
    rag_answer
) -> None:

    print()
    print(title)
    print("-" * 120)

    print(
        "Intent       :",
        rag_answer.intent.value
    )

    print(
        "Answer type  :",
        rag_answer.answer_type
    )

    print(
        "Grounded     :",
        rag_answer.is_grounded
    )

    print(
        "Evidence count:",
        len(
            rag_answer.evidence
        )
    )

    print()
    print(
        rag_answer.answer
    )

    for evidence in (
        rag_answer.evidence
    ):

        print()
        print(
            f"Evidence {evidence.rank}: "
            f"{evidence.chunk_id}"
        )

        print(
            evidence.content
        )


def main() -> None:

    loader = RateJSONLoader()

    documents = loader.load_directory(
        "output/rates"
    )

    documents_by_source = {
        document.source_file: document
        for document in documents
    }

    old_document = documents_by_source[
        OLD_SOURCE_FILE
    ]

    new_document = documents_by_source[
        NEW_SOURCE_FILE
    ]

    comparator = RateComparator()

    comparison_result = comparator.compare(
        old_document=old_document,
        new_document=new_document
    )

    embedding_model = LocalEmbeddingModel(
        batch_size=32,
        normalize_embeddings=True,
        show_progress_bar=False
    )

    vector_store = TariffVectorStore(
        embedding_model=embedding_model,
        persist_directory=(
            PERSIST_DIRECTORY
        ),
        collection_name=(
            COLLECTION_NAME
        )
    )

    assert vector_store.count == 1557, (
        "Run python -m test_vector_store "
        "before this test."
    )

    retriever = TariffRetriever(
        vector_store=vector_store
    )

    answer_generator = (
        TariffAnswerGenerator(
            retriever=retriever,
            comparison_result=(
                comparison_result
            )
        )
    )

    print("=" * 120)
    print("TARIFF ANSWER GENERATOR TEST")
    print("=" * 120)

    rate_answer = answer_generator.answer(
        (
            "What was the Oncor "
            "Residential Customer Charge "
            "in 2023?"
        )
    )

    assert rate_answer.is_grounded

    assert (
        rate_answer.answer_type
        == "RATE_LOOKUP"
    )

    assert "$1.43" in rate_answer.answer

    assert (
        "May 1, 2023"
        in rate_answer.answer
    )

    comparison_answer = (
        answer_generator.answer(
            (
                "How did the Oncor "
                "Residential Customer Charge "
                "change between the old and "
                "new tariffs?"
            )
        )
    )

    assert (
        comparison_answer.is_grounded
    )

    assert (
        comparison_answer.answer_type
        == "RATE_COMPARISON"
    )

    assert (
        "$0.90"
        in comparison_answer.answer
    )

    assert (
        "$1.43"
        in comparison_answer.answer
    )

    assert (
        "58.89%"
        in comparison_answer.answer
    )

    added_riders_answer = (
        answer_generator.answer(
            (
                "Which Riders were added "
                "in the new tariff?"
            )
        )
    )

    assert (
        added_riders_answer.is_grounded
    )

    assert (
        added_riders_answer.answer_type
        == "RIDER_SCHEDULE_CHANGE"
    )

    added_answer_text = (
        added_riders_answer
        .answer
        .upper()
    )

    assert (
        "RIDER ISR - "
        "INTEREST SAVINGS REFUND"
        in added_answer_text
    )

    assert (
        "RIDER MG - "
        "MOBILE GENERATION"
        in added_answer_text
    )

    assert (
        "DISTRIBUTION COST "
        "RECOVERY FACTOR"
        not in added_answer_text
    )

    assert (
        "TRANSMISSION COST "
        "RECOVERY FACTOR"
        not in added_answer_text
    )

    section_answer = (
        answer_generator.answer(
            (
                "Which tariff sections "
                "are not applicable?"
            ),
            top_k=10
        )
    )

    assert section_answer.is_grounded

    assert (
        section_answer.answer_type
        == "SECTION_COVERAGE"
    )

    assert (
        "NOT APPLICABLE"
        in section_answer.answer.upper()
    )

    assert (
        "TRANSITION CHARGE"
        in section_answer.answer.upper()
    )

    assert (
        "COMPETITION TRANSITION CHARGE"
        in section_answer.answer.upper()
    )

    assert (
        "SYSTEM BENEFIT FUND"
        in section_answer.answer.upper()
    )

    print_answer(
        title=(
            "RESIDENTIAL CUSTOMER RATE"
        ),
        rag_answer=rate_answer
    )

    print_answer(
        title=(
            "RESIDENTIAL CUSTOMER "
            "COMPARISON"
        ),
        rag_answer=(
            comparison_answer
        )
    )

    print_answer(
        title="ADDED RIDERS",
        rag_answer=(
            added_riders_answer
        )
    )

    print_answer(
        title=(
            "NOT-APPLICABLE SECTIONS"
        ),
        rag_answer=section_answer
    )

    print()
    print("=" * 120)
    print("ALL ASSERTIONS PASSED")
    print("=" * 120)


if __name__ == "__main__":
    main()