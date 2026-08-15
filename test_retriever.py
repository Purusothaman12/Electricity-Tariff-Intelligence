from src.rag.embedding_model import (
    LocalEmbeddingModel
)
from src.rag.retriever import (
    RetrievalIntent,
    TariffRetriever
)
from src.rag.vector_store import (
    TariffVectorStore
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


def print_results(
    title: str,
    results
) -> None:

    print()
    print(title)
    print("-" * 120)

    for result in results:

        semantic_similarity = (
            f"{result.semantic_similarity:.6f}"
            if (
                result.semantic_similarity
                is not None
            )
            else "N/A"
        )

        print(
            f"Rank {result.rank:<3} "
            f"Vector rank: "
            f"{result.vector_rank:<3} "
            f"Semantic: "
            f"{semantic_similarity:<10} "
            f"Rerank: "
            f"{result.rerank_score:.6f}"
        )

        print(
            "Signals:",
            ", ".join(
                result.matched_signals
            )
        )

        print(
            result.content
        )

        print()


def main() -> None:

    embedding_model = (
        LocalEmbeddingModel(
            batch_size=32,
            normalize_embeddings=True,
            show_progress_bar=False
        )
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
        "before running this test."
    )

    retriever = TariffRetriever(
        vector_store=vector_store
    )

    print("=" * 120)
    print("TARIFF RETRIEVER TEST")
    print("=" * 120)

    rate_query = (
        "What was the Oncor Residential "
        "Customer Charge in 2023?"
    )

    rate_intent = retriever.detect_intent(
        rate_query
    )

    assert (
        rate_intent
        == RetrievalIntent.RATE_LOOKUP
    )

    rate_results = retriever.retrieve(
        query=rate_query,
        top_k=5
    )

    assert rate_results

    top_rate = rate_results[0]

    assert (
        top_rate.chunk_type
        == "RATE"
    )

    assert (
        top_rate.metadata.get(
            "source_file"
        )
        == NEW_SOURCE_FILE
    )

    assert (
        top_rate.metadata.get(
            "schedule_title"
        )
        == "RESIDENTIAL SERVICE"
    )

    assert (
        top_rate.metadata.get(
            "normalized_charge_name"
        )
        == "CUSTOMER CHARGE"
    )

    assert (
        top_rate.metadata.get(
            "value_text"
        )
        == "$1.43"
    )

    comparison_query = (
        "How did the Oncor Residential "
        "Customer Charge change between "
        "the old and new tariffs?"
    )

    comparison_intent = (
        retriever.detect_intent(
            comparison_query
        )
    )

    assert (
        comparison_intent
        == RetrievalIntent.COMPARISON
    )

    comparison_results = (
        retriever.retrieve(
            query=comparison_query,
            top_k=5
        )
    )

    assert comparison_results

    top_comparison = (
        comparison_results[0]
    )

    assert (
        top_comparison.chunk_type
        == "COMPARISON"
    )

    assert (
        top_comparison.metadata.get(
            "schedule_title"
        )
        == "RESIDENTIAL SERVICE"
    )

    assert (
        top_comparison.metadata.get(
            "normalized_charge_name"
        )
        == "CUSTOMER CHARGE"
    )

    assert (
        top_comparison.metadata.get(
            "status"
        )
        == "INCREASED"
    )

    assert (
        top_comparison.metadata.get(
            "old_value_text"
        )
        == "$0.90"
    )

    assert (
        top_comparison.metadata.get(
            "new_value_text"
        )
        == "$1.43"
    )

    added_rider_query = (
        "Which Riders were added in "
        "the new tariff?"
    )

    added_rider_results = (
        retriever.retrieve(
            query=added_rider_query,
            top_k=5
        )
    )

    assert added_rider_results

    top_added_rider = (
        added_rider_results[0]
    )

    assert (
        top_added_rider.chunk_type
        == "COMPARISON"
    )

    assert (
        top_added_rider.metadata.get(
            "status"
        )
        == "ADDED"
    )

    assert (
        top_added_rider.metadata.get(
            "category"
        )
        == "RIDER"
    )

    section_query = (
        "Which tariff sections are "
        "not applicable?"
    )

    section_intent = (
        retriever.detect_intent(
            section_query
        )
    )

    assert (
        section_intent
        == (
            RetrievalIntent
            .SECTION_COVERAGE
        )
    )

    section_results = (
        retriever.retrieve(
            query=section_query,
            top_k=6
        )
    )

    assert section_results

    assert all(
        result.chunk_type
        == "SECTION_COVERAGE"
        for result in section_results
    )

    assert all(
        result.metadata.get(
            "applicability_status"
        )
        == "NOT_APPLICABLE"
        for result in section_results
    )

    print_results(
        title=(
            "RESIDENTIAL CUSTOMER RATE"
        ),
        results=rate_results
    )

    print_results(
        title=(
            "RESIDENTIAL CUSTOMER COMPARISON"
        ),
        results=comparison_results
    )

    print_results(
        title=(
            "ADDED RIDERS"
        ),
        results=added_rider_results
    )

    print_results(
        title=(
            "NOT-APPLICABLE SECTIONS"
        ),
        results=section_results
    )

    print()
    print("=" * 120)
    print("ALL ASSERTIONS PASSED")
    print("=" * 120)


if __name__ == "__main__":
    main()