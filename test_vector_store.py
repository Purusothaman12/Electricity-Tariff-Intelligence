from src.comparison.rate_comparator import (
    RateComparator
)
from src.loaders.rate_json_loader import (
    RateJSONLoader
)
from src.rag.chunk_builder import (
    RAGChunkBuilder
)
from src.rag.embedding_model import (
    LocalEmbeddingModel
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

TEST_PERSIST_DIRECTORY = (
    "output/chroma_test"
)

TEST_COLLECTION_NAME = (
    "electricity_tariff_test"
)


def print_results(
    title: str,
    results
) -> None:

    print()
    print(title)
    print("-" * 120)

    for result in results[:5]:

        similarity = (
            f"{result.similarity:.6f}"
            if result.similarity
            is not None
            else "N/A"
        )

        print(
            f"Rank {result.rank:<3} "
            f"Similarity: {similarity} | "
            f"Type: {result.chunk_type}"
        )

        print(
            result.content
        )

        print()


def main() -> None:

    loader = RateJSONLoader()

    documents = loader.load_directory(
        "output/rates"
    )

    documents_by_source = {
        document.source_file: document
        for document in documents
    }

    assert OLD_SOURCE_FILE in (
        documents_by_source
    )

    assert NEW_SOURCE_FILE in (
        documents_by_source
    )

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

    chunk_builder = RAGChunkBuilder()

    chunks = chunk_builder.build_all(
        documents=documents,
        comparison_result=(
            comparison_result
        )
    )

    embedding_model = (
        LocalEmbeddingModel(
            batch_size=32,
            normalize_embeddings=True,
            show_progress_bar=True
        )
    )

    vector_store = TariffVectorStore(
        embedding_model=embedding_model,
        persist_directory=(
            TEST_PERSIST_DIRECTORY
        ),
        collection_name=(
            TEST_COLLECTION_NAME
        ),
        batch_size=64
    )

    print("=" * 120)
    print("TARIFF VECTOR STORE TEST")
    print("=" * 120)

    print()
    print("INDEXING")
    print("-" * 120)

    print(
        "Chunks to index:",
        len(chunks)
    )

    indexed_count = (
        vector_store.upsert_chunks(
            chunks=chunks,
            rebuild=True
        )
    )

    print(
        "Indexed chunks :",
        indexed_count
    )

    print(
        "Stored chunks  :",
        vector_store.count
    )

    assert (
        indexed_count
        == len(chunks)
    )

    assert (
        vector_store.count
        == len(chunks)
    )

    original_count = (
        vector_store.count
    )

    vector_store.upsert_chunks(
        chunks=chunks[:5],
        rebuild=False
    )

    assert (
        vector_store.count
        == original_count
    ), (
        "Upserting existing IDs should not "
        "increase collection size."
    )

    rate_results = vector_store.search(
        query=(
            "What was the Oncor Residential "
            "Customer Charge effective "
            "May 1, 2023?"
        ),
        n_results=20,
        where={
            "chunk_type": "RATE"
        }
    )

    assert rate_results

    residential_customer_results = [
        result
        for result in rate_results
        if (
            result.metadata.get(
                "source_file"
            )
            == NEW_SOURCE_FILE
            and result.metadata.get(
                "normalized_charge_name"
            )
            == "CUSTOMER CHARGE"
            and result.metadata.get(
                "value_text"
            )
            == "$1.43"
        )
    ]

    assert residential_customer_results, (
        "The 2023 Residential Customer "
        "Charge was not retrieved."
    )

    comparison_results = (
        vector_store.search(
            query=(
                "How did the Oncor Residential "
                "Customer Charge change between "
                "the old and new tariffs?"
            ),
            n_results=20,
            where={
                "chunk_type": "COMPARISON"
            }
        )
    )

    assert comparison_results

    customer_comparison_results = [
        result
        for result in comparison_results
        if (
            result.metadata.get(
                "schedule_title"
            )
            == "RESIDENTIAL SERVICE"
            and result.metadata.get(
                "normalized_charge_name"
            )
            == "CUSTOMER CHARGE"
            and result.metadata.get(
                "status"
            )
            == "INCREASED"
        )
    ]

    assert customer_comparison_results, (
        "The Residential Customer Charge "
        "comparison was not retrieved."
    )

    selected_result = (
        residential_customer_results[0]
    )

    stored_chunk = (
        vector_store.get_chunk(
            selected_result.chunk_id
        )
    )

    assert stored_chunk is not None

    assert (
        stored_chunk.chunk_id
        == selected_result.chunk_id
    )

    assert (
        "$1.43"
        in stored_chunk.content
    )

    reopened_store = TariffVectorStore(
        embedding_model=embedding_model,
        persist_directory=(
            TEST_PERSIST_DIRECTORY
        ),
        collection_name=(
            TEST_COLLECTION_NAME
        ),
        batch_size=64
    )

    assert (
        reopened_store.count
        == len(chunks)
    ), (
        "The persisted collection was not "
        "loaded correctly."
    )

    print_results(
        title=(
            "RESIDENTIAL CUSTOMER RATE RESULTS"
        ),
        results=rate_results
    )

    print_results(
        title=(
            "RESIDENTIAL CUSTOMER "
            "COMPARISON RESULTS"
        ),
        results=comparison_results
    )

    print()
    print("PERSISTENCE CHECK")
    print("-" * 120)

    print(
        "Original store count:",
        vector_store.count
    )

    print(
        "Reopened store count:",
        reopened_store.count
    )

    print()
    print("=" * 120)
    print("ALL ASSERTIONS PASSED")
    print("=" * 120)


if __name__ == "__main__":
    main()