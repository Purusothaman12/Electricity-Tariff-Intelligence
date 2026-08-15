from collections import Counter

from src.comparison.rate_comparator import (
    RateChangeStatus,
    RateComparator
)
from src.loaders.rate_json_loader import (
    RateJSONLoader
)
from src.rag.chunk_builder import (
    RAGChunkBuilder,
    RAGChunkType
)


OLD_SOURCE_FILE = (
    "Oncor_November_27_2017.pdf"
)

NEW_SOURCE_FILE = (
    "Oncor_May_1_2023.pdf"
)

RESIDENTIAL_SECTION_ID = (
    "6.1.1.1.1"
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

    builder = RAGChunkBuilder()

    old_chunks = (
        builder.build_document_chunks(
            old_document
        )
    )

    new_chunks = (
        builder.build_document_chunks(
            new_document
        )
    )

    comparison_chunks = (
        builder.build_comparison_chunks(
            comparison_result
        )
    )

    all_chunks = builder.build_all(
        documents=documents,
        comparison_result=(
            comparison_result
        )
    )

    print("=" * 120)
    print("RAG CHUNK BUILDER TEST")
    print("=" * 120)

    print()
    print("CHUNK COUNTS")
    print("-" * 120)

    print(
        "Old document chunks :",
        len(old_chunks)
    )

    print(
        "New document chunks :",
        len(new_chunks)
    )

    print(
        "Comparison chunks   :",
        len(comparison_chunks)
    )

    print(
        "Total unique chunks :",
        len(all_chunks)
    )

    chunk_type_counts = Counter(
        chunk.chunk_type.value
        for chunk in all_chunks
    )

    print()
    print("CHUNKS BY TYPE")
    print("-" * 120)

    for chunk_type, count in sorted(
        chunk_type_counts.items()
    ):

        print(
            f"{chunk_type:<25}: {count}"
        )

    assert old_chunks
    assert new_chunks
    assert comparison_chunks
    assert all_chunks

    assert (
        len(
            {
                chunk.chunk_id
                for chunk in all_chunks
            }
        )
        == len(all_chunks)
    )

    assert all(
        chunk.content.strip()
        for chunk in all_chunks
    )

    assert all(
        chunk.metadata.get(
            "chunk_type"
        )
        == chunk.chunk_type.value
        for chunk in all_chunks
    )

    rate_chunks = [
        chunk
        for chunk in all_chunks
        if (
            chunk.chunk_type
            == RAGChunkType.RATE
        )
    ]

    section_chunks = [
        chunk
        for chunk in all_chunks
        if (
            chunk.chunk_type
            == RAGChunkType
            .SECTION_COVERAGE
        )
    ]

    assert rate_chunks
    assert section_chunks

    structural_artifacts = [
        chunk
        for chunk in rate_chunks
        if (
            str(
                chunk.metadata.get(
                    "charge_name",
                    ""
                )
            )
            .strip()
            .upper()
            .startswith("COLUMN ")
        )
    ]

    assert not structural_artifacts, (
        "Structural matrix artifacts were "
        "included in RAG chunks."
    )

    residential_rate_chunks = [
        chunk
        for chunk in rate_chunks
        if (
            chunk.metadata.get(
                "source_file"
            )
            == NEW_SOURCE_FILE
            and chunk.metadata.get(
                "schedule_id"
            )
            == RESIDENTIAL_SECTION_ID
            and chunk.metadata.get(
                "normalized_charge_name"
            )
            == "CUSTOMER CHARGE"
            and chunk.metadata.get(
                "value_text"
            )
            == "$1.43"
        )
    ]

    assert residential_rate_chunks, (
        "The 2023 Residential Customer "
        "Charge chunk was not created."
    )

    residential_rate_chunk = (
        residential_rate_chunks[0]
    )

    assert (
        "$1.43"
        in residential_rate_chunk.content
    )

    assert (
        "May 1, 2023"
        in residential_rate_chunk.content
    )

    assert (
        "RESIDENTIAL SERVICE"
        in residential_rate_chunk.content
    )

    residential_comparison_chunks = [
        chunk
        for chunk in comparison_chunks
        if (
            chunk.metadata.get(
                "schedule_title"
            )
            == "RESIDENTIAL SERVICE"
            and chunk.metadata.get(
                "normalized_charge_name"
            )
            == "CUSTOMER CHARGE"
            and chunk.metadata.get(
                "status"
            )
            == RateChangeStatus
            .INCREASED
            .value
        )
    ]

    assert residential_comparison_chunks, (
        "The Residential Customer Charge "
        "comparison chunk was not created."
    )

    comparison_chunk = (
        residential_comparison_chunks[0]
    )

    assert (
        "$0.90"
        in comparison_chunk.content
    )

    assert (
        "$1.43"
        in comparison_chunk.content
    )

    assert (
        "INCREASED"
        in comparison_chunk.content
    )

    not_applicable_chunks = [
        chunk
        for chunk in section_chunks
        if (
            chunk.metadata.get(
                "applicability_status"
            )
            == "NOT_APPLICABLE"
        )
    ]

    assert (
        len(not_applicable_chunks)
        == 6
    ), (
        "Expected three not-applicable "
        "sections from each tariff."
    )

    print()
    print("2023 RESIDENTIAL CUSTOMER CHARGE")
    print("-" * 120)

    print(
        residential_rate_chunk.content
    )

    print()
    print("RESIDENTIAL CUSTOMER COMPARISON")
    print("-" * 120)

    print(
        comparison_chunk.content
    )

    print()
    print("NOT-APPLICABLE SECTION CHUNKS")
    print("-" * 120)

    for chunk in not_applicable_chunks:

        print(
            chunk.metadata.get(
                "source_file"
            ),
            "|",
            chunk.metadata.get(
                "section_id"
            ),
            "|",
            chunk.metadata.get(
                "section_title"
            )
        )

    print()
    print("=" * 120)
    print("ALL ASSERTIONS PASSED")
    print("=" * 120)


if __name__ == "__main__":
    main()