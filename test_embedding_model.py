import math

from src.rag.embedding_model import (
    LocalEmbeddingModel
)


def vector_norm(
    vector: list[float]
) -> float:

    return math.sqrt(
        sum(
            value * value
            for value in vector
        )
    )


def main() -> None:

    model = LocalEmbeddingModel(
        batch_size=4,
        normalize_embeddings=True,
        show_progress_bar=False
    )

    documents = [
        (
            "Tariff rate record. Utility: Oncor. "
            "Schedule: Residential Service. "
            "Charge: Customer Charge. "
            "Rate value: $1.43. "
            "Effective date: May 1, 2023."
        ),
        (
            "Tariff rate record. Utility: Oncor. "
            "Schedule: Residential Service. "
            "Charge: Metering Charge. "
            "Rate value: $2.80. "
            "Effective date: May 1, 2023."
        ),
        (
            "Weather forecast record. Heavy rain "
            "is expected tomorrow afternoon."
        )
    ]

    query = (
        "What was the Oncor Residential "
        "Customer Charge in 2023?"
    )

    print("=" * 120)
    print("LOCAL EMBEDDING MODEL TEST")
    print("=" * 120)

    document_embeddings = (
        model.embed_documents(
            documents
        )
    )

    query_embedding = (
        model.embed_query(
            query
        )
    )

    print()
    print("MODEL DETAILS")
    print("-" * 120)

    print(
        "Model name          :",
        model.model_name
    )

    print(
        "Embedding dimension :",
        model.embedding_dimension
    )

    print(
        "Document vectors    :",
        len(document_embeddings)
    )

    print(
        "Query vector length :",
        len(query_embedding)
    )

    assert len(
        document_embeddings
    ) == len(documents)

    assert all(
        len(vector)
        == model.embedding_dimension
        for vector in document_embeddings
    )

    assert (
        len(query_embedding)
        == model.embedding_dimension
    )

    print()
    print("VECTOR NORMS")
    print("-" * 120)

    for index, embedding in enumerate(
        document_embeddings,
        start=1
    ):

        norm = vector_norm(
            embedding
        )

        print(
            f"Document {index:<3}: "
            f"{norm:.6f}"
        )

        assert math.isclose(
            norm,
            1.0,
            rel_tol=1e-4,
            abs_tol=1e-4
        )

    query_norm = vector_norm(
        query_embedding
    )

    print(
        f"Query       : "
        f"{query_norm:.6f}"
    )

    assert math.isclose(
        query_norm,
        1.0,
        rel_tol=1e-4,
        abs_tol=1e-4
    )

    similarities = [
        model.similarity(
            query_embedding,
            document_embedding
        )
        for document_embedding
        in document_embeddings
    ]

    print()
    print("QUERY SIMILARITIES")
    print("-" * 120)

    for index, similarity in enumerate(
        similarities,
        start=1
    ):

        print(
            f"Document {index:<3}: "
            f"{similarity:.6f}"
        )

    best_document_index = max(
        range(
            len(similarities)
        ),
        key=similarities.__getitem__
    )

    print()
    print(
        "Best matching document:",
        best_document_index + 1
    )

    print(
        documents[
            best_document_index
        ]
    )

    assert best_document_index == 0, (
        "The Residential Customer Charge "
        "document should be the best match."
    )

    assert (
        similarities[0]
        > similarities[2]
    ), (
        "The tariff record should be more "
        "similar than the weather record."
    )

    assert (
        similarities[0]
        > similarities[1]
    ), (
        "The Customer Charge document should "
        "rank above the Metering Charge."
    )

    print()
    print("=" * 120)
    print("ALL ASSERTIONS PASSED")
    print("=" * 120)


if __name__ == "__main__":
    main()