from src.rag.service import (
    TariffRAGService
)


PERSIST_DIRECTORY = (
    "output/chroma_test"
)

COLLECTION_NAME = (
    "electricity_tariff_test"
)

OLLAMA_MODEL = (
    "llama3.2:latest"
)


def main() -> None:

    print("=" * 120)
    print("TARIFF RAG SERVICE WITH OLLAMA TEST")
    print("=" * 120)

    service = TariffRAGService(
        persist_directory=(
            PERSIST_DIRECTORY
        ),
        collection_name=(
            COLLECTION_NAME
        ),
        llm_enabled=True,
        ollama_model=(
            OLLAMA_MODEL
        ),
        llm_timeout_seconds=240.0,
        llm_max_output_tokens=300,
        llm_max_evidence_chunks=1
    )

    service_status = (
        service.get_status()
    )

    llm_status_before = (
        service.get_llm_status()
    )

    print()
    print("SERVICE STATUS")
    print("-" * 120)

    for key, value in (
        service_status.to_dict().items()
    ):

        print(
            f"{key:<25}: {value}"
        )

    print()
    print("LLM STATUS BEFORE QUESTION")
    print("-" * 120)

    for key, value in (
        llm_status_before.items()
    ):

        print(
            f"{key:<25}: {value}"
        )

    assert service_status.ready

    assert (
        service_status.indexed_chunk_count
        == 1557
    )

    assert (
        llm_status_before[
            "enabled"
        ]
        is True
    )

    assert (
        llm_status_before[
            "initialized"
        ]
        is False
    ), (
        "Ollama should be initialized lazily."
    )

    question = (
        "How did the Oncor Residential "
        "Customer Charge change between "
        "the old and new tariffs?"
    )

    print()
    print("QUESTION")
    print("-" * 120)

    print(
        question
    )

    answer = service.ask_with_llm(
        question=question,
        top_k=8
    )

    llm_status_after = (
        service.get_llm_status()
    )

    print()
    print("FINAL ANSWER")
    print("-" * 120)

    print(
        answer.answer
    )

    print()
    print("GENERATION DETAILS")
    print("-" * 120)

    print(
        "Intent             :",
        answer.intent.value
    )

    print(
        "Answer type        :",
        answer.answer_type
    )

    print(
        "Grounded           :",
        answer.is_grounded
    )

    print(
        "Generation method  :",
        answer.generation_method
    )

    print(
        "Model              :",
        answer.model
    )

    print(
        "Validation passed  :",
        answer.validation_passed
    )

    print(
        "Prompt tokens      :",
        answer.prompt_tokens
    )

    print(
        "Output tokens      :",
        answer.output_tokens
    )

    print(
        "Duration           :",
        answer.total_duration_seconds
    )

    print(
        "Evidence count     :",
        len(
            answer.evidence
        )
    )

    print()
    print("VALIDATION NOTES")
    print("-" * 120)

    for note in answer.validation_notes:

        print(
            "-",
            note
        )

    print()
    print("LLM STATUS AFTER QUESTION")
    print("-" * 120)

    for key, value in (
        llm_status_after.items()
    ):

        print(
            f"{key:<25}: {value}"
        )

    assert answer.is_grounded

    assert (
        answer.answer_type
        == "RATE_COMPARISON"
    )

    assert (
        answer.model
        == OLLAMA_MODEL
    )

    assert (
        answer.generation_method
        in {
            "OLLAMA",
            (
                "DETERMINISTIC_"
                "VALIDATION_FALLBACK"
            )
        }
    ), (
        "The Ollama generation path was "
        "not executed."
    )

    assert (
        answer.prompt_tokens
        is not None
    )

    assert (
        answer.output_tokens
        is not None
    )

    assert (
        answer.total_duration_seconds
        is not None
    )

    required_facts = (
        "$0.90",
        "$1.43",
        "October 8, 2018",
        "May 1, 2023",
        "58.89%"
    )

    for fact in required_facts:

        assert fact in answer.answer, (
            "The final answer is missing "
            f"the tariff fact: {fact}"
        )

    assert (
        llm_status_after[
            "initialized"
        ]
        is True
    )

    assert (
        llm_status_after[
            "model"
        ]
        == OLLAMA_MODEL
    )

    assert (
        llm_status_after[
            "max_evidence_chunks"
        ]
        == 1
    )

    print()
    print("=" * 120)
    print("ALL ASSERTIONS PASSED")
    print("=" * 120)


if __name__ == "__main__":
    main()