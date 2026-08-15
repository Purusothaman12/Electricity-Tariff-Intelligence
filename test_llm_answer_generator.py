from src.rag.llm_answer_generator import (
    OllamaTariffAnswerGenerator
)
from src.rag.ollama_client import (
    OllamaLLMClient
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


def main() -> None:

    print("=" * 120)
    print("OLLAMA GROUNDED TARIFF ANSWER TEST")
    print("=" * 120)

    service = TariffRAGService(
        persist_directory=(
            PERSIST_DIRECTORY
        ),
        collection_name=(
            COLLECTION_NAME
        )
    )

    assert (
        service.get_status()
        .indexed_chunk_count
        == 1557
    )

    llm_client = OllamaLLMClient(
        timeout_seconds=240.0,
        max_output_tokens=300
    )

    answer_generator = (
        OllamaTariffAnswerGenerator(
            deterministic_generator=(
                service.answer_generator
            ),
            llm_client=llm_client,
            max_evidence_chunks=1,
            max_output_tokens=300,
            fallback_on_error=True,
            fallback_on_validation_failure=True
        )
    )

    question = (
        "How did the Oncor Residential "
        "Customer Charge change between "
        "the old and new tariffs?"
    )

    print()
    print("QUESTION")
    print("-" * 120)
    print(question)

    answer = answer_generator.answer(
        query=question,
        top_k=8
    )

    print()
    print("FINAL ANSWER")
    print("-" * 120)
    print(answer.answer)

    print()
    print("DETERMINISTIC ANSWER")
    print("-" * 120)
    print(
        answer.deterministic_answer
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
    print("EVIDENCE")
    print("-" * 120)

    for evidence in answer.evidence:

        print()
        print(
            f"Evidence {evidence.rank}: "
            f"{evidence.chunk_id}"
        )

        print(
            evidence.content
        )

    assert answer.is_grounded

    assert (
        answer.answer_type
        == "RATE_COMPARISON"
    )

    assert (
        answer.model
        == "llama3.2:latest"
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
        "The local LLM was not used for "
        "answer generation."
    )

    assert (
        answer.prompt_tokens
        is not None
    ), (
        "No Ollama prompt-token usage was "
        "returned."
    )

    assert (
        answer.output_tokens
        is not None
    ), (
        "No Ollama output-token usage was "
        "returned."
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
            f"the protected fact: {fact}"
        )

    assert (
        "RESIDENTIAL SERVICE"
        in answer.answer.upper()
    )

    assert answer.evidence

    response_dictionary = (
        answer.to_dict()
    )

    assert (
        response_dictionary[
            "generation_method"
        ]
        == answer.generation_method
    )

    assert (
        response_dictionary[
            "model"
        ]
        == "llama3.2:latest"
    )

    print()
    print("=" * 120)
    print("ALL ASSERTIONS PASSED")
    print("=" * 120)


if __name__ == "__main__":
    main()