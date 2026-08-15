import os
import sys
import time

from src.rag.dynamic_answer_generator import (
    DynamicOllamaTariffAnswerGenerator
)
from src.rag.ollama_client import (
    OllamaLLMClient
)
from src.rag.service import (
    TariffRAGService
)


RATES_DIRECTORY = os.getenv(
    "RAG_RATES_DIRECTORY",
    "output/rates"
)

PERSIST_DIRECTORY = os.getenv(
    "RAG_PERSIST_DIRECTORY",
    "output/chroma_test"
)

COLLECTION_NAME = os.getenv(
    "RAG_COLLECTION_NAME",
    "electricity_tariff_test"
)

OLLAMA_MODEL = os.getenv(
    "OLLAMA_MODEL",
    "llama3.2:latest"
)


def print_header() -> None:

    print()
    print("=" * 90)
    print("ELECTRICITY TARIFF INTELLIGENCE")
    print("=" * 90)

    print(
        "Ask questions about Oncor electricity tariffs, "
        "rates, charges, Riders and historical changes."
    )

    print()
    print(
        "Examples:"
    )

    print(
        "  What is the Residential Service Customer Charge?"
    )

    print(
        "  What charges are available under Lighting Service?"
    )

    print(
        "  Compare the Residential Transmission System "
        "Charge effective in 2018 and 2023."
    )

    print(
        "  Which Riders were added in the new tariff?"
    )

    print(
        "  Which tariff sections are marked not applicable?"
    )

    print()
    print(
        "Type 'exit' or 'quit' to close the application."
    )

    print("=" * 90)
    print()


def build_application(
) -> DynamicOllamaTariffAnswerGenerator:

    print(
        "Loading tariff intelligence system..."
    )

    service = TariffRAGService(
        rates_directory=(
            RATES_DIRECTORY
        ),
        persist_directory=(
            PERSIST_DIRECTORY
        ),
        collection_name=(
            COLLECTION_NAME
        ),
        llm_enabled=False
    )

    status = service.get_status()

    if not status.ready:

        raise RuntimeError(
            "The tariff RAG service is not ready."
        )

    print(
        f"Structured documents : "
        f"{status.rate_document_count}"
    )

    print(
        f"Indexed chunks       : "
        f"{status.indexed_chunk_count}"
    )

    print(
        f"Comparison records   : "
        f"{status.comparison_count}"
    )

    print(
        f"Embedding model      : "
        f"{status.embedding_model}"
    )

    print(
        f"Ollama model         : "
        f"{OLLAMA_MODEL}"
    )

    print()

    llm_client = OllamaLLMClient(
        model=OLLAMA_MODEL,
        timeout_seconds=240.0,
        max_output_tokens=700
    )

    if not llm_client.health_check():

        raise RuntimeError(
            "Ollama is not available. "
            "Start Ollama before running the application."
        )

    print(
        "Ollama connection    : READY"
    )

    print()

    return DynamicOllamaTariffAnswerGenerator(
        retriever=service.retriever,
        deterministic_generator=(
            service.answer_generator
        ),
        llm_client=llm_client,
        max_output_tokens=700,
        fallback_on_error=True,
        fallback_on_validation_failure=True
    )


def print_answer(
    result
) -> None:

    print()
    print("-" * 90)
    print("ANSWER")
    print("-" * 90)

    print()
    print(
        result.answer
    )

    print()

    print("-" * 90)
    print("RESPONSE INFO")
    print("-" * 90)

    print(
        f"Question type      : "
        f"{result.question_type.value}"
    )

    print(
        f"Retrieval intent   : "
        f"{result.intent.value}"
    )

    print(
        f"Evidence selected  : "
        f"{result.evidence_selection.selected_count}"
    )

    print(
        f"Generation method  : "
        f"{result.generation_method}"
    )

    if result.model:

        print(
            f"Model              : "
            f"{result.model}"
        )

    print(
        f"Grounded           : "
        f"{result.is_grounded}"
    )

    print(
        f"Validation passed  : "
        f"{result.validation_passed}"
    )

    if (
        result.total_duration_seconds
        is not None
    ):

        print(
            f"LLM duration       : "
            f"{result.total_duration_seconds:.2f} sec"
        )

    print("-" * 90)
    print()


def main() -> None:

    print_header()

    try:

        answer_generator = (
            build_application()
        )

    except Exception as error:

        print()
        print(
            "Application startup failed:"
        )

        print(
            error
        )

        sys.exit(
            1
        )

    print(
        "System ready."
    )

    print()

    while True:

        try:

            question = input(
                "You: "
            ).strip()

        except (
            KeyboardInterrupt,
            EOFError
        ):

            print()
            print(
                "Closing Electricity "
                "Tariff Intelligence."
            )

            break

        if not question:

            continue

        if question.lower() in {
            "exit",
            "quit",
            "q"
        }:

            print(
                "Closing Electricity "
                "Tariff Intelligence."
            )

            break

        start_time = time.perf_counter()

        try:

            result = (
                answer_generator.answer(
                    question=question,
                    requested_top_k=8
                )
            )

        except Exception as error:

            print()
            print(
                "Unable to process the question:"
            )

            print(
                error
            )

            print()

            continue

        total_time = (
            time.perf_counter()
            - start_time
        )

        print_answer(
            result
        )

        print(
            f"Total response time : "
            f"{total_time:.2f} sec"
        )

        print()


if __name__ == "__main__":

    main()