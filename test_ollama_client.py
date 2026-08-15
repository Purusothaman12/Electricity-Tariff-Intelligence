from src.rag.ollama_client import (
    OllamaLLMClient
)


def main() -> None:

    print("=" * 120)
    print("OLLAMA LOCAL LLM CLIENT TEST")
    print("=" * 120)

    client = OllamaLLMClient(
        max_output_tokens=100,
        timeout_seconds=180.0
    )

    print()
    print("CLIENT CONFIGURATION")
    print("-" * 120)

    print(
        "Host             :",
        client.host
    )

    print(
        "Model            :",
        client.model
    )

    print(
        "Server available :",
        client.health_check()
    )

    print(
        "Installed models :",
        ", ".join(
            client.list_models()
        )
    )

    assert client.health_check()

    assert (
        client.model
        in client.list_models()
    )

    response = client.generate(
        system_prompt=(
            "Follow the instruction exactly. "
            "Give only the requested words."
        ),
        user_prompt=(
            "Reply with exactly these three "
            "words: OLLAMA CONNECTION WORKING"
        ),
        temperature=0.0,
        max_output_tokens=50
    )

    print()
    print("OLLAMA RESPONSE")
    print("-" * 120)

    print(
        response.text
    )

    print()
    print("RESPONSE DETAILS")
    print("-" * 120)

    print(
        "Model          :",
        response.model
    )

    print(
        "Prompt tokens  :",
        response.prompt_tokens
    )

    print(
        "Output tokens  :",
        response.output_tokens
    )

    print(
        "Duration       :",
        response.total_duration_seconds
    )

    normalized_response = (
        response.text
        .upper()
        .replace(
            ".",
            ""
        )
        .strip()
    )

    assert (
        "OLLAMA CONNECTION WORKING"
        in normalized_response
    )

    assert response.text.strip()
    assert response.model

    print()
    print("=" * 120)
    print("ALL ASSERTIONS PASSED")
    print("=" * 120)


if __name__ == "__main__":
    main()