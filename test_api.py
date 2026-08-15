from fastapi.testclient import TestClient

from src.api.app import create_app
from src.rag.service import TariffRAGService


PERSIST_DIRECTORY = (
    "output/chroma_test"
)

COLLECTION_NAME = (
    "electricity_tariff_test"
)


def build_test_service(
) -> TariffRAGService:

    return TariffRAGService(
        persist_directory=(
            PERSIST_DIRECTORY
        ),
        collection_name=(
            COLLECTION_NAME
        )
    )


def main() -> None:

    app = create_app(
        service_factory=(
            build_test_service
        )
    )

    print("=" * 120)
    print("ELECTRICITY TARIFF API TEST")
    print("=" * 120)

    with TestClient(
        app
    ) as client:

        root_response = client.get(
            "/"
        )

        assert (
            root_response.status_code
            == 200
        )

        root_data = (
            root_response.json()
        )

        assert (
            root_data[
                "ask_endpoint"
            ]
            == "/ask"
        )

        health_response = client.get(
            "/health"
        )

        assert (
            health_response.status_code
            == 200
        )

        health_data = (
            health_response.json()
        )

        assert (
            health_data[
                "ready"
            ]
            is True
        )

        assert (
            health_data[
                "indexed_chunk_count"
            ]
            == 1557
        )

        assert (
            health_data[
                "old_rate_count"
            ]
            == 536
        )

        assert (
            health_data[
                "new_rate_count"
            ]
            == 658
        )

        rate_response = client.post(
            "/ask",
            json={
                "question": (
                    "What was the Oncor "
                    "Residential Customer Charge "
                    "in 2023?"
                ),
                "top_k": 8
            }
        )

        assert (
            rate_response.status_code
            == 200
        )

        rate_data = (
            rate_response.json()
        )

        assert (
            rate_data[
                "is_grounded"
            ]
            is True
        )

        assert (
            rate_data[
                "answer_type"
            ]
            == "RATE_LOOKUP"
        )

        assert (
            "$1.43"
            in rate_data[
                "answer"
            ]
        )

        comparison_response = (
            client.post(
                "/ask",
                json={
                    "question": (
                        "How did the Residential "
                        "Customer Charge change "
                        "between the old and "
                        "new tariffs?"
                    ),
                    "top_k": 8
                }
            )
        )

        assert (
            comparison_response.status_code
            == 200
        )

        comparison_data = (
            comparison_response.json()
        )

        assert (
            comparison_data[
                "answer_type"
            ]
            == "RATE_COMPARISON"
        )

        assert (
            "$0.90"
            in comparison_data[
                "answer"
            ]
        )

        assert (
            "$1.43"
            in comparison_data[
                "answer"
            ]
        )

        rider_response = client.post(
            "/ask",
            json={
                "question": (
                    "Which Riders were added "
                    "in the new tariff?"
                )
            }
        )

        assert (
            rider_response.status_code
            == 200
        )

        rider_data = (
            rider_response.json()
        )

        assert (
            rider_data[
                "answer_type"
            ]
            == "RIDER_SCHEDULE_CHANGE"
        )

        assert (
            "RIDER ISR"
            in rider_data[
                "answer"
            ]
        )

        assert (
            "RIDER MG"
            in rider_data[
                "answer"
            ]
        )

        retrieve_response = (
            client.post(
                "/retrieve",
                json={
                    "question": (
                        "Residential Customer "
                        "Charge comparison"
                    ),
                    "top_k": 3,
                    "intent": "COMPARISON"
                }
            )
        )

        assert (
            retrieve_response.status_code
            == 200
        )

        retrieve_data = (
            retrieve_response.json()
        )

        assert (
            retrieve_data[
                "intent"
            ]
            == "COMPARISON"
        )

        assert (
            retrieve_data[
                "result_count"
            ]
            == 3
        )

        first_result = (
            retrieve_data[
                "results"
            ][0]
        )

        assert (
            first_result[
                "metadata"
            ][
                "schedule_title"
            ]
            == "RESIDENTIAL SERVICE"
        )

        assert (
            first_result[
                "metadata"
            ][
                "normalized_charge_name"
            ]
            == "CUSTOMER CHARGE"
        )

        validation_response = (
            client.post(
                "/ask",
                json={
                    "question": "",
                    "top_k": 0
                }
            )
        )

        assert (
            validation_response.status_code
            == 422
        )

        docs_response = client.get(
            "/docs"
        )

        assert (
            docs_response.status_code
            == 200
        )

        print()
        print("HEALTH")
        print("-" * 120)

        for key, value in (
            health_data.items()
        ):

            print(
                f"{key:<25}: {value}"
            )

        print()
        print("RATE LOOKUP")
        print("-" * 120)
        print(
            rate_data[
                "answer"
            ]
        )

        print()
        print("RATE COMPARISON")
        print("-" * 120)
        print(
            comparison_data[
                "answer"
            ]
        )

        print()
        print("ADDED RIDERS")
        print("-" * 120)
        print(
            rider_data[
                "answer"
            ]
        )

        print()
        print("RETRIEVAL")
        print("-" * 120)

        print(
            "Intent:",
            retrieve_data[
                "intent"
            ]
        )

        print(
            "Result count:",
            retrieve_data[
                "result_count"
            ]
        )

        print(
            "Top result:",
            first_result[
                "content"
            ]
        )

        print()
        print(
            "Validation status:",
            validation_response
            .status_code
        )

        print(
            "Swagger documentation status:",
            docs_response.status_code
        )

    print()
    print("=" * 120)
    print("ALL ASSERTIONS PASSED")
    print("=" * 120)


if __name__ == "__main__":
    main()