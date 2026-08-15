import json

from src.exporters.rate_json_exporter import (
    RateJSONExporter
)
from src.pipelines.tariff_rate_pipeline import (
    TariffRatePipeline
)


RESIDENTIAL_SECTION_ID = (
    "6.1.1.1.1"
)

EXPECTED_NOT_APPLICABLE_IDS = {
    "6.1.1.2.1",
    "6.1.1.3.1",
    "6.1.1.4.1"
}


def main():

    pipeline = TariffRatePipeline(
        max_pages_per_batch=3
    )

    exporter = RateJSONExporter()

    results = pipeline.process_directory(
        "data"
    )

    exported_paths = (
        exporter.export_results(
            results=results,
            output_directory="output/rates"
        )
    )

    print()
    print("=" * 120)
    print("RATE JSON EXPORT TEST")
    print("=" * 120)

    assert results

    assert len(exported_paths) == len(
        results
    )

    for result, exported_path in zip(
        results,
        exported_paths
    ):

        assert exported_path.exists()

        with exported_path.open(
            mode="r",
            encoding="utf-8"
        ) as input_file:

            payload = json.load(
                input_file
            )

        coverage_summary = payload[
            "coverage_summary"
        ]

        print()
        print("-" * 120)

        print(
            "Source File              :",
            result.source_file
        )

        print(
            "Exported Path            :",
            exported_path
        )

        print(
            "Schema Version           :",
            payload["schema_version"]
        )

        print(
            "Final Rates              :",
            len(payload["rates"])
        )

        print(
            "Total Sections           :",
            coverage_summary[
                "total_sections"
            ]
        )

        print(
            "Applicable Sections      :",
            coverage_summary[
                "applicable_sections"
            ]
        )

        print(
            "Not Applicable Sections  :",
            coverage_summary[
                "not_applicable_sections"
            ]
        )

        print(
            "Unknown Sections         :",
            coverage_summary[
                "unknown_sections"
            ]
        )

        print(
            "Sections With Rates      :",
            coverage_summary[
                "sections_with_rates"
            ]
        )

        print(
            "Review Required Sections :",
            coverage_summary[
                "review_required_sections"
            ]
        )

        print(
            "Unresolved Rates         :",
            coverage_summary[
                "unresolved_rate_count"
            ]
        )

        assert (
            payload["schema_version"]
            == "1.1"
        )

        assert (
            payload["source_document"][
                "file_name"
            ]
            == result.source_file
        )

        assert (
            len(payload["rates"])
            == len(result.final_rates)
        )

        assert (
            payload["summary"][
                "final_rates"
            ]
            == len(result.final_rates)
        )

        assert (
            coverage_summary[
                "total_sections"
            ]
            == len(result.sections)
        )

        assert (
            coverage_summary[
                "not_applicable_sections"
            ]
            == 3
        )

        assert (
            coverage_summary[
                "review_required_sections"
            ]
            == 0
        )

        assert (
            coverage_summary[
                "applicable_without_rates"
            ]
            == 0
        )

        assert (
            coverage_summary[
                "unknown_without_rates"
            ]
            == 0
        )

        assert not payload[
            "review_sections"
        ]

        assert not payload[
            "missing_sections"
        ]

        not_applicable_sections = payload[
            "not_applicable_sections"
        ]

        not_applicable_ids = {
            section["section_id"]
            for section
            in not_applicable_sections
        }

        assert (
            not_applicable_ids
            == EXPECTED_NOT_APPLICABLE_IDS
        )

        print()
        print("NOT APPLICABLE SECTIONS")
        print("-" * 120)

        for section in (
            not_applicable_sections
        ):

            print(
                section["section_id"],
                "|",
                section["title"],
                "|",
                section[
                    "applicability_match"
                ]
            )

            assert (
                section[
                    "coverage_status"
                ]
                == "NOT_APPLICABLE"
            )

            assert (
                section[
                    "rate_count"
                ]
                == 0
            )

            assert not section[
                "extraction_expected"
            ]

        residential_rates = [
            rate
            for rate in payload["rates"]
            if (
                rate["schedule_id"]
                == RESIDENTIAL_SECTION_ID
            )
        ]

        assert residential_rates

        residential_names = {
            rate[
                "normalized_charge_name"
            ]
            for rate
            in residential_rates
        }

        required_names = {
            "CUSTOMER CHARGE",
            "METERING CHARGE",
            "DISTRIBUTION SYSTEM CHARGE"
        }

        assert required_names.issubset(
            residential_names
        )

        print()
        print("RESIDENTIAL BASE CHARGES")
        print("-" * 120)

        for rate in residential_rates:

            if (
                rate[
                    "normalized_charge_name"
                ]
                not in required_names
            ):
                continue

            print(
                f"{rate['charge_name']:<35}"
                f"{rate['value_text']:<15}"
                f"{rate['unit']:<35}"
                f"{rate['effective_date']:<20}"
                f"{rate['source_method']}"
            )

    print()
    print("=" * 120)
    print("ALL JSON EXPORT ASSERTIONS PASSED")
    print("=" * 120)


if __name__ == "__main__":
    main()