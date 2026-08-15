from collections import Counter

from src.pipelines.tariff_rate_pipeline import (
    TariffRatePipeline
)


RESIDENTIAL_SECTION_ID = "6.1.1.1.1"

PRIMARY_SMALL_SECTION_ID = (
    "6.1.1.1.4"
)


def main():

    pipeline = TariffRatePipeline(
        max_pages_per_batch=3
    )

    results = pipeline.process_directory(
        "data"
    )

    print()
    print("=" * 120)
    print("END-TO-END PIPELINE RESULTS")
    print("=" * 120)

    assert results

    for result in results:

        summary = result.to_summary()

        print()
        print("=" * 120)
        print(
            "SOURCE FILE:",
            result.source_file
        )
        print("=" * 120)

        for key, value in summary.items():

            print(
                f"{key:<25}:",
                value
            )

        assert result.sections
        assert result.tables
        assert result.final_rates

        residential_rates = [
            rate
            for rate in result.final_rates
            if (
                rate.schedule_id
                == RESIDENTIAL_SECTION_ID
            )
        ]

        assert residential_rates

        residential_charge_names = {
            rate.normalized_charge_name
            for rate in residential_rates
        }

        required_residential_charges = {
            "CUSTOMER CHARGE",
            "METERING CHARGE",
            "DISTRIBUTION SYSTEM CHARGE"
        }

        missing_residential_charges = (
            required_residential_charges
            - residential_charge_names
        )

        assert not missing_residential_charges, (
            "Missing Residential charges: "
            f"{missing_residential_charges}"
        )

        print()
        print("RESIDENTIAL BASE CHARGES")
        print("-" * 120)

        for rate in residential_rates:

            if (
                rate.normalized_charge_name
                not in required_residential_charges
            ):
                continue

            print(
                f"{rate.charge_name:<35}"
                f"{rate.value_text:<15}"
                f"{rate.unit:<35}"
                f"{rate.effective_date:<20}"
                f"{rate.source_method}"
            )

        primary_small_rates = [
            rate
            for rate in result.final_rates
            if (
                rate.schedule_id
                == PRIMARY_SMALL_SECTION_ID
            )
        ]

        assert primary_small_rates

        unresolved_primary_rates = [
            rate
            for rate in primary_small_rates
            if not rate.effective_date
        ]

        assert not unresolved_primary_rates, (
            "Primary <=10 kW still has "
            "unresolved effective dates."
        )

        resolution_counts = Counter(
            rate.metadata.get(
                "effective_date_resolution",
                ""
            )
            for rate in primary_small_rates
        )

        print()
        print(
            "PRIMARY <=10 KW DATE RESOLUTION:",
            dict(resolution_counts)
        )

        merged_records = [
            rate
            for rate in result.final_rates
            if (
                "+"
                in rate.source_method
            )
        ]

        print(
            "TEXT + DOCLING MERGED RECORDS:",
            len(merged_records)
        )

        assert merged_records

        historical_rates = [
            rate
            for rate in result.final_rates
            if (
                rate.category == "RIDER"
                and rate.effective_date
                and rate.attributes.get(
                    "column_header"
                )
            )
        ]

        print(
            "HISTORICAL MATRIX RATES:",
            len(historical_rates)
        )

        assert historical_rates

    print()
    print("=" * 120)
    print("ALL END-TO-END ASSERTIONS PASSED")
    print("=" * 120)


if __name__ == "__main__":
    main()