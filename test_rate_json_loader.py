from src.loaders.rate_json_loader import (
    RateJSONLoader
)


RESIDENTIAL_SECTION_ID = "6.1.1.1.1"


EXPECTED_RESULTS = {
    "Oncor_May_1_2023.pdf": {
        "rates": 658,
        "unresolved": 23,
        "not_applicable": 3
    },
    "Oncor_November_27_2017.pdf": {
        "rates": 536,
        "unresolved": 41,
        "not_applicable": 3
    }
}


REQUIRED_RESIDENTIAL_CHARGES = {
    "CUSTOMER CHARGE",
    "METERING CHARGE",
    "DISTRIBUTION SYSTEM CHARGE"
}


def main() -> None:

    loader = RateJSONLoader()

    documents = loader.load_directory(
        "output/rates"
    )

    print("=" * 120)
    print("RATE JSON LOADER TEST")
    print("=" * 120)

    assert len(documents) == len(
        EXPECTED_RESULTS
    ), (
        "Unexpected number of loaded documents. "
        f"Expected {len(EXPECTED_RESULTS)}, "
        f"received {len(documents)}."
    )

    loaded_source_files = {
        document.source_file
        for document in documents
    }

    assert (
        loaded_source_files
        == set(EXPECTED_RESULTS)
    ), (
        "Loaded source files do not match the "
        "expected files.\n"
        f"Expected: {set(EXPECTED_RESULTS)}\n"
        f"Received: {loaded_source_files}"
    )

    for document in documents:

        expected = EXPECTED_RESULTS[
            document.source_file
        ]

        print()
        print("-" * 120)
        print(
            "SOURCE FILE:",
            document.source_file
        )
        print("-" * 120)

        summary = document.to_summary()

        for key, value in summary.items():

            print(
                f"{key:<30}: {value}"
            )

        assert (
            document.schema_version
            == "1.1"
        ), (
            "Unexpected schema version for "
            f"{document.source_file}: "
            f"{document.schema_version}"
        )

        assert (
            document.rate_count
            == expected["rates"]
        ), (
            "Unexpected rate count for "
            f"{document.source_file}. "
            f"Expected {expected['rates']}, "
            f"received {document.rate_count}."
        )

        assert (
            document.unresolved_rate_count
            == expected["unresolved"]
        ), (
            "Unexpected unresolved-rate count for "
            f"{document.source_file}. "
            f"Expected {expected['unresolved']}, "
            f"received "
            f"{document.unresolved_rate_count}."
        )

        assert (
            len(
                document.not_applicable_sections
            )
            == expected["not_applicable"]
        ), (
            "Unexpected not-applicable section count "
            f"for {document.source_file}."
        )

        assert not document.review_sections, (
            "The loaded document contains sections "
            "that still require review: "
            f"{document.source_file}"
        )

        assert all(
            rate.source_file
            == document.source_file
            for rate in document.rates
        ), (
            "At least one rate contains an incorrect "
            "source_file value in "
            f"{document.source_file}."
        )

        residential_rates = (
            document.get_schedule_rates(
                RESIDENTIAL_SECTION_ID
            )
        )

        assert residential_rates, (
            "No Residential Service rates were found "
            f"in {document.source_file}."
        )

        residential_names = {
            rate.normalized_charge_name
            for rate in residential_rates
        }

        missing_residential_charges = (
            REQUIRED_RESIDENTIAL_CHARGES
            - residential_names
        )

        assert not missing_residential_charges, (
            "Missing Residential Service charges in "
            f"{document.source_file}: "
            f"{missing_residential_charges}"
        )

        print()
        print("RESIDENTIAL BASE CHARGES")
        print("-" * 120)

        printed_charge_count = 0

        for rate in residential_rates:

            if (
                rate.normalized_charge_name
                not in REQUIRED_RESIDENTIAL_CHARGES
            ):
                continue

            resolution_method = (
                rate.metadata.get(
                    "effective_date_resolution",
                    ""
                )
            )

            numeric_value = rate.numeric_value

            print(
                f"{rate.charge_name:<35}"
                f"{rate.value_text:<15}"
                f"{rate.unit:<35}"
                f"{rate.effective_date:<22}"
                f"{resolution_method}"
            )

            assert numeric_value is not None, (
                "The numeric value could not be parsed "
                f"for {rate.charge_name} in "
                f"{document.source_file}."
            )

            assert rate.effective_date, (
                "Residential base charge has no "
                "effective date: "
                f"{rate.charge_name}"
            )

            printed_charge_count += 1

        assert (
            printed_charge_count
            >= len(
                REQUIRED_RESIDENTIAL_CHARGES
            )
        )

        resolution_total = sum(
            document
            .date_resolution_counts
            .values()
        )

        assert (
            resolution_total
            == document.rate_count
        ), (
            "Effective-date resolution counts do not "
            "equal the total rate count for "
            f"{document.source_file}. "
            f"Resolution total: {resolution_total}, "
            f"rate count: {document.rate_count}."
        )

        unresolved_rates = (
            document.unresolved_rates
        )

        assert all(
            not rate.effective_date
            for rate in unresolved_rates
        )

        assert (
            len(unresolved_rates)
            == expected["unresolved"]
        )

        print()
        print("DATE RESOLUTION COUNTS")
        print("-" * 120)

        for method, count in sorted(
            document
            .date_resolution_counts
            .items()
        ):

            print(
                f"{method:<30}: {count}"
            )

    print()
    print("=" * 120)
    print("ALL ASSERTIONS PASSED")
    print("=" * 120)


if __name__ == "__main__":
    main()