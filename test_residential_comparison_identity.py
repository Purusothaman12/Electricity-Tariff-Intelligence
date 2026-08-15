from src.comparison.rate_comparator import (
    RateComparator
)
from src.loaders.rate_json_loader import (
    RateJSONLoader
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

REQUIRED_CHARGES = {
    "CUSTOMER CHARGE",
    "METERING CHARGE",
    "DISTRIBUTION SYSTEM CHARGE"
}


def print_rate_details(
    label: str,
    rate,
    comparator: RateComparator
) -> None:

    print()
    print("-" * 120)
    print(label)
    print("-" * 120)

    print(
        "Schedule ID       :",
        rate.schedule_id
    )

    print(
        "Schedule Title    :",
        rate.schedule_title
    )

    print(
        "Category          :",
        rate.category
    )

    print(
        "Charge Name       :",
        rate.charge_name
    )

    print(
        "Normalized Charge :",
        rate.normalized_charge_name
    )

    print(
        "Value             :",
        rate.value_text
    )

    print(
        "Unit              :",
        rate.unit
    )

    print(
        "Normalized Unit   :",
        rate.normalized_unit
    )

    print(
        "Effective Date    :",
        rate.effective_date
    )

    print(
        "Source Method     :",
        rate.source_method
    )

    print(
        "Table Structure   :",
        rate.metadata.get(
            "table_structure",
            ""
        )
    )

    print(
        "Context Heading   :",
        rate.attributes.get(
            "context_heading",
            ""
        )
    )

    print(
        "Parent Charge     :",
        rate.attributes.get(
            "parent_charge",
            ""
        )
    )

    print(
        "Row Label         :",
        rate.attributes.get(
            "row_label",
            ""
        )
    )

    print(
        "Column Header     :",
        rate.attributes.get(
            "column_header",
            ""
        )
    )

    print()
    print("COMPARATOR COMPONENTS")
    print("-" * 120)

    print(
        "Schedule Identity :",
        comparator._schedule_identity(
            rate
        )
    )

    print(
        "Comparison Unit   :",
        comparator._comparison_unit(
            rate
        )
    )

    print(
        "Context Identity  :",
        comparator._context_identity(
            rate
        )
    )

    print(
        "Matrix Row        :",
        comparator._matrix_row_identity(
            rate
        )
    )

    print(
        "Matrix Column     :",
        comparator._matrix_column_identity(
            rate
        )
    )

    print()
    print(
        "Final Identity    :",
        comparator._create_identity(
            rate
        )
    )


def get_relevant_rates(
    document
):

    return [
        rate
        for rate in document.rates
        if (
            rate.schedule_id
            == RESIDENTIAL_SECTION_ID
            and rate.normalized_charge_name
            in REQUIRED_CHARGES
        )
    ]


def main() -> None:

    loader = RateJSONLoader()

    documents = loader.load_directory(
        "output/rates"
    )

    documents_by_source = {
        document.source_file: document
        for document in documents
    }

    old_document = documents_by_source[
        OLD_SOURCE_FILE
    ]

    new_document = documents_by_source[
        NEW_SOURCE_FILE
    ]

    comparator = RateComparator()

    old_rates = get_relevant_rates(
        old_document
    )

    new_rates = get_relevant_rates(
        new_document
    )

    print("=" * 120)
    print(
        "RESIDENTIAL COMPARISON IDENTITY "
        "DIAGNOSTIC"
    )
    print("=" * 120)

    print()
    print(
        "Old matching records:",
        len(old_rates)
    )

    print(
        "New matching records:",
        len(new_rates)
    )

    for rate in old_rates:

        print_rate_details(
            label="OLD RECORD",
            rate=rate,
            comparator=comparator
        )

    for rate in new_rates:

        print_rate_details(
            label="NEW RECORD",
            rate=rate,
            comparator=comparator
        )

    (
        old_snapshot,
        old_artifacts
    ) = comparator._build_current_snapshot(
        old_document.rates
    )

    (
        new_snapshot,
        new_artifacts
    ) = comparator._build_current_snapshot(
        new_document.rates
    )

    old_snapshot_records = {
        identity: rate
        for identity, rate
        in old_snapshot.items()
        if (
            rate.schedule_id
            == RESIDENTIAL_SECTION_ID
            and rate.normalized_charge_name
            in REQUIRED_CHARGES
        )
    }

    new_snapshot_records = {
        identity: rate
        for identity, rate
        in new_snapshot.items()
        if (
            rate.schedule_id
            == RESIDENTIAL_SECTION_ID
            and rate.normalized_charge_name
            in REQUIRED_CHARGES
        )
    }

    print()
    print("=" * 120)
    print("SELECTED OLD SNAPSHOT RECORDS")
    print("=" * 120)

    for identity, rate in (
        old_snapshot_records.items()
    ):

        print()
        print(
            rate.normalized_charge_name
        )

        print(
            "Identity:",
            identity
        )

        print(
            "Value   :",
            rate.value_text
        )

        print(
            "Date    :",
            rate.effective_date
        )

    print()
    print("=" * 120)
    print("SELECTED NEW SNAPSHOT RECORDS")
    print("=" * 120)

    for identity, rate in (
        new_snapshot_records.items()
    ):

        print()
        print(
            rate.normalized_charge_name
        )

        print(
            "Identity:",
            identity
        )

        print(
            "Value   :",
            rate.value_text
        )

        print(
            "Date    :",
            rate.effective_date
        )

    common_identities = (
        set(old_snapshot_records)
        & set(new_snapshot_records)
    )

    old_only_identities = (
        set(old_snapshot_records)
        - set(new_snapshot_records)
    )

    new_only_identities = (
        set(new_snapshot_records)
        - set(old_snapshot_records)
    )

    print()
    print("=" * 120)
    print("IDENTITY MATCH SUMMARY")
    print("=" * 120)

    print(
        "Old artifacts excluded:",
        old_artifacts
    )

    print(
        "New artifacts excluded:",
        new_artifacts
    )

    print(
        "Common identities:",
        len(common_identities)
    )

    print(
        "Old-only identities:",
        len(old_only_identities)
    )

    print(
        "New-only identities:",
        len(new_only_identities)
    )

    print()
    print("COMMON IDENTITIES")
    print("-" * 120)

    for identity in sorted(
        common_identities
    ):

        print(identity)

    print()
    print("OLD-ONLY IDENTITIES")
    print("-" * 120)

    for identity in sorted(
        old_only_identities
    ):

        print(identity)

    print()
    print("NEW-ONLY IDENTITIES")
    print("-" * 120)

    for identity in sorted(
        new_only_identities
    ):

        print(identity)

    print()
    print("=" * 120)
    print("DIAGNOSTIC COMPLETED")
    print("=" * 120)


if __name__ == "__main__":
    main()