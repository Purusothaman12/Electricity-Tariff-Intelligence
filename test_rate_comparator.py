from decimal import Decimal

from src.comparison.rate_comparator import (
    RateChangeStatus,
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


EXPECTED_RESIDENTIAL_CHANGES = {
    "CUSTOMER CHARGE": {
        "old": Decimal("0.90"),
        "new": Decimal("1.43")
    },
    "METERING CHARGE": {
        "old": Decimal("2.52"),
        "new": Decimal("2.80")
    },
    "DISTRIBUTION SYSTEM CHARGE": {
        "old": Decimal("0.019841"),
        "new": Decimal("0.025344")
    }
}


def main() -> None:

    loader = RateJSONLoader()

    documents = loader.load_directory(
        "output/rates"
    )

    documents_by_source = {
        document.source_file: document
        for document in documents
    }

    assert (
        OLD_SOURCE_FILE
        in documents_by_source
    )

    assert (
        NEW_SOURCE_FILE
        in documents_by_source
    )

    old_document = documents_by_source[
        OLD_SOURCE_FILE
    ]

    new_document = documents_by_source[
        NEW_SOURCE_FILE
    ]

    comparator = RateComparator()

    result = comparator.compare(
        old_document=old_document,
        new_document=new_document
    )

    print("=" * 120)
    print("RATE COMPARATOR TEST")
    print("=" * 120)

    print()
    print("COMPARISON SUMMARY")
    print("-" * 120)

    for key, value in (
        result.to_summary().items()
    ):

        print(
            f"{key:<30}: {value}"
        )

    assert result.comparisons

    assert result.old_snapshot_count > 0
    assert result.new_snapshot_count > 0

    assert (
        result.changed_count > 0
    )

    residential_comparisons = (
        result.get_schedule_comparisons(
            RESIDENTIAL_SECTION_ID
        )
    )

    assert residential_comparisons

    print()
    print("RESIDENTIAL BASE RATE CHANGES")
    print("-" * 120)

    for charge_name, expected in (
        EXPECTED_RESIDENTIAL_CHANGES.items()
    ):

        matches = [
            comparison
            for comparison
            in residential_comparisons
            if (
                comparison
                .normalized_charge_name
                == charge_name
                and comparison.old_rate
                is not None
                and comparison.new_rate
                is not None
            )
        ]

        assert matches, (
            "No comparison was found for "
            f"{charge_name}."
        )

        comparison = matches[0]

        print(
            f"{comparison.charge_name:<35}"
            f"{str(comparison.old_value):<15}"
            f"{str(comparison.new_value):<15}"
            f"{str(comparison.absolute_change):<18}"
            f"{comparison.status.value:<15}"
            f"{comparison.old_effective_date}"
            f" -> "
            f"{comparison.new_effective_date}"
        )

        assert (
            comparison.old_value
            == expected["old"]
        )

        assert (
            comparison.new_value
            == expected["new"]
        )

        assert (
            comparison.status
            == RateChangeStatus.INCREASED
        )

        assert (
            comparison.absolute_change
            == (
                expected["new"]
                - expected["old"]
            )
        )

        assert comparison.percent_change is not None

        assert (
            comparison.old_effective_date
            == "October 8, 2018"
        )

        assert (
            comparison.new_effective_date
            == "May 1, 2023"
        )

    structural_artifacts = [
        comparison
        for comparison in result.comparisons
        if (
            comparison.charge_name
            .strip()
            .upper()
            .startswith("COLUMN ")
        )
    ]

    assert not structural_artifacts, (
        "Structural table artifacts were included "
        "in the comparison."
    )

    comparison_total = sum(
        result.status_counts.values()
    )

    assert (
        comparison_total
        == result.comparison_count
    )

    print()
    print("STATUS COUNTS")
    print("-" * 120)

    for status, count in (
        result.status_counts.items()
    ):

        print(
            f"{status:<20}: {count}"
        )

    print()
    print("=" * 120)
    print("ALL ASSERTIONS PASSED")
    print("=" * 120)


if __name__ == "__main__":
    main()