from src.models.rate import RateItem
from src.parsing.effective_date_resolver import (
    EffectiveDateResolver
)


SOURCE_FILE = "Oncor_May_1_2023.pdf"


def create_rate(
    schedule_id: str,
    schedule_title: str,
    category: str,
    charge_name: str,
    effective_date: str
) -> RateItem:

    return RateItem(
        schedule_id=schedule_id,
        schedule_title=schedule_title,
        category=category,
        source_file=SOURCE_FILE,
        charge_name=charge_name,
        value_text="$1.00",
        unit="per Retail Customer",
        source_method="TEXT",
        effective_date=effective_date
    )


def main():

    resolver = EffectiveDateResolver(
        minimum_sections=2,
        minimum_consensus_ratio=0.75
    )

    rate_items = [
        create_rate(
            schedule_id="6.1.1.1.1",
            schedule_title="RESIDENTIAL SERVICE",
            category="NORMAL_SCHEDULE",
            charge_name="Customer Charge",
            effective_date="May 1, 2023"
        ),
        create_rate(
            schedule_id="6.1.1.1.2",
            schedule_title=(
                "SECONDARY SERVICE LESS THAN "
                "OR EQUAL TO 10 KW"
            ),
            category="NORMAL_SCHEDULE",
            charge_name="Customer Charge",
            effective_date="May 1, 2023"
        ),
        create_rate(
            schedule_id="6.1.1.1.3",
            schedule_title=(
                "SECONDARY SERVICE GREATER "
                "THAN 10 KW"
            ),
            category="NORMAL_SCHEDULE",
            charge_name="Customer Charge",
            effective_date="May 1, 2023"
        ),
        create_rate(
            schedule_id="6.1.1.1.4",
            schedule_title=(
                "PRIMARY SERVICE LESS THAN "
                "OR EQUAL TO 10 KW"
            ),
            category="NORMAL_SCHEDULE",
            charge_name="Customer Charge",
            effective_date=""
        ),
        create_rate(
            schedule_id="6.1.1.1.5",
            schedule_title=(
                "PRIMARY SERVICE GREATER "
                "THAN 10 KW"
            ),
            category="NORMAL_SCHEDULE",
            charge_name="Customer Charge",
            effective_date="May 1, 2023"
        ),
        create_rate(
            schedule_id="6.1.1.1.5",
            schedule_title=(
                "PRIMARY SERVICE GREATER "
                "THAN 10 KW"
            ),
            category="NORMAL_SCHEDULE",
            charge_name="Metering Charge",
            effective_date=""
        ),
        create_rate(
            schedule_id="6.1.1.6.1",
            schedule_title="RIDER TEST ONE",
            category="RIDER",
            charge_name="Residential Service",
            effective_date="March 1, 2025"
        ),
        create_rate(
            schedule_id="6.1.1.6.2",
            schedule_title="RIDER TEST TWO",
            category="RIDER",
            charge_name="Residential Service",
            effective_date="March 1, 2024"
        ),
        create_rate(
            schedule_id="6.1.1.6.3",
            schedule_title="RIDER TEST THREE",
            category="RIDER",
            charge_name="Residential Service",
            effective_date=""
        )
    ]

    resolved_items = resolver.resolve(
        rate_items
    )

    print("=" * 110)
    print("EFFECTIVE DATE RESOLVER TEST")
    print("=" * 110)

    for rate_item in resolved_items:

        print()
        print(
            "Schedule :",
            rate_item.schedule_title
        )

        print(
            "Charge   :",
            rate_item.charge_name
        )

        print(
            "Date     :",
            rate_item.effective_date
        )

        print(
            "Resolution:",
            rate_item.metadata.get(
                "effective_date_resolution"
            )
        )

        print("-" * 110)

    primary_small = next(
        rate
        for rate in resolved_items
        if rate.schedule_id == "6.1.1.1.4"
    )

    assert (
        primary_small.effective_date
        == "May 1, 2023"
    )

    assert (
        primary_small.metadata[
            "effective_date_resolution"
        ]
        == "CATEGORY_CONSENSUS"
    )

    primary_metering = next(
        rate
        for rate in resolved_items
        if (
            rate.schedule_id == "6.1.1.1.5"
            and rate.charge_name
            == "Metering Charge"
        )
    )

    assert (
        primary_metering.effective_date
        == "May 1, 2023"
    )

    assert (
        primary_metering.metadata[
            "effective_date_resolution"
        ]
        == "SCHEDULE_CONSENSUS"
    )

    unresolved_rider = next(
        rate
        for rate in resolved_items
        if rate.schedule_id == "6.1.1.6.3"
    )

    assert unresolved_rider.effective_date == ""

    assert (
        unresolved_rider.metadata[
            "effective_date_resolution"
        ]
        == "UNRESOLVED"
    )

    print()
    print("=" * 110)
    print("ALL ASSERTIONS PASSED")
    print("=" * 110)


if __name__ == "__main__":
    main()