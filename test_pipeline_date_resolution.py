from src.models.rate import RateItem
from src.models.section import Section
from src.pipelines.tariff_rate_pipeline import (
    TariffRatePipeline
)


SOURCE_FILE = "Test_Tariff.pdf"


def create_rate(
    schedule_id: str,
    schedule_title: str,
    category: str,
    charge_name: str,
    value_text: str,
    effective_date: str = "",
    table_structure: str = "ROW",
    row_label: str = "",
    column_header: str = ""
) -> RateItem:

    return RateItem(
        schedule_id=schedule_id,
        schedule_title=schedule_title,
        category=category,
        source_file=SOURCE_FILE,
        charge_name=charge_name,
        value_text=value_text,
        unit="",
        source_method="DOCLING_ACCURATE",
        effective_date=effective_date,
        attributes={
            "row_label": row_label,
            "column_header": column_header
        },
        metadata={
            "table_structure": (
                table_structure
            )
        }
    )


def create_section(
    section_id: str,
    title: str,
    category: str,
    text: str
) -> Section:

    return Section(
        section_id=section_id,
        title=title,
        start_page=1,
        end_page=2,
        source_file=SOURCE_FILE,
        category=category,
        text=text
    )


def main():

    pipeline = TariffRatePipeline(
        max_pages_per_batch=3
    )

    rates = [
        create_rate(
            schedule_id="6.1.1.1.1",
            schedule_title="RESIDENTIAL SERVICE",
            category="NORMAL_SCHEDULE",
            charge_name="Customer Charge",
            value_text="$1.43",
            effective_date="May 1, 2023"
        ),
        create_rate(
            schedule_id="6.1.1.1.2",
            schedule_title="SECONDARY SERVICE",
            category="NORMAL_SCHEDULE",
            charge_name="Customer Charge",
            value_text="$2.00",
            effective_date="May 1, 2023"
        ),
        create_rate(
            schedule_id="6.1.1.1.3",
            schedule_title="PRIMARY SERVICE",
            category="NORMAL_SCHEDULE",
            charge_name="Customer Charge",
            value_text="$3.00",
            effective_date="May 1, 2023"
        ),
        create_rate(
            schedule_id="6.1.1.1.4",
            schedule_title="TRANSMISSION SERVICE",
            category="NORMAL_SCHEDULE",
            charge_name="Customer Charge",
            value_text="$4.00"
        ),
        create_rate(
            schedule_id="6.1.1.6.1",
            schedule_title="RIDER TCRF",
            category="RIDER",
            charge_name="Residential Service",
            value_text="45.88067225%"
        ),
        create_rate(
            schedule_id="6.1.1.6.1",
            schedule_title="RIDER TCRF",
            category="RIDER",
            charge_name="Column 6",
            value_text="0.000514",
            table_structure="MATRIX",
            row_label="Effective Date",
            column_header="($/kWh)"
        ),
        create_rate(
            schedule_id="6.1.1.6.2",
            schedule_title="RIDER TEST TWO",
            category="RIDER",
            charge_name="Residential Service",
            value_text="$0.010000",
            effective_date="March 1, 2025"
        ),
        create_rate(
            schedule_id="6.1.1.6.3",
            schedule_title="RIDER TEST THREE",
            category="RIDER",
            charge_name="Residential Service",
            value_text="$0.020000",
            effective_date="September 1, 2024"
        ),
        create_rate(
            schedule_id="6.1.1.6.4",
            schedule_title="NARRATIVE RIDER",
            category="RIDER",
            charge_name="Refund Factor",
            value_text="$0.009732"
        )
    ]

    sections = [
        create_section(
            section_id="6.1.1.6.1",
            title="RIDER TCRF",
            category="RIDER",
            text=(
                "6.1.1 Delivery System Charges\n"
                "Effective Date: "
                "September 1, 2025 "
                "Revision: Fifty-One\n"
                "Effective Date ($/kWh)\n"
                "March 1, 2025 0.018796"
            )
        ),
        create_section(
            section_id="6.1.1.6.4",
            title="NARRATIVE RIDER",
            category="RIDER",
            text=(
                "The refund is expected to be "
                "fully collected by "
                "December 21, 2018."
            )
        )
    ]

    resolved_rates = (
        pipeline.resolve_effective_dates(
            merged_rates=rates,
            sections=sections
        )
    )

    print("=" * 115)
    print("PIPELINE DATE RESOLUTION TEST")
    print("=" * 115)

    for rate in resolved_rates:

        print()
        print(
            "Schedule  :",
            rate.schedule_id,
            "|",
            rate.schedule_title
        )

        print(
            "Charge    :",
            rate.charge_name
        )

        print(
            "Value     :",
            rate.value_text
        )

        print(
            "Date      :",
            rate.effective_date
        )

        print(
            "Resolution:",
            rate.metadata.get(
                "effective_date_resolution",
                ""
            )
        )

        print(
            "Skipped   :",
            rate.metadata.get(
                "section_effective_date_"
                "resolution_skipped",
                False
            )
        )

        print("-" * 115)

    normal_missing = next(
        rate
        for rate in resolved_rates
        if rate.schedule_id == "6.1.1.1.4"
    )

    assert (
        normal_missing.effective_date
        == "May 1, 2023"
    )

    assert (
        normal_missing.metadata[
            "effective_date_resolution"
        ]
        == "CATEGORY_CONSENSUS"
    )

    tcrf_current = next(
        rate
        for rate in resolved_rates
        if (
            rate.schedule_id
            == "6.1.1.6.1"
            and rate.charge_name
            == "Residential Service"
        )
    )

    assert (
        tcrf_current.effective_date
        == "September 1, 2025"
    )

    assert (
        tcrf_current.metadata[
            "effective_date_resolution"
        ]
        == "SECTION_HEADER"
    )

    matrix_header = next(
        rate
        for rate in resolved_rates
        if rate.charge_name == "Column 6"
    )

    assert not matrix_header.effective_date

    assert (
        matrix_header.metadata[
            "section_effective_date_"
            "resolution_skipped"
        ]
        is True
    )

    narrative_rate = next(
        rate
        for rate in resolved_rates
        if rate.schedule_id == "6.1.1.6.4"
    )

    assert not narrative_rate.effective_date

    assert (
        narrative_rate.metadata[
            "effective_date_resolution"
        ]
        == "UNRESOLVED"
    )

    print()
    print("=" * 115)
    print("ALL ASSERTIONS PASSED")
    print("=" * 115)


if __name__ == "__main__":
    main()