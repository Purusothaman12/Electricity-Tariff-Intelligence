from src.models.rate import RateItem
from src.models.section import Section
from src.parsing.section_effective_date_resolver import (
    SectionEffectiveDateResolver
)


SOURCE_FILE = "Test_Tariff.pdf"


def create_section(
    section_id: str,
    title: str,
    text: str
) -> Section:

    return Section(
        section_id=section_id,
        title=title,
        start_page=1,
        end_page=2,
        source_file=SOURCE_FILE,
        category="RIDER",
        text=text
    )


def create_rate(
    section_id: str,
    charge_name: str,
    value_text: str,
    effective_date: str = "",
    table_structure: str = "ROW",
    row_label: str = "",
    column_header: str = ""
) -> RateItem:

    return RateItem(
        schedule_id=section_id,
        schedule_title="TEST RIDER",
        category="RIDER",
        source_file=SOURCE_FILE,
        charge_name=charge_name,
        value_text=value_text,
        unit="",
        source_method="DOCLING_ACCURATE",
        page_number=1,
        table_index=1,
        row_index=1,
        effective_date=effective_date,
        attributes={
            "row_label": row_label,
            "column_header": column_header
        },
        metadata={
            "table_structure": (
                table_structure
            ),
            "effective_date_resolved": False,
            "effective_date_resolution": (
                "UNRESOLVED"
            )
        }
    )


def main():

    resolver = (
        SectionEffectiveDateResolver()
    )

    sections = [
        create_section(
            section_id="6.1.1.6.1",
            title="RIDER TCRF",
            text=(
                "6.1.1 Delivery System Charges\n"
                "Effective Date: "
                "September 1, 2025 "
                "Revision: Fifty-One\n"
                "Effective Date ($/kWh)\n"
                "March 1, 2025 0.018796\n"
                "Effective Date: "
                "September 1, 2025 "
                "Revision: Fifty-One"
            )
        ),
        create_section(
            section_id="6.1.1.6.6",
            title="RIDER CSR",
            text=(
                "6.1.1 Delivery System Charges\n"
                "Effective Date: "
                "August 27, 2018 "
                "Revision: One\n"
                "Residential Service "
                "31.122307%"
            )
        ),
        create_section(
            section_id="6.1.1.5.1",
            title="RIDER NDC",
            text=(
                "Nuclear Decommissioning "
                "Charge Factor\n"
                "Residential Service "
                "$0.000218 per kWh"
            )
        ),
        create_section(
            section_id="6.1.1.6.9",
            title="CONFLICTING SECTION",
            text=(
                "Effective Date: "
                "January 1, 2024\n"
                "Effective Date: "
                "February 1, 2024"
            )
        ),
        create_section(
            section_id="6.1.1.6.10",
            title="NARRATIVE DATE SECTION",
            text=(
                "The amount is expected to be "
                "fully billed by December 21, 2018."
            )
        )
    ]

    rates = [
        create_rate(
            section_id="6.1.1.6.1",
            charge_name=(
                "Residential Service"
            ),
            value_text="45.88067225%",
            table_structure="ROW"
        ),
        create_rate(
            section_id="6.1.1.6.1",
            charge_name=(
                "Residential Service"
            ),
            value_text="0.018796",
            effective_date=(
                "March 1, 2025"
            ),
            table_structure="MATRIX",
            row_label=(
                "March 1, 2025"
            ),
            column_header="($/kWh)"
        ),
        create_rate(
            section_id="6.1.1.6.1",
            charge_name="Column 6",
            value_text="0.000514",
            table_structure="MATRIX",
            row_label="Effective Date",
            column_header="($/kWh)"
        ),
        create_rate(
            section_id="6.1.1.6.6",
            charge_name="CSRAF",
            value_text="31.122307%",
            table_structure="MATRIX",
            row_label=(
                "Residential Service"
            ),
            column_header="CSRAF"
        ),
        create_rate(
            section_id="6.1.1.5.1",
            charge_name=(
                "Nuclear Decommissioning "
                "Charge Factor"
            ),
            value_text="$0.000218",
            table_structure="ROW"
        ),
        create_rate(
            section_id="6.1.1.6.9",
            charge_name="Customer Charge",
            value_text="$1.00",
            table_structure="ROW"
        ),
        create_rate(
            section_id="6.1.1.6.10",
            charge_name="Refund Factor",
            value_text="$0.009732",
            table_structure="ROW"
        )
    ]

    resolved_rates = resolver.resolve(
        rate_items=rates,
        sections=sections
    )

    print("=" * 115)
    print(
        "SECTION EFFECTIVE DATE RESOLVER TEST"
    )
    print("=" * 115)

    for rate in resolved_rates:

        print()
        print(
            "Section   :",
            rate.schedule_id
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

        print(
            "Skip Reason:",
            rate.metadata.get(
                "section_effective_date_"
                "skip_reason",
                ""
            )
        )

        print("-" * 115)

    current_tcrf = resolved_rates[0]

    assert (
        current_tcrf.effective_date
        == "September 1, 2025"
    )

    assert (
        current_tcrf.metadata[
            "effective_date_resolution"
        ]
        == "SECTION_HEADER"
    )

    historical_tcrf = resolved_rates[1]

    assert (
        historical_tcrf.effective_date
        == "March 1, 2025"
    )

    matrix_header = resolved_rates[2]

    assert not matrix_header.effective_date

    assert (
        matrix_header.metadata[
            "section_effective_date_"
            "resolution_skipped"
        ]
        is True
    )

    assert (
        matrix_header.metadata[
            "section_effective_date_"
            "skip_reason"
        ]
        == "STRUCTURAL_MATRIX_HEADER"
    )

    csr_rate = resolved_rates[3]

    assert (
        csr_rate.effective_date
        == "August 27, 2018"
    )

    assert (
        csr_rate.metadata[
            "effective_date_resolution"
        ]
        == "SECTION_HEADER"
    )

    ndc_rate = resolved_rates[4]

    assert not ndc_rate.effective_date

    conflicting_rate = resolved_rates[5]

    assert not conflicting_rate.effective_date

    narrative_rate = resolved_rates[6]

    assert not narrative_rate.effective_date

    extracted_tcrf_date = (
        resolver.extract_section_effective_date(
            sections[0]
        )
    )

    assert (
        extracted_tcrf_date
        == "September 1, 2025"
    )

    assert not (
        resolver.extract_section_effective_date(
            sections[3]
        )
    )

    assert not (
        resolver.extract_section_effective_date(
            sections[4]
        )
    )

    print()
    print("=" * 115)
    print("ALL ASSERTIONS PASSED")
    print("=" * 115)


if __name__ == "__main__":
    main()