from src.models.rate import RateItem
from src.parsing.rate_merger import RateMerger


def create_rate(
    charge_name: str,
    value_text: str,
    unit: str,
    source_method: str,
    effective_date: str = "",
    context: str = "",
    page_number: int | None = None,
    table_index: int | None = None
) -> RateItem:

    attributes = {}

    if context:
        attributes[
            "context_heading"
        ] = context

    return RateItem(
        schedule_id="6.1.1.1.1",
        schedule_title="RESIDENTIAL SERVICE",
        category="NORMAL_SCHEDULE",
        source_file="Oncor_May_1_2023.pdf",
        charge_name=charge_name,
        value_text=value_text,
        unit=unit,
        source_method=source_method,
        effective_date=effective_date,
        page_number=page_number,
        table_index=table_index,
        attributes=attributes
    )


def main():

    merger = RateMerger()

    text_rates = [
        create_rate(
            charge_name="Customer Charge",
            value_text="$1.43",
            unit="per Retail Customer",
            source_method="TEXT",
            effective_date="May 1, 2023",
            context="I. Base Rate Charges"
        ),
        create_rate(
            charge_name="Distribution System Charge",
            value_text="$0.025344",
            unit="per kWh",
            source_method="TEXT",
            effective_date="May 1, 2023",
            context="I. Base Rate Charges"
        ),
        create_rate(
            charge_name="Customer Charge",
            value_text="$2.27",
            unit="per Retail Customer",
            source_method="TEXT",
            effective_date="May 1, 2023",
            context=(
                "I. Metered Facilities - "
                "Non-Company Owned"
            )
        ),
        create_rate(
            charge_name="Customer Charge",
            value_text="$2.28",
            unit="per Retail Customer",
            source_method="TEXT",
            effective_date="May 1, 2023",
            context=(
                "I. Metered Facilities - "
                "Company-Owned"
            )
        )
    ]

    table_rates = [
        create_rate(
            charge_name="Customer Charge",
            value_text="$1.43",
            unit="per Retail Customer",
            source_method="DOCLING_ACCURATE",
            page_number=67,
            table_index=1
        ),
        create_rate(
            charge_name="Distribution System Charge",
            value_text="$0.025344",
            unit="per kWh",
            source_method="DOCLING_ACCURATE",
            page_number=67,
            table_index=1
        )
    ]

    merged_rates = merger.merge(
        table_rates=table_rates,
        text_rates=text_rates
    )

    print("=" * 110)
    print("RATE MERGER TEST")
    print("=" * 110)

    for rate in merged_rates:

        print()
        print(
            "Charge       :",
            rate.charge_name
        )

        print(
            "Value        :",
            rate.value_text
        )

        print(
            "Unit         :",
            rate.unit
        )

        print(
            "Effective Date:",
            rate.effective_date
        )

        print(
            "Context      :",
            rate.attributes.get(
                "context_heading",
                ""
            )
        )

        print(
            "Source Method:",
            rate.source_method
        )

        print(
            "Page         :",
            rate.page_number
        )

        print(
            "Merged Count :",
            rate.metadata.get(
                "source_record_count"
            )
        )

        print("-" * 110)

    assert len(merged_rates) == 4

    residential_customer = next(
        rate
        for rate in merged_rates
        if (
            rate.charge_name
            == "Customer Charge"
            and rate.value_text == "$1.43"
        )
    )

    assert (
        residential_customer.effective_date
        == "May 1, 2023"
    )

    assert residential_customer.page_number == 67

    assert (
        "TEXT"
        in residential_customer.source_method
    )

    assert (
        "DOCLING_ACCURATE"
        in residential_customer.source_method
    )

    lighting_contexts = {
        rate.attributes.get(
            "context_heading"
        )
        for rate in merged_rates
        if rate.value_text in {
            "$2.27",
            "$2.28"
        }
    }

    assert len(lighting_contexts) == 2

    print()
    print("=" * 110)
    print("ALL ASSERTIONS PASSED")
    print("=" * 110)


if __name__ == "__main__":
    main()