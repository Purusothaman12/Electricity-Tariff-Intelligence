from src.models.rate import RateItem


def main():

    rate_items = [
        RateItem(
            schedule_id="6.1.1.1.1",
            schedule_title="RESIDENTIAL SERVICE",
            category="NORMAL_SCHEDULE",
            source_file="Oncor_May_1_2023.pdf",
            charge_name="Customer Charge",
            value_text="$1.43",
            unit="per Retail Customer",
            source_method="DOCLING",
            page_number=67,
            table_index=1,
            row_index=1
        ),
        RateItem(
            schedule_id="6.1.1.1.1",
            schedule_title="RESIDENTIAL SERVICE",
            category="NORMAL_SCHEDULE",
            source_file="Oncor_May_1_2023.pdf",
            charge_name="Distribution System Charge",
            value_text="$0.025344",
            unit="per kWh",
            source_method="DOCLING",
            page_number=67,
            table_index=1,
            row_index=3
        ),
        RateItem(
            schedule_id="6.1.1.1.1",
            schedule_title="RESIDENTIAL SERVICE",
            category="NORMAL_SCHEDULE",
            source_file="Oncor_May_1_2023.pdf",
            charge_name=(
                "II. Nuclear Decommissioning Charge:"
            ),
            value_text="See Rider NDC",
            unit="per kWh",
            source_method="DOCLING",
            page_number=67,
            table_index=1,
            row_index=4
        ),
        RateItem(
            schedule_id="6.1.1.6.3",
            schedule_title=(
                "RIDER EECRF - ENERGY EFFICIENCY "
                "COST RECOVERY FACTOR"
            ),
            category="RIDER",
            source_file="Oncor_May_1_2023.pdf",
            charge_name="Residential Service",
            value_text="(0.000196)",
            unit="$/kWh",
            source_method="DOCLING",
            page_number=100,
            table_index=1,
            row_index=2,
            effective_date="March 1, 2025"
        )
    ]

    print("=" * 100)
    print("RATE MODEL TEST")
    print("=" * 100)

    for rate in rate_items:

        print()
        print(
            "Charge Name     :",
            rate.charge_name
        )

        print(
            "Normalized Name :",
            rate.normalized_charge_name
        )

        print(
            "Value Text      :",
            rate.value_text
        )

        print(
            "Numeric Value   :",
            rate.numeric_value
        )

        print(
            "Value Kind      :",
            rate.value_kind
        )

        print(
            "Unit            :",
            rate.normalized_unit
        )

        print(
            "Is Reference    :",
            rate.is_reference
        )

        print(
            "Comparison Key  :",
            rate.comparison_key
        )

        print("-" * 100)

    print("=" * 100)
    print("TEST COMPLETED")
    print("=" * 100)


if __name__ == "__main__":
    main()