from src.models.table import ExtractedTable


def main():

    table = ExtractedTable(
        schedule_id="6.1.1.6.3",
        schedule_title=(
            "RIDER EECRF - ENERGY EFFICIENCY "
            "COST RECOVERY FACTOR"
        ),
        category="RIDER",
        source_file="Oncor_May_1_2023.pdf",
        page_number=100,
        table_index=1,
        extraction_method="PDFPLUMBER",
        rows=[
            [
                "Service",
                "Unit",
                "Value"
            ],
            [
                "Residential Service",
                "$/kWh",
                "0.001137"
            ],
            [
                "Secondary Service <= 10 kW",
                "$/kWh",
                "(0.000196)"
            ]
        ],
        metadata={
            "test": True
        }
    )

    print("=" * 80)
    print("TABLE MODEL TEST")
    print("=" * 80)

    print("Schedule ID      :", table.schedule_id)
    print("Schedule Title   :", table.schedule_title)
    print("Category         :", table.category)
    print("Source File      :", table.source_file)
    print("Page Number      :", table.page_number)
    print("Table Index      :", table.table_index)
    print("Extraction Method:", table.extraction_method)
    print("Row Count        :", table.row_count)
    print("Column Count     :", table.column_count)
    print("Is Empty         :", table.is_empty)

    print("-" * 80)
    print("ROWS")
    print("-" * 80)

    for row in table.rows:
        print(row)

    print("=" * 80)
    print("TEST COMPLETED")
    print("=" * 80)


if __name__ == "__main__":
    main()