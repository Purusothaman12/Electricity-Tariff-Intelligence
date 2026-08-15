from src.ingestion.document_extractor import DocumentExtractor
from src.ingestion.pdf_loader import PDFLoader
from src.parsing.section_parser import SectionParser
from src.parsing.table_rate_parser import TableRateParser
from src.parsing.toc_parser import TOCParser
from src.table_extraction.docling_extractor import (
    DoclingTableExtractor
)


TEST_TITLES = {
    "RESIDENTIAL SERVICE",
    (
        "RIDER EECRF - ENERGY EFFICIENCY "
        "COST RECOVERY FACTOR"
    )
}


def main():

    loader = PDFLoader()
    document_extractor = DocumentExtractor()
    toc_parser = TOCParser()
    section_parser = SectionParser()

    table_extractor = DoclingTableExtractor(
        max_pages_per_batch=3
    )

    rate_parser = TableRateParser()

    documents = loader.load("data")

    print("=" * 110)
    print("TABLE RATE PARSER TEST")
    print("=" * 110)

    for document in documents:

        document = document_extractor.extract(
            document
        )

        toc_entries = toc_parser.parse(
            document
        )

        sections = section_parser.parse(
            document=document,
            toc_entries=toc_entries
        )

        test_sections = [
            section
            for section in sections
            if section.title in TEST_TITLES
        ]

        print()
        print("=" * 110)
        print("SOURCE FILE:", document.file_name)
        print("=" * 110)

        for section in test_sections:

            tables = table_extractor.extract_section(
                pdf_path=document.file_path,
                section=section
            )

            rate_items = rate_parser.parse(
                tables
            )

            print()
            print("-" * 110)
            print(
                section.section_id,
                "|",
                section.title
            )

            print(
                "Tables Found:",
                len(tables)
            )

            print(
                "Rates Parsed:",
                len(rate_items)
            )

            print("-" * 110)

            display_items = rate_items

            if len(rate_items) > 20:
                display_items = rate_items[:20]

            for rate in display_items:

                print(
                    f"{rate.charge_name:<55}"
                    f"{rate.value_text:<18}"
                    f"{rate.unit:<25}"
                    f"{rate.effective_date}"
                )

            if len(rate_items) > 20:

                print(
                    f"... {len(rate_items) - 20} "
                    f"additional rate items"
                )

    print()
    print("=" * 110)
    print("TEST COMPLETED")
    print("=" * 110)


if __name__ == "__main__":
    main()
    