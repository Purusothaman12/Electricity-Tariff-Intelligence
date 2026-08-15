from src.ingestion.document_extractor import DocumentExtractor
from src.ingestion.pdf_loader import PDFLoader
from src.parsing.section_parser import SectionParser
from src.parsing.text_rate_parser import TextRateParser
from src.parsing.toc_parser import TOCParser


def main():

    loader = PDFLoader()
    document_extractor = DocumentExtractor()
    toc_parser = TOCParser()
    section_parser = SectionParser()
    text_rate_parser = TextRateParser()

    documents = loader.load("data")

    print("=" * 120)
    print("TEXT RATE PARSER TEST")
    print("=" * 120)

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

        normal_sections = [
            section
            for section in sections
            if section.category
            == "NORMAL_SCHEDULE"
        ]

        print()
        print("=" * 120)
        print("SOURCE FILE:", document.file_name)
        print("=" * 120)

        for section in normal_sections:

            rate_items = (
                text_rate_parser.parse_section(
                    section
                )
            )

            print()
            print("-" * 120)

            print(
                section.section_id,
                "|",
                section.title
            )

            print(
                "Pages:",
                section.start_page,
                "->",
                section.end_page
            )

            print(
                "Effective Date:",
                (
                    rate_items[0].effective_date
                    if rate_items
                    else ""
                )
            )

            print(
                "Rates Parsed:",
                len(rate_items)
            )

            print("-" * 120)

            display_items = rate_items[:30]

            for rate_item in display_items:

                context = (
                    rate_item.attributes.get(
                        "context_heading",
                        ""
                    )
                )

                print(
                    f"{rate_item.charge_name:<65}"
                    f"{rate_item.value_text:<18}"
                    f"{rate_item.unit:<40}"
                    f"{context}"
                )

            if len(rate_items) > 30:

                print(
                    f"... {len(rate_items) - 30} "
                    f"additional rate items"
                )

    print()
    print("=" * 120)
    print("TEST COMPLETED")
    print("=" * 120)


if __name__ == "__main__":
    main()