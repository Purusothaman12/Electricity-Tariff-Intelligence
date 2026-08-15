from src.ingestion.document_extractor import DocumentExtractor
from src.ingestion.pdf_loader import PDFLoader
from src.parsing.section_parser import SectionParser
from src.parsing.toc_parser import TOCParser
from src.table_extraction.pdfplumber_extractor import (
    PDFPlumberExtractor
)


TARGET_SECTIONS = {
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
    table_extractor = PDFPlumberExtractor()

    documents = loader.load("data")

    print("=" * 100)
    print("PDFPLUMBER EXTRACTOR TEST")
    print("=" * 100)

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

        selected_sections = [
            section
            for section in sections
            if section.title in TARGET_SECTIONS
        ]

        print()
        print("=" * 100)
        print(document.file_name)
        print("=" * 100)

        for section in selected_sections:

            tables = table_extractor.extract_section(
                pdf_path=document.file_path,
                section=section
            )

            print()
            print("-" * 100)
            print("Section       :", section.title)
            print(
                "Pages         :",
                section.start_page,
                "->",
                section.end_page
            )
            print("Tables Found  :", len(tables))
            print("-" * 100)

            for table in tables:

                print()
                print(
                    f"Page {table.page_number} "
                    f"| Table {table.table_index} "
                    f"| {table.extraction_method}"
                )

                print(
                    "Rows:",
                    table.row_count,
                    "| Columns:",
                    table.column_count
                )

                print("-" * 100)

                for row in table.rows[:10]:
                    print(row)

                if table.row_count > 10:
                    print(
                        f"... {table.row_count - 10} "
                        f"more rows"
                    )

    print()
    print("=" * 100)
    print("TEST COMPLETED")
    print("=" * 100)


if __name__ == "__main__":
    main()