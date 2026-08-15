from collections import defaultdict

from src.ingestion.document_extractor import DocumentExtractor
from src.ingestion.pdf_loader import PDFLoader
from src.parsing.section_parser import SectionParser
from src.parsing.toc_parser import TOCParser
from src.table_extraction.docling_extractor import (
    DoclingTableExtractor
)


def main():

    loader = PDFLoader()
    document_extractor = DocumentExtractor()
    toc_parser = TOCParser()
    section_parser = SectionParser()
    table_extractor = DoclingTableExtractor()

    documents = loader.load("data")

    print("=" * 120)
    print("AUTOMATIC DOCLING EXTRACTION TEST")
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

        print()
        print("=" * 120)
        print("SOURCE FILE:", document.file_name)
        print("SECTIONS   :", len(sections))
        print("=" * 120)

        tables = table_extractor.extract(
            pdf_path=document.file_path,
            sections=sections
        )

        tables_by_section = defaultdict(list)

        for table in tables:

            tables_by_section[
                table.schedule_id
            ].append(table)

        sections_with_tables = 0
        sections_without_tables = 0

        for section in sections:

            section_tables = tables_by_section.get(
                section.section_id,
                []
            )

            print()
            print("-" * 120)

            print(
                f"{section.section_id} "
                f"| {section.title}"
            )

            print(
                "Category       :",
                section.category
            )

            print(
                "Section Pages  :",
                section.start_page,
                "->",
                section.end_page
            )

            print(
                "Tables Detected:",
                len(section_tables)
            )

            if not section_tables:

                sections_without_tables += 1

                print(
                    "Extraction Route: TEXT"
                )

                continue

            sections_with_tables += 1

            print(
                "Extraction Route: DOCLING TABLE"
            )

            for table in section_tables:

                print()
                print(
                    f"  Page {table.page_number}"
                    f" | Table {table.table_index}"
                    f" | Rows {table.row_count}"
                    f" | Columns {table.column_count}"
                )

                if table.rows:

                    header_preview = (
                        table.rows[0][:5]
                    )

                    print(
                        "  Header Preview:",
                        header_preview
                    )

                if len(table.rows) > 1:

                    data_preview = (
                        table.rows[1][:5]
                    )

                    print(
                        "  Data Preview  :",
                        data_preview
                    )

        print()
        print("=" * 120)
        print("DOCUMENT SUMMARY")
        print("=" * 120)

        print(
            "Total Sections        :",
            len(sections)
        )

        print(
            "Sections With Tables  :",
            sections_with_tables
        )

        print(
            "Sections Without Tables:",
            sections_without_tables
        )

        print(
            "Total Tables Detected :",
            len(tables)
        )

    print()
    print("=" * 120)
    print("TEST COMPLETED")
    print("=" * 120)


if __name__ == "__main__":
    main()