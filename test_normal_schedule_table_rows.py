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

    table_extractor = DoclingTableExtractor(
        max_pages_per_batch=3
    )

    documents = loader.load("data")

    print("=" * 120)
    print("NORMAL SCHEDULE FULL TABLE ROW TEST")
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
            if section.category == "NORMAL_SCHEDULE"
        ]

        tables = table_extractor.extract(
            pdf_path=document.file_path,
            sections=normal_sections
        )

        tables_by_section = defaultdict(list)

        for table in tables:

            tables_by_section[
                table.schedule_id
            ].append(table)

        print()
        print("=" * 120)
        print("SOURCE FILE:", document.file_name)
        print("=" * 120)

        for section in normal_sections:

            section_tables = tables_by_section.get(
                section.section_id,
                []
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
                "Tables Found:",
                len(section_tables)
            )

            if not section_tables:

                print(
                    "No structured table detected."
                )

                print(
                    "This section will require "
                    "text-based rate extraction."
                )

                continue

            for table in section_tables:

                print()
                print(
                    f"PAGE {table.page_number} "
                    f"| TABLE {table.table_index} "
                    f"| ROWS {table.row_count} "
                    f"| COLUMNS {table.column_count}"
                )

                print("-" * 120)

                for row_number, row in enumerate(
                    table.rows,
                    start=1
                ):

                    print(
                        f"{row_number:>3}:",
                        row
                    )

    print()
    print("=" * 120)
    print("TEST COMPLETED")
    print("=" * 120)


if __name__ == "__main__":
    main()