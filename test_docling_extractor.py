from src.ingestion.document_extractor import DocumentExtractor
from src.ingestion.pdf_loader import PDFLoader
from src.parsing.section_parser import SectionParser
from src.parsing.toc_parser import TOCParser
from src.table_extraction.docling_extractor import (
    DoclingTableExtractor
)


TARGET_TITLE = (
    "RIDER EECRF - ENERGY EFFICIENCY "
    "COST RECOVERY FACTOR"
)


def main():

    loader = PDFLoader()
    document_extractor = DocumentExtractor()
    toc_parser = TOCParser()
    section_parser = SectionParser()
    table_extractor = DoclingTableExtractor()

    documents = loader.load("data")

    print("=" * 110)
    print("DOCLING EXTRACTOR TEST")
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

        eecrf_section = next(
            (
                section
                for section in sections
                if section.title == TARGET_TITLE
            ),
            None
        )

        if eecrf_section is None:

            print(
                "EECRF section not found:",
                document.file_name
            )

            continue

        tables = table_extractor.extract_section(
            pdf_path=document.file_path,
            section=eecrf_section
        )

        print()
        print("=" * 110)
        print("Source File  :", document.file_name)
        print("Tables Found :", len(tables))
        print("=" * 110)

        for table in tables:

            print()
            print("-" * 110)

            print(
                f"Page {table.page_number} "
                f"| Table {table.table_index}"
            )

            print(
                "Method :",
                table.extraction_method
            )

            print(
                "Rows   :",
                table.row_count
            )

            print(
                "Columns:",
                table.column_count
            )

            print("-" * 110)

            for row in table.rows:
                print(row)

    print()
    print("=" * 110)
    print("TEST COMPLETED")
    print("=" * 110)


if __name__ == "__main__":
    main()