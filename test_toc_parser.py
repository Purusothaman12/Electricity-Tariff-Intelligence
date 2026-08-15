from src.ingestion.pdf_loader import PDFLoader
from src.ingestion.document_extractor import DocumentExtractor
from src.parsing.toc_parser import TOCParser


def main():

    loader = PDFLoader()
    document_extractor = DocumentExtractor()
    toc_parser = TOCParser()

    documents = loader.load("data")

    print("=" * 90)
    print("TOC PARSER TEST")
    print("=" * 90)

    for document in documents:

        document = document_extractor.extract(
            document
        )

        entries = toc_parser.parse(
            document
        )

        print()
        print("=" * 90)
        print(document.file_name)
        print("=" * 90)

        print(
            "Schedule and rider entries found:",
            len(entries)
        )

        print("-" * 90)

        for entry in entries:

            print(
                f"{entry.section_id:<15} "
                f"{entry.title:<60} "
                f"Page {entry.start_page}"
            )

    print()
    print("=" * 90)
    print("TEST COMPLETED")
    print("=" * 90)


if __name__ == "__main__":
    main()