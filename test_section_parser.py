from src.ingestion.pdf_loader import PDFLoader
from src.ingestion.document_extractor import DocumentExtractor
from src.parsing.toc_parser import TOCParser
from src.parsing.section_parser import SectionParser


def main():

    loader = PDFLoader()
    document_extractor = DocumentExtractor()
    toc_parser = TOCParser()
    section_parser = SectionParser()

    documents = loader.load("data")

    print("=" * 110)
    print("SECTION PARSER TEST")
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

        print()
        print("=" * 110)
        print(document.file_name)
        print("=" * 110)

        print(
            "Sections extracted:",
            len(sections)
        )

        print("-" * 110)

        for section in sections:

            preview = (
                section.text
                .replace("\n", " ")
                .strip()
            )

            print(
                f"{section.section_id:<15}"
                f"{section.title:<60}"
                f"{section.start_page:>4} -> "
                f"{section.end_page:<4}"
                f"Chars: {len(section.text):>7}"
            )

            print(
                "Preview:",
                preview[:120]
            )

            print("-" * 110)

    print("=" * 110)
    print("TEST COMPLETED")
    print("=" * 110)


if __name__ == "__main__":
    main()