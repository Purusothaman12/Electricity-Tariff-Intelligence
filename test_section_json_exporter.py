import json

from src.exporters.section_json_exporter import SectionJSONExporter
from src.ingestion.document_extractor import DocumentExtractor
from src.ingestion.pdf_loader import PDFLoader
from src.parsing.section_parser import SectionParser
from src.parsing.toc_parser import TOCParser


def main():

    loader = PDFLoader()
    document_extractor = DocumentExtractor()
    toc_parser = TOCParser()
    section_parser = SectionParser()
    exporter = SectionJSONExporter()

    documents = loader.load("data")

    print("=" * 90)
    print("SECTION JSON EXPORTER TEST")
    print("=" * 90)

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

        output_file = exporter.export(
            document=document,
            sections=sections
        )

        with output_file.open(
            mode="r",
            encoding="utf-8"
        ) as file:

            saved_data = json.load(file)

        print()
        print("Source File   :", document.file_name)
        print("Output File   :", output_file)
        print(
            "Sections Saved:",
            saved_data["document"]["section_count"]
        )

        first_section = saved_data["sections"][0]

        print(
            "First Section :",
            first_section["section_id"],
            first_section["title"]
        )

        print(
            "First Category:",
            first_section["category"]
        )

        print(
            "First Pages   :",
            first_section["start_page"],
            "->",
            first_section["end_page"]
        )

        print("-" * 90)

    print("=" * 90)
    print("TEST COMPLETED")
    print("=" * 90)


if __name__ == "__main__":
    main()