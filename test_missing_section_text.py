import json

from pathlib import Path

from src.ingestion.document_extractor import (
    DocumentExtractor
)
from src.ingestion.pdf_loader import PDFLoader
from src.parsing.section_parser import (
    SectionParser
)
from src.parsing.toc_parser import TOCParser


def print_numbered_text(
    text: str
) -> None:

    lines = text.splitlines()

    for line_number, line in enumerate(
        lines,
        start=1
    ):

        print(
            f"{line_number:>4}: {line}"
        )


def main():

    rates_directory = Path(
        "output/rates"
    )

    json_paths = sorted(
        rates_directory.glob(
            "*_rates.json"
        )
    )

    if not json_paths:

        raise FileNotFoundError(
            "No exported rate JSON files found "
            "inside output/rates."
        )

    loader = PDFLoader()
    document_extractor = DocumentExtractor()
    toc_parser = TOCParser()
    section_parser = SectionParser()

    documents = loader.load(
        "data"
    )

    documents_by_name = {
        document.file_name: document
        for document in documents
    }

    print("=" * 120)
    print("MISSING SECTION TEXT TEST")
    print("=" * 120)

    for json_path in json_paths:

        with json_path.open(
            mode="r",
            encoding="utf-8"
        ) as input_file:

            payload = json.load(
                input_file
            )

        source_file = payload[
            "source_document"
        ][
            "file_name"
        ]

        missing_sections = payload.get(
            "missing_sections",
            []
        )

        print()
        print("=" * 120)
        print(
            "SOURCE FILE:",
            source_file
        )
        print("=" * 120)

        if not missing_sections:

            print(
                "No missing sections recorded."
            )

            continue

        document = documents_by_name.get(
            source_file
        )

        if document is None:

            raise FileNotFoundError(
                f"Source PDF not found in data: "
                f"{source_file}"
            )

        document = (
            document_extractor.extract(
                document
            )
        )

        toc_entries = toc_parser.parse(
            document
        )

        sections = section_parser.parse(
            document=document,
            toc_entries=toc_entries
        )

        sections_by_id = {
            section.section_id: section
            for section in sections
        }

        for missing_section in missing_sections:

            section_id = missing_section[
                "section_id"
            ]

            section = sections_by_id.get(
                section_id
            )

            print()
            print("-" * 120)

            print(
                section_id,
                "|",
                missing_section["title"]
            )

            print(
                "Category:",
                missing_section["category"]
            )

            if section is None:

                print(
                    "Section could not be found "
                    "in parsed document."
                )

                continue

            print(
                "Pages:",
                section.start_page,
                "->",
                section.end_page
            )

            print(
                "Character Count:",
                len(section.text)
            )

            print("-" * 120)

            if not section.text.strip():

                print(
                    "SECTION TEXT IS EMPTY"
                )

                continue

            print_numbered_text(
                section.text
            )

    print()
    print("=" * 120)
    print("TEST COMPLETED")
    print("=" * 120)


if __name__ == "__main__":
    main()