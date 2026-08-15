from pathlib import Path

from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import (
    PdfPipelineOptions,
    TableFormerMode
)
from docling.document_converter import (
    DocumentConverter,
    PdfFormatOption
)

from src.ingestion.document_extractor import DocumentExtractor
from src.ingestion.pdf_loader import PDFLoader
from src.parsing.section_parser import SectionParser
from src.parsing.toc_parser import TOCParser


TARGET_TITLE = (
    "RIDER EECRF - ENERGY EFFICIENCY "
    "COST RECOVERY FACTOR"
)

OUTPUT_DIRECTORY = Path(
    "output/docling_test"
)


def create_converter() -> DocumentConverter:
    """
    Creates a Docling converter configured for
    complex, text-based tariff tables.
    """

    pipeline_options = PdfPipelineOptions(
        do_ocr=False,
        do_table_structure=True
    )

    pipeline_options.table_structure_options.mode = (
        TableFormerMode.ACCURATE
    )

    # Use TableFormer's predicted cells instead of mapping
    # everything back to the PDF text cells. This can reduce
    # merging of nearby columns in complex tables.
    pipeline_options.table_structure_options.do_cell_matching = (
        False
    )

    return DocumentConverter(
        format_options={
            InputFormat.PDF: PdfFormatOption(
                pipeline_options=pipeline_options
            )
        }
    )


def find_eecrf_section(
    document,
    document_extractor,
    toc_parser,
    section_parser
):
    """
    Finds the EECRF section and its actual physical PDF pages.
    """

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

    for section in sections:

        if section.title == TARGET_TITLE:
            return section

    return None


def get_table_page_number(table) -> int | None:
    """
    Returns the physical source page stored by Docling.
    """

    provenance = getattr(
        table,
        "prov",
        None
    )

    if not provenance:
        return None

    return getattr(
        provenance[0],
        "page_no",
        None
    )


def clean_file_name(value: str) -> str:
    """
    Creates a filesystem-safe CSV filename.
    """

    cleaned = []

    for character in value:

        if character.isalnum():
            cleaned.append(character)

        elif character in {" ", "-", "_"}:
            cleaned.append("_")

    return "".join(cleaned).strip("_")


def main():

    OUTPUT_DIRECTORY.mkdir(
        parents=True,
        exist_ok=True
    )

    loader = PDFLoader()
    document_extractor = DocumentExtractor()
    toc_parser = TOCParser()
    section_parser = SectionParser()
    converter = create_converter()

    documents = loader.load("data")

    print("=" * 110)
    print("DOCLING EECRF TABLE TEST")
    print("=" * 110)

    for document in documents:

        section = find_eecrf_section(
            document=document,
            document_extractor=document_extractor,
            toc_parser=toc_parser,
            section_parser=section_parser
        )

        if section is None:

            print()
            print(
                "EECRF section not found:",
                document.file_name
            )

            continue

        print()
        print("=" * 110)
        print("Source File :", document.file_name)
        print("Section     :", section.title)
        print(
            "Physical Pages:",
            section.start_page,
            "->",
            section.end_page
        )
        print("=" * 110)

        result = converter.convert(
            source=document.file_path,
            page_range=(
                section.start_page,
                section.end_page
            )
        )

        tables = result.document.tables

        print(
            "Docling Tables Found:",
            len(tables)
        )

        if not tables:

            print(
                "No tables were detected by Docling."
            )

            continue

        document_name = clean_file_name(
            Path(document.file_name).stem
        )

        for table_index, table in enumerate(
            tables,
            start=1
        ):

            dataframe = table.export_to_dataframe(
                doc=result.document
            )

            page_number = get_table_page_number(
                table
            )

            print()
            print("-" * 110)
            print(
                f"TABLE {table_index} "
                f"| PAGE {page_number}"
            )
            print("-" * 110)

            print(
                "Rows   :",
                dataframe.shape[0]
            )

            print(
                "Columns:",
                dataframe.shape[1]
            )

            print()
            print(
                dataframe.to_string(
                    index=False
                )
            )

            output_file = (
                OUTPUT_DIRECTORY
                / (
                    f"{document_name}"
                    f"_eecrf_table_{table_index}.csv"
                )
            )

            dataframe.to_csv(
                output_file,
                index=False,
                encoding="utf-8-sig"
            )

            print()
            print(
                "CSV Saved:",
                output_file
            )

    print()
    print("=" * 110)
    print("TEST COMPLETED")
    print("=" * 110)


if __name__ == "__main__":
    main()