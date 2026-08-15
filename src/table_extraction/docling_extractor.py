from pathlib import Path

import pandas as pd

from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import (
    PdfPipelineOptions,
    TableFormerMode
)
from docling.document_converter import (
    DocumentConverter,
    PdfFormatOption
)

from src.models.section import Section
from src.models.table import ExtractedTable


class DoclingTableExtractor:
    """
    Automatically extracts tables from every supplied tariff section.

    Each section is processed separately in small page batches.
    This prevents memory failures when a tariff contains many pages.

    No schedule titles, rider names, section IDs or table dimensions
    are hard-coded.
    """

    def __init__(
        self,
        max_pages_per_batch: int = 3
    ) -> None:

        if max_pages_per_batch < 1:
            raise ValueError(
                "max_pages_per_batch must be at least 1."
            )

        self.max_pages_per_batch = (
            max_pages_per_batch
        )

        self.converter = self._create_converter()

    def extract(
        self,
        pdf_path: str,
        sections: list[Section]
    ) -> list[ExtractedTable]:
        """
        Extracts tables from all supplied sections.

        Every section is processed independently so that failure in
        one section does not prevent later sections from being read.
        """

        path = self._validate_pdf_path(
            pdf_path
        )

        if not sections:
            return []

        sorted_sections = sorted(
            sections,
            key=lambda section: (
                section.start_page,
                section.end_page,
                section.section_id
            )
        )

        extracted_tables = []

        for section in sorted_sections:

            section_tables = (
                self._extract_section_batches(
                    pdf_path=path,
                    section=section
                )
            )

            extracted_tables.extend(
                section_tables
            )

        extracted_tables.sort(
            key=lambda table: (
                table.page_number,
                table.schedule_id,
                table.table_index
            )
        )

        return extracted_tables

    def extract_section(
        self,
        pdf_path: str,
        section: Section
    ) -> list[ExtractedTable]:
        """
        Extracts all tables from one section.
        """

        path = self._validate_pdf_path(
            pdf_path
        )

        return self._extract_section_batches(
            pdf_path=path,
            section=section
        )

    def _extract_section_batches(
        self,
        pdf_path: Path,
        section: Section
    ) -> list[ExtractedTable]:

        print()
        print(
            "Scanning section:",
            section.section_id,
            "|",
            section.title
        )

        print(
            "Section pages:",
            section.start_page,
            "->",
            section.end_page
        )

        batches = self._build_section_batches(
            start_page=section.start_page,
            end_page=section.end_page
        )

        extracted_tables = []
        table_counter = 0

        for batch_start, batch_end in batches:

            print(
                "Docling batch:",
                batch_start,
                "->",
                batch_end
            )

            try:

                result = self.converter.convert(
                    source=str(pdf_path),
                    page_range=(
                        batch_start,
                        batch_end
                    )
                )

            except Exception as error:

                print(
                    "Docling batch failed:",
                    batch_start,
                    "->",
                    batch_end,
                    "|",
                    type(error).__name__,
                    str(error)
                )

                continue

            docling_tables = (
                result.document.tables
                or []
            )

            for docling_table in docling_tables:

                page_number = self._get_page_number(
                    docling_table
                )

                if page_number is None:
                    page_number = batch_start

                if not (
                    section.start_page
                    <= page_number
                    <= section.end_page
                ):
                    continue

                try:

                    dataframe = (
                        docling_table.export_to_dataframe(
                            doc=result.document
                        )
                    )

                except Exception as error:

                    print(
                        "Table export failed on page",
                        page_number,
                        "|",
                        type(error).__name__,
                        str(error)
                    )

                    continue

                if not self._is_meaningful_dataframe(
                    dataframe
                ):
                    continue

                rows = self._dataframe_to_rows(
                    dataframe
                )

                if not rows:
                    continue

                table_counter += 1

                table = ExtractedTable(
                    schedule_id=section.section_id,
                    schedule_title=section.title,
                    category=section.category,
                    source_file=section.source_file,
                    page_number=page_number,
                    table_index=table_counter,
                    extraction_method=(
                        "DOCLING_ACCURATE"
                    ),
                    rows=rows,
                    metadata={
                        "section_start_page": (
                            section.start_page
                        ),
                        "section_end_page": (
                            section.end_page
                        ),
                        "batch_start_page": (
                            batch_start
                        ),
                        "batch_end_page": (
                            batch_end
                        ),
                        "dataframe_row_count": (
                            int(dataframe.shape[0])
                        ),
                        "dataframe_column_count": (
                            int(dataframe.shape[1])
                        ),
                        "automatic_detection": True
                    }
                )

                if table.is_empty:
                    continue

                extracted_tables.append(
                    table
                )

        return extracted_tables

    def _build_section_batches(
        self,
        start_page: int,
        end_page: int
    ) -> list[tuple[int, int]]:
        """
        Divides a section into small page batches.

        Example with batch size 3:

        82 -> 90 becomes:
        82 -> 84
        85 -> 87
        88 -> 90
        """

        batches = []

        current_page = start_page

        while current_page <= end_page:

            batch_end = min(
                current_page
                + self.max_pages_per_batch
                - 1,
                end_page
            )

            batches.append(
                (
                    current_page,
                    batch_end
                )
            )

            current_page = batch_end + 1

        return batches

    def _create_converter(
        self
    ) -> DocumentConverter:

        pipeline_options = PdfPipelineOptions(
            do_ocr=False,
            do_table_structure=True
        )

        pipeline_options.table_structure_options.mode = (
            TableFormerMode.ACCURATE
        )

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

    def _validate_pdf_path(
        self,
        pdf_path: str
    ) -> Path:

        path = Path(pdf_path)

        if not path.exists():
            raise FileNotFoundError(
                f"PDF file not found: {path}"
            )

        if not path.is_file():
            raise ValueError(
                f"PDF path is not a file: {path}"
            )

        return path

    def _is_meaningful_dataframe(
        self,
        dataframe: pd.DataFrame | None
    ) -> bool:

        if dataframe is None:
            return False

        if dataframe.empty:
            return False

        row_count = int(
            dataframe.shape[0]
        )

        column_count = int(
            dataframe.shape[1]
        )

        if row_count < 1:
            return False

        if column_count < 2:
            return False

        non_empty_cells = 0

        for column in dataframe.columns:

            if self._clean_value(column):
                non_empty_cells += 1

        for row in dataframe.itertuples(
            index=False,
            name=None
        ):

            for value in row:

                if self._clean_value(value):
                    non_empty_cells += 1

        return non_empty_cells >= 4

    def _dataframe_to_rows(
        self,
        dataframe: pd.DataFrame
    ) -> list[list[str]]:

        rows = []

        header_row = [
            self._clean_value(column)
            for column in dataframe.columns
        ]

        if (
            any(header_row)
            and not self._is_generated_header(
                header_row
            )
        ):
            rows.append(
                header_row
            )

        for dataframe_row in dataframe.itertuples(
            index=False,
            name=None
        ):

            cleaned_row = [
                self._clean_value(value)
                for value in dataframe_row
            ]

            if any(cleaned_row):
                rows.append(
                    cleaned_row
                )

        return rows

    def _is_generated_header(
        self,
        header_row: list[str]
    ) -> bool:
        """
        Detects automatically generated headers such as:

        ['0', '1', '2']
        ['0', '1', '2', '3']

        These are not real tariff table headers.
        """

        if not header_row:
            return False

        expected_headers = [
            str(index)
            for index in range(
                len(header_row)
            )
        ]

        return header_row == expected_headers

    def _clean_value(
        self,
        value
    ) -> str:

        if value is None:
            return ""

        try:

            if pd.isna(value):
                return ""

        except (TypeError, ValueError):
            pass

        text = str(value)

        text = text.replace(
            "\n",
            " "
        )

        text = text.replace(
            "\r",
            " "
        )

        text = " ".join(
            text.split()
        )

        return text.strip()

    def _get_page_number(
        self,
        docling_table
    ) -> int | None:

        provenance = getattr(
            docling_table,
            "prov",
            None
        )

        if not provenance:
            return None

        page_number = getattr(
            provenance[0],
            "page_no",
            None
        )

        if page_number is None:
            return None

        return int(page_number)