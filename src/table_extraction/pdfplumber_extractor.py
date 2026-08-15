from pathlib import Path

import pdfplumber

from src.models.section import Section
from src.models.table import ExtractedTable


class PDFPlumberExtractor:
    """
    Extracts structured tables from tariff PDF pages using pdfplumber.

    The extractor first uses pdfplumber's default table detection.
    When no table is found, it tries text-alignment detection.
    """

    def extract(
        self,
        pdf_path: str,
        sections: list[Section]
    ) -> list[ExtractedTable]:

        extracted_tables = []

        for section in sections:

            section_tables = self.extract_section(
                pdf_path=pdf_path,
                section=section
            )

            extracted_tables.extend(
                section_tables
            )

        return extracted_tables

    def extract_section(
        self,
        pdf_path: str,
        section: Section
    ) -> list[ExtractedTable]:

        path = Path(pdf_path)

        if not path.exists():
            raise FileNotFoundError(
                f"PDF file not found: {path}"
            )

        tables = []

        with pdfplumber.open(path) as pdf:

            start_page = max(
                1,
                section.start_page
            )

            end_page = min(
                section.end_page,
                len(pdf.pages)
            )

            for page_number in range(
                start_page,
                end_page + 1
            ):

                pdf_page = pdf.pages[
                    page_number - 1
                ]

                page_tables, method = (
                    self._extract_page_tables(
                        pdf_page
                    )
                )

                for table_index, rows in enumerate(
                    page_tables,
                    start=1
                ):

                    if not self._is_valid_table(rows):
                        continue

                    table = ExtractedTable(
                        schedule_id=section.section_id,
                        schedule_title=section.title,
                        category=section.category,
                        source_file=section.source_file,
                        page_number=page_number,
                        table_index=table_index,
                        extraction_method=method,
                        rows=rows,
                        metadata={
                            "page_width": float(
                                pdf_page.width
                            ),
                            "page_height": float(
                                pdf_page.height
                            )
                        }
                    )

                    if table.is_empty:
                        continue

                    tables.append(table)

        return tables

    def _extract_page_tables(
        self,
        pdf_page
    ) -> tuple[list[list[list[str]]], str]:

        default_tables = (
            pdf_page.extract_tables()
            or []
        )

        valid_default_tables = [
            table
            for table in default_tables
            if self._is_valid_table(table)
        ]

        if valid_default_tables:

            return (
                valid_default_tables,
                "PDFPLUMBER_DEFAULT"
            )

        text_settings = {
            "vertical_strategy": "text",
            "horizontal_strategy": "text",
            "min_words_vertical": 2,
            "min_words_horizontal": 1,
            "text_x_tolerance": 3,
            "text_y_tolerance": 3,
            "intersection_x_tolerance": 5,
            "intersection_y_tolerance": 5
        }

        text_tables = (
            pdf_page.extract_tables(
                table_settings=text_settings
            )
            or []
        )

        valid_text_tables = [
            table
            for table in text_tables
            if self._is_valid_table(table)
        ]

        return (
            valid_text_tables,
            "PDFPLUMBER_TEXT"
        )

    def _is_valid_table(
        self,
        rows
    ) -> bool:

        if not rows:
            return False

        non_empty_cells = 0

        for row in rows:

            if row is None:
                continue

            for cell in row:

                if cell is None:
                    continue

                if str(cell).strip():
                    non_empty_cells += 1

        return non_empty_cells >= 2