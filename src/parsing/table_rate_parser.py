import re
from typing import Optional

from src.models.rate import RateItem
from src.models.table import ExtractedTable


class TableRateParser:
    """
    Converts generic Docling table rows into RateItem objects.

    It supports two broad table structures:

    1. Row-based charge tables

       Customer Charge | $1.43 | per Retail Customer

    2. Matrix tables

       Effective Date | Residential | Secondary | Primary
       March 1, 2025  | 0.001137    | 0.000223 | 0.000014

    No schedule name, rider name, section ID or fixed table
    dimension is required.
    """

    DATE_PATTERN = re.compile(
        r"\b("
        r"JAN(?:UARY)?|"
        r"FEB(?:RUARY)?|"
        r"MAR(?:CH)?|"
        r"APR(?:IL)?|"
        r"MAY|"
        r"JUN(?:E)?|"
        r"JUL(?:Y)?|"
        r"AUG(?:UST)?|"
        r"SEP(?:T|TEMBER)?|"
        r"OCT(?:OBER)?|"
        r"NOV(?:EMBER)?|"
        r"DEC(?:EMBER)?"
        r")"
        r"\.?\s+"
        r"\d{1,2}"
        r"(?:,|\s)"
        r"\s*\d{4}\b",
        re.IGNORECASE
    )

    NUMERIC_PATTERN = re.compile(
        r"^"
        r"\(?"
        r"\s*"
        r"[+-]?"
        r"\d{1,3}"
        r"(?:,\d{3})*"
        r"(?:\.\d+)?"
        r"\s*"
        r"\)?"
        r"$"
    )

    UNIT_PATTERN = re.compile(
        r"\("
        r"\s*"
        r"\$?"
        r"\s*/"
        r"\s*"
        r"[^)]+"
        r"\)",
        re.IGNORECASE
    )

    def parse(
        self,
        tables: list[ExtractedTable]
    ) -> list[RateItem]:
        """
        Parses every supplied table.
        """

        rate_items = []

        for table in tables:

            table_rate_items = self.parse_table(
                table
            )

            rate_items.extend(
                table_rate_items
            )

        return rate_items

    def parse_table(
        self,
        table: ExtractedTable
    ) -> list[RateItem]:
        """
        Automatically determines whether a table is row-based
        or matrix-based.
        """

        if table.is_empty:
            return []

        if self._is_matrix_table(table):
            return self._parse_matrix_table(
                table
            )

        return self._parse_row_table(
            table
        )

    def _is_matrix_table(
        self,
        table: ExtractedTable
    ) -> bool:
        """
        A matrix table usually has:

        - A descriptive header row
        - Multiple rate values across later rows
        """

        if table.row_count < 2:
            return False

        header_row = table.rows[0]

        non_empty_headers = sum(
            1
            for cell in header_row
            if self._clean_text(cell)
        )

        if non_empty_headers < 2:
            return False

        maximum_rate_values = 0

        for row in table.rows[1:]:

            rate_value_count = 0

            for column_index, cell in enumerate(row):

                header = self._get_cell(
                    header_row,
                    column_index
                )

                if self._is_rate_value(
                    value=cell,
                    header=header,
                    allow_reference=False
                ):
                    rate_value_count += 1

            maximum_rate_values = max(
                maximum_rate_values,
                rate_value_count
            )

        return maximum_rate_values >= 2

    def _parse_row_table(
        self,
        table: ExtractedTable
    ) -> list[RateItem]:
        """
        Parses tables where each row represents one charge.
        """

        rate_items = []
        pending_charge_name = ""

        for row_index, original_row in enumerate(
            table.rows,
            start=1
        ):

            row = self._remove_duplicate_cells(
                original_row
            )

            if not row:
                continue

            value_index = self._find_row_value_index(
                row
            )

            if value_index is None:

                possible_heading = self._build_label(
                    row
                )

                if self._looks_like_charge_heading(
                    possible_heading
                ):
                    pending_charge_name = (
                        possible_heading
                    )

                elif self._looks_like_section_heading(
                    possible_heading
                ):
                    pending_charge_name = ""

                continue

            value_text = row[value_index]

            label_cells = row[
                :value_index
            ]

            unit_cells = row[
                value_index + 1:
            ]

            charge_name = self._build_label(
                label_cells
            )

            unit = self._build_unit(
                unit_cells
            )

            if (
                pending_charge_name
                and not self._looks_like_explicit_charge(
                    charge_name
                )
            ):

                if charge_name:

                    charge_name = (
                        f"{pending_charge_name} - "
                        f"{charge_name}"
                    )

                else:
                    charge_name = pending_charge_name

            elif self._looks_like_explicit_charge(
                charge_name
            ):
                pending_charge_name = ""

            if not charge_name:
                continue

            rate_item = RateItem(
                schedule_id=table.schedule_id,
                schedule_title=table.schedule_title,
                category=table.category,
                source_file=table.source_file,
                charge_name=charge_name,
                value_text=value_text,
                unit=unit,
                source_method=table.extraction_method,
                page_number=table.page_number,
                table_index=table.table_index,
                row_index=row_index,
                metadata={
                    "table_structure": "ROW",
                    "original_row": original_row
                }
            )

            rate_items.append(
                rate_item
            )

        return rate_items

    def _parse_matrix_table(
        self,
        table: ExtractedTable
    ) -> list[RateItem]:
        """
        Parses tables where one row contains values for
        several rate columns.
        """

        header_row = table.rows[0]
        rate_items = []

        for row_index, row in enumerate(
            table.rows[1:],
            start=2
        ):

            if not row:
                continue

            first_cell = self._get_cell(
                row,
                0
            )

            effective_date = ""

            row_label = ""

            if self._looks_like_date(
                first_cell
            ):
                effective_date = first_cell

            elif first_cell:
                row_label = first_cell

            for column_index in range(
                1,
                max(
                    len(header_row),
                    len(row)
                )
            ):

                value_text = self._get_cell(
                    row,
                    column_index
                )

                header_text = self._get_cell(
                    header_row,
                    column_index
                )

                if not self._is_rate_value(
                    value=value_text,
                    header=header_text,
                    allow_reference=True
                ):
                    continue

                charge_name, unit = (
                    self._split_header(
                        header_text
                    )
                )

                if not charge_name:

                    charge_name = (
                        f"Column {column_index + 1}"
                    )

                attributes = {}

                if row_label:
                    attributes["row_label"] = (
                        row_label
                    )

                attributes["column_header"] = (
                    header_text
                )

                rate_item = RateItem(
                    schedule_id=table.schedule_id,
                    schedule_title=table.schedule_title,
                    category=table.category,
                    source_file=table.source_file,
                    charge_name=charge_name,
                    value_text=value_text,
                    unit=unit,
                    source_method=table.extraction_method,
                    page_number=table.page_number,
                    table_index=table.table_index,
                    row_index=row_index,
                    effective_date=effective_date,
                    attributes=attributes,
                    metadata={
                        "table_structure": "MATRIX",
                        "column_index": column_index,
                        "original_row": row
                    }
                )

                rate_items.append(
                    rate_item
                )

        return rate_items

    def _find_row_value_index(
        self,
        row: list[str]
    ) -> Optional[int]:

        for index, cell in enumerate(row):

            if self._is_reference_value(cell):
                return index

            if self._contains_currency(cell):
                return index

            if self._contains_percentage(cell):
                return index

            if self._is_parenthesized_number(cell):
                return index

            if self._is_table_reference(cell):
                return index

        return None

    def _is_rate_value(
        self,
        value: str,
        header: str,
        allow_reference: bool
    ) -> bool:

        value = self._clean_text(
            value
        )

        header = self._clean_text(
            header
        )

        if not value:
            return False

        if allow_reference:

            if self._is_reference_value(value):
                return True

            if self._is_table_reference(value):
                return True

        if self._contains_currency(value):
            return True

        if self._contains_percentage(value):
            return True

        if self._is_parenthesized_number(value):
            return True

        if (
            self._is_plain_number(value)
            and self._is_rate_header(header)
        ):
            return True

        return False

    def _is_rate_header(
        self,
        header: str
    ) -> bool:

        normalized_header = (
            header.upper()
        )

        indicators = (
            "$",
            "%",
            "RATE",
            "CHARGE",
            "CREDIT",
            "FACTOR",
            "PRICE",
            "COST"
        )

        return any(
            indicator in normalized_header
            for indicator in indicators
        )

    def _contains_currency(
        self,
        value: str
    ) -> bool:

        value = self._clean_text(
            value
        )

        return bool(
            re.match(
                r"^\(?\s*\$\s*"
                r"[+-]?"
                r"\d"
                r"[\d,]*"
                r"(?:\.\d+)?"
                r"\s*\)?"
                r"(?:\s+.*)?$",
                value
            )
        )

    def _contains_percentage(
        self,
        value: str
    ) -> bool:

        value = self._clean_text(
            value
        )

        return bool(
            re.match(
                r"^\(?\s*"
                r"[+-]?"
                r"\d"
                r"[\d,]*"
                r"(?:\.\d+)?"
                r"\s*%"
                r"\s*\)?$",
                value
            )
        )

    def _is_parenthesized_number(
        self,
        value: str
    ) -> bool:

        value = self._clean_text(
            value
        )

        return bool(
            re.match(
                r"^\(\s*"
                r"\$?"
                r"\s*"
                r"\d"
                r"[\d,]*"
                r"(?:\.\d+)?"
                r"\s*"
                r"%?"
                r"\s*\)$",
                value
            )
        )

    def _is_plain_number(
        self,
        value: str
    ) -> bool:

        value = self._clean_text(
            value
        )

        return bool(
            self.NUMERIC_PATTERN.match(
                value
            )
        )

    def _is_reference_value(
        self,
        value: str
    ) -> bool:

        return bool(
            re.match(
                r"^SEE\s+RIDER\b",
                self._clean_text(value),
                re.IGNORECASE
            )
        )

    def _is_table_reference(
        self,
        value: str
    ) -> bool:

        normalized = (
            self._clean_text(value)
            .upper()
        )

        return normalized in {
            "SEE TABLE",
            "SEE TABLE BELOW"
        }

    def _split_header(
        self,
        header: str
    ) -> tuple[str, str]:

        header = self._clean_text(
            header
        )

        unit_matches = list(
            self.UNIT_PATTERN.finditer(
                header
            )
        )

        unit = ""

        if unit_matches:
            unit = unit_matches[-1].group(0)

        charge_name = self.UNIT_PATTERN.sub(
            " ",
            header
        )

        charge_name = re.sub(
            r"\s*\.\s*",
            " ",
            charge_name
        )

        charge_name = re.sub(
            r"\s+",
            " ",
            charge_name
        )

        return (
            charge_name.strip(" .:-"),
            unit.strip()
        )

    def _build_label(
        self,
        cells: list[str]
    ) -> str:

        cleaned_cells = []

        for cell in cells:

            cleaned = self._clean_text(
                cell
            )

            if not cleaned:
                continue

            if (
                cleaned_cells
                and cleaned.upper()
                == cleaned_cells[-1].upper()
            ):
                continue

            cleaned_cells.append(
                cleaned
            )

        return " - ".join(
            cleaned_cells
        )

    def _build_unit(
        self,
        cells: list[str]
    ) -> str:

        cleaned_cells = []

        for cell in cells:

            cleaned = self._clean_text(
                cell
            )

            if not cleaned:
                continue

            if (
                cleaned_cells
                and cleaned.upper()
                == cleaned_cells[-1].upper()
            ):
                continue

            cleaned_cells.append(
                cleaned
            )

        return " ".join(
            cleaned_cells
        )

    def _remove_duplicate_cells(
        self,
        row: list[str]
    ) -> list[str]:

        cleaned_row = []

        for cell in row:

            cleaned = self._clean_text(
                cell
            )

            if not cleaned:
                continue

            if (
                cleaned_row
                and cleaned.upper()
                == cleaned_row[-1].upper()
            ):
                continue

            cleaned_row.append(
                cleaned
            )

        return cleaned_row

    def _looks_like_date(
        self,
        value: str
    ) -> bool:

        return bool(
            self.DATE_PATTERN.search(
                self._clean_text(value)
            )
        )

    def _looks_like_charge_heading(
        self,
        value: str
    ) -> bool:

        normalized = value.upper()

        indicators = (
            "CHARGE",
            "FACTOR",
            "CREDIT",
            "SURCHARGE",
            "REFUND"
        )

        return any(
            indicator in normalized
            for indicator in indicators
        )

    def _looks_like_explicit_charge(
        self,
        value: str
    ) -> bool:

        return self._looks_like_charge_heading(
            value
        )

    def _looks_like_section_heading(
        self,
        value: str
    ) -> bool:

        normalized = value.upper()

        return normalized in {
            "OTHER CHARGES",
            "OTHER CHARGES OR CREDITS",
            "OTHER CHARGES AND CREDITS"
        }

    def _get_cell(
        self,
        row: list[str],
        index: int
    ) -> str:

        if index < 0:
            return ""

        if index >= len(row):
            return ""

        return self._clean_text(
            row[index]
        )

    def _clean_text(
        self,
        value
    ) -> str:

        if value is None:
            return ""

        text = str(value)

        text = text.replace(
            "\n",
            " "
        )

        text = text.replace(
            "\r",
            " "
        )

        text = re.sub(
            r"\s+",
            " ",
            text
        )

        return text.strip()