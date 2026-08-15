from dataclasses import dataclass, field
from typing import Any


@dataclass(slots=True)
class ExtractedTable:
    """
    Represents one table extracted from a tariff PDF page.
    """

    schedule_id: str
    schedule_title: str
    category: str
    source_file: str
    page_number: int
    table_index: int
    extraction_method: str
    rows: list[list[str]] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:

        self.schedule_id = self.schedule_id.strip()
        self.schedule_title = self.schedule_title.strip()
        self.category = self.category.strip().upper()
        self.source_file = self.source_file.strip()
        self.extraction_method = (
            self.extraction_method.strip().upper()
        )

        if not self.schedule_id:
            raise ValueError(
                "Schedule ID cannot be empty."
            )

        if not self.schedule_title:
            raise ValueError(
                "Schedule title cannot be empty."
            )

        if not self.source_file:
            raise ValueError(
                "Source file cannot be empty."
            )

        if self.page_number < 1:
            raise ValueError(
                "Page number must be greater than or equal to 1."
            )

        if self.table_index < 1:
            raise ValueError(
                "Table index must be greater than or equal to 1."
            )

        self.rows = self._normalize_rows(
            self.rows
        )

    @property
    def row_count(self) -> int:
        """
        Returns the number of rows in the table.
        """

        return len(self.rows)

    @property
    def column_count(self) -> int:
        """
        Returns the maximum number of columns found in any row.
        """

        return max(
            (
                len(row)
                for row in self.rows
            ),
            default=0
        )

    @property
    def is_empty(self) -> bool:
        """
        Returns True when the table has no meaningful cell values.
        """

        if not self.rows:
            return True

        return not any(
            cell.strip()
            for row in self.rows
            for cell in row
        )

    def get_row(
        self,
        row_index: int
    ) -> list[str] | None:
        """
        Returns a row using a zero-based index.
        """

        if row_index < 0:
            return None

        if row_index >= len(self.rows):
            return None

        return self.rows[row_index]

    def get_cell(
        self,
        row_index: int,
        column_index: int
    ) -> str | None:
        """
        Returns one cell using zero-based row and column indexes.
        """

        row = self.get_row(
            row_index
        )

        if row is None:
            return None

        if column_index < 0:
            return None

        if column_index >= len(row):
            return None

        return row[column_index]

    def to_dict(self) -> dict[str, Any]:
        """
        Converts the extracted table into a serializable dictionary.
        """

        return {
            "schedule_id": self.schedule_id,
            "schedule_title": self.schedule_title,
            "category": self.category,
            "source_file": self.source_file,
            "page_number": self.page_number,
            "table_index": self.table_index,
            "extraction_method": self.extraction_method,
            "row_count": self.row_count,
            "column_count": self.column_count,
            "rows": self.rows,
            "metadata": self.metadata
        }

    def _normalize_rows(
        self,
        rows: list[list[Any]]
    ) -> list[list[str]]:
        """
        Converts every table cell into cleaned text.
        """

        normalized_rows = []

        for row in rows:

            if row is None:
                continue

            normalized_row = []

            for cell in row:

                if cell is None:
                    value = ""

                else:
                    value = str(cell)

                    value = value.replace(
                        "\n",
                        " "
                    )

                    value = value.replace(
                        "\r",
                        " "
                    )

                    value = " ".join(
                        value.split()
                    )

                normalized_row.append(
                    value.strip()
                )

            if any(normalized_row):
                normalized_rows.append(
                    normalized_row
                )

        return normalized_rows