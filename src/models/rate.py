import re

from dataclasses import dataclass, field
from decimal import Decimal, InvalidOperation
from typing import Any


@dataclass(slots=True)
class RateItem:
    """
    Represents one tariff charge, credit, factor or rider reference.

    Examples:

    Customer Charge | $1.43 | per Retail Customer

    Distribution System Charge | $0.025344 | per kWh

    Nuclear Decommissioning Charge | See Rider NDC | per kWh
    """

    schedule_id: str
    schedule_title: str
    category: str
    source_file: str
    charge_name: str
    value_text: str
    unit: str
    source_method: str

    page_number: int | None = None
    table_index: int | None = None
    row_index: int | None = None
    effective_date: str = ""

    attributes: dict[str, str] = field(
        default_factory=dict
    )

    metadata: dict[str, Any] = field(
        default_factory=dict
    )

    def __post_init__(self) -> None:

        self.schedule_id = self.schedule_id.strip()

        self.schedule_title = (
            self.schedule_title.strip()
        )

        self.category = (
            self.category.strip().upper()
        )

        self.source_file = self.source_file.strip()

        self.charge_name = self._clean_text(
            self.charge_name
        )

        self.value_text = self._clean_text(
            self.value_text
        )

        self.unit = self._clean_text(
            self.unit
        )

        self.source_method = (
            self.source_method
            .strip()
            .upper()
        )

        self.effective_date = self._clean_text(
            self.effective_date
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

        if not self.charge_name:
            raise ValueError(
                "Charge name cannot be empty."
            )

        if not self.source_method:
            raise ValueError(
                "Source method cannot be empty."
            )

        if (
            self.page_number is not None
            and self.page_number < 1
        ):
            raise ValueError(
                "Page number must be at least 1."
            )

        if (
            self.table_index is not None
            and self.table_index < 1
        ):
            raise ValueError(
                "Table index must be at least 1."
            )

        if (
            self.row_index is not None
            and self.row_index < 1
        ):
            raise ValueError(
                "Row index must be at least 1."
            )

        self.attributes = {
            self._clean_text(key): self._clean_text(value)
            for key, value in self.attributes.items()
            if self._clean_text(key)
        }

    @property
    def normalized_charge_name(self) -> str:
        """
        Produces a stable charge name for comparison.

        Example:

        'II. Nuclear Decommissioning Charge:'
        becomes
        'NUCLEAR DECOMMISSIONING CHARGE'
        """

        name = self.charge_name.upper()

        name = re.sub(
            r"^[IVXLCDM]+\.\s*",
            "",
            name
        )

        name = re.sub(
            r"^\d+\.\s*",
            "",
            name
        )

        name = name.replace(
            "\u2013",
            "-"
        )

        name = name.replace(
            "\u2014",
            "-"
        )

        name = re.sub(
            r"\s+",
            " ",
            name
        )

        return name.strip(
            " .:-"
        )

    @property
    def normalized_unit(self) -> str:
        """
        Produces a stable unit for comparison.
        """

        unit = self.unit.upper()

        unit = unit.replace(
            "\u2013",
            "-"
        )

        unit = unit.replace(
            "\u2014",
            "-"
        )

        unit = re.sub(
            r"\s+",
            " ",
            unit
        )

        return unit.strip(
            " .:"
        )

    @property
    def numeric_value(self) -> Decimal | None:
        """
        Converts a numeric rate into Decimal.

        Supported examples:

        $1.43
        0.025344
        (0.000196)
        45.88067225%
        $1.57 per month
        """

        text = self.value_text.strip()

        if not text:
            return None

        if self.is_reference:
            return None

        if text.upper() in {
            "N.A.",
            "N.A",
            "NA",
            "N/A",
            "NOT APPLICABLE"
        }:
            return None

        match = re.match(
            r"^"
            r"(?P<open_parenthesis>\()?"
            r"\s*"
            r"\$?"
            r"\s*"
            r"(?P<number>"
            r"[+-]?"
            r"\d{1,3}"
            r"(?:,\d{3})*"
            r"(?:\.\d+)?"
            r"|"
            r"[+-]?\d+"
            r"(?:\.\d+)?"
            r")"
            r"\s*"
            r"%?"
            r"\s*"
            r"(?P<close_parenthesis>\))?"
            r"(?:\s+.*)?"
            r"$",
            text
        )

        if not match:
            return None

        number_text = (
            match.group("number")
            .replace(",", "")
        )

        try:
            number = Decimal(
                number_text
            )

        except InvalidOperation:
            return None

        if (
            match.group("open_parenthesis")
            and match.group("close_parenthesis")
        ):
            number = -abs(number)

        return number

    @property
    def is_reference(self) -> bool:
        """
        Returns True for values such as 'See Rider EECRF'.
        """

        return bool(
            re.match(
                r"^SEE\s+RIDER\b",
                self.value_text,
                re.IGNORECASE
            )
        )

    @property
    def value_kind(self) -> str:
        """
        Classifies the value as NUMERIC, REFERENCE or TEXT.
        """

        if self.is_reference:
            return "REFERENCE"

        if self.numeric_value is not None:
            return "NUMERIC"

        return "TEXT"

    @property
    def comparison_key(self) -> str:
        """
        Returns a stable key for merging table and text results.
        """

        attribute_text = "|".join(
            f"{key.upper()}={value.upper()}"
            for key, value in sorted(
                self.attributes.items()
            )
        )

        parts = [
            self.schedule_title.upper(),
            self.normalized_charge_name,
            self.normalized_unit,
            attribute_text
        ]

        return "||".join(parts)

    def to_dict(self) -> dict[str, Any]:

        numeric_value = self.numeric_value

        return {
            "schedule_id": self.schedule_id,
            "schedule_title": self.schedule_title,
            "category": self.category,
            "source_file": self.source_file,
            "charge_name": self.charge_name,
            "normalized_charge_name": (
                self.normalized_charge_name
            ),
            "value_text": self.value_text,
            "numeric_value": (
                str(numeric_value)
                if numeric_value is not None
                else None
            ),
            "value_kind": self.value_kind,
            "unit": self.unit,
            "normalized_unit": self.normalized_unit,
            "source_method": self.source_method,
            "page_number": self.page_number,
            "table_index": self.table_index,
            "row_index": self.row_index,
            "effective_date": self.effective_date,
            "attributes": self.attributes,
            "metadata": self.metadata,
            "comparison_key": self.comparison_key
        }

    def _clean_text(
        self,
        value: Any
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