import re

from dataclasses import replace
from typing import Any

from src.models.rate import RateItem
from src.models.section import Section


class SectionEffectiveDateResolver:
    """
    Resolves missing rate effective dates using an explicit
    section-level tariff header.

    Only dates appearing in this structure are accepted:

        Effective Date: September 1, 2025

    Dates appearing inside ordinary narrative sentences are
    intentionally ignored.

    Historical matrix records that already contain a row-level
    effective date are never changed.
    """

    MONTH_PATTERN = (
        r"(?:"
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
    )

    EXPLICIT_EFFECTIVE_DATE_PATTERN = re.compile(
        r"\b"
        r"EFFECTIVE\s+DATE"
        r"\s*:\s*"
        r"(?P<date>"
        + MONTH_PATTERN
        + r"\.?\s+"
        r"\d{1,2}"
        r",?\s+"
        r"\d{4}"
        r")",
        re.IGNORECASE
    )

    GENERIC_COLUMN_PATTERN = re.compile(
        r"^(?:COLUMN|COL)\s*\d+$",
        re.IGNORECASE
    )

    EFFECTIVE_DATE_ROW_PATTERN = re.compile(
        r"^\(?\s*EFFECTIVE\s+DATE\s*\)?$",
        re.IGNORECASE
    )

    def resolve(
        self,
        rate_items: list[RateItem],
        sections: list[Section]
    ) -> list[RateItem]:
        """
        Returns copied rate items with safe section-level
        effective dates applied.
        """

        section_dates = (
            self._build_section_date_map(
                sections
            )
        )

        resolved_rates = []

        for rate in rate_items:

            copied_rate = self._copy_rate(
                rate
            )

            if copied_rate.effective_date:

                resolved_rates.append(
                    copied_rate
                )

                continue

            section_key = self._create_section_key(
                source_file=(
                    copied_rate.source_file
                ),
                section_id=(
                    copied_rate.schedule_id
                )
            )

            section_date = section_dates.get(
                section_key,
                ""
            )

            if not section_date:

                resolved_rates.append(
                    copied_rate
                )

                continue

            if self._is_structural_matrix_header(
                copied_rate
            ):

                copied_rate.metadata[
                    "section_effective_date_candidate"
                ] = section_date

                copied_rate.metadata[
                    "section_effective_date_resolution_skipped"
                ] = True

                copied_rate.metadata[
                    "section_effective_date_skip_reason"
                ] = "STRUCTURAL_MATRIX_HEADER"

                resolved_rates.append(
                    copied_rate
                )

                continue

            metadata = dict(
                copied_rate.metadata
            )

            metadata[
                "effective_date_resolved"
            ] = True

            metadata[
                "effective_date_resolution"
            ] = "SECTION_HEADER"

            metadata[
                "effective_date_original"
            ] = copied_rate.effective_date

            metadata[
                "section_effective_date"
            ] = section_date

            metadata[
                "section_effective_date_source"
            ] = "EXPLICIT_HEADER"

            resolved_rates.append(
                replace(
                    copied_rate,
                    effective_date=section_date,
                    metadata=metadata
                )
            )

        return resolved_rates

    def extract_section_effective_date(
        self,
        section: Section
    ) -> str:
        """
        Returns one explicit effective date when all matching
        section headers agree.

        Returns an empty string when:

        - No explicit header exists
        - Multiple conflicting header dates exist
        """

        matches = []

        for match in (
            self.EXPLICIT_EFFECTIVE_DATE_PATTERN
            .finditer(
                section.text
            )
        ):

            date_value = self._clean_date(
                match.group(
                    "date"
                )
            )

            if not date_value:
                continue

            normalized_date = (
                self._normalize_text(
                    date_value
                )
            )

            if any(
                self._normalize_text(
                    existing_date
                )
                == normalized_date
                for existing_date in matches
            ):
                continue

            matches.append(
                date_value
            )

        if len(matches) != 1:
            return ""

        return matches[0]

    def _build_section_date_map(
        self,
        sections: list[Section]
    ) -> dict[tuple[str, str], str]:

        section_dates = {}

        for section in sections:

            effective_date = (
                self.extract_section_effective_date(
                    section
                )
            )

            if not effective_date:
                continue

            section_key = (
                self._create_section_key(
                    source_file=(
                        section.source_file
                    ),
                    section_id=(
                        section.section_id
                    )
                )
            )

            section_dates[
                section_key
            ] = effective_date

        return section_dates

    def _is_structural_matrix_header(
        self,
        rate: RateItem
    ) -> bool:
        """
        Detects high-confidence matrix header artifacts.

        Example:

            Charge      : Column 6
            Row Label   : Effective Date
            Column Header: ($/kWh)

        Such a record represents table structure rather than
        an actual tariff rate.
        """

        table_structure = (
            self._normalize_text(
                rate.metadata.get(
                    "table_structure",
                    ""
                )
            )
        )

        if table_structure != "MATRIX":
            return False

        row_label = self._clean_text(
            rate.attributes.get(
                "row_label",
                ""
            )
        )

        charge_name = self._clean_text(
            rate.charge_name
        )

        if (
            self.EFFECTIVE_DATE_ROW_PATTERN
            .fullmatch(
                row_label
            )
        ):
            return True

        if (
            self.GENERIC_COLUMN_PATTERN
            .fullmatch(
                charge_name
            )
            and self._looks_like_unit_header(
                rate.attributes.get(
                    "column_header",
                    ""
                )
            )
        ):
            return True

        return False

    def _looks_like_unit_header(
        self,
        value: Any
    ) -> bool:

        normalized = self._normalize_text(
            value
        )

        if not normalized:
            return False

        unit_indicators = (
            "$/KWH",
            "$/KW",
            "$/BILLING KW",
            "$/NCP KW",
            "$/4CP KW",
            "PERCENT",
            "%"
        )

        return any(
            indicator in normalized
            for indicator in unit_indicators
        )

    def _create_section_key(
        self,
        source_file: str,
        section_id: str
    ) -> tuple[str, str]:

        return (
            self._normalize_text(
                source_file
            ),
            self._normalize_text(
                section_id
            )
        )

    def _copy_rate(
        self,
        rate: RateItem
    ) -> RateItem:

        return replace(
            rate,
            attributes=dict(
                rate.attributes
            ),
            metadata=dict(
                rate.metadata
            )
        )

    def _clean_date(
        self,
        value: Any
    ) -> str:

        text = self._clean_text(
            value
        )

        text = re.sub(
            r"\s*,\s*",
            ", ",
            text
        )

        return text

    def _clean_text(
        self,
        value: Any
    ) -> str:

        if value is None:
            return ""

        text = str(value)

        text = text.replace(
            "\u00a0",
            " "
        )

        text = text.replace(
            "\u2013",
            "-"
        )

        text = text.replace(
            "\u2014",
            "-"
        )

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

    def _normalize_text(
        self,
        value: Any
    ) -> str:

        return self._clean_text(
            value
        ).upper()