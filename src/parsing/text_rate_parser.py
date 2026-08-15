import re
from typing import Optional

from src.models.rate import RateItem
from src.models.section import Section


class TextRateParser:
    """
    Extracts tariff charges directly from section text.

    Supported structures:

    - Currency charges
    - Percentage factors
    - Parenthesized credits
    - Rider references
    - Table references
    - Wrapped values and units
    - Multiple contexts inside one tariff section

    The parser does not depend on fixed schedule titles,
    section IDs or rider names.
    """

    RIDER_REFERENCE_PATTERN = re.compile(
        r"^"
        r"(?P<label>.+?)"
        r"\s+"
        r"(?P<value>"
        r"SEE\s+RIDER\s+"
        r"[A-Z0-9][A-Z0-9_-]*"
        r")"
        r"(?:"
        r"\s+"
        r"(?P<unit>PER\s+.+)"
        r")?"
        r"$",
        re.IGNORECASE
    )

    TABLE_REFERENCE_PATTERN = re.compile(
        r"^"
        r"(?P<label>.+?)"
        r"\s+"
        r"(?P<value>"
        r"SEE\s+TABLE"
        r"(?:\s+BELOW)?"
        r")"
        r"$",
        re.IGNORECASE
    )

    NUMERIC_RATE_PATTERN = re.compile(
        r"^"
        r"(?P<label>.+?)"
        r"\s+"
        r"(?:PLUS\s+)?"
        r"(?P<value>"
        r"\(\s*\$?\s*[+-]?"
        r"\d[\d,]*(?:\.\d+)?"
        r"\s*%?\s*\)"
        r"|"
        r"[+-]?"
        r"\d[\d,]*(?:\.\d+)?"
        r"\s*%"
        r"|"
        r"\$\s*[+-]?"
        r"\d[\d,]*(?:\.\d+)?"
        r")"
        r"(?:"
        r"\s+"
        r"(?P<unit>PER\s+.+)"
        r")?"
        r"$",
        re.IGNORECASE
    )

    EFFECTIVE_DATE_PATTERN = re.compile(
        r"EFFECTIVE\s+DATE\s*:\s*"
        r"(?P<date>"
        r"(?:"
        r"JAN(?:UARY)?|"
        r"FEB(?:RUARY)?|"
        r"MAR(?:CH)?|"
        r"APR(?:IL)?|"
        r"MAY|"
        r"JUN(?:E)?|"
        r"JUL(?:Y)?|"
        r"AUG(?:UST)?|"
        r"SEP(?:TEMBER)?|"
        r"OCT(?:OBER)?|"
        r"NOV(?:EMBER)?|"
        r"DEC(?:EMBER)?"
        r")"
        r"\.?\s+"
        r"\d{1,2},\s+"
        r"\d{4}"
        r")",
        re.IGNORECASE
    )

    VALUE_INDICATOR_PATTERN = re.compile(
        r"("
        r"\$\s*\(?\s*\d"
        r"|"
        r"\d[\d,]*(?:\.\d+)?\s*%"
        r"|"
        r"\bSEE\s+RIDER\b"
        r"|"
        r"\bSEE\s+TABLE\b"
        r")",
        re.IGNORECASE
    )

    DELIVERY_SHEET_PATTERN = re.compile(
        r"^"
        r"(?:\d+(?:\.\d+)+\s+)?"
        r"DELIVERY\s+SYSTEM\s+CHARGES"
        r"\s+SHEET\b",
        re.IGNORECASE
    )

    PAGE_HEADER_PATTERN = re.compile(
        r"^"
        r"(?:"
        r"SHEET\s*:?\s*\d"
        r"|"
        r"PAGE\s+\d+\s+OF\s+\d+"
        r"|"
        r"REVISION\s*:?\s*\w+"
        r")"
        r"$",
        re.IGNORECASE
    )

    def parse(
        self,
        sections: list[Section]
    ) -> list[RateItem]:
        """
        Parses multiple tariff sections.
        """

        rate_items = []

        for section in sections:

            section_rate_items = (
                self.parse_section(
                    section
                )
            )

            rate_items.extend(
                section_rate_items
            )

        return rate_items

    def parse_section(
        self,
        section: Section
    ) -> list[RateItem]:
        """
        Extracts rate items from one tariff section.
        """

        if not section.text.strip():
            return []

        prepared_lines = self._prepare_lines(
            section.text
        )

        effective_date = (
            self._extract_effective_date(
                section.text
            )
        )

        rate_items = []

        current_context = ""
        pending_charge = ""
        pending_line_number = None

        for line_number, line in prepared_lines:

            if self._is_noise_line(
                line
            ):
                continue

            parsed_value = self._parse_value_line(
                line
            )

            if parsed_value is not None:

                label = parsed_value["label"]
                value_text = parsed_value["value"]
                unit = parsed_value["unit"]

                explicit_charge = (
                    self._looks_like_rate_label(
                        label
                    )
                )

                parent_charge = ""

                if (
                    pending_charge
                    and not explicit_charge
                ):

                    parent_charge = pending_charge

                    if label:

                        charge_name = (
                            f"{pending_charge} - "
                            f"{label}"
                        )

                    else:

                        charge_name = (
                            pending_charge
                        )

                else:

                    charge_name = label

                charge_name = (
                    self._clean_charge_name(
                        charge_name
                    )
                )

                if not charge_name:
                    continue

                if self._is_invalid_charge_name(
                    charge_name
                ):
                    continue

                attributes = {}

                if current_context:

                    attributes[
                        "context_heading"
                    ] = current_context

                if parent_charge:

                    attributes[
                        "parent_charge"
                    ] = parent_charge

                rate_item = RateItem(
                    schedule_id=section.section_id,
                    schedule_title=section.title,
                    category=section.category,
                    source_file=section.source_file,
                    charge_name=charge_name,
                    value_text=value_text,
                    unit=unit,
                    source_method="TEXT",
                    effective_date=effective_date,
                    attributes=attributes,
                    metadata={
                        "text_line_number": (
                            line_number
                        ),
                        "original_line": line,
                        "section_start_page": (
                            section.start_page
                        ),
                        "section_end_page": (
                            section.end_page
                        ),
                        "parser_scope": (
                            "SECTION_TEXT"
                        )
                    }
                )

                rate_items.append(
                    rate_item
                )

                if explicit_charge:

                    pending_charge = ""
                    pending_line_number = None

                continue

            if self._is_context_heading(
                line
            ):

                current_context = (
                    self._clean_context_heading(
                        line
                    )
                )

                pending_charge = ""
                pending_line_number = None

                continue

            if self._looks_like_pending_charge(
                line
            ):

                pending_charge = (
                    self._clean_charge_name(
                        line
                    )
                )

                pending_line_number = (
                    line_number
                )

                continue

            if (
                pending_line_number is not None
                and line_number
                > pending_line_number + 3
            ):

                pending_charge = ""
                pending_line_number = None

        return self._remove_duplicates(
            rate_items
        )

    def _parse_value_line(
        self,
        line: str
    ) -> Optional[dict[str, str]]:
        """
        Attempts rider-reference, table-reference and
        numeric-rate patterns in that order.
        """

        match = (
            self.RIDER_REFERENCE_PATTERN.match(
                line
            )
        )

        if match:

            label = self._clean_text(
                match.group("label")
            )

            if self._is_invalid_charge_name(
                label
            ):
                return None

            return {
                "label": label,
                "value": self._clean_text(
                    match.group("value")
                ),
                "unit": self._clean_text(
                    match.group("unit") or ""
                )
            }

        match = (
            self.TABLE_REFERENCE_PATTERN.match(
                line
            )
        )

        if match:

            label = self._clean_text(
                match.group("label")
            )

            if self._is_invalid_charge_name(
                label
            ):
                return None

            return {
                "label": label,
                "value": self._clean_text(
                    match.group("value")
                ),
                "unit": ""
            }

        match = (
            self.NUMERIC_RATE_PATTERN.match(
                line
            )
        )

        if not match:
            return None

        label = self._clean_text(
            match.group("label")
        )

        value_text = self._clean_text(
            match.group("value")
        )

        unit = self._clean_text(
            match.group("unit") or ""
        )

        if not self._looks_like_rate_label(
            label
        ):
            return None

        if self._is_invalid_charge_name(
            label
        ):
            return None

        return {
            "label": label,
            "value": value_text,
            "unit": unit
        }

    def _prepare_lines(
        self,
        text: str
    ) -> list[tuple[int, str]]:
        """
        Cleans and joins predictable wrapped lines.

        Example:

        Distribution System Charge $3.70 per billing
        kW

        becomes:

        Distribution System Charge $3.70 per billing kW
        """

        raw_lines = text.splitlines()

        cleaned_lines = [
            self._clean_text(line)
            for line in raw_lines
        ]

        prepared_lines = []

        index = 0

        while index < len(cleaned_lines):

            line_number = index + 1

            current_line = cleaned_lines[
                index
            ]

            if not current_line:

                index += 1
                continue

            while (
                index + 1
                < len(cleaned_lines)
                and self._needs_continuation(
                    current_line
                )
            ):

                next_line = cleaned_lines[
                    index + 1
                ]

                if not self._is_safe_continuation(
                    next_line
                ):
                    break

                current_line = self._clean_text(
                    f"{current_line} {next_line}"
                )

                index += 1

            prepared_lines.append(
                (
                    line_number,
                    current_line
                )
            )

            index += 1

        return prepared_lines

    def _needs_continuation(
        self,
        line: str
    ) -> bool:

        normalized = (
            line.upper()
            .strip(" .,:;")
        )

        incomplete_endings = (
            " PER",
            " RETAIL",
            " BILLING",
            " AS",
            " RIDER"
        )

        if normalized.endswith(
            incomplete_endings
        ):
            return True

        if re.search(
            r"\bSEE\s+RIDER$",
            normalized
        ):
            return True

        if (
            line.count("(")
            > line.count(")")
        ):
            return True

        return False

    def _is_safe_continuation(
        self,
        line: str
    ) -> bool:

        if not line:
            return False

        if len(line.split()) > 8:
            return False

        if re.fullmatch(
            r"\d+(?:\.\d+)?",
            line
        ):
            return False

        if line.upper().startswith(
            "EFFECTIVE DATE"
        ):
            return False

        if re.match(
            r"^6(?:\.\d+)+",
            line
        ):
            return False

        if self.VALUE_INDICATOR_PATTERN.search(
            line
        ):
            return False

        return True

    def _extract_effective_date(
        self,
        text: str
    ) -> str:

        normalized_text = text.replace(
            "\n",
            " "
        )

        normalized_text = re.sub(
            r"\s+",
            " ",
            normalized_text
        )

        match = (
            self.EFFECTIVE_DATE_PATTERN.search(
                normalized_text
            )
        )

        if not match:
            return ""

        return self._clean_text(
            match.group("date")
        )

    def _looks_like_rate_label(
        self,
        label: str
    ) -> bool:

        normalized = (
            self._clean_text(label)
            .upper()
        )

        indicators = (
            "CHARGE",
            "CREDIT",
            "FACTOR",
            "RATE",
            "SURCHARGE",
            "REFUND",
            "FEE",
            "COST",
            "PRICE",
            "POINT OF DELIVERY",
            "POINTS OF DELIVERY",
            "POD",
            "EXTRA SPAN",
            "EXTRA SPANS"
        )

        return any(
            indicator in normalized
            for indicator in indicators
        )

    def _looks_like_pending_charge(
        self,
        line: str
    ) -> bool:

        normalized = (
            self._clean_text(line)
            .upper()
            .strip(" .:")
        )

        if not normalized:
            return False

        if self._is_invalid_charge_name(
            normalized
        ):
            return False

        if self.VALUE_INDICATOR_PATTERN.search(
            normalized
        ):
            return False

        if normalized in {
            "OTHER CHARGES",
            "OTHER CHARGES OR CREDITS",
            "OTHER CHARGES AND CREDITS",
            "BASE RATE CHARGES",
            "TRANSMISSION AND DISTRIBUTION CHARGES"
        }:
            return False

        singular_indicators = (
            " CHARGE",
            " FACTOR",
            " CREDIT",
            " SURCHARGE",
            " REFUND"
        )

        return normalized.endswith(
            singular_indicators
        )

    def _is_context_heading(
        self,
        line: str
    ) -> bool:

        normalized = self._clean_text(
            line
        )

        if not normalized:
            return False

        if self.VALUE_INDICATOR_PATTERN.search(
            normalized
        ):
            return False

        if self._is_noise_line(
            normalized
        ):
            return False

        uppercase = normalized.upper()

        if uppercase in {
            "OTHER CHARGES",
            "OTHER CHARGES OR CREDITS",
            "OTHER CHARGES AND CREDITS",
            "MONTHLY RATE",
            "COMPANY SPECIFIC APPLICATIONS"
        }:
            return True

        if re.match(
            r"^[IVXLCDM]+\.\s+",
            normalized,
            re.IGNORECASE
        ):
            return True

        letter_count = sum(
            character.isalpha()
            for character in normalized
        )

        uppercase_count = sum(
            character.isupper()
            for character in normalized
        )

        if letter_count == 0:
            return False

        uppercase_ratio = (
            uppercase_count
            / letter_count
        )

        return (
            uppercase_ratio >= 0.90
            and len(normalized.split()) <= 12
        )

    def _is_invalid_charge_name(
        self,
        charge_name: str
    ) -> bool:
        """
        Rejects page headers and structural metadata that
        contain words such as 'Charges' but are not actual rates.
        """

        normalized = self._clean_text(
            charge_name
        )

        if not normalized:
            return True

        if self.DELIVERY_SHEET_PATTERN.match(
            normalized
        ):
            return True

        if self.PAGE_HEADER_PATTERN.match(
            normalized
        ):
            return True

        uppercase = normalized.upper()

        invalid_phrases = (
            "DELIVERY SYSTEM CHARGES SHEET",
            "APPLICABLE: ENTIRE CERTIFIED",
            "TARIFF FOR RETAIL DELIVERY SERVICE",
            "EFFECTIVE DATE:",
            "REVISION:"
        )

        if any(
            phrase in uppercase
            for phrase in invalid_phrases
        ):
            return True

        return False

    def _clean_context_heading(
        self,
        line: str
    ) -> str:

        cleaned = self._clean_text(
            line
        )

        return cleaned.strip(
            " .:"
        )

    def _clean_charge_name(
        self,
        charge_name: str
    ) -> str:

        charge_name = self._clean_text(
            charge_name
        )

        charge_name = re.sub(
            r"\s*:\s*$",
            "",
            charge_name
        )

        return charge_name.strip(
            " ."
        )

    def _is_noise_line(
        self,
        line: str
    ) -> bool:

        normalized = self._clean_text(
            line
        )

        if not normalized:
            return True

        if re.fullmatch(
            r"\d+(?:\.\d+)?",
            normalized
        ):
            return True

        if self.DELIVERY_SHEET_PATTERN.match(
            normalized
        ):
            return True

        if self.PAGE_HEADER_PATTERN.match(
            normalized
        ):
            return True

        uppercase = normalized.upper()

        if uppercase.startswith(
            "TARIFF FOR RETAIL DELIVERY SERVICE"
        ):
            return True

        if uppercase.startswith(
            "ONCOR ELECTRIC DELIVERY"
        ):
            return True

        return False

    def _remove_duplicates(
        self,
        rate_items: list[RateItem]
    ) -> list[RateItem]:

        unique_items = []
        seen_keys = set()

        for rate_item in rate_items:

            duplicate_key = (
                rate_item.comparison_key,
                rate_item.value_text.upper(),
                rate_item.effective_date.upper()
            )

            if duplicate_key in seen_keys:
                continue

            seen_keys.add(
                duplicate_key
            )

            unique_items.append(
                rate_item
            )

        return unique_items

    def _clean_text(
        self,
        value
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