import re

from dataclasses import dataclass
from enum import StrEnum
from typing import Any

from src.models.section import Section


class SectionApplicability(StrEnum):
    """
    Applicability status of a tariff section.
    """

    APPLICABLE = "APPLICABLE"
    NOT_APPLICABLE = "NOT_APPLICABLE"
    UNKNOWN = "UNKNOWN"


@dataclass(slots=True)
class SectionApplicabilityResult:
    """
    Stores the applicability classification for one section.
    """

    section_id: str
    section_title: str
    source_file: str
    category: str
    status: SectionApplicability
    reason: str
    matched_text: str = ""
    substantive_lines: list[str] | None = None

    def __post_init__(self) -> None:

        if self.substantive_lines is None:
            self.substantive_lines = []

    def to_dict(self) -> dict[str, Any]:

        return {
            "section_id": self.section_id,
            "section_title": self.section_title,
            "source_file": self.source_file,
            "category": self.category,
            "status": self.status.value,
            "reason": self.reason,
            "matched_text": self.matched_text,
            "substantive_lines": list(
                self.substantive_lines
            )
        }


class SectionApplicabilityParser:
    """
    Determines whether a tariff section is applicable.

    A section is classified as NOT_APPLICABLE only when its
    substantive content consists entirely of explicit
    non-applicability markers.

    This prevents ordinary legal sentences such as:

        "This provision is not applicable to some customers"

    from incorrectly marking the entire section as unavailable.
    """

    NOT_APPLICABLE_PATTERNS = (
        re.compile(
            r"^NOT\s+APPLICABLE[.!]?$",
            re.IGNORECASE
        ),
        re.compile(
            r"^NOT\s+CURRENTLY\s+APPLICABLE[.!]?$",
            re.IGNORECASE
        ),
        re.compile(
            r"^NO\s+LONGER\s+APPLICABLE[.!]?$",
            re.IGNORECASE
        ),
        re.compile(
            r"^N\s*/\s*A[.!]?$",
            re.IGNORECASE
        )
    )

    PAGE_NUMBER_PATTERN = re.compile(
        r"^\d+$"
    )

    SECTION_ID_PATTERN = re.compile(
        r"^\d+(?:\.\d+)+\b"
    )

    def parse(
        self,
        section: Section
    ) -> SectionApplicabilityResult:
        """
        Classifies one tariff section.
        """

        substantive_lines = (
            self._get_substantive_lines(
                section
            )
        )

        if not substantive_lines:

            return SectionApplicabilityResult(
                section_id=section.section_id,
                section_title=section.title,
                source_file=section.source_file,
                category=section.category,
                status=(
                    SectionApplicability.UNKNOWN
                ),
                reason=(
                    "No substantive section content "
                    "was available."
                ),
                substantive_lines=[]
            )

        applicability_matches = [
            line
            for line in substantive_lines
            if self._is_not_applicable_marker(
                line
            )
        ]

        if (
            applicability_matches
            and len(applicability_matches)
            == len(substantive_lines)
        ):

            return SectionApplicabilityResult(
                section_id=section.section_id,
                section_title=section.title,
                source_file=section.source_file,
                category=section.category,
                status=(
                    SectionApplicability
                    .NOT_APPLICABLE
                ),
                reason=(
                    "The complete substantive section "
                    "content explicitly states that the "
                    "section is not applicable."
                ),
                matched_text=(
                    applicability_matches[0]
                ),
                substantive_lines=(
                    substantive_lines
                )
            )

        return SectionApplicabilityResult(
            section_id=section.section_id,
            section_title=section.title,
            source_file=section.source_file,
            category=section.category,
            status=(
                SectionApplicability.APPLICABLE
            ),
            reason=(
                "The section contains substantive tariff "
                "content and is not limited to an explicit "
                "non-applicability marker."
            ),
            substantive_lines=(
                substantive_lines
            )
        )

    def parse_many(
        self,
        sections: list[Section]
    ) -> list[SectionApplicabilityResult]:
        """
        Classifies multiple tariff sections.
        """

        return [
            self.parse(section)
            for section in sections
        ]

    def _get_substantive_lines(
        self,
        section: Section
    ) -> list[str]:
        """
        Removes structural content such as:

        - Section headings
        - Printed page numbers
        - Repeated tariff page headers
        """

        substantive_lines = []

        normalized_title = self._normalize_text(
            section.title
        )

        for original_line in (
            section.text.splitlines()
        ):

            line = self._clean_text(
                original_line
            )

            if not line:
                continue

            if self.PAGE_NUMBER_PATTERN.fullmatch(
                line
            ):
                continue

            if self._is_section_heading(
                line=line,
                section_id=section.section_id,
                normalized_title=normalized_title
            ):
                continue

            if self._is_page_header(
                line
            ):
                continue

            substantive_lines.append(
                line
            )

        return substantive_lines

    def _is_section_heading(
        self,
        line: str,
        section_id: str,
        normalized_title: str
    ) -> bool:

        normalized_line = self._normalize_text(
            line
        )

        normalized_section_id = (
            self._normalize_text(
                section_id
            )
        )

        if normalized_line.startswith(
            normalized_section_id
        ):
            return True

        if (
            normalized_title
            and normalized_title
            in normalized_line
            and self.SECTION_ID_PATTERN.match(
                line
            )
        ):
            return True

        return False

    def _is_page_header(
        self,
        line: str
    ) -> bool:

        normalized = self._normalize_text(
            line
        )

        header_prefixes = (
            "TARIFF FOR RETAIL DELIVERY SERVICE",
            "ONCOR ELECTRIC DELIVERY COMPANY",
            "APPLICABLE: ENTIRE CERTIFIED SERVICE AREA",
            "EFFECTIVE DATE:",
            "REVISION:",
            "SHEET:",
            "PAGE "
        )

        return normalized.startswith(
            header_prefixes
        )

    def _is_not_applicable_marker(
        self,
        line: str
    ) -> bool:

        cleaned = self._clean_text(
            line
        )

        return any(
            pattern.fullmatch(cleaned)
            for pattern
            in self.NOT_APPLICABLE_PATTERNS
        )

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