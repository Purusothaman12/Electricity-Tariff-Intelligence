import re

from src.models.document import Document
from src.models.section import TOCEntry


class TOCParser:
    """
    Extracts tariff schedule and rider entries from
    the Table of Contents pages.

    Examples:

    6.1.1.1.1 RESIDENTIAL SERVICE ........ 67
    6.1.1.13.1 RIDER EECRF ............... 100
    """

    ENTRY_PATTERN = re.compile(
        r"(?P<section_id>6(?:\.\d+){4})"
        r"\s+"
        r"(?P<title>.*?)"
        r"\s+\.{1,}\s+"
        r"(?P<start_page>\d+)",
        re.IGNORECASE
    )

    def __init__(
        self,
        max_toc_pages: int = 15
    ) -> None:

        if max_toc_pages < 1:
            raise ValueError(
                "max_toc_pages must be at least 1."
            )

        self.max_toc_pages = max_toc_pages

    def parse(
        self,
        document: Document
    ) -> list[TOCEntry]:

        if not document.pages:
            raise ValueError(
                f"No extracted pages found in "
                f"{document.file_name}."
            )

        toc_text = self._collect_toc_text(
            document
        )

        matches = self.ENTRY_PATTERN.finditer(
            toc_text
        )

        entries = []
        seen_section_ids = set()

        for match in matches:

            section_id = (
                match.group("section_id")
                .strip()
            )

            if section_id in seen_section_ids:
                continue

            title = self._clean_title(
                match.group("title")
            )

            start_page = int(
                match.group("start_page")
            )

            if not title:
                continue

            entry = TOCEntry(
                section_id=section_id,
                title=title,
                start_page=start_page
            )

            entries.append(entry)
            seen_section_ids.add(section_id)

        entries.sort(
            key=lambda entry: (
                entry.start_page,
                entry.section_id
            )
        )

        return entries

    def _collect_toc_text(
        self,
        document: Document
    ) -> str:

        pages = document.pages[
            :self.max_toc_pages
        ]

        combined_text = "\n".join(
            page.text
            for page in pages
            if page.text
        )

        combined_text = combined_text.replace(
            "\u00a0",
            " "
        )

        combined_text = combined_text.replace(
            "\u2013",
            "-"
        )

        combined_text = combined_text.replace(
            "\u2014",
            "-"
        )

        combined_text = re.sub(
            r"\s+",
            " ",
            combined_text
        )

        return combined_text.strip()

    def _clean_title(
        self,
        title: str
    ) -> str:

        title = title.replace(
            "\u00a0",
            " "
        )

        title = title.replace(
            "\u2013",
            "-"
        )

        title = title.replace(
            "\u2014",
            "-"
        )

        title = re.sub(
            r"\s+",
            " ",
            title
        )

        return title.strip(
            " .:-"
        ).upper()