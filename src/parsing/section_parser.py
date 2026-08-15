import re
from typing import Optional

from src.models.document import Document
from src.models.section import Section, TOCEntry
from src.parsing.schedule_classifier import ScheduleClassifier


class SectionParser:
    """
    Extracts complete tariff sections using TOC entries.

    TOC page numbers may represent printed tariff page labels rather
    than physical PDF page numbers. Inserted pages such as 100.1 and
    100.2 can therefore move a section several physical pages forward.
    """

    SECTION_HEADING_PATTERN = re.compile(
        r"(?m)^[ \t]*"
        r"(?P<section_id>6(?:\.\d+){1,6})"
        r"[ \t]+"
        r"(?P<title>[^\n\r]+)"
    )

    def __init__(
        self,
        search_back_pages: int = 2,
        search_forward_pages: int = 20
    ) -> None:

        if search_back_pages < 0:
            raise ValueError(
                "search_back_pages cannot be negative."
            )

        if search_forward_pages < 1:
            raise ValueError(
                "search_forward_pages must be at least 1."
            )

        self.search_back_pages = search_back_pages
        self.search_forward_pages = search_forward_pages

        self.classifier = ScheduleClassifier()

    def parse(
        self,
        document: Document,
        toc_entries: list[TOCEntry]
    ) -> list[Section]:

        if not document.pages:
            raise ValueError(
                f"No extracted pages found in "
                f"{document.file_name}."
            )

        if not toc_entries:
            return []

        entries = sorted(
            toc_entries,
            key=lambda entry: (
                entry.start_page,
                self._section_id_key(
                    entry.section_id
                )
            )
        )

        located_entries = []

        for entry in entries:

            location = self._locate_entry(
                document=document,
                entry=entry
            )

            if location is None:

                search_start = max(
                    1,
                    entry.start_page
                    - self.search_back_pages
                )

                search_end = min(
                    document.page_count,
                    entry.start_page
                    + self.search_forward_pages
                )

                raise ValueError(
                    f"Could not locate section heading "
                    f"'{entry.full_title}' in "
                    f"{document.file_name}. "
                    f"Searched physical PDF pages "
                    f"{search_start} through {search_end}."
                )

            located_entries.append(
                {
                    "entry": entry,
                    "page_number": location[0],
                    "offset": location[1]
                }
            )

        # Actual physical page order is more reliable than
        # TOC page labels such as 100.1, 100.2 and 101.1.
        located_entries.sort(
            key=lambda item: (
                item["page_number"],
                item["offset"],
                self._section_id_key(
                    item["entry"].section_id
                )
            )
        )

        sections = []

        for index, located in enumerate(
            located_entries
        ):

            entry = located["entry"]

            start_page = located["page_number"]
            start_offset = located["offset"]

            if index + 1 < len(located_entries):

                next_location = located_entries[
                    index + 1
                ]

                boundary_page = (
                    next_location["page_number"]
                )

                boundary_offset = (
                    next_location["offset"]
                )

            else:

                boundary = (
                    self._find_next_section_boundary(
                        document=document,
                        current_entry=entry,
                        start_page=start_page,
                        start_offset=start_offset
                    )
                )

                if boundary is None:

                    boundary_page = (
                        document.page_count + 1
                    )

                    boundary_offset = 0

                else:

                    boundary_page = boundary[0]
                    boundary_offset = boundary[1]

            section_text, end_page = (
                self._extract_section_text(
                    document=document,
                    start_page=start_page,
                    start_offset=start_offset,
                    boundary_page=boundary_page,
                    boundary_offset=boundary_offset
                )
            )

            category = self.classifier.classify(
                entry
            )

            section = Section(
                section_id=entry.section_id,
                title=entry.title,
                start_page=start_page,
                end_page=end_page,
                source_file=document.file_name,
                category=category,
                text=section_text.strip()
            )

            sections.append(section)

        return sections

    def _locate_entry(
        self,
        document: Document,
        entry: TOCEntry
    ) -> Optional[tuple[int, int]]:

        search_start = max(
            1,
            entry.start_page
            - self.search_back_pages
        )

        search_end = min(
            document.page_count,
            entry.start_page
            + self.search_forward_pages
        )

        section_pattern = re.compile(
            rf"(?<![\d.])"
            rf"{re.escape(entry.section_id)}"
            rf"(?![\d.])",
            re.IGNORECASE
        )

        candidates = []

        for page_number in range(
            search_start,
            search_end + 1
        ):

            page = document.get_page(
                page_number
            )

            if page is None:
                continue

            for match in section_pattern.finditer(
                page.text
            ):

                window = page.text[
                    match.start():
                    match.start() + 500
                ]

                title_score = (
                    self._calculate_title_score(
                        expected_title=entry.title,
                        candidate_text=window
                    )
                )

                if title_score < 0.40:
                    continue

                page_distance = abs(
                    page_number
                    - entry.start_page
                )

                candidates.append(
                    {
                        "page_number": page_number,
                        "offset": match.start(),
                        "title_score": title_score,
                        "page_distance": page_distance
                    }
                )

        if not candidates:
            return None

        candidates.sort(
            key=lambda candidate: (
                -candidate["title_score"],
                candidate["page_distance"],
                candidate["page_number"],
                candidate["offset"]
            )
        )

        best_candidate = candidates[0]

        return (
            best_candidate["page_number"],
            best_candidate["offset"]
        )

    def _calculate_title_score(
        self,
        expected_title: str,
        candidate_text: str
    ) -> float:

        normalized_title = self._normalize_text(
            expected_title
        )

        normalized_candidate = self._normalize_text(
            candidate_text
        )

        if normalized_title in normalized_candidate:
            return 1.0

        title_words = [
            word
            for word in normalized_title.split()
            if len(word) >= 3
        ]

        if not title_words:
            return 0.0

        matched_words = sum(
            1
            for word in title_words
            if word in normalized_candidate
        )

        return matched_words / len(title_words)

    def _find_next_section_boundary(
        self,
        document: Document,
        current_entry: TOCEntry,
        start_page: int,
        start_offset: int
    ) -> Optional[tuple[int, int]]:

        current_key = self._section_id_key(
            current_entry.section_id
        )

        for page_number in range(
            start_page,
            document.page_count + 1
        ):

            page = document.get_page(
                page_number
            )

            if page is None:
                continue

            if page_number == start_page:

                search_offset = (
                    start_offset
                    + len(current_entry.section_id)
                )

            else:

                search_offset = 0

            page_text = page.text[
                search_offset:
            ]

            for match in (
                self.SECTION_HEADING_PATTERN.finditer(
                    page_text
                )
            ):

                candidate_id = (
                    match.group("section_id")
                    .strip()
                )

                candidate_title = (
                    match.group("title")
                    .strip()
                )

                candidate_key = (
                    self._section_id_key(
                        candidate_id
                    )
                )

                if candidate_id == current_entry.section_id:
                    continue

                # Repeated parent headers such as
                # "6.1.1 Delivery System Charges"
                # must not end the current rider.
                if self._is_parent_section(
                    parent_key=candidate_key,
                    child_key=current_key
                ):
                    continue

                if self._is_parent_section(
                    parent_key=current_key,
                    child_key=candidate_key
                ):
                    continue

                if candidate_key <= current_key:
                    continue

                if len(candidate_title) > 180:
                    continue

                absolute_offset = (
                    search_offset
                    + match.start()
                )

                return (
                    page_number,
                    absolute_offset
                )

        return None

    def _extract_section_text(
        self,
        document: Document,
        start_page: int,
        start_offset: int,
        boundary_page: int,
        boundary_offset: int
    ) -> tuple[str, int]:

        extracted_parts = []

        # Two sections can begin on the same physical page.
        if boundary_page == start_page:

            page = document.get_page(
                start_page
            )

            if page is None:
                return "", start_page

            text = page.text[
                start_offset:boundary_offset
            ]

            return (
                text.strip(),
                start_page
            )

        # The next section normally begins after the standard
        # page header on a new physical PDF page. Therefore,
        # that boundary page belongs to the next section.
        if boundary_page <= document.page_count:

            end_page = boundary_page - 1

        else:

            end_page = document.page_count

        end_page = max(
            start_page,
            end_page
        )

        for page_number in range(
            start_page,
            end_page + 1
        ):

            page = document.get_page(
                page_number
            )

            if page is None:
                continue

            if page_number == start_page:

                page_text = page.text[
                    start_offset:
                ]

            else:

                page_text = page.text

            if page_text.strip():

                extracted_parts.append(
                    page_text.strip()
                )

        return (
            "\n\n".join(extracted_parts),
            end_page
        )

    def _section_id_key(
        self,
        section_id: str
    ) -> tuple[int, ...]:

        return tuple(
            int(part)
            for part in section_id.split(".")
        )

    def _is_parent_section(
        self,
        parent_key: tuple[int, ...],
        child_key: tuple[int, ...]
    ) -> bool:

        if len(parent_key) >= len(child_key):
            return False

        return (
            child_key[:len(parent_key)]
            == parent_key
        )

    def _normalize_text(
        self,
        text: str
    ) -> str:

        text = text.upper()

        text = text.replace(
            "\u2013",
            "-"
        )

        text = text.replace(
            "\u2014",
            "-"
        )

        text = re.sub(
            r"\s*-\s*",
            " - ",
            text
        )

        text = re.sub(
            r"\s+",
            " ",
            text
        )

        return text.strip()