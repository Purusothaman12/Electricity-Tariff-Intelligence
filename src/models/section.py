from dataclasses import dataclass


@dataclass(slots=True)
class TOCEntry:
    """
    Represents one schedule or rider entry found
    in the PDF table of contents.
    """

    section_id: str
    title: str
    start_page: int

    def __post_init__(self) -> None:

        self.section_id = self.section_id.strip()
        self.title = self.title.strip()

        if not self.section_id:
            raise ValueError(
                "TOC section ID cannot be empty."
            )

        if not self.title:
            raise ValueError(
                "TOC title cannot be empty."
            )

        if self.start_page < 1:
            raise ValueError(
                "TOC start page must be greater than or equal to 1."
            )

    @property
    def full_title(self) -> str:

        return f"{self.section_id} {self.title}"


@dataclass(slots=True)
class Section:
    """
    Represents an extracted tariff schedule or rider section.
    """

    section_id: str
    title: str
    start_page: int
    end_page: int
    source_file: str = ""
    category: str = "UNCLASSIFIED"
    text: str = ""

    def __post_init__(self) -> None:

        self.section_id = self.section_id.strip()
        self.title = self.title.strip()
        self.category = self.category.strip().upper()

        if not self.section_id:
            raise ValueError(
                "Section ID cannot be empty."
            )

        if not self.title:
            raise ValueError(
                "Section title cannot be empty."
            )

        if self.start_page < 1:
            raise ValueError(
                "Section start page must be greater than or equal to 1."
            )

        if self.end_page < self.start_page:
            raise ValueError(
                "Section end page cannot be before the start page."
            )

    @property
    def page_count(self) -> int:

        return self.end_page - self.start_page + 1

    @property
    def full_title(self) -> str:

        return f"{self.section_id} {self.title}"

    def contains_page(self, page_number: int) -> bool:

        return self.start_page <= page_number <= self.end_page