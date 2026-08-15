from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass(slots=True)
class Page:
    """
    Represents one page extracted from a PDF document.
    """

    page_number: int
    text: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:

        if self.page_number < 1:
            raise ValueError(
                "Page number must be greater than or equal to 1."
            )

        if self.text is None:
            self.text = ""

        if not isinstance(self.text, str):
            self.text = str(self.text)


@dataclass(slots=True)
class Document:
    """
    Represents one tariff PDF and its extracted pages.
    """

    file_name: str
    file_path: str
    pages: list[Page] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:

        if not self.file_path:
            raise ValueError(
                "Document file path cannot be empty."
            )

        self.file_path = str(
            Path(self.file_path)
        )

        if not self.file_name:
            self.file_name = Path(
                self.file_path
            ).name

    @property
    def page_count(self) -> int:
        """
        Returns the total number of extracted pages.
        """

        return len(self.pages)

    def add_page(self, page: Page) -> None:
        """
        Adds a page while preventing duplicate page numbers.
        """

        existing_page_numbers = {
            existing_page.page_number
            for existing_page in self.pages
        }

        if page.page_number in existing_page_numbers:
            raise ValueError(
                f"Page {page.page_number} already exists "
                f"in {self.file_name}."
            )

        self.pages.append(page)

        self.pages.sort(
            key=lambda item: item.page_number
        )

    def get_page(self, page_number: int) -> Page | None:
        """
        Returns a page by its one-based page number.
        """

        for page in self.pages:

            if page.page_number == page_number:
                return page

        return None