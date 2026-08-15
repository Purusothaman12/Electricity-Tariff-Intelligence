import json
import re
from pathlib import Path

from src.models.document import Document
from src.models.section import Section


class SectionJSONExporter:
    """
    Saves extracted tariff sections as a structured JSON file.
    """

    def __init__(
        self,
        output_directory: str = "output/extracted"
    ) -> None:

        self.output_directory = Path(
            output_directory
        )

    def export(
        self,
        document: Document,
        sections: list[Section]
    ) -> Path:

        self.output_directory.mkdir(
            parents=True,
            exist_ok=True
        )

        output_file = (
            self.output_directory
            / f"{Path(document.file_name).stem}_sections.json"
        )

        payload = {
            "document": {
                "file_name": document.file_name,
                "file_path": document.file_path,
                "page_count": document.page_count,
                "section_count": len(sections)
            },
            "sections": [
                self._section_to_dict(section)
                for section in sections
            ]
        }

        with output_file.open(
            mode="w",
            encoding="utf-8"
        ) as file:

            json.dump(
                payload,
                file,
                indent=2,
                ensure_ascii=False
            )

        return output_file

    def _section_to_dict(
        self,
        section: Section
    ) -> dict:

        return {
            "section_id": section.section_id,
            "title": section.title,
            "normalized_title": self._normalize_title(
                section.title
            ),
            "category": section.category,
            "source_file": section.source_file,
            "start_page": section.start_page,
            "end_page": section.end_page,
            "page_count": section.page_count,
            "character_count": len(section.text),
            "text": section.text
        }

    def _normalize_title(
        self,
        title: str
    ) -> str:

        normalized = title.upper()

        normalized = normalized.replace(
            "\u2013",
            "-"
        )

        normalized = normalized.replace(
            "\u2014",
            "-"
        )

        normalized = re.sub(
            r"\s*-\s*",
            " - ",
            normalized
        )

        normalized = re.sub(
            r"\s+",
            " ",
            normalized
        )

        return normalized.strip()