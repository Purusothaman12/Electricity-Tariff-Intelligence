import json
import re

from collections import Counter
from pathlib import Path
from typing import Any

from src.ingestion.document_extractor import (
    DocumentExtractor
)
from src.ingestion.pdf_loader import PDFLoader
from src.parsing.section_parser import (
    SectionParser
)
from src.parsing.toc_parser import TOCParser


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


EFFECTIVE_DATE_PATTERN = re.compile(
    r"\b"
    r"EFFECTIVE"
    r"(?:\s+DATE)?"
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


GENERIC_DATE_PATTERN = re.compile(
    r"\b"
    + MONTH_PATTERN
    + r"\.?\s+"
    r"\d{1,2}"
    r",?\s+"
    r"\d{4}"
    r"\b",
    re.IGNORECASE
)


def clean_text(
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

    text = re.sub(
        r"\s+",
        " ",
        text
    )

    return text.strip()


def get_table_structure(
    rate: dict[str, Any]
) -> str:

    metadata = rate.get(
        "metadata",
        {}
    )

    if not isinstance(
        metadata,
        dict
    ):
        return "UNKNOWN"

    return clean_text(
        metadata.get(
            "table_structure",
            "UNKNOWN"
        )
    ).upper()


def extract_effective_dates(
    text: str
) -> list[str]:

    normalized_text = clean_text(
        text
    )

    dates = []

    for match in (
        EFFECTIVE_DATE_PATTERN.finditer(
            normalized_text
        )
    ):

        effective_date = clean_text(
            match.group("date")
        )

        if effective_date not in dates:

            dates.append(
                effective_date
            )

    return dates


def extract_all_dates(
    text: str
) -> list[str]:

    normalized_text = clean_text(
        text
    )

    dates = []

    for match in (
        GENERIC_DATE_PATTERN.finditer(
            normalized_text
        )
    ):

        date_value = clean_text(
            match.group(0)
        )

        if date_value not in dates:

            dates.append(
                date_value
            )

    return dates


def get_relevant_lines(
    text: str
) -> list[tuple[int, str]]:

    relevant_lines = []

    indicators = (
        "EFFECTIVE",
        "REVISION",
        "APPLICABLE:",
        "SHEET:",
        "PAGE "
    )

    for line_number, original_line in enumerate(
        text.splitlines(),
        start=1
    ):

        line = clean_text(
            original_line
        )

        if not line:
            continue

        uppercase_line = line.upper()

        if any(
            indicator in uppercase_line
            for indicator in indicators
        ):

            relevant_lines.append(
                (
                    line_number,
                    line
                )
            )

            continue

        if GENERIC_DATE_PATTERN.search(
            line
        ):

            relevant_lines.append(
                (
                    line_number,
                    line
                )
            )

    return relevant_lines


def main():

    rates_directory = Path(
        "output/rates"
    )

    json_paths = sorted(
        rates_directory.glob(
            "*_rates.json"
        )
    )

    if not json_paths:

        raise FileNotFoundError(
            "No rate JSON files found inside "
            "output/rates."
        )

    loader = PDFLoader()
    document_extractor = (
        DocumentExtractor()
    )

    toc_parser = TOCParser()
    section_parser = SectionParser()

    documents = loader.load(
        "data"
    )

    documents_by_name = {
        document.file_name: document
        for document in documents
    }

    print("=" * 120)
    print(
        "UNRESOLVED SECTION EFFECTIVE DATE TEST"
    )
    print("=" * 120)

    for json_path in json_paths:

        with json_path.open(
            mode="r",
            encoding="utf-8"
        ) as input_file:

            payload = json.load(
                input_file
            )

        source_file = payload[
            "source_document"
        ][
            "file_name"
        ]

        unresolved_rates = payload.get(
            "unresolved_rates",
            []
        )

        unresolved_by_section = {}

        for rate in unresolved_rates:

            section_id = rate.get(
                "schedule_id",
                ""
            )

            if not section_id:
                continue

            unresolved_by_section.setdefault(
                section_id,
                []
            )

            unresolved_by_section[
                section_id
            ].append(
                rate
            )

        print()
        print("=" * 120)
        print(
            "SOURCE FILE:",
            source_file
        )
        print("=" * 120)

        print(
            "Unresolved Rates:",
            len(unresolved_rates)
        )

        print(
            "Affected Sections:",
            len(unresolved_by_section)
        )

        document = documents_by_name.get(
            source_file
        )

        if document is None:

            raise FileNotFoundError(
                "The source PDF was not found "
                f"inside data: {source_file}"
            )

        document = (
            document_extractor.extract(
                document
            )
        )

        toc_entries = toc_parser.parse(
            document
        )

        sections = section_parser.parse(
            document=document,
            toc_entries=toc_entries
        )

        sections_by_id = {
            section.section_id: section
            for section in sections
        }

        ordered_sections = sorted(
            unresolved_by_section.items(),
            key=lambda item: (
                -len(item[1]),
                item[0]
            )
        )

        for section_id, section_rates in (
            ordered_sections
        ):

            section = sections_by_id.get(
                section_id
            )

            print()
            print("-" * 120)

            if section is None:

                print(
                    section_id,
                    "| SECTION NOT FOUND"
                )

                continue

            structure_counts = Counter(
                get_table_structure(
                    rate
                )
                for rate in section_rates
            )

            method_counts = Counter(
                clean_text(
                    rate.get(
                        "source_method",
                        ""
                    )
                ).upper()
                for rate in section_rates
            )

            effective_dates = (
                extract_effective_dates(
                    section.text
                )
            )

            all_dates = extract_all_dates(
                section.text
            )

            relevant_lines = (
                get_relevant_lines(
                    section.text
                )
            )

            print(
                section.section_id,
                "|",
                section.title
            )

            print(
                "Pages:",
                section.start_page,
                "->",
                section.end_page
            )

            print(
                "Unresolved Count:",
                len(section_rates)
            )

            print(
                "Table Structures:",
                dict(structure_counts)
            )

            print(
                "Source Methods:",
                dict(method_counts)
            )

            print(
                "Explicit Effective Dates:",
                effective_dates
            )

            print(
                "All Dates in Section:",
                all_dates
            )

            print()
            print(
                "Relevant section lines:"
            )

            if not relevant_lines:

                print(
                    "  No date, revision or "
                    "applicability lines found."
                )

            for line_number, line in (
                relevant_lines
            ):

                print(
                    f"{line_number:>4}: {line}"
                )

            print()
            print(
                "First unresolved records:"
            )

            for rate in section_rates[:5]:

                attributes = rate.get(
                    "attributes",
                    {}
                )

                if not isinstance(
                    attributes,
                    dict
                ):
                    attributes = {}

                print(
                    " ",
                    rate.get(
                        "charge_name",
                        ""
                    ),
                    "|",
                    rate.get(
                        "value_text",
                        ""
                    ),
                    "| Structure:",
                    get_table_structure(
                        rate
                    ),
                    "| Row Label:",
                    attributes.get(
                        "row_label",
                        ""
                    )
                )

    print()
    print("=" * 120)
    print("TEST COMPLETED")
    print("=" * 120)


if __name__ == "__main__":
    main()