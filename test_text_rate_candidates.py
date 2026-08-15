import re

from src.ingestion.document_extractor import DocumentExtractor
from src.ingestion.pdf_loader import PDFLoader
from src.parsing.section_parser import SectionParser
from src.parsing.toc_parser import TOCParser


class TextRateCandidateFinder:
    """
    Finds lines that may contain tariff charges, values,
    units or rider references.

    This is a diagnostic utility. It does not extract final
    RateItem objects.
    """

    CANDIDATE_PATTERN = re.compile(
        r"("
        r"\$\s*\(?\d"
        r"|"
        r"\(?\d+(?:\.\d+)?\)?\s*%"
        r"|"
        r"\bCUSTOMER\s+CHARGE\b"
        r"|"
        r"\bMETERING\s+CHARGE\b"
        r"|"
        r"\bTRANSMISSION\s+SYSTEM\s+CHARGE\b"
        r"|"
        r"\bDISTRIBUTION\s+SYSTEM\s+CHARGE\b"
        r"|"
        r"\bNUCLEAR\s+DECOMMISSIONING\b"
        r"|"
        r"\bCOST\s+RECOVERY\s+FACTOR\b"
        r"|"
        r"\bSURCHARGE\b"
        r"|"
        r"\bREFUND\b"
        r"|"
        r"\bCREDIT\b"
        r"|"
        r"\bSEE\s+RIDER\b"
        r"|"
        r"\bSEE\s+TABLE\b"
        r"|"
        r"\bPER\s+KWH\b"
        r"|"
        r"\bPER\s+RETAIL\s+CUSTOMER\b"
        r"|"
        r"\bPER\s+NCP\s+KW\b"
        r"|"
        r"\bPER\s+4CP\s+KW\b"
        r"|"
        r"\bBILLING\s+KW\b"
        r")",
        re.IGNORECASE
    )

    def find(
        self,
        text: str,
        context_lines: int = 1
    ) -> list[tuple[int, str]]:

        lines = [
            self._clean_text(line)
            for line in text.splitlines()
        ]

        selected_indexes = set()

        for index, line in enumerate(lines):

            if not line:
                continue

            if not self.CANDIDATE_PATTERN.search(
                line
            ):
                continue

            start_index = max(
                0,
                index - context_lines
            )

            end_index = min(
                len(lines) - 1,
                index + context_lines
            )

            for candidate_index in range(
                start_index,
                end_index + 1
            ):

                if lines[candidate_index]:
                    selected_indexes.add(
                        candidate_index
                    )

        return [
            (
                index + 1,
                lines[index]
            )
            for index in sorted(
                selected_indexes
            )
        ]

    def _clean_text(
        self,
        text: str
    ) -> str:

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


def main():

    loader = PDFLoader()
    document_extractor = DocumentExtractor()
    toc_parser = TOCParser()
    section_parser = SectionParser()

    candidate_finder = (
        TextRateCandidateFinder()
    )

    documents = loader.load("data")

    print("=" * 120)
    print("TEXT RATE CANDIDATE TEST")
    print("=" * 120)

    for document in documents:

        document = document_extractor.extract(
            document
        )

        toc_entries = toc_parser.parse(
            document
        )

        sections = section_parser.parse(
            document=document,
            toc_entries=toc_entries
        )

        normal_sections = [
            section
            for section in sections
            if section.category
            == "NORMAL_SCHEDULE"
        ]

        print()
        print("=" * 120)
        print("SOURCE FILE:", document.file_name)
        print("=" * 120)

        for section in normal_sections:

            candidates = candidate_finder.find(
                text=section.text,
                context_lines=1
            )

            print()
            print("-" * 120)

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
                "Candidate Lines:",
                len(candidates)
            )

            print("-" * 120)

            if not candidates:

                print(
                    "No rate candidate lines found."
                )

                continue

            previous_line_number = None

            for line_number, line in candidates:

                if (
                    previous_line_number is not None
                    and line_number
                    > previous_line_number + 1
                ):
                    print("...")

                print(
                    f"{line_number:>4}: {line}"
                )

                previous_line_number = (
                    line_number
                )

    print()
    print("=" * 120)
    print("TEST COMPLETED")
    print("=" * 120)


if __name__ == "__main__":
    main()
    