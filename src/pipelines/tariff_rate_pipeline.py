from collections import Counter
from dataclasses import dataclass

from src.ingestion.document_extractor import DocumentExtractor
from src.ingestion.pdf_loader import PDFLoader
from src.models.document import Document
from src.models.rate import RateItem
from src.models.section import Section
from src.models.table import ExtractedTable
from src.parsing.effective_date_resolver import (
    EffectiveDateResolver
)
from src.parsing.rate_merger import RateMerger
from src.parsing.section_applicability_parser import (
    SectionApplicability,
    SectionApplicabilityParser,
    SectionApplicabilityResult
)
from src.parsing.section_effective_date_resolver import (
    SectionEffectiveDateResolver
)
from src.parsing.section_parser import SectionParser
from src.parsing.table_rate_parser import TableRateParser
from src.parsing.text_rate_parser import TextRateParser
from src.parsing.toc_parser import TOCParser
from src.table_extraction.docling_extractor import (
    DoclingTableExtractor
)


@dataclass(slots=True)
class TariffPipelineResult:
    """
    Stores the complete processing result for one tariff PDF.
    """

    document: Document
    sections: list[Section]

    section_applicability: list[
        SectionApplicabilityResult
    ]

    tables: list[ExtractedTable]
    table_rates: list[RateItem]
    text_rates: list[RateItem]
    merged_rates: list[RateItem]
    final_rates: list[RateItem]

    @property
    def source_file(self) -> str:

        return self.document.file_name

    @property
    def unresolved_date_count(self) -> int:

        return sum(
            1
            for rate in self.final_rates
            if not rate.effective_date
        )

    @property
    def category_counts(self) -> dict[str, int]:

        counts = Counter(
            rate.category
            for rate in self.final_rates
        )

        return dict(counts)

    @property
    def schedule_count(self) -> int:

        return len(
            {
                (
                    rate.schedule_id,
                    rate.schedule_title
                )
                for rate in self.final_rates
            }
        )

    @property
    def applicability_counts(
        self
    ) -> dict[str, int]:

        counts = Counter(
            result.status.value
            for result
            in self.section_applicability
        )

        return {
            status.value: counts.get(
                status.value,
                0
            )
            for status in SectionApplicability
        }

    @property
    def effective_date_resolution_counts(
        self
    ) -> dict[str, int]:

        counts = Counter(
            rate.metadata.get(
                "effective_date_resolution",
                "UNKNOWN"
            )
            for rate in self.final_rates
        )

        return dict(counts)

    @property
    def applicable_section_count(self) -> int:

        return self.applicability_counts[
            SectionApplicability
            .APPLICABLE
            .value
        ]

    @property
    def not_applicable_section_count(
        self
    ) -> int:

        return self.applicability_counts[
            SectionApplicability
            .NOT_APPLICABLE
            .value
        ]

    @property
    def unknown_section_count(self) -> int:

        return self.applicability_counts[
            SectionApplicability
            .UNKNOWN
            .value
        ]

    @property
    def processable_section_count(self) -> int:
        """
        Applicable and unknown sections are processable.

        Explicitly not-applicable sections are skipped.
        """

        return (
            self.applicable_section_count
            + self.unknown_section_count
        )

    @property
    def not_applicable_section_ids(
        self
    ) -> list[str]:

        return [
            result.section_id
            for result
            in self.section_applicability
            if (
                result.status
                == SectionApplicability
                .NOT_APPLICABLE
            )
        ]

    def get_applicability(
        self,
        section_id: str
    ) -> SectionApplicabilityResult | None:

        for result in self.section_applicability:

            if result.section_id == section_id:
                return result

        return None

    def to_summary(self) -> dict:

        return {
            "source_file": self.source_file,
            "document_pages": (
                self.document.page_count
            ),
            "sections": len(self.sections),
            "applicable_sections": (
                self.applicable_section_count
            ),
            "not_applicable_sections": (
                self.not_applicable_section_count
            ),
            "unknown_sections": (
                self.unknown_section_count
            ),
            "processable_sections": (
                self.processable_section_count
            ),
            "tables": len(self.tables),
            "table_rates": len(
                self.table_rates
            ),
            "text_rates": len(
                self.text_rates
            ),
            "merged_rates": len(
                self.merged_rates
            ),
            "final_rates": len(
                self.final_rates
            ),
            "schedules": self.schedule_count,
            "unresolved_dates": (
                self.unresolved_date_count
            ),
            "category_counts": (
                self.category_counts
            ),
            "applicability_counts": (
                self.applicability_counts
            ),
            "effective_date_resolution_counts": (
                self.effective_date_resolution_counts
            )
        }


class TariffRatePipeline:
    """
    Runs the complete tariff-rate extraction workflow.

    Workflow:

    PDF loading
        ↓
    Page text extraction
        ↓
    TOC parsing
        ↓
    Section parsing
        ↓
    Section applicability classification
        ↓
    Skip explicitly not-applicable sections
        ↓
    Docling table extraction
        ↓
    Table rate parsing
        ↓
    Text rate parsing
        ↓
    Rate merging
        ↓
    Schedule/category date consensus
        ↓
    Explicit section-header date resolution
    """

    def __init__(
        self,
        max_pages_per_batch: int = 3
    ) -> None:

        self.pdf_loader = PDFLoader()

        self.document_extractor = (
            DocumentExtractor()
        )

        self.toc_parser = TOCParser()

        self.section_parser = (
            SectionParser()
        )

        self.section_applicability_parser = (
            SectionApplicabilityParser()
        )

        self.table_extractor = (
            DoclingTableExtractor(
                max_pages_per_batch=(
                    max_pages_per_batch
                )
            )
        )

        self.table_rate_parser = (
            TableRateParser()
        )

        self.text_rate_parser = (
            TextRateParser()
        )

        self.rate_merger = RateMerger()

        self.effective_date_resolver = (
            EffectiveDateResolver()
        )

        self.section_effective_date_resolver = (
            SectionEffectiveDateResolver()
        )

    def process_directory(
        self,
        data_directory: str
    ) -> list[TariffPipelineResult]:
        """
        Processes every PDF found in a directory.
        """

        documents = self.pdf_loader.load(
            data_directory
        )

        results = []

        for document in documents:

            result = self.process_document(
                document
            )

            results.append(
                result
            )

        return results

    def process_document(
        self,
        document: Document
    ) -> TariffPipelineResult:
        """
        Processes one tariff PDF.
        """

        print()
        print("=" * 120)
        print(
            "PROCESSING TARIFF:",
            document.file_name
        )
        print("=" * 120)

        extracted_document = (
            self.document_extractor.extract(
                document
            )
        )

        toc_entries = self.toc_parser.parse(
            extracted_document
        )

        sections = self.section_parser.parse(
            document=extracted_document,
            toc_entries=toc_entries
        )

        print(
            "Sections identified:",
            len(sections)
        )

        applicability_results = (
            self.section_applicability_parser
            .parse_many(
                sections
            )
        )

        applicability_counts = Counter(
            result.status.value
            for result
            in applicability_results
        )

        print(
            "Applicable sections:",
            applicability_counts.get(
                SectionApplicability
                .APPLICABLE
                .value,
                0
            )
        )

        print(
            "Not applicable sections:",
            applicability_counts.get(
                SectionApplicability
                .NOT_APPLICABLE
                .value,
                0
            )
        )

        print(
            "Unknown sections:",
            applicability_counts.get(
                SectionApplicability
                .UNKNOWN
                .value,
                0
            )
        )

        processable_sections = (
            self.get_processable_sections(
                sections=sections,
                applicability_results=(
                    applicability_results
                )
            )
        )

        print(
            "Sections sent for extraction:",
            len(processable_sections)
        )

        tables = self.table_extractor.extract(
            pdf_path=(
                extracted_document.file_path
            ),
            sections=processable_sections
        )

        print(
            "Tables extracted:",
            len(tables)
        )

        table_rates = (
            self.table_rate_parser.parse(
                tables
            )
        )

        print(
            "Table rates parsed:",
            len(table_rates)
        )

        text_rates = (
            self.text_rate_parser.parse(
                processable_sections
            )
        )

        print(
            "Text rates parsed:",
            len(text_rates)
        )

        merged_rates = self.rate_merger.merge(
            table_rates=table_rates,
            text_rates=text_rates
        )

        print(
            "Rates after merging:",
            len(merged_rates)
        )

        final_rates = (
            self.resolve_effective_dates(
                merged_rates=merged_rates,
                sections=sections
            )
        )

        unresolved_count = sum(
            1
            for rate in final_rates
            if not rate.effective_date
        )

        resolution_counts = Counter(
            rate.metadata.get(
                "effective_date_resolution",
                "UNKNOWN"
            )
            for rate in final_rates
        )

        print(
            "Final rates:",
            len(final_rates)
        )

        print(
            "Unresolved dates:",
            unresolved_count
        )

        print(
            "Date resolution methods:",
            dict(resolution_counts)
        )

        return TariffPipelineResult(
            document=extracted_document,
            sections=sections,
            section_applicability=(
                applicability_results
            ),
            tables=tables,
            table_rates=table_rates,
            text_rates=text_rates,
            merged_rates=merged_rates,
            final_rates=final_rates
        )

    def resolve_effective_dates(
        self,
        merged_rates: list[RateItem],
        sections: list[Section]
    ) -> list[RateItem]:
        """
        Runs the two date-resolution stages.

        Stage 1:
            Schedule and category consensus.

        Stage 2:
            Explicit section-header dates.

        Section-header resolution is performed last so its
        provenance remains SECTION_HEADER.
        """

        consensus_resolved_rates = (
            self.effective_date_resolver.resolve(
                merged_rates
            )
        )

        return (
            self.section_effective_date_resolver
            .resolve(
                rate_items=(
                    consensus_resolved_rates
                ),
                sections=sections
            )
        )

    def get_processable_sections(
        self,
        sections: list[Section],
        applicability_results: list[
            SectionApplicabilityResult
        ]
    ) -> list[Section]:
        """
        Returns sections that should be sent to rate extraction.

        UNKNOWN sections are included so potentially useful
        content is not silently discarded.
        """

        status_by_section_id = {
            result.section_id: result.status
            for result
            in applicability_results
        }

        return [
            section
            for section in sections
            if (
                status_by_section_id.get(
                    section.section_id,
                    SectionApplicability.UNKNOWN
                )
                != SectionApplicability
                .NOT_APPLICABLE
            )
        ]