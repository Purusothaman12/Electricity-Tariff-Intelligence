import json
import re

from collections import Counter
from pathlib import Path
from typing import Any

from src.parsing.section_applicability_parser import (
    SectionApplicability
)
from src.pipelines.tariff_rate_pipeline import (
    TariffPipelineResult
)


class RateJSONExporter:
    """
    Exports final tariff-rate records to structured JSON.

    The exported coverage information distinguishes between:

    - EXTRACTED
    - NOT_APPLICABLE
    - REVIEW_REQUIRED

    A section explicitly marked NOT APPLICABLE is not considered
    a missing extraction.
    """

    SCHEMA_VERSION = "1.1"

    def export_result(
        self,
        result: TariffPipelineResult,
        output_directory: str = "output/rates"
    ) -> Path:
        """
        Exports one processed tariff document.
        """

        output_path = Path(
            output_directory
        )

        output_path.mkdir(
            parents=True,
            exist_ok=True
        )

        file_stem = self._safe_file_stem(
            result.source_file
        )

        destination = (
            output_path
            / f"{file_stem}_rates.json"
        )

        payload = self._build_payload(
            result
        )

        temporary_destination = (
            destination.with_suffix(
                ".json.tmp"
            )
        )

        with temporary_destination.open(
            mode="w",
            encoding="utf-8"
        ) as output_file:

            json.dump(
                payload,
                output_file,
                indent=2,
                ensure_ascii=False,
                default=str
            )

        temporary_destination.replace(
            destination
        )

        return destination

    def export_results(
        self,
        results: list[TariffPipelineResult],
        output_directory: str = "output/rates"
    ) -> list[Path]:
        """
        Exports multiple processed tariff documents.
        """

        exported_paths = []

        for result in results:

            exported_path = self.export_result(
                result=result,
                output_directory=output_directory
            )

            exported_paths.append(
                exported_path
            )

        return exported_paths

    def _build_payload(
        self,
        result: TariffPipelineResult
    ) -> dict[str, Any]:

        rate_counts = Counter(
            rate.schedule_id
            for rate in result.final_rates
        )

        table_counts = Counter(
            table.schedule_id
            for table in result.tables
        )

        unresolved_counts = Counter(
            rate.schedule_id
            for rate in result.final_rates
            if not rate.effective_date
        )

        merged_counts = Counter(
            rate.schedule_id
            for rate in result.final_rates
            if "+" in rate.source_method
        )

        applicability_by_section_id = {
            applicability.section_id: applicability
            for applicability
            in result.section_applicability
        }

        section_coverage = []

        for section in result.sections:

            rate_count = rate_counts.get(
                section.section_id,
                0
            )

            table_count = table_counts.get(
                section.section_id,
                0
            )

            unresolved_count = (
                unresolved_counts.get(
                    section.section_id,
                    0
                )
            )

            merged_count = merged_counts.get(
                section.section_id,
                0
            )

            applicability = (
                applicability_by_section_id.get(
                    section.section_id
                )
            )

            if applicability is None:

                applicability_status = (
                    SectionApplicability
                    .UNKNOWN
                    .value
                )

                applicability_reason = (
                    "No applicability result "
                    "was available."
                )

                applicability_match = ""

                substantive_lines = []

            else:

                applicability_status = (
                    applicability.status.value
                )

                applicability_reason = (
                    applicability.reason
                )

                applicability_match = (
                    applicability.matched_text
                )

                substantive_lines = list(
                    applicability.substantive_lines
                )

            extraction_expected = (
                applicability_status
                != SectionApplicability
                .NOT_APPLICABLE
                .value
            )

            coverage_status = (
                self._get_coverage_status(
                    applicability_status=(
                        applicability_status
                    ),
                    rate_count=rate_count
                )
            )

            section_coverage.append(
                {
                    "section_id": (
                        section.section_id
                    ),
                    "title": section.title,
                    "category": section.category,
                    "start_page": (
                        section.start_page
                    ),
                    "end_page": section.end_page,
                    "applicability_status": (
                        applicability_status
                    ),
                    "applicability_reason": (
                        applicability_reason
                    ),
                    "applicability_match": (
                        applicability_match
                    ),
                    "substantive_lines": (
                        substantive_lines
                    ),
                    "extraction_expected": (
                        extraction_expected
                    ),
                    "coverage_status": (
                        coverage_status
                    ),
                    "table_count": table_count,
                    "rate_count": rate_count,
                    "merged_rate_count": (
                        merged_count
                    ),
                    "unresolved_date_count": (
                        unresolved_count
                    ),
                    "has_tables": (
                        table_count > 0
                    ),
                    "has_rates": (
                        rate_count > 0
                    )
                }
            )

        extracted_sections = [
            section
            for section in section_coverage
            if (
                section["coverage_status"]
                == "EXTRACTED"
            )
        ]

        not_applicable_sections = [
            section
            for section in section_coverage
            if (
                section["coverage_status"]
                == "NOT_APPLICABLE"
            )
        ]

        review_sections = [
            section
            for section in section_coverage
            if (
                section["coverage_status"]
                == "REVIEW_REQUIRED"
            )
        ]

        applicable_without_rates = [
            section
            for section in review_sections
            if (
                section[
                    "applicability_status"
                ]
                == SectionApplicability
                .APPLICABLE
                .value
            )
        ]

        unknown_without_rates = [
            section
            for section in review_sections
            if (
                section[
                    "applicability_status"
                ]
                == SectionApplicability
                .UNKNOWN
                .value
            )
        ]

        unresolved_rates = [
            rate.to_dict()
            for rate in result.final_rates
            if not rate.effective_date
        ]

        final_rates = [
            rate.to_dict()
            for rate in result.final_rates
        ]

        applicability_results = [
            applicability.to_dict()
            for applicability
            in result.section_applicability
        ]

        return {
            "schema_version": (
                self.SCHEMA_VERSION
            ),
            "source_document": {
                "file_name": (
                    result.document.file_name
                ),
                "file_path": (
                    result.document.file_path
                ),
                "page_count": (
                    result.document.page_count
                ),
                "metadata": dict(
                    result.document.metadata
                )
            },
            "summary": result.to_summary(),
            "coverage_summary": {
                "total_sections": len(
                    result.sections
                ),
                "applicable_sections": (
                    result.applicable_section_count
                ),
                "not_applicable_sections": (
                    result.not_applicable_section_count
                ),
                "unknown_sections": (
                    result.unknown_section_count
                ),
                "processable_sections": (
                    result.processable_section_count
                ),
                "sections_with_rates": len(
                    extracted_sections
                ),
                "sections_without_rates": (
                    len(result.sections)
                    - len(extracted_sections)
                ),
                "review_required_sections": len(
                    review_sections
                ),
                "applicable_without_rates": len(
                    applicable_without_rates
                ),
                "unknown_without_rates": len(
                    unknown_without_rates
                ),
                "sections_with_tables": sum(
                    1
                    for section in section_coverage
                    if section["has_tables"]
                ),
                "unresolved_rate_count": len(
                    unresolved_rates
                )
            },
            "section_applicability": (
                applicability_results
            ),
            "section_coverage": (
                section_coverage
            ),
            "extracted_sections": (
                extracted_sections
            ),
            "not_applicable_sections": (
                not_applicable_sections
            ),
            "review_sections": (
                review_sections
            ),
            "missing_sections": (
                review_sections
            ),
            "unresolved_rates": (
                unresolved_rates
            ),
            "rates": final_rates
        }

    def _get_coverage_status(
        self,
        applicability_status: str,
        rate_count: int
    ) -> str:
        """
        Determines the section's final extraction status.
        """

        if (
            applicability_status
            == SectionApplicability
            .NOT_APPLICABLE
            .value
        ):
            return "NOT_APPLICABLE"

        if rate_count > 0:
            return "EXTRACTED"

        return "REVIEW_REQUIRED"

    def _safe_file_stem(
        self,
        source_file: str
    ) -> str:

        file_stem = Path(
            source_file
        ).stem

        file_stem = re.sub(
            r"[^A-Za-z0-9_-]+",
            "_",
            file_stem
        )

        file_stem = file_stem.strip(
            "_"
        )

        if not file_stem:
            return "tariff"

        return file_stem