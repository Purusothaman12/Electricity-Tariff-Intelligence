import hashlib
import re

from dataclasses import dataclass
from enum import StrEnum
from typing import Any
from typing import Iterable

from src.comparison.rate_comparator import (
    RateComparisonRecord,
    TariffComparisonResult
)
from src.loaders.rate_json_loader import (
    LoadedRateDocument
)
from src.models.rate import RateItem


class RAGChunkType(StrEnum):
    """
    Types of searchable chunks stored in the RAG system.
    """

    RATE = "RATE"
    COMPARISON = "COMPARISON"
    SECTION_COVERAGE = "SECTION_COVERAGE"


@dataclass(slots=True)
class RAGChunk:
    """
    Represents one searchable tariff knowledge chunk.
    """

    chunk_id: str
    chunk_type: RAGChunkType
    content: str
    metadata: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:

        return {
            "chunk_id": self.chunk_id,
            "chunk_type": (
                self.chunk_type.value
            ),
            "content": self.content,
            "metadata": dict(
                self.metadata
            )
        }


class RAGChunkBuilder:
    """
    Converts structured tariff information into searchable text
    chunks with metadata.

    Supported chunk types:

    1. RATE
       One tariff rate, reference, Rider value or historical value.

    2. COMPARISON
       One logical old-versus-new rate comparison.

    3. SECTION_COVERAGE
       Applicability and extraction coverage for one tariff section.
    """

    GENERIC_COLUMN_PATTERN = re.compile(
        r"^(?:COLUMN|COL)\s*\d+$",
        re.IGNORECASE
    )

    EFFECTIVE_DATE_LABEL_PATTERN = re.compile(
        r"^\(?\s*EFFECTIVE\s+DATE\s*\)?$",
        re.IGNORECASE
    )

    def build_document_chunks(
        self,
        document: LoadedRateDocument
    ) -> list[RAGChunk]:
        """
        Builds RATE and SECTION_COVERAGE chunks for one tariff
        document.
        """

        chunks = []

        for rate in document.rates:

            if self._is_structural_artifact(
                rate
            ):
                continue

            chunks.append(
                self._build_rate_chunk(
                    document=document,
                    rate=rate
                )
            )

        for section_data in (
            document.section_coverage
        ):

            if not isinstance(
                section_data,
                dict
            ):
                continue

            chunks.append(
                self._build_section_chunk(
                    document=document,
                    section_data=section_data
                )
            )

        return self._deduplicate_chunks(
            chunks
        )

    def build_comparison_chunks(
        self,
        comparison_result: TariffComparisonResult
    ) -> list[RAGChunk]:
        """
        Builds searchable chunks from tariff comparison records.
        """

        chunks = [
            self._build_comparison_chunk(
                comparison_result=(
                    comparison_result
                ),
                comparison=comparison
            )
            for comparison
            in comparison_result.comparisons
        ]

        return self._deduplicate_chunks(
            chunks
        )

    def build_all(
        self,
        documents: Iterable[
            LoadedRateDocument
        ],
        comparison_result: (
            TariffComparisonResult | None
        ) = None
    ) -> list[RAGChunk]:
        """
        Builds and combines all available RAG chunks.
        """

        chunks = []

        for document in documents:

            chunks.extend(
                self.build_document_chunks(
                    document
                )
            )

        if comparison_result is not None:

            chunks.extend(
                self.build_comparison_chunks(
                    comparison_result
                )
            )

        return self._deduplicate_chunks(
            chunks
        )

    def _build_rate_chunk(
        self,
        document: LoadedRateDocument,
        rate: RateItem
    ) -> RAGChunk:

        utility = self._get_utility_name(
            document
        )

        context = self._build_rate_context(
            rate
        )

        effective_date = (
            rate.effective_date
            or "Not available"
        )

        unit = (
            rate.unit
            or rate.normalized_unit
            or "Not specified"
        )

        resolution_method = str(
            rate.metadata.get(
                "effective_date_resolution",
                "UNKNOWN"
            )
        )

        content_lines = [
            "Tariff rate record.",
            f"Utility: {utility}.",
            (
                "Source tariff document: "
                f"{rate.source_file}."
            ),
            (
                "Schedule: "
                f"{rate.schedule_title} "
                f"({rate.schedule_id})."
            ),
            f"Category: {rate.category}.",
            f"Charge: {rate.charge_name}.",
            f"Rate value: {rate.value_text}.",
            f"Unit: {unit}.",
            (
                "Effective date: "
                f"{effective_date}."
            ),
            (
                "Value type: "
                f"{rate.value_kind}."
            )
        ]

        if context:

            content_lines.append(
                f"Context: {context}."
            )

        if rate.is_reference:

            content_lines.append(
                "This record is a reference to "
                "another tariff schedule or Rider."
            )

        content_lines.extend(
            [
                (
                    "Date resolution method: "
                    f"{resolution_method}."
                ),
                (
                    "Extraction source: "
                    f"{rate.source_method}."
                )
            ]
        )

        metadata = {
            "chunk_type": (
                RAGChunkType.RATE.value
            ),
            "utility": utility,
            "source_file": rate.source_file,
            "schema_version": (
                document.schema_version
            ),
            "schedule_id": rate.schedule_id,
            "schedule_title": (
                rate.schedule_title
            ),
            "category": rate.category,
            "charge_name": rate.charge_name,
            "normalized_charge_name": (
                rate.normalized_charge_name
            ),
            "value_text": rate.value_text,
            "numeric_value": (
                str(rate.numeric_value)
                if rate.numeric_value
                is not None
                else None
            ),
            "unit": rate.unit,
            "normalized_unit": (
                rate.normalized_unit
            ),
            "effective_date": (
                rate.effective_date
            ),
            "value_kind": rate.value_kind,
            "is_reference": (
                rate.is_reference
            ),
            "source_method": (
                rate.source_method
            ),
            "page_number": (
                rate.page_number
            ),
            "table_index": (
                rate.table_index
            ),
            "row_index": rate.row_index,
            "date_resolution": (
                resolution_method
            ),
            "context": context,
            "context_heading": (
                rate.attributes.get(
                    "context_heading",
                    ""
                )
            ),
            "parent_charge": (
                rate.attributes.get(
                    "parent_charge",
                    ""
                )
            ),
            "row_label": (
                rate.attributes.get(
                    "row_label",
                    ""
                )
            ),
            "column_header": (
                rate.attributes.get(
                    "column_header",
                    ""
                )
            )
        }

        chunk_id = self._create_chunk_id(
            namespace="rate",
            parts=[
                rate.source_file,
                rate.schedule_id,
                rate.schedule_title,
                rate.charge_name,
                rate.value_text,
                rate.unit,
                rate.effective_date,
                context,
                rate.page_number,
                rate.table_index,
                rate.row_index
            ]
        )

        return RAGChunk(
            chunk_id=chunk_id,
            chunk_type=RAGChunkType.RATE,
            content=" ".join(
                content_lines
            ),
            metadata=metadata
        )

    def _build_comparison_chunk(
        self,
        comparison_result: (
            TariffComparisonResult
        ),
        comparison: RateComparisonRecord
    ) -> RAGChunk:

        old_rate = comparison.old_rate
        new_rate = comparison.new_rate

        old_value = (
            old_rate.value_text
            if old_rate is not None
            else "Not present"
        )

        new_value = (
            new_rate.value_text
            if new_rate is not None
            else "Not present"
        )

        old_date = (
            old_rate.effective_date
            if (
                old_rate is not None
                and old_rate.effective_date
            )
            else "Not available"
        )

        new_date = (
            new_rate.effective_date
            if (
                new_rate is not None
                and new_rate.effective_date
            )
            else "Not available"
        )

        absolute_change = (
            str(comparison.absolute_change)
            if comparison.absolute_change
            is not None
            else "Not applicable"
        )

        percent_change = (
            str(comparison.percent_change)
            if comparison.percent_change
            is not None
            else "Not applicable"
        )

        content_lines = [
            "Tariff rate comparison.",
            (
                "Old tariff document: "
                f"{comparison_result.old_document.source_file}."
            ),
            (
                "New tariff document: "
                f"{comparison_result.new_document.source_file}."
            ),
            (
                "Schedule: "
                f"{comparison.schedule_title}."
            ),
            (
                "Charge: "
                f"{comparison.charge_name}."
            ),
            (
                "Comparison status: "
                f"{comparison.status.value}."
            ),
            (
                "Old rate: "
                f"{old_value}, effective "
                f"{old_date}."
            ),
            (
                "New rate: "
                f"{new_value}, effective "
                f"{new_date}."
            ),
            (
                "Absolute change: "
                f"{absolute_change}."
            ),
            (
                "Percentage change: "
                f"{percent_change}."
            )
        ]

        if comparison.unit:

            content_lines.append(
                f"Unit: {comparison.unit}."
            )

        if comparison.context:

            content_lines.append(
                (
                    "Comparison context: "
                    f"{comparison.context}."
                )
            )

        normalized_charge_name = ""

        if new_rate is not None:

            normalized_charge_name = (
                new_rate.normalized_charge_name
            )

        elif old_rate is not None:

            normalized_charge_name = (
                old_rate.normalized_charge_name
            )

        metadata = {
            "chunk_type": (
                RAGChunkType.COMPARISON.value
            ),
            "old_source_file": (
                comparison_result
                .old_document
                .source_file
            ),
            "new_source_file": (
                comparison_result
                .new_document
                .source_file
            ),
            "schedule_id": (
                comparison.schedule_id
            ),
            "schedule_title": (
                comparison.schedule_title
            ),
            "category": (
                comparison.category
            ),
            "charge_name": (
                comparison.charge_name
            ),
            "normalized_charge_name": (
                normalized_charge_name
            ),
            "unit": comparison.unit,
            "context": (
                comparison.context
            ),
            "status": (
                comparison.status.value
            ),
            "old_value_text": (
                old_rate.value_text
                if old_rate is not None
                else None
            ),
            "new_value_text": (
                new_rate.value_text
                if new_rate is not None
                else None
            ),
            "old_numeric_value": (
                str(old_rate.numeric_value)
                if (
                    old_rate is not None
                    and old_rate.numeric_value
                    is not None
                )
                else None
            ),
            "new_numeric_value": (
                str(new_rate.numeric_value)
                if (
                    new_rate is not None
                    and new_rate.numeric_value
                    is not None
                )
                else None
            ),
            "old_effective_date": (
                old_rate.effective_date
                if old_rate is not None
                else ""
            ),
            "new_effective_date": (
                new_rate.effective_date
                if new_rate is not None
                else ""
            ),
            "absolute_change": (
                str(comparison.absolute_change)
                if comparison.absolute_change
                is not None
                else None
            ),
            "percent_change": (
                str(comparison.percent_change)
                if comparison.percent_change
                is not None
                else None
            )
        }

        chunk_id = self._create_chunk_id(
            namespace="comparison",
            parts=[
                comparison_result
                .old_document
                .source_file,
                comparison_result
                .new_document
                .source_file,
                *comparison.identity,
                comparison.status.value,
                old_value,
                new_value
            ]
        )

        return RAGChunk(
            chunk_id=chunk_id,
            chunk_type=(
                RAGChunkType.COMPARISON
            ),
            content=" ".join(
                content_lines
            ),
            metadata=metadata
        )

    def _build_section_chunk(
        self,
        document: LoadedRateDocument,
        section_data: dict[str, Any]
    ) -> RAGChunk:

        utility = self._get_utility_name(
            document
        )

        section_id = self._string(
            section_data.get(
                "section_id"
            )
        )

        section_title = self._string(
            section_data.get(
                "section_title",
                section_data.get(
                    "title",
                    ""
                )
            )
        )

        category = self._string(
            section_data.get(
                "category"
            )
        )

        applicability_status = (
            self._string(
                section_data.get(
                    "applicability_status"
                )
            )
            or "UNKNOWN"
        )

        coverage_status = (
            self._string(
                section_data.get(
                    "coverage_status"
                )
            )
            or "UNKNOWN"
        )

        applicability_reason = self._string(
            section_data.get(
                "applicability_reason"
            )
        )

        rate_count = section_data.get(
            "rate_count",
            0
        )

        table_count = section_data.get(
            "table_count",
            0
        )

        content_lines = [
            "Tariff section coverage record.",
            f"Utility: {utility}.",
            (
                "Source tariff document: "
                f"{document.source_file}."
            ),
            (
                "Section: "
                f"{section_title} "
                f"({section_id})."
            ),
            f"Category: {category}.",
            (
                "Applicability status: "
                f"{applicability_status}."
            ),
            (
                "Extraction coverage status: "
                f"{coverage_status}."
            ),
            (
                "Extracted rate count: "
                f"{rate_count}."
            ),
            (
                "Extracted table count: "
                f"{table_count}."
            )
        ]

        if applicability_reason:

            content_lines.append(
                (
                    "Applicability reason: "
                    f"{applicability_reason}."
                )
            )

        metadata = {
            "chunk_type": (
                RAGChunkType
                .SECTION_COVERAGE
                .value
            ),
            "utility": utility,
            "source_file": (
                document.source_file
            ),
            "schema_version": (
                document.schema_version
            ),
            "section_id": section_id,
            "section_title": section_title,
            "category": category,
            "applicability_status": (
                applicability_status
            ),
            "coverage_status": (
                coverage_status
            ),
            "applicability_reason": (
                applicability_reason
            ),
            "rate_count": rate_count,
            "table_count": table_count
        }

        chunk_id = self._create_chunk_id(
            namespace="section",
            parts=[
                document.source_file,
                section_id,
                section_title,
                applicability_status,
                coverage_status
            ]
        )

        return RAGChunk(
            chunk_id=chunk_id,
            chunk_type=(
                RAGChunkType
                .SECTION_COVERAGE
            ),
            content=" ".join(
                content_lines
            ),
            metadata=metadata
        )

    def _build_rate_context(
        self,
        rate: RateItem
    ) -> str:

        context_values = []

        for key in (
            "context_heading",
            "parent_charge",
            "row_label",
            "column_header"
        ):

            value = self._string(
                rate.attributes.get(
                    key
                )
            )

            if not value:
                continue

            if value not in context_values:

                context_values.append(
                    value
                )

        return " | ".join(
            context_values
        )

    def _is_structural_artifact(
        self,
        rate: RateItem
    ) -> bool:
        """
        Removes only high-confidence table-structure artifacts.
        """

        if rate.metadata.get(
            "section_effective_date_"
            "resolution_skipped",
            False
        ):

            return True

        table_structure = self._string(
            rate.metadata.get(
                "table_structure"
            )
        ).upper()

        if table_structure != "MATRIX":
            return False

        charge_name = self._string(
            rate.charge_name
        )

        row_label = self._string(
            rate.attributes.get(
                "row_label"
            )
        )

        if self.GENERIC_COLUMN_PATTERN.fullmatch(
            charge_name
        ):

            return True

        if (
            self.EFFECTIVE_DATE_LABEL_PATTERN
            .fullmatch(
                row_label
            )
        ):

            return True

        return False

    def _get_utility_name(
        self,
        document: LoadedRateDocument
    ) -> str:

        for key in (
            "utility",
            "utility_name",
            "company",
            "company_name"
        ):

            value = self._string(
                document.source_document.get(
                    key
                )
            )

            if value:
                return value

        source_file = document.source_file

        if not source_file:
            return "Unknown Utility"

        utility_name = source_file.split(
            "_",
            maxsplit=1
        )[0]

        return utility_name.replace(
            "-",
            " "
        ).strip()

    def _create_chunk_id(
        self,
        namespace: str,
        parts: Iterable[Any]
    ) -> str:

        normalized_parts = [
            self._string(
                part
            ).upper()
            for part in parts
        ]

        raw_identity = "||".join(
            normalized_parts
        )

        digest = hashlib.sha256(
            raw_identity.encode(
                "utf-8"
            )
        ).hexdigest()[:24]

        return (
            f"{namespace.lower()}_"
            f"{digest}"
        )

    def _deduplicate_chunks(
        self,
        chunks: Iterable[RAGChunk]
    ) -> list[RAGChunk]:

        unique_chunks = {}

        for chunk in chunks:

            unique_chunks[
                chunk.chunk_id
            ] = chunk

        return sorted(
            unique_chunks.values(),
            key=lambda chunk: (
                chunk.chunk_type.value,
                chunk.chunk_id
            )
        )

    def _string(
        self,
        value: Any
    ) -> str:

        if value is None:
            return ""

        text = str(
            value
        )

        text = text.replace(
            "\u00a0",
            " "
        )

        text = re.sub(
            r"\s+",
            " ",
            text
        )

        return text.strip()