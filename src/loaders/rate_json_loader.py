import json

from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from src.models.rate import RateItem


@dataclass(slots=True)
class LoadedRateDocument:
    """
    Represents one tariff-rate JSON export loaded into memory.
    """

    json_path: Path
    schema_version: str
    source_document: dict[str, Any]
    summary: dict[str, Any]
    coverage_summary: dict[str, Any]
    section_coverage: list[dict[str, Any]]
    not_applicable_sections: list[dict[str, Any]]
    review_sections: list[dict[str, Any]]
    rates: list[RateItem]

    @property
    def source_file(self) -> str:

        return str(
            self.source_document.get(
                "file_name",
                ""
            )
        )

    @property
    def rate_count(self) -> int:

        return len(
            self.rates
        )

    @property
    def unresolved_rates(self) -> list[RateItem]:

        return [
            rate
            for rate in self.rates
            if not rate.effective_date
        ]

    @property
    def unresolved_rate_count(self) -> int:

        return len(
            self.unresolved_rates
        )

    @property
    def schedule_count(self) -> int:

        return len(
            {
                (
                    rate.schedule_id,
                    rate.schedule_title
                )
                for rate in self.rates
            }
        )

    @property
    def category_counts(self) -> dict[str, int]:

        counts = Counter(
            rate.category
            for rate in self.rates
        )

        return dict(
            counts
        )

    @property
    def date_resolution_counts(
        self
    ) -> dict[str, int]:

        counts = Counter(
            str(
                rate.metadata.get(
                    "effective_date_resolution",
                    "UNKNOWN"
                )
            )
            for rate in self.rates
        )

        return dict(
            counts
        )

    def get_schedule_rates(
        self,
        schedule_id: str
    ) -> list[RateItem]:
        """
        Returns every rate belonging to one schedule ID.
        """

        normalized_schedule_id = (
            schedule_id.strip().upper()
        )

        return [
            rate
            for rate in self.rates
            if (
                rate.schedule_id
                .strip()
                .upper()
                == normalized_schedule_id
            )
        ]

    def to_summary(self) -> dict[str, Any]:

        return {
            "source_file": self.source_file,
            "json_path": str(
                self.json_path
            ),
            "schema_version": (
                self.schema_version
            ),
            "rates": self.rate_count,
            "schedules": self.schedule_count,
            "unresolved_rates": (
                self.unresolved_rate_count
            ),
            "category_counts": (
                self.category_counts
            ),
            "date_resolution_counts": (
                self.date_resolution_counts
            ),
            "not_applicable_sections": len(
                self.not_applicable_sections
            ),
            "review_sections": len(
                self.review_sections
            )
        }


class RateJSONLoader:
    """
    Loads tariff-rate JSON exports created by RateJSONExporter.

    Derived properties such as numeric_value, value_kind and
    normalized_charge_name are recalculated by RateItem rather
    than copied from the JSON.
    """

    SUPPORTED_SCHEMA_VERSIONS = {
        "1.1"
    }

    def load_directory(
        self,
        input_directory: str = "output/rates"
    ) -> list[LoadedRateDocument]:
        """
        Loads every *_rates.json file from a directory.
        """

        directory = Path(
            input_directory
        )

        if not directory.exists():

            raise FileNotFoundError(
                "Rate JSON directory does not exist: "
                f"{directory}"
            )

        if not directory.is_dir():

            raise NotADirectoryError(
                "Rate JSON path is not a directory: "
                f"{directory}"
            )

        json_paths = sorted(
            directory.glob(
                "*_rates.json"
            ),
            key=lambda path: (
                path.name.lower()
            )
        )

        if not json_paths:

            raise FileNotFoundError(
                "No *_rates.json files were found "
                f"inside: {directory}"
            )

        return [
            self.load_file(
                json_path
            )
            for json_path in json_paths
        ]

    def load_file(
        self,
        json_path: str | Path
    ) -> LoadedRateDocument:
        """
        Loads one tariff-rate JSON file.
        """

        path = Path(
            json_path
        )

        if not path.exists():

            raise FileNotFoundError(
                "Rate JSON file does not exist: "
                f"{path}"
            )

        if not path.is_file():

            raise IsADirectoryError(
                "Expected a JSON file but received "
                f"a directory: {path}"
            )

        try:

            with path.open(
                mode="r",
                encoding="utf-8"
            ) as input_file:

                payload = json.load(
                    input_file
                )

        except json.JSONDecodeError as error:

            raise ValueError(
                "Invalid JSON content in "
                f"{path}: {error}"
            ) from error

        self._validate_payload(
            payload=payload,
            json_path=path
        )

        source_document = dict(
            payload.get(
                "source_document",
                {}
            )
        )

        source_file = str(
            source_document.get(
                "file_name",
                ""
            )
        ).strip()

        rates = [
            self._deserialize_rate(
                rate_data=rate_data,
                fallback_source_file=(
                    source_file
                )
            )
            for rate_data in payload.get(
                "rates",
                []
            )
        ]

        return LoadedRateDocument(
            json_path=path,
            schema_version=str(
                payload.get(
                    "schema_version",
                    ""
                )
            ),
            source_document=(
                source_document
            ),
            summary=dict(
                payload.get(
                    "summary",
                    {}
                )
            ),
            coverage_summary=dict(
                payload.get(
                    "coverage_summary",
                    {}
                )
            ),
            section_coverage=self._copy_list(
                payload.get(
                    "section_coverage",
                    []
                )
            ),
            not_applicable_sections=(
                self._copy_list(
                    payload.get(
                        "not_applicable_sections",
                        []
                    )
                )
            ),
            review_sections=self._copy_list(
                payload.get(
                    "review_sections",
                    []
                )
            ),
            rates=rates
        )

    def _validate_payload(
        self,
        payload: Any,
        json_path: Path
    ) -> None:

        if not isinstance(
            payload,
            dict
        ):

            raise ValueError(
                "The JSON root must be an object: "
                f"{json_path}"
            )

        schema_version = str(
            payload.get(
                "schema_version",
                ""
            )
        ).strip()

        if (
            schema_version
            not in self.SUPPORTED_SCHEMA_VERSIONS
        ):

            supported_versions = ", ".join(
                sorted(
                    self.SUPPORTED_SCHEMA_VERSIONS
                )
            )

            raise ValueError(
                "Unsupported rate JSON schema "
                f"version '{schema_version}' in "
                f"{json_path}. Supported versions: "
                f"{supported_versions}"
            )

        source_document = payload.get(
            "source_document"
        )

        if not isinstance(
            source_document,
            dict
        ):

            raise ValueError(
                "source_document must be an "
                f"object in: {json_path}"
            )

        source_file = str(
            source_document.get(
                "file_name",
                ""
            )
        ).strip()

        if not source_file:

            raise ValueError(
                "source_document.file_name is "
                f"missing in: {json_path}"
            )

        rates = payload.get(
            "rates"
        )

        if not isinstance(
            rates,
            list
        ):

            raise ValueError(
                "rates must be a list in: "
                f"{json_path}"
            )

        for index, rate_data in enumerate(
            rates
        ):

            if not isinstance(
                rate_data,
                dict
            ):

                raise ValueError(
                    "Each rate must be an object. "
                    f"Invalid item at index {index} "
                    f"in: {json_path}"
                )

    def _deserialize_rate(
        self,
        rate_data: dict[str, Any],
        fallback_source_file: str
    ) -> RateItem:

        attributes = rate_data.get(
            "attributes",
            {}
        )

        metadata = rate_data.get(
            "metadata",
            {}
        )

        if not isinstance(
            attributes,
            dict
        ):
            attributes = {}

        if not isinstance(
            metadata,
            dict
        ):
            metadata = {}

        source_file = self._string(
            rate_data.get(
                "source_file"
            )
        )

        if not source_file:
            source_file = (
                fallback_source_file
            )

        return RateItem(
            schedule_id=self._string(
                rate_data.get(
                    "schedule_id"
                )
            ),
            schedule_title=self._string(
                rate_data.get(
                    "schedule_title"
                )
            ),
            category=self._string(
                rate_data.get(
                    "category"
                )
            ),
            source_file=source_file,
            charge_name=self._string(
                rate_data.get(
                    "charge_name"
                )
            ),
            value_text=self._string(
                rate_data.get(
                    "value_text"
                )
            ),
            unit=self._string(
                rate_data.get(
                    "unit"
                )
            ),
            source_method=self._string(
                rate_data.get(
                    "source_method"
                )
            ),
            page_number=self._optional_integer(
                rate_data.get(
                    "page_number"
                )
            ),
            table_index=self._optional_integer(
                rate_data.get(
                    "table_index"
                )
            ),
            row_index=self._optional_integer(
                rate_data.get(
                    "row_index"
                )
            ),
            effective_date=self._string(
                rate_data.get(
                    "effective_date"
                )
            ),
            attributes=dict(
                attributes
            ),
            metadata=dict(
                metadata
            )
        )

    def _copy_list(
        self,
        value: Any
    ) -> list[dict[str, Any]]:

        if not isinstance(
            value,
            list
        ):
            return []

        copied_items = []

        for item in value:

            if isinstance(
                item,
                dict
            ):

                copied_items.append(
                    dict(item)
                )

        return copied_items

    def _optional_integer(
        self,
        value: Any
    ) -> int | None:

        if value is None:
            return None

        if isinstance(
            value,
            bool
        ):
            return None

        try:

            return int(
                value
            )

        except (
            TypeError,
            ValueError
        ):

            return None

    def _string(
        self,
        value: Any
    ) -> str:

        if value is None:
            return ""

        return str(
            value
        ).strip()