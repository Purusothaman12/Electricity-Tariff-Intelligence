import re

from collections import Counter
from dataclasses import dataclass
from datetime import date
from datetime import datetime
from decimal import Decimal
from enum import StrEnum
from typing import Any

from src.comparison.charge_identity_normalizer import (
    ChargeIdentityNormalizer
)
from src.loaders.rate_json_loader import (
    LoadedRateDocument
)
from src.models.rate import RateItem


class RateChangeStatus(StrEnum):
    """
    Describes how a logical rate changed between two tariffs.
    """

    ADDED = "ADDED"
    REMOVED = "REMOVED"
    INCREASED = "INCREASED"
    DECREASED = "DECREASED"
    UNCHANGED = "UNCHANGED"
    CHANGED = "CHANGED"


@dataclass(slots=True)
class RateComparisonRecord:
    """
    Represents one logical tariff-rate comparison.
    """

    identity: tuple[str, ...]
    schedule_id: str
    schedule_title: str
    category: str
    charge_name: str
    unit: str
    context: str
    old_rate: RateItem | None
    new_rate: RateItem | None
    status: RateChangeStatus
    absolute_change: Decimal | None = None
    percent_change: Decimal | None = None

    @property
    def normalized_charge_name(self) -> str:

        if self.new_rate is not None:

            return (
                self.new_rate
                .normalized_charge_name
            )

        if self.old_rate is not None:

            return (
                self.old_rate
                .normalized_charge_name
            )

        return ""

    @property
    def old_value(self) -> Decimal | None:

        if self.old_rate is None:
            return None

        return self.old_rate.numeric_value

    @property
    def new_value(self) -> Decimal | None:

        if self.new_rate is None:
            return None

        return self.new_rate.numeric_value

    @property
    def old_effective_date(self) -> str:

        if self.old_rate is None:
            return ""

        return self.old_rate.effective_date

    @property
    def new_effective_date(self) -> str:

        if self.new_rate is None:
            return ""

        return self.new_rate.effective_date

    def to_dict(self) -> dict[str, Any]:

        return {
            "comparison_key": list(
                self.identity
            ),
            "schedule_id": self.schedule_id,
            "schedule_title": (
                self.schedule_title
            ),
            "category": self.category,
            "charge_name": self.charge_name,
            "normalized_charge_name": (
                self.normalized_charge_name
            ),
            "unit": self.unit,
            "context": self.context,
            "status": self.status.value,
            "old_value": (
                str(self.old_value)
                if self.old_value is not None
                else None
            ),
            "new_value": (
                str(self.new_value)
                if self.new_value is not None
                else None
            ),
            "absolute_change": (
                str(self.absolute_change)
                if self.absolute_change is not None
                else None
            ),
            "percent_change": (
                str(self.percent_change)
                if self.percent_change is not None
                else None
            ),
            "old_effective_date": (
                self.old_effective_date
            ),
            "new_effective_date": (
                self.new_effective_date
            ),
            "old_rate": (
                self.old_rate.to_dict()
                if self.old_rate is not None
                else None
            ),
            "new_rate": (
                self.new_rate.to_dict()
                if self.new_rate is not None
                else None
            )
        }


@dataclass(slots=True)
class TariffComparisonResult:
    """
    Stores the complete comparison between two tariffs.
    """

    old_document: LoadedRateDocument
    new_document: LoadedRateDocument
    comparisons: list[RateComparisonRecord]
    old_snapshot_count: int
    new_snapshot_count: int
    excluded_old_artifacts: int
    excluded_new_artifacts: int

    @property
    def status_counts(self) -> dict[str, int]:

        counts = Counter(
            comparison.status.value
            for comparison
            in self.comparisons
        )

        return {
            status.value: counts.get(
                status.value,
                0
            )
            for status in RateChangeStatus
        }

    @property
    def comparison_count(self) -> int:

        return len(
            self.comparisons
        )

    @property
    def changed_count(self) -> int:

        changed_statuses = {
            RateChangeStatus.INCREASED,
            RateChangeStatus.DECREASED,
            RateChangeStatus.CHANGED
        }

        return sum(
            1
            for comparison in self.comparisons
            if (
                comparison.status
                in changed_statuses
            )
        )

    @property
    def added_count(self) -> int:

        return self.status_counts[
            RateChangeStatus.ADDED.value
        ]

    @property
    def removed_count(self) -> int:

        return self.status_counts[
            RateChangeStatus.REMOVED.value
        ]

    @property
    def unchanged_count(self) -> int:

        return self.status_counts[
            RateChangeStatus.UNCHANGED.value
        ]

    def get_schedule_comparisons(
        self,
        schedule_id: str
    ) -> list[RateComparisonRecord]:
        """
        Returns comparisons where the old or new rate belongs
        to the supplied schedule ID.
        """

        normalized_schedule_id = (
            schedule_id.strip().upper()
        )

        results = []

        for comparison in self.comparisons:

            old_schedule_id = ""

            if comparison.old_rate is not None:

                old_schedule_id = (
                    comparison.old_rate
                    .schedule_id
                    .strip()
                    .upper()
                )

            new_schedule_id = ""

            if comparison.new_rate is not None:

                new_schedule_id = (
                    comparison.new_rate
                    .schedule_id
                    .strip()
                    .upper()
                )

            if normalized_schedule_id in {
                old_schedule_id,
                new_schedule_id
            }:

                results.append(
                    comparison
                )

        return results

    def to_summary(self) -> dict[str, Any]:

        return {
            "old_source_file": (
                self.old_document.source_file
            ),
            "new_source_file": (
                self.new_document.source_file
            ),
            "old_snapshot_rates": (
                self.old_snapshot_count
            ),
            "new_snapshot_rates": (
                self.new_snapshot_count
            ),
            "comparison_records": (
                self.comparison_count
            ),
            "changed_records": (
                self.changed_count
            ),
            "added_records": (
                self.added_count
            ),
            "removed_records": (
                self.removed_count
            ),
            "unchanged_records": (
                self.unchanged_count
            ),
            "excluded_old_artifacts": (
                self.excluded_old_artifacts
            ),
            "excluded_new_artifacts": (
                self.excluded_new_artifacts
            ),
            "status_counts": (
                self.status_counts
            )
        }


class RateComparator:
    """
    Compares the latest logical-rate snapshot from two tariff
    documents.

    Matching rules:

    - Physical section IDs are excluded because Rider IDs can
      move between tariff versions.
    - ChargeIdentityNormalizer creates stable cross-document
      charge identities.
    - Units are ignored for reference records such as
      'See Rider DCRF'.
    - Generic section headings are excluded.
    - Meaningful Lighting and matrix contexts are retained.
    - Historical effective dates are excluded from identity.
    """

    GENERIC_COLUMN_PATTERN = re.compile(
        r"^(?:COLUMN|COL)\s*\d+$",
        re.IGNORECASE
    )

    EFFECTIVE_DATE_LABEL_PATTERN = re.compile(
        r"^\(?\s*EFFECTIVE\s+DATE\s*\)?$",
        re.IGNORECASE
    )

    ROMAN_HEADING_PATTERN = re.compile(
        r"^[IVXLCDM]+\.\s*",
        re.IGNORECASE
    )

    DATE_RESOLUTION_PRIORITY = {
        "EXPLICIT": 5,
        "SECTION_HEADER": 4,
        "SCHEDULE_CONSENSUS": 3,
        "CATEGORY_CONSENSUS": 2,
        "UNRESOLVED": 0,
        "UNKNOWN": 0
    }

    GENERIC_CONTEXTS = {
        "BASE RATE CHARGE",
        "BASE RATE CHARGES",
        "BASE CHARGE",
        "BASE CHARGES",
        "TRANSMISSION AND DISTRIBUTION CHARGE",
        "TRANSMISSION AND DISTRIBUTION CHARGES",
        "DELIVERY SYSTEM CHARGE",
        "DELIVERY SYSTEM CHARGES",
        "RATE SCHEDULE",
        "RATE SCHEDULES",
        "RATE",
        "RATES",
        "CHARGE",
        "CHARGES",
        "APPLICABLE"
    }

    UNIT_ONLY_HEADERS = {
        "$/KWH",
        "$/KW",
        "$/BILLINGKW",
        "$/DISTRIBUTIONSYSTEMBILLINGKW",
        "$/NCPKW",
        "$/4CPKW",
        "PERKWH",
        "PERKW",
        "PERBILLINGKW",
        "PERDISTRIBUTIONSYSTEMBILLINGKW",
        "PERNCPKW",
        "PER4CPKW",
        "%",
        "PERCENT"
    }

    def __init__(
        self,
        charge_identity_normalizer: (
            ChargeIdentityNormalizer | None
        ) = None
    ) -> None:

        self.charge_identity_normalizer = (
            charge_identity_normalizer
            or ChargeIdentityNormalizer()
        )

    def compare(
        self,
        old_document: LoadedRateDocument,
        new_document: LoadedRateDocument
    ) -> TariffComparisonResult:
        """
        Compares two loaded tariff documents.
        """

        (
            old_snapshot,
            excluded_old_artifacts
        ) = self._build_current_snapshot(
            old_document.rates
        )

        (
            new_snapshot,
            excluded_new_artifacts
        ) = self._build_current_snapshot(
            new_document.rates
        )

        all_identities = sorted(
            set(old_snapshot)
            | set(new_snapshot)
        )

        comparisons = []

        for identity in all_identities:

            comparisons.append(
                self._compare_rate_pair(
                    identity=identity,
                    old_rate=old_snapshot.get(
                        identity
                    ),
                    new_rate=new_snapshot.get(
                        identity
                    )
                )
            )

        comparisons.sort(
            key=self._comparison_sort_key
        )

        return TariffComparisonResult(
            old_document=old_document,
            new_document=new_document,
            comparisons=comparisons,
            old_snapshot_count=len(
                old_snapshot
            ),
            new_snapshot_count=len(
                new_snapshot
            ),
            excluded_old_artifacts=(
                excluded_old_artifacts
            ),
            excluded_new_artifacts=(
                excluded_new_artifacts
            )
        )

    def _build_current_snapshot(
        self,
        rates: list[RateItem]
    ) -> tuple[
        dict[tuple[str, ...], RateItem],
        int
    ]:
        """
        Groups rates by logical identity and selects the latest
        trustworthy record from each group.
        """

        grouped_rates: dict[
            tuple[str, ...],
            list[RateItem]
        ] = {}

        excluded_artifact_count = 0

        for rate in rates:

            if self._is_structural_artifact(
                rate
            ):

                excluded_artifact_count += 1

                continue

            identity = self._create_identity(
                rate
            )

            grouped_rates.setdefault(
                identity,
                []
            )

            grouped_rates[
                identity
            ].append(
                rate
            )

        snapshot = {}

        for identity, candidates in (
            grouped_rates.items()
        ):

            snapshot[
                identity
            ] = max(
                candidates,
                key=self._selection_score
            )

        return (
            snapshot,
            excluded_artifact_count
        )

    def _compare_rate_pair(
        self,
        identity: tuple[str, ...],
        old_rate: RateItem | None,
        new_rate: RateItem | None
    ) -> RateComparisonRecord:

        representative_rate = (
            new_rate
            if new_rate is not None
            else old_rate
        )

        if representative_rate is None:

            raise ValueError(
                "At least one rate is required "
                "for comparison."
            )

        if old_rate is None:

            status = RateChangeStatus.ADDED
            absolute_change = None
            percent_change = None

        elif new_rate is None:

            status = RateChangeStatus.REMOVED
            absolute_change = None
            percent_change = None

        else:

            (
                status,
                absolute_change,
                percent_change
            ) = self._compare_values(
                old_rate=old_rate,
                new_rate=new_rate
            )

        return RateComparisonRecord(
            identity=identity,
            schedule_id=(
                representative_rate.schedule_id
            ),
            schedule_title=(
                representative_rate.schedule_title
            ),
            category=(
                representative_rate.category
            ),
            charge_name=(
                representative_rate.charge_name
            ),
            unit=self._comparison_unit(
                representative_rate
            ),
            context=self._comparison_context(
                representative_rate
            ),
            old_rate=old_rate,
            new_rate=new_rate,
            status=status,
            absolute_change=absolute_change,
            percent_change=percent_change
        )

    def _compare_values(
        self,
        old_rate: RateItem,
        new_rate: RateItem
    ) -> tuple[
        RateChangeStatus,
        Decimal | None,
        Decimal | None
    ]:

        old_numeric = old_rate.numeric_value
        new_numeric = new_rate.numeric_value

        if (
            old_numeric is not None
            and new_numeric is not None
        ):

            absolute_change = (
                new_numeric
                - old_numeric
            )

            if absolute_change > 0:

                status = (
                    RateChangeStatus.INCREASED
                )

            elif absolute_change < 0:

                status = (
                    RateChangeStatus.DECREASED
                )

            else:

                status = (
                    RateChangeStatus.UNCHANGED
                )

            percent_change = None

            if old_numeric != 0:

                percent_change = (
                    absolute_change
                    / abs(old_numeric)
                    * Decimal("100")
                )

            return (
                status,
                absolute_change,
                percent_change
            )

        old_text = self._normalize_value_text(
            old_rate.value_text
        )

        new_text = self._normalize_value_text(
            new_rate.value_text
        )

        if old_text == new_text:

            status = (
                RateChangeStatus.UNCHANGED
            )

        else:

            status = (
                RateChangeStatus.CHANGED
            )

        return (
            status,
            None,
            None
        )

    def _create_identity(
        self,
        rate: RateItem
    ) -> tuple[str, ...]:
        """
        Creates a stable cross-document logical identity.
        """

        charge_identity = (
            self.charge_identity_normalizer
            .normalize(
                rate
            )
        )

        return (
            self._schedule_identity(
                rate
            ),
            self._normalize_text(
                rate.category
            ),
            self._normalize_text(
                charge_identity
            ),
            self._normalize_text(
                self._comparison_unit(
                    rate
                )
            ),
            self._context_identity(
                rate
            ),
            self._matrix_row_identity(
                rate
            ),
            self._matrix_column_identity(
                rate
            )
        )

    def _schedule_identity(
        self,
        rate: RateItem
    ) -> str:
        """
        Builds a stable identity from the schedule title.
        """

        title = self._normalize_text(
            rate.schedule_title
        )

        if not title:

            return self._normalize_text(
                rate.schedule_id
            )

        title = re.sub(
            r"^RIDER\s+",
            "",
            title
        )

        title = re.sub(
            r"\s*-\s*",
            " - ",
            title
        )

        title = re.sub(
            r"\s+",
            " ",
            title
        )

        return title.strip()

    def _context_identity(
        self,
        rate: RateItem
    ) -> str:
        """
        Keeps meaningful sub-rate contexts while removing generic
        headings whose wording may change between tariff versions.
        """

        context_values = []

        for key in (
            "context_heading",
            "parent_charge"
        ):

            raw_value = rate.attributes.get(
                key,
                ""
            )

            canonical_value = (
                self._canonical_context(
                    raw_value
                )
            )

            if not canonical_value:
                continue

            if (
                canonical_value
                not in context_values
            ):

                context_values.append(
                    canonical_value
                )

        return " | ".join(
            context_values
        )

    def _canonical_context(
        self,
        value: Any
    ) -> str:
        """
        Normalizes and filters one context heading.
        """

        normalized_value = (
            self._normalize_text(
                value
            )
        )

        if not normalized_value:
            return ""

        normalized_value = (
            self.ROMAN_HEADING_PATTERN.sub(
                "",
                normalized_value
            )
        )

        normalized_value = re.sub(
            r"^\d+(?:\.\d+)*[.)]?\s*",
            "",
            normalized_value
        )

        normalized_value = re.sub(
            r"\s+",
            " ",
            normalized_value
        ).strip()

        if (
            normalized_value
            in self.GENERIC_CONTEXTS
        ):

            return ""

        return normalized_value

    def _matrix_row_identity(
        self,
        rate: RateItem
    ) -> str:
        """
        Uses row labels only for meaningful matrix dimensions.
        """

        if (
            self._table_structure(
                rate
            )
            != "MATRIX"
        ):
            return ""

        value = self._clean_text(
            rate.attributes.get(
                "row_label",
                ""
            )
        )

        if not value:
            return ""

        if (
            self.EFFECTIVE_DATE_LABEL_PATTERN
            .fullmatch(
                value
            )
        ):
            return ""

        if self._parse_date(
            value
        ) is not None:
            return ""

        normalized_value = (
            self._normalize_text(
                value
            )
        )

        if (
            normalized_value
            == self._normalize_text(
                rate.normalized_charge_name
            )
        ):
            return ""

        return normalized_value

    def _matrix_column_identity(
        self,
        rate: RateItem
    ) -> str:
        """
        Uses column labels only for meaningful matrix dimensions.
        """

        if (
            self._table_structure(
                rate
            )
            != "MATRIX"
        ):
            return ""

        value = self._clean_text(
            rate.attributes.get(
                "column_header",
                ""
            )
        )

        if not value:
            return ""

        if (
            self.EFFECTIVE_DATE_LABEL_PATTERN
            .fullmatch(
                value
            )
        ):
            return ""

        if self._parse_date(
            value
        ) is not None:
            return ""

        normalized_value = (
            self._normalize_text(
                value
            )
        )

        if (
            normalized_value
            == self._normalize_text(
                rate.normalized_charge_name
            )
        ):
            return ""

        if self._is_unit_only_header(
            normalized_value
        ):

            return ""

        return normalized_value

    def _comparison_unit(
        self,
        rate: RateItem
    ) -> str:
        """
        Returns the normalized comparison unit.

        Reference values do not use a unit in their identity.
        """

        if rate.is_reference:
            return ""

        normalized_unit = (
            rate.normalized_unit
        )

        if normalized_unit:
            return normalized_unit

        value_text = self._normalize_text(
            rate.value_text
        )

        if "%" in value_text:
            return "%"

        unit_patterns = (
            (
                r"PER\s+RETAIL\s+CUSTOMER",
                "PER RETAIL CUSTOMER"
            ),
            (
                r"PER\s+MONTH",
                "PER MONTH"
            ),
            (
                r"PER\s+DISTRIBUTION\s+SYSTEM"
                r"\s+BILLING\s+KW",
                "PER DISTRIBUTION SYSTEM BILLING KW"
            ),
            (
                r"PER\s+BILLING\s+KW",
                "PER BILLING KW"
            ),
            (
                r"PER\s+NCP\s+KW",
                "PER NCP KW"
            ),
            (
                r"PER\s+4CP\s+KW",
                "PER 4CP KW"
            ),
            (
                r"PER\s+KWH",
                "PER KWH"
            ),
            (
                r"PER\s+KW",
                "PER KW"
            )
        )

        for pattern, unit in unit_patterns:

            if re.search(
                pattern,
                value_text
            ):

                return unit

        return ""

    def _comparison_context(
        self,
        rate: RateItem
    ) -> str:

        context_values = []

        context_identity = (
            self._context_identity(
                rate
            )
        )

        if context_identity:

            context_values.append(
                context_identity
            )

        matrix_row = (
            self._matrix_row_identity(
                rate
            )
        )

        if (
            matrix_row
            and matrix_row
            not in context_values
        ):

            context_values.append(
                matrix_row
            )

        matrix_column = (
            self._matrix_column_identity(
                rate
            )
        )

        if (
            matrix_column
            and matrix_column
            not in context_values
        ):

            context_values.append(
                matrix_column
            )

        return " | ".join(
            context_values
        )

    def _is_structural_artifact(
        self,
        rate: RateItem
    ) -> bool:

        if rate.metadata.get(
            "section_effective_date_"
            "resolution_skipped",
            False
        ):

            return True

        if (
            self._table_structure(
                rate
            )
            != "MATRIX"
        ):
            return False

        charge_name = self._clean_text(
            rate.charge_name
        )

        row_label = self._clean_text(
            rate.attributes.get(
                "row_label",
                ""
            )
        )

        if (
            self.EFFECTIVE_DATE_LABEL_PATTERN
            .fullmatch(
                row_label
            )
        ):

            return True

        if (
            self.GENERIC_COLUMN_PATTERN
            .fullmatch(
                charge_name
            )
        ):

            return True

        return False

    def _selection_score(
        self,
        rate: RateItem
    ) -> tuple[
        int,
        int,
        int,
        int,
        int,
        int
    ]:

        parsed_date = self._parse_date(
            rate.effective_date
        )

        date_score = (
            parsed_date.toordinal()
            if parsed_date is not None
            else 0
        )

        resolution_method = (
            self._normalize_text(
                rate.metadata.get(
                    "effective_date_resolution",
                    "UNKNOWN"
                )
            )
        )

        resolution_score = (
            self.DATE_RESOLUTION_PRIORITY.get(
                resolution_method,
                0
            )
        )

        merged_method_score = (
            1
            if "+"
            in rate.source_method
            else 0
        )

        numeric_score = (
            1
            if rate.numeric_value is not None
            else 0
        )

        page_score = (
            rate.page_number
            if rate.page_number is not None
            else 0
        )

        row_score = (
            rate.row_index
            if rate.row_index is not None
            else 0
        )

        return (
            date_score,
            resolution_score,
            merged_method_score,
            numeric_score,
            page_score,
            row_score
        )

    def _table_structure(
        self,
        rate: RateItem
    ) -> str:

        return self._normalize_text(
            rate.metadata.get(
                "table_structure",
                ""
            )
        )

    def _is_unit_only_header(
        self,
        value: str
    ) -> bool:

        compact_value = re.sub(
            r"[\s()]",
            "",
            self._normalize_text(
                value
            )
        )

        return (
            compact_value
            in self.UNIT_ONLY_HEADERS
        )

    def _parse_date(
        self,
        value: Any
    ) -> date | None:

        text = self._clean_text(
            value
        )

        if not text:
            return None

        text = re.sub(
            r"\bSEPT\.?",
            "SEP",
            text,
            flags=re.IGNORECASE
        )

        text = text.replace(
            ".",
            ""
        )

        text = re.sub(
            r"\s+",
            " ",
            text
        )

        date_formats = (
            "%B %d, %Y",
            "%b %d, %Y",
            "%B %d %Y",
            "%b %d %Y"
        )

        for date_format in date_formats:

            try:

                return datetime.strptime(
                    text,
                    date_format
                ).date()

            except ValueError:

                continue

        return None

    def _normalize_value_text(
        self,
        value: Any
    ) -> str:

        text = self._normalize_text(
            value
        )

        text = text.replace(
            "$",
            ""
        )

        text = text.replace(
            ",",
            ""
        )

        return text

    def _comparison_sort_key(
        self,
        comparison: RateComparisonRecord
    ) -> tuple[str, ...]:

        return (
            self._normalize_text(
                comparison.category
            ),
            self._normalize_text(
                comparison.schedule_title
            ),
            self._normalize_text(
                comparison.charge_name
            ),
            self._normalize_text(
                comparison.unit
            ),
            self._normalize_text(
                comparison.context
            )
        )

    def _clean_text(
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

    def _normalize_text(
        self,
        value: Any
    ) -> str:

        return self._clean_text(
            value
        ).upper()