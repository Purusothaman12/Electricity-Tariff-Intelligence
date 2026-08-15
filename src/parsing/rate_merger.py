import re

from dataclasses import replace
from decimal import Decimal

from src.models.rate import RateItem


class RateMerger:
    """
    Merges rate items extracted through different methods.

    Records are merged only when they belong to the same:

    - Source document
    - Schedule
    - Category
    - Charge
    - Unit
    - Value
    - Effective date
    - Meaningful rate context

    This prevents identical rates from different tariff documents
    from being incorrectly combined.
    """

    def merge(
        self,
        table_rates: list[RateItem],
        text_rates: list[RateItem]
    ) -> list[RateItem]:

        merged_rates: list[RateItem] = []

        all_rates = [
            *text_rates,
            *table_rates
        ]

        for candidate in all_rates:

            matching_index = (
                self._find_matching_rate(
                    existing_rates=merged_rates,
                    candidate=candidate
                )
            )

            if matching_index is None:

                merged_rates.append(
                    self._copy_rate(
                        candidate
                    )
                )

                continue

            existing = merged_rates[
                matching_index
            ]

            merged_rates[
                matching_index
            ] = self._merge_pair(
                first=existing,
                second=candidate
            )

        merged_rates.sort(
            key=self._sort_key
        )

        return merged_rates

    def _find_matching_rate(
        self,
        existing_rates: list[RateItem],
        candidate: RateItem
    ) -> int | None:

        for index, existing in enumerate(
            existing_rates
        ):

            if self._are_same_rate(
                first=existing,
                second=candidate
            ):
                return index

        return None

    def _are_same_rate(
        self,
        first: RateItem,
        second: RateItem
    ) -> bool:

        if (
            self._normalize_text(
                first.source_file
            )
            != self._normalize_text(
                second.source_file
            )
        ):
            return False

        if (
            self._normalize_text(
                first.category
            )
            != self._normalize_text(
                second.category
            )
        ):
            return False

        if (
            self._normalize_text(
                first.schedule_id
            )
            != self._normalize_text(
                second.schedule_id
            )
        ):
            return False

        if (
            self._normalize_text(
                first.schedule_title
            )
            != self._normalize_text(
                second.schedule_title
            )
        ):
            return False

        if (
            first.normalized_charge_name
            != second.normalized_charge_name
        ):
            return False

        if not self._units_compatible(
            first.normalized_unit,
            second.normalized_unit
        ):
            return False

        if not self._values_equal(
            first,
            second
        ):
            return False

        if not self._dates_compatible(
            first.effective_date,
            second.effective_date
        ):
            return False

        if not self._attributes_compatible(
            first.attributes,
            second.attributes
        ):
            return False

        return True

    def _units_compatible(
        self,
        first_unit: str,
        second_unit: str
    ) -> bool:

        first_unit = self._normalize_text(
            first_unit
        )

        second_unit = self._normalize_text(
            second_unit
        )

        if not first_unit or not second_unit:
            return True

        return first_unit == second_unit

    def _values_equal(
        self,
        first: RateItem,
        second: RateItem
    ) -> bool:

        first_numeric = (
            first.numeric_value
        )

        second_numeric = (
            second.numeric_value
        )

        if (
            first_numeric is not None
            and second_numeric is not None
        ):

            return self._decimals_equal(
                first_numeric,
                second_numeric
            )

        first_value = self._normalize_value(
            first.value_text
        )

        second_value = self._normalize_value(
            second.value_text
        )

        return first_value == second_value

    def _dates_compatible(
        self,
        first_date: str,
        second_date: str
    ) -> bool:

        first_date = self._normalize_text(
            first_date
        )

        second_date = self._normalize_text(
            second_date
        )

        if not first_date or not second_date:
            return True

        return first_date == second_date

    def _attributes_compatible(
        self,
        first_attributes: dict[str, str],
        second_attributes: dict[str, str]
    ) -> bool:
        """
        Keeps rates separate when both contain different
        meaningful contexts.

        Example:

        Customer Charge
        context = Non-Company Owned

        remains separate from:

        Customer Charge
        context = Company-Owned
        """

        meaningful_keys = {
            "context_heading",
            "row_label",
            "parent_charge"
        }

        for key in meaningful_keys:

            first_value = (
                self._normalize_text(
                    first_attributes.get(
                        key,
                        ""
                    )
                )
            )

            second_value = (
                self._normalize_text(
                    second_attributes.get(
                        key,
                        ""
                    )
                )
            )

            if (
                first_value
                and second_value
                and first_value
                != second_value
            ):
                return False

        return True

    def _merge_pair(
        self,
        first: RateItem,
        second: RateItem
    ) -> RateItem:

        preferred, alternate = (
            self._choose_preferred(
                first,
                second
            )
        )

        merged_attributes = dict(
            alternate.attributes
        )

        merged_attributes.update(
            preferred.attributes
        )

        source_methods = (
            self._merge_source_methods(
                first.source_method,
                second.source_method
            )
        )

        merged_metadata = dict(
            alternate.metadata
        )

        merged_metadata.update(
            preferred.metadata
        )

        merged_metadata[
            "merged"
        ] = True

        merged_metadata[
            "merged_source_methods"
        ] = source_methods.split("+")

        merged_metadata[
            "source_record_count"
        ] = (
            self._source_record_count(
                first
            )
            + self._source_record_count(
                second
            )
        )

        merged_unit = self._choose_unit(
            preferred.unit,
            alternate.unit
        )

        merged_date = (
            preferred.effective_date
            or alternate.effective_date
        )

        merged_page_number = (
            preferred.page_number
            if preferred.page_number is not None
            else alternate.page_number
        )

        merged_table_index = (
            preferred.table_index
            if preferred.table_index is not None
            else alternate.table_index
        )

        merged_row_index = (
            preferred.row_index
            if preferred.row_index is not None
            else alternate.row_index
        )

        return replace(
            preferred,
            unit=merged_unit,
            source_method=source_methods,
            page_number=merged_page_number,
            table_index=merged_table_index,
            row_index=merged_row_index,
            effective_date=merged_date,
            attributes=merged_attributes,
            metadata=merged_metadata
        )

    def _choose_preferred(
        self,
        first: RateItem,
        second: RateItem
    ) -> tuple[RateItem, RateItem]:

        first_score = (
            self._completeness_score(
                first
            )
        )

        second_score = (
            self._completeness_score(
                second
            )
        )

        if second_score > first_score:
            return second, first

        return first, second

    def _completeness_score(
        self,
        rate: RateItem
    ) -> int:

        score = 0

        if rate.effective_date:
            score += 5

        if rate.unit:
            score += 3

        if rate.page_number is not None:
            score += 2

        if rate.table_index is not None:
            score += 2

        if rate.row_index is not None:
            score += 1

        if rate.attributes.get(
            "context_heading"
        ):
            score += 4

        if rate.attributes.get(
            "row_label"
        ):
            score += 3

        if rate.attributes.get(
            "parent_charge"
        ):
            score += 2

        if (
            "TEXT"
            in rate.source_method.upper()
        ):
            score += 1

        return score

    def _choose_unit(
        self,
        preferred_unit: str,
        alternate_unit: str
    ) -> str:

        preferred_unit = (
            preferred_unit.strip()
        )

        alternate_unit = (
            alternate_unit.strip()
        )

        if preferred_unit:
            return preferred_unit

        return alternate_unit

    def _merge_source_methods(
        self,
        first_method: str,
        second_method: str
    ) -> str:

        methods = []

        for source_method in (
            first_method,
            second_method
        ):

            for method in source_method.split(
                "+"
            ):

                method = (
                    method.strip().upper()
                )

                if not method:
                    continue

                if method not in methods:

                    methods.append(
                        method
                    )

        return "+".join(
            methods
        )

    def _source_record_count(
        self,
        rate: RateItem
    ) -> int:

        count = rate.metadata.get(
            "source_record_count",
            1
        )

        try:
            return int(count)

        except (TypeError, ValueError):
            return 1

    def _copy_rate(
        self,
        rate: RateItem
    ) -> RateItem:

        metadata = dict(
            rate.metadata
        )

        metadata.setdefault(
            "source_record_count",
            1
        )

        return replace(
            rate,
            attributes=dict(
                rate.attributes
            ),
            metadata=metadata
        )

    def _normalize_value(
        self,
        value: str
    ) -> str:

        value = self._normalize_text(
            value
        )

        value = value.replace(
            "$",
            ""
        )

        value = value.replace(
            ",",
            ""
        )

        return value

    def _normalize_text(
        self,
        value: str
    ) -> str:

        if not value:
            return ""

        value = str(value)

        value = value.replace(
            "\u2013",
            "-"
        )

        value = value.replace(
            "\u2014",
            "-"
        )

        value = re.sub(
            r"\s+",
            " ",
            value
        )

        return value.strip().upper()

    def _decimals_equal(
        self,
        first: Decimal,
        second: Decimal
    ) -> bool:

        return first == second

    def _sort_key(
        self,
        rate: RateItem
    ) -> tuple:

        return (
            self._normalize_text(
                rate.source_file
            ),
            self._normalize_text(
                rate.category
            ),
            self._normalize_text(
                rate.schedule_id
            ),
            self._normalize_text(
                rate.effective_date
            ),
            rate.page_number or 0,
            rate.table_index or 0,
            rate.row_index or 0,
            rate.normalized_charge_name,
            self._normalize_text(
                rate.attributes.get(
                    "context_heading",
                    ""
                )
            )
        )