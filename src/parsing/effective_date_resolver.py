import re

from collections import Counter
from dataclasses import replace

from src.models.rate import RateItem


class EffectiveDateResolver:
    """
    Resolves missing effective dates without relying on filenames,
    schedule titles or fixed section IDs.

    Resolution order:

    1. Schedule-level consensus
       A missing rate inherits the date already found in the same
       schedule.

    2. Category-level consensus
       A schedule with no date may inherit the dominant date used by
       other schedules in the same document and category.

    Category-level inheritance occurs only when the evidence is
    strong enough. This prevents historical rider dates from being
    incorrectly overwritten.
    """

    def __init__(
        self,
        minimum_sections: int = 2,
        minimum_consensus_ratio: float = 0.75
    ) -> None:

        if minimum_sections < 1:
            raise ValueError(
                "minimum_sections must be at least 1."
            )

        if not 0 < minimum_consensus_ratio <= 1:
            raise ValueError(
                "minimum_consensus_ratio must be "
                "greater than 0 and at most 1."
            )

        self.minimum_sections = minimum_sections

        self.minimum_consensus_ratio = (
            minimum_consensus_ratio
        )

    def resolve(
        self,
        rate_items: list[RateItem]
    ) -> list[RateItem]:
        """
        Returns new RateItem objects with missing effective dates
        resolved where reliable evidence exists.
        """

        if not rate_items:
            return []

        schedule_dates = (
            self._build_schedule_dates(
                rate_items
            )
        )

        category_consensus = (
            self._build_category_consensus(
                schedule_dates
            )
        )

        resolved_items = []

        for rate_item in rate_items:

            if rate_item.effective_date:

                resolved_items.append(
                    self._copy_rate(
                        rate_item
                    )
                )

                continue

            schedule_key = (
                self._schedule_key(
                    rate_item
                )
            )

            schedule_date = (
                self._get_schedule_consensus(
                    schedule_dates.get(
                        schedule_key,
                        []
                    )
                )
            )

            if schedule_date:

                resolved_items.append(
                    self._apply_resolved_date(
                        rate_item=rate_item,
                        effective_date=schedule_date,
                        resolution_method=(
                            "SCHEDULE_CONSENSUS"
                        )
                    )
                )

                continue

            category_key = (
                self._category_key(
                    rate_item
                )
            )

            category_date = (
                category_consensus.get(
                    category_key,
                    ""
                )
            )

            if category_date:

                resolved_items.append(
                    self._apply_resolved_date(
                        rate_item=rate_item,
                        effective_date=category_date,
                        resolution_method=(
                            "CATEGORY_CONSENSUS"
                        )
                    )
                )

                continue

            unresolved_item = (
                self._copy_rate(
                    rate_item
                )
            )

            unresolved_item.metadata[
                "effective_date_resolved"
            ] = False

            unresolved_item.metadata[
                "effective_date_resolution"
            ] = "UNRESOLVED"

            resolved_items.append(
                unresolved_item
            )

        return resolved_items

    def _build_schedule_dates(
        self,
        rate_items: list[RateItem]
    ) -> dict[
        tuple[str, str, str],
        list[str]
    ]:
        """
        Collects explicit dates for each schedule.
        """

        schedule_dates: dict[
            tuple[str, str, str],
            list[str]
        ] = {}

        for rate_item in rate_items:

            effective_date = (
                self._clean_date(
                    rate_item.effective_date
                )
            )

            if not effective_date:
                continue

            schedule_key = (
                self._schedule_key(
                    rate_item
                )
            )

            schedule_dates.setdefault(
                schedule_key,
                []
            )

            schedule_dates[
                schedule_key
            ].append(
                effective_date
            )

        return schedule_dates

    def _build_category_consensus(
        self,
        schedule_dates: dict[
            tuple[str, str, str],
            list[str]
        ]
    ) -> dict[
        tuple[str, str],
        str
    ]:
        """
        Determines a dominant date across schedules belonging to
        the same source file and category.

        Each schedule contributes only one vote. Historical matrix
        rows do not produce hundreds of duplicate votes.
        """

        category_votes: dict[
            tuple[str, str],
            list[str]
        ] = {}

        for schedule_key, dates in (
            schedule_dates.items()
        ):

            source_file = schedule_key[0]
            category = schedule_key[1]

            schedule_date = (
                self._get_schedule_consensus(
                    dates
                )
            )

            if not schedule_date:
                continue

            category_key = (
                source_file,
                category
            )

            category_votes.setdefault(
                category_key,
                []
            )

            category_votes[
                category_key
            ].append(
                schedule_date
            )

        consensus_results = {}

        for category_key, dates in (
            category_votes.items()
        ):

            consensus_date = (
                self._get_category_consensus(
                    dates
                )
            )

            if consensus_date:

                consensus_results[
                    category_key
                ] = consensus_date

        return consensus_results

    def _get_schedule_consensus(
        self,
        dates: list[str]
    ) -> str:
        """
        Returns a date only when one unique date exists inside
        the schedule.

        Historical rider schedules containing several dates do not
        receive a schedule-wide inherited date.
        """

        normalized_dates = {
            self._clean_date(date)
            for date in dates
            if self._clean_date(date)
        }

        if len(normalized_dates) != 1:
            return ""

        return next(
            iter(normalized_dates)
        )

    def _get_category_consensus(
        self,
        dates: list[str]
    ) -> str:
        """
        Returns the dominant category date when enough schedules
        agree.
        """

        cleaned_dates = [
            self._clean_date(date)
            for date in dates
            if self._clean_date(date)
        ]

        if (
            len(cleaned_dates)
            < self.minimum_sections
        ):
            return ""

        date_counts = Counter(
            cleaned_dates
        )

        ordered_counts = (
            date_counts.most_common()
        )

        dominant_date = (
            ordered_counts[0][0]
        )

        dominant_count = (
            ordered_counts[0][1]
        )

        if len(ordered_counts) > 1:

            second_count = (
                ordered_counts[1][1]
            )

            if dominant_count == second_count:
                return ""

        consensus_ratio = (
            dominant_count
            / len(cleaned_dates)
        )

        if (
            consensus_ratio
            < self.minimum_consensus_ratio
        ):
            return ""

        return dominant_date

    def _apply_resolved_date(
        self,
        rate_item: RateItem,
        effective_date: str,
        resolution_method: str
    ) -> RateItem:

        metadata = dict(
            rate_item.metadata
        )

        metadata[
            "effective_date_resolved"
        ] = True

        metadata[
            "effective_date_resolution"
        ] = resolution_method

        metadata[
            "effective_date_original"
        ] = rate_item.effective_date

        return replace(
            rate_item,
            effective_date=effective_date,
            attributes=dict(
                rate_item.attributes
            ),
            metadata=metadata
        )

    def _copy_rate(
        self,
        rate_item: RateItem
    ) -> RateItem:

        metadata = dict(
            rate_item.metadata
        )

        metadata.setdefault(
            "effective_date_resolved",
            False
        )

        metadata.setdefault(
            "effective_date_resolution",
            "EXPLICIT"
            if rate_item.effective_date
            else "UNRESOLVED"
        )

        return replace(
            rate_item,
            attributes=dict(
                rate_item.attributes
            ),
            metadata=metadata
        )

    def _schedule_key(
        self,
        rate_item: RateItem
    ) -> tuple[str, str, str]:

        return (
            self._normalize_text(
                rate_item.source_file
            ),
            self._normalize_text(
                rate_item.category
            ),
            self._normalize_text(
                rate_item.schedule_id
            )
        )

    def _category_key(
        self,
        rate_item: RateItem
    ) -> tuple[str, str]:

        return (
            self._normalize_text(
                rate_item.source_file
            ),
            self._normalize_text(
                rate_item.category
            )
        )

    def _clean_date(
        self,
        date_value: str
    ) -> str:

        if not date_value:
            return ""

        cleaned = re.sub(
            r"\s+",
            " ",
            str(date_value)
        )

        return cleaned.strip()

    def _normalize_text(
        self,
        value: str
    ) -> str:

        if not value:
            return ""

        normalized = re.sub(
            r"\s+",
            " ",
            str(value)
        )

        return normalized.strip().upper()