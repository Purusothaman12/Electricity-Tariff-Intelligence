import re

from dataclasses import dataclass
from typing import Any

from src.rag.question_planner import (
    TariffQuestionPlan,
    TariffQuestionType
)
from src.rag.retriever import (
    RetrievalResult
)


@dataclass(slots=True)
class EvidenceSelection:
    """
    Evidence selected for one user question.
    """

    question: str
    question_type: TariffQuestionType
    candidate_count: int
    selected_count: int
    selected_source_file: str | None
    target_terms: tuple[str, ...]
    results: list[RetrievalResult]
    reason: str

    def to_dict(
        self
    ) -> dict[str, Any]:

        return {
            "question": self.question,
            "question_type": (
                self.question_type.value
            ),
            "candidate_count": (
                self.candidate_count
            ),
            "selected_count": (
                self.selected_count
            ),
            "selected_source_file": (
                self.selected_source_file
            ),
            "target_terms": list(
                self.target_terms
            ),
            "reason": self.reason,
            "results": [
                result.to_dict()
                for result in self.results
            ]
        }


class TariffEvidenceSelector:
    """
    Selects relevant evidence after semantic retrieval.

    RATE_LOOKUP:
        Select one specific rate, preferring the latest tariff
        unless the user requests a year.

    RATE_LIST:
        Detect the requested schedule and select multiple unique
        rates only from that schedule.

    RATE_COMPARISON:
        Score comparison records using both schedule identity and
        charge identity.

    RIDER_CHANGE:
        Select Rider comparison records.

    SECTION_COVERAGE:
        Select section applicability records.

    OUT_OF_SCOPE:
        Select no evidence.
    """

    GENERIC_QUERY_WORDS = {
        "a",
        "all",
        "an",
        "and",
        "are",
        "available",
        "be",
        "between",
        "charge",
        "charges",
        "compare",
        "effective",
        "for",
        "from",
        "in",
        "is",
        "of",
        "on",
        "rate",
        "rates",
        "service",
        "show",
        "the",
        "to",
        "under",
        "was",
        "were",
        "what",
        "which",
        "with"
    }

    SCHEDULE_FIELDS = (
        "schedule_title",
        "service_name",
        "schedule_name"
    )

    CHARGE_FIELDS = (
        "charge_name",
        "normalized_charge_name",
        "rate_name"
    )

    VALUE_FIELDS = (
        "value_text",
        "new_value_text",
        "old_value_text",
        "rate_value",
        "value"
    )

    DATE_FIELDS = (
        "effective_date",
        "new_effective_date",
        "old_effective_date"
    )

    def select(
        self,
        plan: TariffQuestionPlan,
        results: list[RetrievalResult]
    ) -> EvidenceSelection:
        """
        Selects evidence according to the question plan.
        """

        if not isinstance(
            plan,
            TariffQuestionPlan
        ):

            raise TypeError(
                "plan must be a TariffQuestionPlan."
            )

        if not isinstance(
            results,
            list
        ):

            raise TypeError(
                "results must be a list."
            )

        valid_results = [
            result
            for result in results
            if self._is_valid_result(
                result
            )
        ]

        unique_results = (
            self._deduplicate_by_chunk(
                valid_results
            )
        )

        candidate_count = len(
            unique_results
        )

        target_terms = (
            self._extract_target_terms(
                plan.question
            )
        )

        if (
            plan.question_type
            == TariffQuestionType.OUT_OF_SCOPE
            or not plan.allow_llm
        ):

            return EvidenceSelection(
                question=plan.question,
                question_type=(
                    plan.question_type
                ),
                candidate_count=(
                    candidate_count
                ),
                selected_count=0,
                selected_source_file=None,
                target_terms=target_terms,
                results=[],
                reason=(
                    "The question is outside the "
                    "indexed electricity tariff scope."
                )
            )

        if not unique_results:

            return EvidenceSelection(
                question=plan.question,
                question_type=(
                    plan.question_type
                ),
                candidate_count=0,
                selected_count=0,
                selected_source_file=None,
                target_terms=target_terms,
                results=[],
                reason=(
                    "Retrieval returned no usable "
                    "tariff evidence."
                )
            )

        if (
            plan.question_type
            == TariffQuestionType.RATE_LOOKUP
        ):

            return self._select_rate_lookup(
                plan=plan,
                results=unique_results,
                candidate_count=(
                    candidate_count
                ),
                target_terms=target_terms
            )

        if (
            plan.question_type
            == TariffQuestionType.RATE_LIST
        ):

            return self._select_rate_list(
                plan=plan,
                results=unique_results,
                candidate_count=(
                    candidate_count
                ),
                target_terms=target_terms
            )

        if (
            plan.question_type
            == TariffQuestionType.RATE_COMPARISON
        ):

            return self._select_rate_comparison(
                plan=plan,
                results=unique_results,
                candidate_count=(
                    candidate_count
                ),
                target_terms=target_terms
            )

        if (
            plan.question_type
            == TariffQuestionType.RIDER_CHANGE
        ):

            selected_results = (
                self._select_rider_records(
                    results=unique_results,
                    limit=(
                        plan.prompt_evidence_limit
                    )
                )
            )

            return self._build_selection(
                plan=plan,
                candidate_count=(
                    candidate_count
                ),
                target_terms=target_terms,
                selected_results=(
                    selected_results
                ),
                selected_source_file=None,
                reason=(
                    "Rider comparison evidence was "
                    "selected for the Rider-change "
                    "question."
                )
            )

        if (
            plan.question_type
            == TariffQuestionType.SECTION_COVERAGE
        ):

            selected_results = (
                unique_results[
                    :plan.prompt_evidence_limit
                ]
            )

            return self._build_selection(
                plan=plan,
                candidate_count=(
                    candidate_count
                ),
                target_terms=target_terms,
                selected_results=(
                    selected_results
                ),
                selected_source_file=None,
                reason=(
                    "Section applicability evidence "
                    "was selected."
                )
            )

        selected_results = (
            self._deduplicate_rate_identity(
                unique_results
            )[
                :plan.prompt_evidence_limit
            ]
        )

        return self._build_selection(
            plan=plan,
            candidate_count=(
                candidate_count
            ),
            target_terms=target_terms,
            selected_results=(
                selected_results
            ),
            selected_source_file=None,
            reason=(
                "The highest-ranked unique tariff "
                "evidence records were selected."
            )
        )

    def _select_rate_lookup(
        self,
        plan: TariffQuestionPlan,
        results: list[RetrievalResult],
        candidate_count: int,
        target_terms: tuple[str, ...]
    ) -> EvidenceSelection:
        """
        Selects one specific rate record.
        """

        preferred_source_file = (
            self._select_preferred_source_file(
                results=results,
                requested_years=(
                    plan.extracted_years
                )
            )
        )

        scored_results = []

        for position, result in enumerate(
            results
        ):

            score = (
                self._calculate_rate_lookup_score(
                    result=result,
                    target_terms=target_terms,
                    requested_years=(
                        plan.extracted_years
                    )
                )
            )

            if (
                preferred_source_file
                and self._get_primary_source_file(
                    result
                )
                == preferred_source_file
            ):

                score += 150

            scored_results.append(
                (
                    score,
                    position,
                    result
                )
            )

        scored_results.sort(
            key=lambda item: (
                -item[0],
                item[1]
            )
        )

        selected_result = (
            scored_results[0][2]
        )

        selected_source_file = (
            self._get_primary_source_file(
                selected_result
            )
        )

        if plan.extracted_years:

            reason = (
                "The most relevant rate record "
                "matching the requested effective "
                "year was selected."
            )

        else:

            reason = (
                "The most relevant rate record from "
                "the latest available tariff source "
                "was selected."
            )

        return self._build_selection(
            plan=plan,
            candidate_count=(
                candidate_count
            ),
            target_terms=target_terms,
            selected_results=[
                selected_result
            ],
            selected_source_file=(
                selected_source_file
            ),
            reason=reason
        )

    def _calculate_rate_lookup_score(
        self,
        result: RetrievalResult,
        target_terms: tuple[str, ...],
        requested_years: tuple[int, ...]
    ) -> int:

        metadata = self._get_metadata(
            result
        )

        schedule_title = (
            self._first_metadata_value(
                metadata,
                self.SCHEDULE_FIELDS
            )
        )

        charge_name = (
            self._first_metadata_value(
                metadata,
                self.CHARGE_FIELDS
            )
        )

        effective_date = (
            self._first_metadata_value(
                metadata,
                self.DATE_FIELDS
            )
        )

        content = self._get_content(
            result
        )

        normalized_schedule = (
            self._normalize_text(
                schedule_title
            )
        )

        normalized_charge = (
            self._normalize_text(
                charge_name
            )
        )

        normalized_content = (
            self._normalize_text(
                content
            )
        )

        score = 0

        for term in target_terms:

            normalized_term = (
                self._normalize_text(
                    term
                )
            )

            if not normalized_term:
                continue

            if normalized_term in normalized_schedule:
                score += 60

            if normalized_term in normalized_charge:
                score += 50

            if normalized_term in normalized_content:
                score += 5

        year_text = " ".join(
            [
                effective_date,
                self._get_primary_source_file(
                    result
                ),
                content
            ]
        )

        for requested_year in requested_years:

            if str(
                requested_year
            ) in effective_date:

                score += 150

            elif str(
                requested_year
            ) in year_text:

                score += 80

        if schedule_title:
            score += 2

        if charge_name:
            score += 2

        if effective_date:
            score += 1

        return score

    def _select_rate_list(
        self,
        plan: TariffQuestionPlan,
        results: list[RetrievalResult],
        candidate_count: int,
        target_terms: tuple[str, ...]
    ) -> EvidenceSelection:
        """
        Selects multiple rates from one requested schedule.

        Schedule-title filtering prevents Rider rows whose charge
        name happens to contain the requested schedule name from
        being included.
        """

        target_schedule_title = (
            self._select_target_schedule_title(
                results=results,
                target_terms=target_terms
            )
        )

        schedule_results = results

        if target_schedule_title:

            normalized_target_schedule = (
                self._normalize_text(
                    target_schedule_title
                )
            )

            exact_schedule_results = [
                result
                for result in results
                if self._normalize_text(
                    self._get_schedule_title(
                        result
                    )
                )
                == normalized_target_schedule
            ]

            if exact_schedule_results:

                schedule_results = (
                    exact_schedule_results
                )

        preferred_source_file = (
            self._select_preferred_source_file(
                results=schedule_results,
                requested_years=(
                    plan.extracted_years
                )
            )
        )

        if preferred_source_file:

            same_source_results = [
                result
                for result in schedule_results
                if (
                    self._get_primary_source_file(
                        result
                    )
                    == preferred_source_file
                )
            ]

            if same_source_results:

                schedule_results = (
                    same_source_results
                )

        rate_results = [
            result
            for result in schedule_results
            if self._get_charge_name(
                result
            )
        ]

        if rate_results:

            schedule_results = rate_results

        scored_results = []

        for position, result in enumerate(
            schedule_results
        ):

            score = (
                self._calculate_rate_list_score(
                    result=result,
                    target_terms=target_terms,
                    requested_years=(
                        plan.extracted_years
                    )
                )
            )

            scored_results.append(
                (
                    score,
                    position,
                    result
                )
            )

        scored_results.sort(
            key=lambda item: (
                -item[0],
                item[1]
            )
        )

        sorted_results = [
            result
            for _, _, result
            in scored_results
        ]

        unique_results = (
            self._deduplicate_rate_identity(
                sorted_results
            )
        )

        selected_results = (
            unique_results[
                :plan.prompt_evidence_limit
            ]
        )

        return self._build_selection(
            plan=plan,
            candidate_count=(
                candidate_count
            ),
            target_terms=target_terms,
            selected_results=(
                selected_results
            ),
            selected_source_file=(
                preferred_source_file
            ),
            reason=(
                "Multiple unique charge records "
                "from the requested tariff schedule "
                "were selected."
            )
        )

    def _select_target_schedule_title(
        self,
        results: list[RetrievalResult],
        target_terms: tuple[str, ...]
    ) -> str:
        """
        Identifies the schedule title that best matches the user
        question.
        """

        schedule_scores: dict[
            str,
            int
        ] = {}

        original_titles: dict[
            str,
            str
        ] = {}

        for result in results:

            schedule_title = (
                self._get_schedule_title(
                    result
                )
            )

            if not schedule_title:
                continue

            normalized_schedule = (
                self._normalize_text(
                    schedule_title
                )
            )

            score = 0

            for term in target_terms:

                normalized_term = (
                    self._normalize_text(
                        term
                    )
                )

                if (
                    normalized_term
                    and normalized_term
                    in normalized_schedule
                ):

                    score += 100

            if score <= 0:
                continue

            schedule_scores[
                normalized_schedule
            ] = max(
                schedule_scores.get(
                    normalized_schedule,
                    0
                ),
                score
            )

            original_titles[
                normalized_schedule
            ] = schedule_title

        if not schedule_scores:
            return ""

        selected_normalized_title = max(
            schedule_scores,
            key=lambda title: (
                schedule_scores[
                    title
                ],
                -len(
                    title
                )
            )
        )

        return original_titles[
            selected_normalized_title
        ]

    def _calculate_rate_list_score(
        self,
        result: RetrievalResult,
        target_terms: tuple[str, ...],
        requested_years: tuple[int, ...]
    ) -> int:

        metadata = self._get_metadata(
            result
        )

        schedule_title = (
            self._get_schedule_title(
                result
            )
        )

        charge_name = (
            self._get_charge_name(
                result
            )
        )

        content = self._get_content(
            result
        )

        normalized_schedule = (
            self._normalize_text(
                schedule_title
            )
        )

        normalized_charge = (
            self._normalize_text(
                charge_name
            )
        )

        normalized_content = (
            self._normalize_text(
                content
            )
        )

        score = 0

        for term in target_terms:

            normalized_term = (
                self._normalize_text(
                    term
                )
            )

            if normalized_term in normalized_schedule:
                score += 100

            elif normalized_term in normalized_charge:
                score += 20

            elif normalized_term in normalized_content:
                score += 5

        date_text = " ".join(
            [
                self._first_metadata_value(
                    metadata,
                    self.DATE_FIELDS
                ),
                self._get_primary_source_file(
                    result
                ),
                content
            ]
        )

        for requested_year in requested_years:

            if str(
                requested_year
            ) in date_text:

                score += 50

        return score

    def _select_rate_comparison(
        self,
        plan: TariffQuestionPlan,
        results: list[RetrievalResult],
        candidate_count: int,
        target_terms: tuple[str, ...]
    ) -> EvidenceSelection:
        """
        Selects one comparison record using both the requested
        schedule and requested charge.

        Example:

            Residential Transmission System Charge

        should match:

            Schedule: RESIDENTIAL SERVICE
            Charge: Transmission System Charge

        and not:

            Schedule: TRANSMISSION SERVICE
            Charge: Transmission System Charge
        """

        scored_results = []

        for position, result in enumerate(
            results
        ):

            score = (
                self._calculate_comparison_score(
                    result=result,
                    target_terms=target_terms,
                    requested_years=(
                        plan.extracted_years
                    )
                )
            )

            scored_results.append(
                (
                    score,
                    position,
                    result
                )
            )

        scored_results.sort(
            key=lambda item: (
                -item[0],
                item[1]
            )
        )

        selected_result = (
            scored_results[0][2]
        )

        return self._build_selection(
            plan=plan,
            candidate_count=(
                candidate_count
            ),
            target_terms=target_terms,
            selected_results=[
                selected_result
            ],
            selected_source_file=(
                self._get_primary_source_file(
                    selected_result
                )
            ),
            reason=(
                "The comparison record whose schedule "
                "and charge identities best match the "
                "question was selected."
            )
        )

    def _calculate_comparison_score(
        self,
        result: RetrievalResult,
        target_terms: tuple[str, ...],
        requested_years: tuple[int, ...]
    ) -> int:
        """
        Scores comparison evidence.

        A query term found in the schedule but not the charge is
        treated as a schedule discriminator.

        For example:

            residential

        identifies RESIDENTIAL SERVICE, while:

            transmission
            system

        identify Transmission System Charge.
        """

        metadata = self._get_metadata(
            result
        )

        schedule_title = (
            self._get_schedule_title(
                result
            )
        )

        charge_name = (
            self._get_charge_name(
                result
            )
        )

        content = self._get_content(
            result
        )

        normalized_schedule = (
            self._normalize_text(
                schedule_title
            )
        )

        normalized_charge = (
            self._normalize_text(
                charge_name
            )
        )

        normalized_content = (
            self._normalize_text(
                content
            )
        )

        score = 0

        for term in target_terms:

            normalized_term = (
                self._normalize_text(
                    term
                )
            )

            if not normalized_term:
                continue

            schedule_match = (
                normalized_term
                in normalized_schedule
            )

            charge_match = (
                normalized_term
                in normalized_charge
            )

            content_match = (
                normalized_term
                in normalized_content
            )

            if schedule_match:
                score += 50

            if charge_match:
                score += 70

            if (
                schedule_match
                and not charge_match
            ):

                # Strong schedule discriminator.
                score += 150

            if content_match:
                score += 5

        for requested_year in requested_years:

            if str(
                requested_year
            ) in content:

                score += 30

        normalized_content_upper = (
            content.upper()
        )

        if (
            "OLD RATE" in normalized_content_upper
            and "NEW RATE" in normalized_content_upper
        ):

            score += 10

        if schedule_title:
            score += 2

        if charge_name:
            score += 2

        return score

    def _select_rider_records(
        self,
        results: list[RetrievalResult],
        limit: int
    ) -> list[RetrievalResult]:
        """
        Selects Rider-related comparison records.
        """

        selected_results = []

        for result in results:

            metadata = self._get_metadata(
                result
            )

            searchable_text = (
                self._normalize_text(
                    " ".join(
                        [
                            self._get_content(
                                result
                            ),
                            self._get_schedule_title(
                                result
                            ),
                            self._clean_text(
                                metadata.get(
                                    "category"
                                )
                            )
                        ]
                    )
                )
            )

            if "rider" not in searchable_text:
                continue

            selected_results.append(
                result
            )

            if len(
                selected_results
            ) >= limit:

                break

        return self._deduplicate_rate_identity(
            selected_results
        )

    def _select_preferred_source_file(
        self,
        results: list[RetrievalResult],
        requested_years: tuple[int, ...]
    ) -> str | None:
        """
        Chooses a tariff source by:

        1. Requested effective year.
        2. Latest year in the source filename.
        """

        grouped_results: dict[
            str,
            list[RetrievalResult]
        ] = {}

        for result in results:

            source_file = (
                self._get_primary_source_file(
                    result
                )
            )

            if not source_file:
                continue

            grouped_results.setdefault(
                source_file,
                []
            ).append(
                result
            )

        if not grouped_results:
            return None

        scored_sources = []

        for source_file, source_results in (
            grouped_results.items()
        ):

            requested_year_score = 0

            for result in source_results:

                metadata = self._get_metadata(
                    result
                )

                date_text = " ".join(
                    self._clean_text(
                        metadata.get(
                            field_name
                        )
                    )
                    for field_name
                    in self.DATE_FIELDS
                )

                searchable_text = " ".join(
                    [
                        source_file,
                        date_text,
                        self._get_content(
                            result
                        )
                    ]
                )

                for requested_year in requested_years:

                    year_text = str(
                        requested_year
                    )

                    if year_text in date_text:
                        requested_year_score += 100

                    elif year_text in searchable_text:
                        requested_year_score += 40

            source_year = (
                self._extract_latest_year(
                    source_file
                )
            )

            scored_sources.append(
                (
                    requested_year_score,
                    source_year,
                    source_file
                )
            )

        scored_sources.sort(
            key=lambda item: (
                item[0],
                item[1],
                item[2]
            ),
            reverse=True
        )

        return scored_sources[0][2]

    def _deduplicate_by_chunk(
        self,
        results: list[RetrievalResult]
    ) -> list[RetrievalResult]:

        unique_results = []
        seen_keys = set()

        for result in results:

            key = (
                self._get_chunk_id(
                    result
                )
                or self._normalize_text(
                    self._get_content(
                        result
                    )
                )
            )

            if not key:
                continue

            if key in seen_keys:
                continue

            seen_keys.add(
                key
            )

            unique_results.append(
                result
            )

        return unique_results

    def _deduplicate_rate_identity(
        self,
        results: list[RetrievalResult]
    ) -> list[RetrievalResult]:

        unique_results = []
        seen_keys = set()

        for result in results:

            metadata = self._get_metadata(
                result
            )

            identity_key = (
                self._normalize_text(
                    self._get_schedule_title(
                        result
                    )
                ),
                self._normalize_text(
                    self._get_charge_name(
                        result
                    )
                ),
                self._normalize_text(
                    self._first_metadata_value(
                        metadata,
                        self.VALUE_FIELDS
                    )
                ),
                self._normalize_text(
                    self._first_metadata_value(
                        metadata,
                        self.DATE_FIELDS
                    )
                ),
                self._normalize_text(
                    self._get_primary_source_file(
                        result
                    )
                )
            )

            if not any(
                identity_key
            ):

                identity_key = (
                    self._normalize_text(
                        self._get_content(
                            result
                        )
                    ),
                )

            if identity_key in seen_keys:
                continue

            seen_keys.add(
                identity_key
            )

            unique_results.append(
                result
            )

        return unique_results

    def _extract_target_terms(
        self,
        question: str
    ) -> tuple[str, ...]:

        normalized_question = (
            self._normalize_text(
                question
            )
        )

        raw_tokens = re.findall(
            r"[a-z0-9]+",
            normalized_question
        )

        target_terms = [
            token
            for token in raw_tokens
            if (
                token
                not in self.GENERIC_QUERY_WORDS
                and len(token) > 1
                and not token.isdigit()
            )
        ]

        return tuple(
            dict.fromkeys(
                target_terms
            )
        )

    def _get_schedule_title(
        self,
        result: RetrievalResult
    ) -> str:

        return self._first_metadata_value(
            self._get_metadata(
                result
            ),
            self.SCHEDULE_FIELDS
        )

    def _get_charge_name(
        self,
        result: RetrievalResult
    ) -> str:

        return self._first_metadata_value(
            self._get_metadata(
                result
            ),
            self.CHARGE_FIELDS
        )

    def _get_primary_source_file(
        self,
        result: RetrievalResult
    ) -> str:

        metadata = self._get_metadata(
            result
        )

        source_file = (
            self._clean_text(
                metadata.get(
                    "source_file"
                )
            )
        )

        if source_file:
            return source_file

        new_source_file = (
            self._clean_text(
                metadata.get(
                    "new_source_file"
                )
            )
        )

        if new_source_file:
            return new_source_file

        return self._clean_text(
            metadata.get(
                "old_source_file"
            )
        )

    def _get_metadata(
        self,
        result: RetrievalResult
    ) -> dict[str, Any]:

        metadata = getattr(
            result,
            "metadata",
            None
        )

        if isinstance(
            metadata,
            dict
        ):

            return metadata

        result_dictionary = (
            result.to_dict()
        )

        dictionary_metadata = (
            result_dictionary.get(
                "metadata",
                {}
            )
        )

        if isinstance(
            dictionary_metadata,
            dict
        ):

            return dictionary_metadata

        return {}

    def _get_content(
        self,
        result: RetrievalResult
    ) -> str:

        content = getattr(
            result,
            "content",
            None
        )

        if content is None:

            content = (
                result.to_dict().get(
                    "content",
                    ""
                )
            )

        return self._clean_text(
            content
        )

    def _get_chunk_id(
        self,
        result: RetrievalResult
    ) -> str:

        chunk_id = getattr(
            result,
            "chunk_id",
            None
        )

        if chunk_id is None:

            chunk_id = (
                result.to_dict().get(
                    "chunk_id",
                    ""
                )
            )

        return self._clean_text(
            chunk_id
        )

    def _first_metadata_value(
        self,
        metadata: dict[str, Any],
        field_names: tuple[str, ...]
    ) -> str:

        for field_name in field_names:

            value = self._clean_text(
                metadata.get(
                    field_name
                )
            )

            if value:
                return value

        return ""

    def _extract_latest_year(
        self,
        value: str
    ) -> int:
        """
        Extracts years from filenames containing underscores.

        Example:

            Oncor_May_1_2023.pdf
                -> 2023
        """

        years = re.findall(
            r"(?<!\d)(?:19|20)\d{2}(?!\d)",
            value
        )

        return max(
            (
                int(
                    year
                )
                for year in years
            ),
            default=0
        )

    def _build_selection(
        self,
        plan: TariffQuestionPlan,
        candidate_count: int,
        target_terms: tuple[str, ...],
        selected_results: list[RetrievalResult],
        selected_source_file: str | None,
        reason: str
    ) -> EvidenceSelection:

        return EvidenceSelection(
            question=plan.question,
            question_type=(
                plan.question_type
            ),
            candidate_count=(
                candidate_count
            ),
            selected_count=len(
                selected_results
            ),
            selected_source_file=(
                selected_source_file
            ),
            target_terms=target_terms,
            results=selected_results,
            reason=reason
        )

    def _is_valid_result(
        self,
        result: Any
    ) -> bool:

        if result is None:
            return False

        return bool(
            self._get_content(
                result
            )
        )

    def _normalize_text(
        self,
        value: Any
    ) -> str:

        cleaned_value = (
            self._clean_text(
                value
            ).lower()
        )

        cleaned_value = (
            cleaned_value
            .replace(
                "\u2013",
                "-"
            )
            .replace(
                "\u2014",
                "-"
            )
        )

        cleaned_value = re.sub(
            r"[^a-z0-9.$%/\-\s]",
            " ",
            cleaned_value
        )

        cleaned_value = re.sub(
            r"\s+",
            " ",
            cleaned_value
        )

        return cleaned_value.strip()

    def _clean_text(
        self,
        value: Any
    ) -> str:

        if value is None:
            return ""

        return " ".join(
            str(
                value
            ).split()
        )