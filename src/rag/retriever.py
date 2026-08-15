import re

from dataclasses import dataclass
from enum import StrEnum
from typing import Any

from src.rag.vector_store import (
    TariffVectorStore,
    VectorSearchResult
)


class RetrievalIntent(StrEnum):
    """
    Describes the type of tariff information requested.
    """

    RATE_LOOKUP = "RATE_LOOKUP"
    COMPARISON = "COMPARISON"
    SECTION_COVERAGE = "SECTION_COVERAGE"
    GENERAL = "GENERAL"


@dataclass(slots=True)
class RetrievalResult:
    """
    Represents one reranked tariff retrieval result.
    """

    rank: int
    vector_rank: int
    chunk_id: str
    content: str
    metadata: dict[str, Any]
    semantic_similarity: float | None
    rerank_score: float
    matched_signals: tuple[str, ...]

    @property
    def chunk_type(self) -> str:

        return str(
            self.metadata.get(
                "chunk_type",
                ""
            )
        )

    def to_dict(self) -> dict[str, Any]:

        return {
            "rank": self.rank,
            "vector_rank": self.vector_rank,
            "chunk_id": self.chunk_id,
            "content": self.content,
            "metadata": dict(
                self.metadata
            ),
            "semantic_similarity": (
                self.semantic_similarity
            ),
            "rerank_score": (
                self.rerank_score
            ),
            "matched_signals": list(
                self.matched_signals
            )
        }


class TariffRetriever:
    """
    Retrieves tariff evidence using two stages:

    1. Semantic vector search
    2. Metadata-aware reranking

    Reranking prioritizes exact tariff concepts such as:

    - Residential Service
    - Customer Charge
    - Effective year
    - Added or removed Riders
    - Increased or decreased rates
    - Not-applicable sections
    """

    COMPARISON_TERMS = {
        "COMPARE",
        "COMPARISON",
        "CHANGE",
        "CHANGED",
        "DIFFERENCE",
        "DIFFERENCES",
        "BETWEEN",
        "OLD AND NEW",
        "PREVIOUS AND CURRENT"
    }

    COMPARISON_STATUS_TERMS = {
        "ADD",
        "ADDED",
        "INTRODUCED",
        "NEW RIDER",
        "NEW RIDERS",
        "REMOVE",
        "REMOVED",
        "RETIRED",
        "DISCONTINUED",
        "INCREASE",
        "INCREASED",
        "HIGHER",
        "ROSE",
        "DECREASE",
        "DECREASED",
        "LOWER",
        "FELL",
        "REDUCED",
        "UNCHANGED",
        "NO CHANGE"
    }

    SECTION_TERMS = {
        "SECTION",
        "SECTIONS",
        "APPLICABLE",
        "APPLICABILITY",
        "NOT APPLICABLE",
        "COVERAGE",
        "EXTRACTED",
        "EXTRACTION"
    }

    STOP_WORDS = {
        "A",
        "AN",
        "AND",
        "ARE",
        "AS",
        "AT",
        "BE",
        "BETWEEN",
        "BY",
        "DID",
        "DO",
        "DOES",
        "FOR",
        "FROM",
        "HOW",
        "IN",
        "IS",
        "IT",
        "OF",
        "ON",
        "OR",
        "THE",
        "TO",
        "WAS",
        "WERE",
        "WHAT",
        "WHEN",
        "WHICH",
        "WHO",
        "WITH",
        "TARIFF",
        "TARIFFS",
        "RATE",
        "RATES",
        "SCHEDULE",
        "SERVICE"
    }

    STATUS_QUERY_TERMS = {
        "ADDED": {
            "ADD",
            "ADDED",
            "INTRODUCED",
            "NEW RIDER",
            "NEW RIDERS"
        },
        "REMOVED": {
            "REMOVE",
            "REMOVED",
            "RETIRED",
            "DISCONTINUED"
        },
        "INCREASED": {
            "INCREASE",
            "INCREASED",
            "HIGHER",
            "ROSE"
        },
        "DECREASED": {
            "DECREASE",
            "DECREASED",
            "LOWER",
            "FELL",
            "REDUCED"
        },
        "UNCHANGED": {
            "UNCHANGED",
            "SAME RATE",
            "NO CHANGE"
        }
    }

    def __init__(
        self,
        vector_store: TariffVectorStore,
        candidate_multiplier: int = 8,
        minimum_candidate_count: int = 40
    ) -> None:

        if candidate_multiplier <= 0:

            raise ValueError(
                "candidate_multiplier must be "
                "greater than zero."
            )

        if minimum_candidate_count <= 0:

            raise ValueError(
                "minimum_candidate_count must be "
                "greater than zero."
            )

        self.vector_store = vector_store

        self.candidate_multiplier = (
            candidate_multiplier
        )

        self.minimum_candidate_count = (
            minimum_candidate_count
        )

    def detect_intent(
        self,
        query: str
    ) -> RetrievalIntent:
        """
        Detects the most likely retrieval intent.

        Added, removed, increased and decreased questions are
        comparison questions because their answers come from
        COMPARISON chunks.
        """

        normalized_query = (
            self._normalize_text(
                query
            )
        )

        if not normalized_query:

            raise ValueError(
                "The query cannot be empty."
            )

        if self._contains_any_phrase(
            normalized_query,
            self.SECTION_TERMS
        ):

            return (
                RetrievalIntent
                .SECTION_COVERAGE
            )

        if self._contains_any_phrase(
            normalized_query,
            self.COMPARISON_TERMS
        ):

            return (
                RetrievalIntent
                .COMPARISON
            )

        if self._contains_any_phrase(
            normalized_query,
            self.COMPARISON_STATUS_TERMS
        ):

            return (
                RetrievalIntent
                .COMPARISON
            )

        return RetrievalIntent.RATE_LOOKUP

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        intent: RetrievalIntent | None = None,
        where: dict[str, Any] | None = None
    ) -> list[RetrievalResult]:
        """
        Retrieves and reranks tariff chunks.
        """

        cleaned_query = " ".join(
            query.split()
        )

        if not cleaned_query:

            raise ValueError(
                "The query cannot be empty."
            )

        if top_k <= 0:

            raise ValueError(
                "top_k must be greater than zero."
            )

        resolved_intent = (
            intent
            or self.detect_intent(
                cleaned_query
            )
        )

        intent_filter = (
            self._intent_filter(
                resolved_intent
            )
        )

        combined_filter = (
            self._combine_filters(
                intent_filter,
                where
            )
        )

        candidate_count = max(
            top_k
            * self.candidate_multiplier,
            self.minimum_candidate_count
        )

        candidate_count = min(
            candidate_count,
            self.vector_store.count
        )

        vector_results = (
            self.vector_store.search(
                query=cleaned_query,
                n_results=candidate_count,
                where=combined_filter
            )
        )

        reranked_results = []

        for vector_result in (
            vector_results
        ):

            (
                rerank_score,
                matched_signals
            ) = self._calculate_rerank_score(
                query=cleaned_query,
                intent=resolved_intent,
                result=vector_result
            )

            reranked_results.append(
                RetrievalResult(
                    rank=0,
                    vector_rank=(
                        vector_result.rank
                    ),
                    chunk_id=(
                        vector_result.chunk_id
                    ),
                    content=(
                        vector_result.content
                    ),
                    metadata=dict(
                        vector_result.metadata
                    ),
                    semantic_similarity=(
                        vector_result.similarity
                    ),
                    rerank_score=(
                        rerank_score
                    ),
                    matched_signals=tuple(
                        matched_signals
                    )
                )
            )

        reranked_results.sort(
            key=lambda result: (
                result.rerank_score,
                (
                    result.semantic_similarity
                    if (
                        result.semantic_similarity
                        is not None
                    )
                    else -1.0
                ),
                -result.vector_rank
            ),
            reverse=True
        )

        selected_results = (
            reranked_results[:top_k]
        )

        for rank, result in enumerate(
            selected_results,
            start=1
        ):

            result.rank = rank

        return selected_results

    def _calculate_rerank_score(
        self,
        query: str,
        intent: RetrievalIntent,
        result: VectorSearchResult
    ) -> tuple[float, list[str]]:
        """
        Combines semantic similarity with metadata signals.
        """

        normalized_query = (
            self._normalize_text(
                query
            )
        )

        metadata = result.metadata

        similarity = (
            result.similarity
            if result.similarity is not None
            else 0.0
        )

        score = similarity
        matched_signals = []

        schedule_title = self._normalize_text(
            metadata.get(
                "schedule_title",
                ""
            )
        )

        charge_name = self._normalize_text(
            metadata.get(
                "normalized_charge_name",
                metadata.get(
                    "charge_name",
                    ""
                )
            )
        )

        category = self._normalize_text(
            metadata.get(
                "category",
                ""
            )
        )

        status = self._normalize_text(
            metadata.get(
                "status",
                ""
            )
        )

        applicability_status = (
            self._normalize_text(
                metadata.get(
                    "applicability_status",
                    ""
                )
            )
        )

        utility = self._normalize_text(
            metadata.get(
                "utility",
                ""
            )
        )

        context = self._normalize_text(
            metadata.get(
                "context",
                ""
            )
        )

        source_text = self._normalize_text(
            " ".join(
                [
                    self._string(
                        metadata.get(
                            "source_file",
                            ""
                        )
                    ),
                    self._string(
                        metadata.get(
                            "old_source_file",
                            ""
                        )
                    ),
                    self._string(
                        metadata.get(
                            "new_source_file",
                            ""
                        )
                    ),
                    self._string(
                        metadata.get(
                            "effective_date",
                            ""
                        )
                    ),
                    self._string(
                        metadata.get(
                            "old_effective_date",
                            ""
                        )
                    ),
                    self._string(
                        metadata.get(
                            "new_effective_date",
                            ""
                        )
                    )
                ]
            )
        )

        query_tokens = self._tokenize(
            normalized_query
        )

        schedule_tokens = (
            self._significant_tokens(
                schedule_title
            )
        )

        if (
            schedule_tokens
            and schedule_tokens.issubset(
                query_tokens
            )
        ):

            score += 0.18

            matched_signals.append(
                "schedule_exact"
            )

        elif self._token_overlap_ratio(
            normalized_query,
            schedule_title
        ) >= 0.50:

            score += 0.10

            matched_signals.append(
                "schedule_partial"
            )

        charge_tokens = (
            self._significant_tokens(
                charge_name
            )
        )

        if (
            charge_tokens
            and charge_tokens.issubset(
                query_tokens
            )
        ):

            score += 0.24

            matched_signals.append(
                "charge_exact"
            )

        elif self._token_overlap_ratio(
            normalized_query,
            charge_name
        ) >= 0.50:

            score += 0.12

            matched_signals.append(
                "charge_partial"
            )

        query_years = set(
            re.findall(
                r"\b(?:19|20)\d{2}\b",
                normalized_query
            )
        )

        matching_years = {
            year
            for year in query_years
            if year in source_text
        }

        if matching_years:

            score += (
                0.10
                * len(matching_years)
            )

            matched_signals.append(
                "year_match"
            )

        if (
            utility
            and self._contains_phrase(
                normalized_query,
                utility
            )
        ):

            score += 0.05

            matched_signals.append(
                "utility_match"
            )

        requested_status = (
            self._detect_requested_status(
                normalized_query
            )
        )

        if (
            requested_status
            and status == requested_status
        ):

            score += 0.30

            matched_signals.append(
                "status_match"
            )

        if (
            "RIDER"
            in query_tokens
            or "RIDERS"
            in query_tokens
        ):

            if category == "RIDER":

                score += 0.20

                matched_signals.append(
                    "rider_category"
                )

            else:

                score -= 0.10

        if (
            self._contains_phrase(
                normalized_query,
                "NOT APPLICABLE"
            )
            and applicability_status
            == "NOT_APPLICABLE"
        ):

            score += 0.35

            matched_signals.append(
                "not_applicable"
            )

        if (
            context
            and self._token_overlap_ratio(
                normalized_query,
                context
            ) >= 0.60
        ):

            score += 0.05

            matched_signals.append(
                "context_match"
            )

        expected_chunk_type = (
            self._chunk_type_for_intent(
                intent
            )
        )

        if (
            expected_chunk_type
            and result.chunk_type
            == expected_chunk_type
        ):

            score += 0.03

            matched_signals.append(
                "intent_type"
            )

        return (
            score,
            matched_signals
        )

    def _intent_filter(
        self,
        intent: RetrievalIntent
    ) -> dict[str, Any] | None:

        chunk_type = (
            self._chunk_type_for_intent(
                intent
            )
        )

        if not chunk_type:
            return None

        return {
            "chunk_type": chunk_type
        }

    def _chunk_type_for_intent(
        self,
        intent: RetrievalIntent
    ) -> str:

        mapping = {
            RetrievalIntent.RATE_LOOKUP: (
                "RATE"
            ),
            RetrievalIntent.COMPARISON: (
                "COMPARISON"
            ),
            RetrievalIntent
            .SECTION_COVERAGE: (
                "SECTION_COVERAGE"
            )
        }

        return mapping.get(
            intent,
            ""
        )

    def _combine_filters(
        self,
        left_filter: dict[str, Any] | None,
        right_filter: dict[str, Any] | None
    ) -> dict[str, Any] | None:

        if (
            left_filter is None
            and right_filter is None
        ):

            return None

        if left_filter is None:
            return right_filter

        if right_filter is None:
            return left_filter

        return {
            "$and": [
                left_filter,
                right_filter
            ]
        }

    def _detect_requested_status(
        self,
        query: str
    ) -> str:

        for status, terms in (
            self.STATUS_QUERY_TERMS.items()
        ):

            if self._contains_any_phrase(
                query,
                terms
            ):

                return status

        return ""

    def _contains_any_phrase(
        self,
        text: str,
        phrases: set[str]
    ) -> bool:

        return any(
            self._contains_phrase(
                text,
                phrase
            )
            for phrase in phrases
        )

    def _contains_phrase(
        self,
        text: str,
        phrase: str
    ) -> bool:
        """
        Performs phrase matching using word boundaries.

        This prevents short terms such as ADD from accidentally
        matching unrelated longer words.
        """

        normalized_text = (
            self._normalize_text(
                text
            )
        )

        normalized_phrase = (
            self._normalize_text(
                phrase
            )
        )

        if not normalized_phrase:
            return False

        escaped_phrase = re.escape(
            normalized_phrase
        )

        escaped_phrase = (
            escaped_phrase.replace(
                r"\ ",
                r"\s+"
            )
        )

        pattern = (
            r"(?<![A-Z0-9])"
            + escaped_phrase
            + r"(?![A-Z0-9])"
        )

        return bool(
            re.search(
                pattern,
                normalized_text
            )
        )

    def _token_overlap_ratio(
        self,
        query: str,
        candidate: str
    ) -> float:

        candidate_tokens = (
            self._significant_tokens(
                candidate
            )
        )

        if not candidate_tokens:
            return 0.0

        query_tokens = self._tokenize(
            query
        )

        overlap = (
            query_tokens
            & candidate_tokens
        )

        return (
            len(overlap)
            / len(candidate_tokens)
        )

    def _significant_tokens(
        self,
        value: str
    ) -> set[str]:

        return {
            token
            for token in self._tokenize(
                value
            )
            if (
                token not in self.STOP_WORDS
                and len(token) > 1
            )
        }

    def _tokenize(
        self,
        value: str
    ) -> set[str]:

        return set(
            re.findall(
                r"[A-Z0-9]+",
                self._normalize_text(
                    value
                )
            )
        )

    def _normalize_text(
        self,
        value: Any
    ) -> str:

        text = self._string(
            value
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

        return text.strip().upper()

    def _string(
        self,
        value: Any
    ) -> str:

        if value is None:
            return ""

        return str(
            value
        )