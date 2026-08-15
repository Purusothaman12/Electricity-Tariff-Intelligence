import re

from collections import defaultdict
from dataclasses import dataclass
from decimal import Decimal
from typing import Any

from src.comparison.rate_comparator import (
    RateComparisonRecord,
    TariffComparisonResult
)
from src.rag.retriever import (
    RetrievalIntent,
    RetrievalResult,
    TariffRetriever
)


@dataclass(slots=True)
class AnswerEvidence:
    """
    Represents one retrieved chunk used to support an answer.
    """

    rank: int
    chunk_id: str
    content: str
    metadata: dict[str, Any]
    score: float | None

    def to_dict(self) -> dict[str, Any]:

        return {
            "rank": self.rank,
            "chunk_id": self.chunk_id,
            "content": self.content,
            "metadata": dict(
                self.metadata
            ),
            "score": self.score
        }


@dataclass(slots=True)
class RAGAnswer:
    """
    Represents the complete grounded response to one question.
    """

    query: str
    intent: RetrievalIntent
    answer_type: str
    answer: str
    evidence: list[AnswerEvidence]
    is_grounded: bool

    def to_dict(self) -> dict[str, Any]:

        return {
            "query": self.query,
            "intent": self.intent.value,
            "answer_type": self.answer_type,
            "answer": self.answer,
            "is_grounded": self.is_grounded,
            "evidence": [
                item.to_dict()
                for item in self.evidence
            ]
        }


class TariffAnswerGenerator:
    """
    Produces grounded tariff answers from retrieved evidence.

    The generator uses:

    1. Semantic retrieval for relevant evidence.
    2. Metadata for exact values, dates and statuses.
    3. Structured comparison data for schedule-level questions.

    It never calculates an answer from unsupported free text.
    """

    ADDED_TERMS = {
        "ADD",
        "ADDED",
        "INTRODUCED",
        "NEW RIDER",
        "NEW RIDERS"
    }

    REMOVED_TERMS = {
        "REMOVE",
        "REMOVED",
        "RETIRED",
        "DISCONTINUED"
    }

    def __init__(
        self,
        retriever: TariffRetriever,
        comparison_result: (
            TariffComparisonResult | None
        ) = None
    ) -> None:

        self.retriever = retriever

        self.comparison_result = (
            comparison_result
        )

    def answer(
        self,
        query: str,
        top_k: int = 8
    ) -> RAGAnswer:
        """
        Answers one tariff question using retrieved evidence.
        """

        cleaned_query = " ".join(
            query.split()
        )

        if not cleaned_query:

            raise ValueError(
                "The question cannot be empty."
            )

        if top_k <= 0:

            raise ValueError(
                "top_k must be greater than zero."
            )

        intent = self.retriever.detect_intent(
            cleaned_query
        )

        requested_status = (
            self._detect_schedule_status(
                cleaned_query
            )
        )

        if (
            intent
            == RetrievalIntent.COMPARISON
            and requested_status
            in {
                "ADDED",
                "REMOVED"
            }
            and self._mentions_rider(
                cleaned_query
            )
            and self.comparison_result
            is not None
        ):

            return (
                self._answer_complete_rider_changes(
                    query=cleaned_query,
                    intent=intent,
                    status=requested_status
                )
            )

        results = self.retriever.retrieve(
            query=cleaned_query,
            top_k=top_k,
            intent=intent
        )

        if not results:

            return self._insufficient_answer(
                query=cleaned_query,
                intent=intent
            )

        if (
            intent
            == RetrievalIntent.RATE_LOOKUP
        ):

            return self._answer_rate_lookup(
                query=cleaned_query,
                intent=intent,
                results=results
            )

        if (
            intent
            == RetrievalIntent.COMPARISON
        ):

            return self._answer_comparison(
                query=cleaned_query,
                intent=intent,
                results=results
            )

        if (
            intent
            == RetrievalIntent
            .SECTION_COVERAGE
        ):

            return self._answer_section_coverage(
                query=cleaned_query,
                intent=intent,
                results=results
            )

        return self._answer_general(
            query=cleaned_query,
            intent=intent,
            results=results
        )

    def _answer_rate_lookup(
        self,
        query: str,
        intent: RetrievalIntent,
        results: list[RetrievalResult]
    ) -> RAGAnswer:

        top_result = results[0]
        metadata = top_result.metadata

        utility = self._value(
            metadata,
            "utility",
            default="The utility"
        )

        schedule_title = self._value(
            metadata,
            "schedule_title",
            default="the selected schedule"
        )

        charge_name = self._value(
            metadata,
            "charge_name",
            default="the selected charge"
        )

        value_text = self._value(
            metadata,
            "value_text",
            default="not available"
        )

        unit = self._value(
            metadata,
            "unit"
        )

        effective_date = self._value(
            metadata,
            "effective_date",
            default="an unspecified date"
        )

        source_file = self._value(
            metadata,
            "source_file",
            default="the tariff document"
        )

        rate_text = value_text

        if (
            unit
            and unit.upper()
            not in value_text.upper()
        ):

            rate_text = (
                f"{value_text} {unit}"
            )

        answer_text = (
            f"For {utility}'s "
            f"{schedule_title}, the "
            f"{charge_name} was "
            f"{rate_text}, effective "
            f"{effective_date}. "
            f"The value was retrieved from "
            f"{source_file}."
        )

        return RAGAnswer(
            query=query,
            intent=intent,
            answer_type="RATE_LOOKUP",
            answer=answer_text,
            evidence=self._build_evidence(
                results[:3]
            ),
            is_grounded=True
        )

    def _answer_comparison(
        self,
        query: str,
        intent: RetrievalIntent,
        results: list[RetrievalResult]
    ) -> RAGAnswer:

        top_result = results[0]
        metadata = top_result.metadata

        schedule_title = self._value(
            metadata,
            "schedule_title",
            default="The selected schedule"
        )

        charge_name = self._value(
            metadata,
            "charge_name",
            default="the selected charge"
        )

        status = self._value(
            metadata,
            "status",
            default="CHANGED"
        )

        old_value = self._value(
            metadata,
            "old_value_text",
            default="not present"
        )

        new_value = self._value(
            metadata,
            "new_value_text",
            default="not present"
        )

        old_date = self._value(
            metadata,
            "old_effective_date",
            default="an unspecified date"
        )

        new_date = self._value(
            metadata,
            "new_effective_date",
            default="an unspecified date"
        )

        absolute_change = self._value(
            metadata,
            "absolute_change"
        )

        percent_change = self._format_percent(
            metadata.get(
                "percent_change"
            )
        )

        status_text = status.lower()

        answer_parts = [
            (
                f"For {schedule_title}, the "
                f"{charge_name} {status_text} "
                f"from {old_value}, effective "
                f"{old_date}, to {new_value}, "
                f"effective {new_date}."
            )
        ]

        change_parts = []

        if absolute_change:

            change_parts.append(
                (
                    "an absolute change of "
                    f"{absolute_change}"
                )
            )

        if percent_change:

            change_parts.append(
                (
                    "a percentage change of "
                    f"{percent_change}%"
                )
            )

        if change_parts:

            answer_parts.append(
                (
                    "This represents "
                    + " and ".join(
                        change_parts
                    )
                    + "."
                )
            )

        return RAGAnswer(
            query=query,
            intent=intent,
            answer_type="RATE_COMPARISON",
            answer=" ".join(
                answer_parts
            ),
            evidence=self._build_evidence(
                results[:3]
            ),
            is_grounded=True
        )

    def _answer_complete_rider_changes(
        self,
        query: str,
        intent: RetrievalIntent,
        status: str
    ) -> RAGAnswer:
        """
        Finds complete Rider schedule additions or removals.

        A schedule is considered completely added only when every
        comparison record for that Rider lacks an old rate.

        A schedule is considered completely removed only when every
        comparison record for that Rider lacks a new rate.
        """

        if self.comparison_result is None:

            return self._insufficient_answer(
                query=query,
                intent=intent
            )

        grouped_comparisons: dict[
            str,
            list[RateComparisonRecord]
        ] = defaultdict(
            list
        )

        display_titles = {}

        for comparison in (
            self.comparison_result.comparisons
        ):

            if (
                self._normalize_text(
                    comparison.category
                )
                != "RIDER"
            ):
                continue

            title = (
                comparison.schedule_title
                .strip()
            )

            if not title:
                continue

            normalized_title = (
                self._normalize_text(
                    title
                )
            )

            grouped_comparisons[
                normalized_title
            ].append(
                comparison
            )

            display_titles[
                normalized_title
            ] = title

        matching_titles = []

        for (
            normalized_title,
            comparisons
        ) in grouped_comparisons.items():

            if not comparisons:
                continue

            if status == "ADDED":

                is_complete_change = all(
                    comparison.old_rate
                    is None
                    and comparison.new_rate
                    is not None
                    for comparison
                    in comparisons
                )

            else:

                is_complete_change = all(
                    comparison.old_rate
                    is not None
                    and comparison.new_rate
                    is None
                    for comparison
                    in comparisons
                )

            if is_complete_change:

                matching_titles.append(
                    display_titles[
                        normalized_title
                    ]
                )

        matching_titles.sort(
            key=self._normalize_text
        )

        if not matching_titles:

            action_text = (
                "added"
                if status == "ADDED"
                else "removed"
            )

            return RAGAnswer(
                query=query,
                intent=intent,
                answer_type=(
                    "RIDER_SCHEDULE_CHANGE"
                ),
                answer=(
                    "No complete Rider schedules "
                    f"were identified as "
                    f"{action_text}. Individual "
                    "rate rows may still have "
                    "changed."
                ),
                evidence=[],
                is_grounded=True
            )

        action_text = (
            "added"
            if status == "ADDED"
            else "removed"
        )

        tariff_text = (
            "new tariff"
            if status == "ADDED"
            else "older tariff comparison"
        )

        formatted_titles = "; ".join(
            matching_titles
        )

        answer_text = (
            f"The {tariff_text} {action_text} "
            f"{len(matching_titles)} complete "
            f"Rider schedule"
        )

        if len(matching_titles) != 1:

            answer_text += "s"

        answer_text += (
            f": {formatted_titles}. "
            "Riders containing only individual "
            "added or removed rate rows are not "
            "counted as completely new or "
            "completely removed schedules."
        )

        evidence = (
            self._get_schedule_change_evidence(
                titles=matching_titles,
                status=status
            )
        )

        return RAGAnswer(
            query=query,
            intent=intent,
            answer_type=(
                "RIDER_SCHEDULE_CHANGE"
            ),
            answer=answer_text,
            evidence=evidence,
            is_grounded=True
        )

    def _get_schedule_change_evidence(
        self,
        titles: list[str],
        status: str
    ) -> list[AnswerEvidence]:

        evidence = []

        for title in titles:

            vector_results = (
                self.retriever
                .vector_store
                .search(
                    query=(
                        f"{status} Rider "
                        f"{title}"
                    ),
                    n_results=1,
                    where={
                        "$and": [
                            {
                                "chunk_type":
                                "COMPARISON"
                            },
                            {
                                "category":
                                "RIDER"
                            },
                            {
                                "status":
                                status
                            },
                            {
                                "schedule_title":
                                title
                            }
                        ]
                    }
                )
            )

            for result in vector_results:

                evidence.append(
                    AnswerEvidence(
                        rank=len(
                            evidence
                        ) + 1,
                        chunk_id=(
                            result.chunk_id
                        ),
                        content=(
                            result.content
                        ),
                        metadata=dict(
                            result.metadata
                        ),
                        score=(
                            result.similarity
                        )
                    )
                )

        return evidence

    def _answer_section_coverage(
        self,
        query: str,
        intent: RetrievalIntent,
        results: list[RetrievalResult]
    ) -> RAGAnswer:

        normalized_query = (
            self._normalize_text(
                query
            )
        )

        require_not_applicable = (
            "NOT APPLICABLE"
            in normalized_query
        )

        selected_results = []

        seen_sections = set()

        for result in results:

            applicability_status = (
                self._normalize_text(
                    result.metadata.get(
                        "applicability_status",
                        ""
                    )
                )
            )

            if (
                require_not_applicable
                and applicability_status
                != "NOT_APPLICABLE"
            ):

                continue

            identity = (
                self._value(
                    result.metadata,
                    "source_file"
                ),
                self._value(
                    result.metadata,
                    "section_id"
                )
            )

            if identity in seen_sections:
                continue

            seen_sections.add(
                identity
            )

            selected_results.append(
                result
            )

        if not selected_results:

            return self._insufficient_answer(
                query=query,
                intent=intent
            )

        section_descriptions = []

        for result in selected_results:

            metadata = result.metadata

            section_title = self._value(
                metadata,
                "section_title",
                default="Unnamed section"
            )

            section_id = self._value(
                metadata,
                "section_id"
            )

            source_file = self._value(
                metadata,
                "source_file",
                default="Unknown document"
            )

            section_descriptions.append(
                (
                    f"{section_title} "
                    f"({section_id}) in "
                    f"{source_file}"
                )
            )

        if require_not_applicable:

            answer_text = (
                f"{len(section_descriptions)} "
                "section records are marked "
                "not applicable: "
            )

        else:

            answer_text = (
                f"{len(section_descriptions)} "
                "relevant section coverage "
                "records were found: "
            )

        answer_text += (
            "; ".join(
                section_descriptions
            )
            + "."
        )

        return RAGAnswer(
            query=query,
            intent=intent,
            answer_type="SECTION_COVERAGE",
            answer=answer_text,
            evidence=self._build_evidence(
                selected_results
            ),
            is_grounded=True
        )

    def _answer_general(
        self,
        query: str,
        intent: RetrievalIntent,
        results: list[RetrievalResult]
    ) -> RAGAnswer:

        selected_results = results[:3]

        evidence_text = " ".join(
            result.content
            for result in selected_results
        )

        return RAGAnswer(
            query=query,
            intent=intent,
            answer_type="GENERAL_EVIDENCE",
            answer=(
                "The most relevant tariff "
                "evidence is: "
                f"{evidence_text}"
            ),
            evidence=self._build_evidence(
                selected_results
            ),
            is_grounded=True
        )

    def _build_evidence(
        self,
        results: list[RetrievalResult]
    ) -> list[AnswerEvidence]:

        return [
            AnswerEvidence(
                rank=index,
                chunk_id=result.chunk_id,
                content=result.content,
                metadata=dict(
                    result.metadata
                ),
                score=(
                    result.rerank_score
                )
            )
            for index, result in enumerate(
                results,
                start=1
            )
        ]

    def _detect_schedule_status(
        self,
        query: str
    ) -> str:

        normalized_query = (
            self._normalize_text(
                query
            )
        )

        if self._contains_any_phrase(
            normalized_query,
            self.ADDED_TERMS
        ):

            return "ADDED"

        if self._contains_any_phrase(
            normalized_query,
            self.REMOVED_TERMS
        ):

            return "REMOVED"

        return ""

    def _mentions_rider(
        self,
        query: str
    ) -> bool:

        tokens = set(
            re.findall(
                r"[A-Z0-9]+",
                self._normalize_text(
                    query
                )
            )
        )

        return bool(
            {
                "RIDER",
                "RIDERS"
            }
            & tokens
        )

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

        escaped_phrase = re.escape(
            normalized_phrase
        ).replace(
            r"\ ",
            r"\s+"
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

    def _format_percent(
        self,
        value: Any
    ) -> str:

        text = self._string(
            value
        )

        if not text:
            return ""

        try:

            decimal_value = Decimal(
                text
            )

        except Exception:

            return text

        formatted_value = format(
            decimal_value.quantize(
                Decimal("0.01")
            ),
            "f"
        )

        return formatted_value

    def _insufficient_answer(
        self,
        query: str,
        intent: RetrievalIntent
    ) -> RAGAnswer:

        return RAGAnswer(
            query=query,
            intent=intent,
            answer_type=(
                "INSUFFICIENT_EVIDENCE"
            ),
            answer=(
                "The indexed tariff evidence "
                "does not contain enough reliable "
                "information to answer this "
                "question."
            ),
            evidence=[],
            is_grounded=False
        )

    def _value(
        self,
        metadata: dict[str, Any],
        key: str,
        default: str = ""
    ) -> str:

        value = self._string(
            metadata.get(
                key
            )
        )

        return value or default

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
        ).strip()