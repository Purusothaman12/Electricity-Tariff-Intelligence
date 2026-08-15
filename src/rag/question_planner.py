import re

from dataclasses import dataclass
from enum import Enum
from typing import Any

from src.rag.retriever import (
    RetrievalIntent
)


class TariffQuestionType(str, Enum):
    """
    Supported electricity-tariff question categories.
    """

    RATE_LOOKUP = "RATE_LOOKUP"

    RATE_LIST = "RATE_LIST"

    RATE_COMPARISON = "RATE_COMPARISON"

    RIDER_CHANGE = "RIDER_CHANGE"

    SECTION_COVERAGE = "SECTION_COVERAGE"

    GENERAL_TARIFF = "GENERAL_TARIFF"

    OUT_OF_SCOPE = "OUT_OF_SCOPE"


@dataclass(slots=True)
class TariffQuestionPlan:
    """
    Describes how one user question should be processed.
    """

    question: str

    question_type: TariffQuestionType

    retrieval_intent: RetrievalIntent

    retrieval_top_k: int

    prompt_evidence_limit: int

    use_deterministic_answer: bool

    allow_llm: bool

    extracted_years: tuple[int, ...]

    matched_terms: tuple[str, ...]

    reason: str

    def to_dict(
        self
    ) -> dict[str, Any]:

        return {
            "question": self.question,
            "question_type": (
                self.question_type.value
            ),
            "retrieval_intent": (
                self.retrieval_intent.value
            ),
            "retrieval_top_k": (
                self.retrieval_top_k
            ),
            "prompt_evidence_limit": (
                self.prompt_evidence_limit
            ),
            "use_deterministic_answer": (
                self.use_deterministic_answer
            ),
            "allow_llm": self.allow_llm,
            "extracted_years": list(
                self.extracted_years
            ),
            "matched_terms": list(
                self.matched_terms
            ),
            "reason": self.reason
        }


class TariffQuestionPlanner:
    """
    Classifies a natural-language question and determines the
    retrieval and answer-generation strategy.

    Examples:

        What is the Residential Customer Charge?
            -> RATE_LOOKUP

        What charges are available under Lighting Service?
            -> RATE_LIST

        Compare the Residential Transmission System Charge
        in 2018 and 2023.
            -> RATE_COMPARISON

        Which Riders were added?
            -> RIDER_CHANGE

        Which tariff sections are not applicable?
            -> SECTION_COVERAGE

        What is tomorrow's weather?
            -> OUT_OF_SCOPE
    """

    MAX_RETRIEVAL_TOP_K = 50

    STRONG_TARIFF_TERMS = (
        "electricity tariff",
        "utility tariff",
        "tariff rate",
        "tariff rates",
        "rate schedule",
        "rate schedules",
        "customer charge",
        "customer charges",
        "metering charge",
        "metering charges",
        "distribution charge",
        "distribution charges",
        "distribution system charge",
        "distribution system charges",
        "transmission charge",
        "transmission charges",
        "transmission system charge",
        "transmission system charges",
        "residential service",
        "lighting service",
        "retail customer",
        "retail customers",
        "effective date",
        "effective dates",
        "not applicable",
        "oncor",
        "kilowatt hour",
        "kilowatt-hour",
        "per kwh",
        "per kw",
        "rider",
        "riders",
        "rider schedule",
        "rider schedules"
    )

    WEAK_TARIFF_TERMS = (
        "tariff",
        "tariffs",
        "rate",
        "rates",
        "charge",
        "charges",
        "schedule",
        "schedules",
        "section",
        "sections",
        "service",
        "services",
        "customer",
        "customers",
        "residential",
        "lighting",
        "metering",
        "distribution",
        "transmission",
        "effective",
        "applicable",
        "utility",
        "utilities",
        "electricity",
        "kwh",
        "kw",
        "rider",
        "riders"
    )

    LIST_PATTERNS = (
        r"\bwhat\s+charges\b",
        r"\bwhich\s+charges\b",
        r"\blist\s+(?:all\s+)?charges\b",
        r"\bshow\s+(?:all\s+)?charges\b",
        r"\ball\s+charges\b",
        r"\bavailable\s+charges\b",
        r"\bcharges\s+(?:in|under|within|for)\b",
        r"\bwhat\s+rates\b",
        r"\bwhich\s+rates\b",
        r"\blist\s+(?:all\s+)?rates\b",
        r"\bshow\s+(?:all\s+)?rates\b",
        r"\ball\s+rates\b",
        r"\bavailable\s+rates\b",
        r"\bwhat\s+is\s+included\b",
        r"\bwhat\s+are\s+included\b"
    )

    COMPARISON_PATTERNS = (
        r"\bcompare\b",
        r"\bcomparison\b",
        r"\bdifference\b",
        r"\bdifferences\b",
        r"\bchanged\b",
        r"\bchange\s+between\b",
        r"\bincreased\b",
        r"\bdecreased\b",
        r"\bincrease\b",
        r"\bdecrease\b",
        r"\bold\s+and\s+new\b",
        r"\bprevious\s+and\s+current\b",
        r"\bearlier\s+and\s+later\b",
        r"\bversus\b",
        r"\bvs\.?\b"
    )

    RIDER_CHANGE_PATTERNS = (
        r"\badded\s+riders?\b",
        r"\bnew\s+riders?\b",
        r"\bremoved\s+riders?\b",
        r"\bdeleted\s+riders?\b",
        r"\bwhich\s+riders?\s+were\s+added\b",
        r"\bwhich\s+riders?\s+were\s+removed\b",
        r"\bwhat\s+riders?\s+were\s+added\b",
        r"\bwhat\s+riders?\s+were\s+removed\b",
        r"\brider\s+changes?\b",
        r"\briders?\s+added\b",
        r"\briders?\s+removed\b",
        r"\bnew\s+rider\s+schedules?\b",
        r"\bremoved\s+rider\s+schedules?\b"
    )

    SECTION_COVERAGE_PATTERNS = (
        r"\bnot\s+applicable\b",
        r"\bapplicable\s+sections?\b",
        r"\bsection\s+coverage\b",
        r"\bsections?\s+marked\b",
        r"\bwhich\s+sections?\b",
        r"\bsection\s+applicability\b"
    )

    LOOKUP_PATTERNS = (
        r"\bwhat\s+is\b",
        r"\bwhat\s+was\b",
        r"\bhow\s+much\b",
        r"\bfind\b",
        r"\bshow\s+me\b",
        r"\brate\s+for\b",
        r"\bcharge\s+for\b",
        r"\beffective\s+(?:in|on|from)\b",
        r"\bcurrent\s+rate\b",
        r"\bcurrent\s+charge\b"
    )

    def plan(
        self,
        question: str,
        requested_top_k: int = 8
    ) -> TariffQuestionPlan:
        """
        Creates a retrieval and answer-generation plan for one
        user question.
        """

        cleaned_question = (
            self._validate_question(
                question
            )
        )

        validated_top_k = (
            self._validate_top_k(
                requested_top_k
            )
        )

        normalized_question = (
            self._normalize_text(
                cleaned_question
            )
        )

        matched_terms = (
            self._find_tariff_terms(
                normalized_question
            )
        )

        extracted_years = (
            self._extract_years(
                normalized_question
            )
        )

        rider_change_match = (
            self._matches_any(
                normalized_question,
                self.RIDER_CHANGE_PATTERNS
            )
        )

        section_coverage_match = (
            self._matches_any(
                normalized_question,
                self.SECTION_COVERAGE_PATTERNS
            )
        )

        comparison_match = (
            self._matches_any(
                normalized_question,
                self.COMPARISON_PATTERNS
            )
            or len(
                extracted_years
            ) >= 2
        )

        rate_list_match = (
            self._matches_any(
                normalized_question,
                self.LIST_PATTERNS
            )
        )

        rate_lookup_match = (
            self._matches_any(
                normalized_question,
                self.LOOKUP_PATTERNS
            )
        )

        tariff_scope_match = (
            self._is_tariff_question(
                normalized_question,
                matched_terms
            )
        )

        # Rider-change questions are handled before the general
        # scope rejection because their wording may contain only
        # a small number of general tariff terms.
        if rider_change_match:

            return self._build_plan(
                question=cleaned_question,
                question_type=(
                    TariffQuestionType
                    .RIDER_CHANGE
                ),
                retrieval_intent=(
                    RetrievalIntent.COMPARISON
                ),
                requested_top_k=(
                    validated_top_k
                ),
                minimum_retrieval_top_k=20,
                prompt_evidence_limit=12,
                use_deterministic_answer=True,
                extracted_years=(
                    extracted_years
                ),
                matched_terms=(
                    matched_terms
                ),
                reason=(
                    "The question asks which Rider "
                    "schedules were added or removed."
                )
            )

        if (
            section_coverage_match
            and tariff_scope_match
        ):

            return self._build_plan(
                question=cleaned_question,
                question_type=(
                    TariffQuestionType
                    .SECTION_COVERAGE
                ),
                retrieval_intent=(
                    RetrievalIntent
                    .SECTION_COVERAGE
                ),
                requested_top_k=(
                    validated_top_k
                ),
                minimum_retrieval_top_k=25,
                prompt_evidence_limit=20,
                use_deterministic_answer=True,
                extracted_years=(
                    extracted_years
                ),
                matched_terms=(
                    matched_terms
                ),
                reason=(
                    "The question asks about section "
                    "applicability or coverage."
                )
            )

        if not tariff_scope_match:

            return TariffQuestionPlan(
                question=cleaned_question,
                question_type=(
                    TariffQuestionType
                    .OUT_OF_SCOPE
                ),
                retrieval_intent=(
                    RetrievalIntent.GENERAL
                ),
                retrieval_top_k=0,
                prompt_evidence_limit=0,
                use_deterministic_answer=False,
                allow_llm=False,
                extracted_years=(
                    extracted_years
                ),
                matched_terms=(
                    matched_terms
                ),
                reason=(
                    "The question does not contain "
                    "enough electricity-tariff context."
                )
            )

        if comparison_match:

            return self._build_plan(
                question=cleaned_question,
                question_type=(
                    TariffQuestionType
                    .RATE_COMPARISON
                ),
                retrieval_intent=(
                    RetrievalIntent.COMPARISON
                ),
                requested_top_k=(
                    validated_top_k
                ),
                minimum_retrieval_top_k=8,
                prompt_evidence_limit=1,
                use_deterministic_answer=True,
                extracted_years=(
                    extracted_years
                ),
                matched_terms=(
                    matched_terms
                ),
                reason=(
                    "The question asks for a change "
                    "or comparison between tariff "
                    "rates or effective periods."
                )
            )

        if rate_list_match:

            return self._build_plan(
                question=cleaned_question,
                question_type=(
                    TariffQuestionType
                    .RATE_LIST
                ),
                retrieval_intent=(
                    RetrievalIntent.RATE_LOOKUP
                ),
                requested_top_k=(
                    validated_top_k
                ),
                minimum_retrieval_top_k=50,
                prompt_evidence_limit=20,
                use_deterministic_answer=False,
                extracted_years=(
                    extracted_years
                ),
                matched_terms=(
                    matched_terms
                ),
                reason=(
                    "The question asks for multiple "
                    "charges or rates within a tariff "
                    "service or schedule."
                )
            )

        if rate_lookup_match:

            return self._build_plan(
                question=cleaned_question,
                question_type=(
                    TariffQuestionType
                    .RATE_LOOKUP
                ),
                retrieval_intent=(
                    RetrievalIntent.RATE_LOOKUP
                ),
                requested_top_k=(
                    validated_top_k
                ),
                minimum_retrieval_top_k=8,
                prompt_evidence_limit=1,
                use_deterministic_answer=True,
                extracted_years=(
                    extracted_years
                ),
                matched_terms=(
                    matched_terms
                ),
                reason=(
                    "The question asks for one "
                    "specific tariff rate or charge."
                )
            )

        return self._build_plan(
            question=cleaned_question,
            question_type=(
                TariffQuestionType
                .GENERAL_TARIFF
            ),
            retrieval_intent=(
                RetrievalIntent.GENERAL
            ),
            requested_top_k=(
                validated_top_k
            ),
            minimum_retrieval_top_k=15,
            prompt_evidence_limit=10,
            use_deterministic_answer=False,
            extracted_years=(
                extracted_years
            ),
            matched_terms=(
                matched_terms
            ),
            reason=(
                "The question is tariff-related but "
                "does not match a narrower category."
            )
        )

    def _build_plan(
        self,
        question: str,
        question_type: TariffQuestionType,
        retrieval_intent: RetrievalIntent,
        requested_top_k: int,
        minimum_retrieval_top_k: int,
        prompt_evidence_limit: int,
        use_deterministic_answer: bool,
        extracted_years: tuple[int, ...],
        matched_terms: tuple[str, ...],
        reason: str
    ) -> TariffQuestionPlan:
        """
        Creates a validated question-plan instance.
        """

        retrieval_top_k = min(
            max(
                requested_top_k,
                minimum_retrieval_top_k
            ),
            self.MAX_RETRIEVAL_TOP_K
        )

        return TariffQuestionPlan(
            question=question,
            question_type=question_type,
            retrieval_intent=retrieval_intent,
            retrieval_top_k=(
                retrieval_top_k
            ),
            prompt_evidence_limit=(
                prompt_evidence_limit
            ),
            use_deterministic_answer=(
                use_deterministic_answer
            ),
            allow_llm=True,
            extracted_years=(
                extracted_years
            ),
            matched_terms=(
                matched_terms
            ),
            reason=reason
        )

    def _is_tariff_question(
        self,
        normalized_question: str,
        matched_terms: tuple[str, ...]
    ) -> bool:
        """
        Determines whether the question belongs to the indexed
        electricity-tariff domain.
        """

        strong_matches = [
            term
            for term in matched_terms
            if term in self.STRONG_TARIFF_TERMS
        ]

        if strong_matches:
            return True

        weak_matches = [
            term
            for term in matched_terms
            if term in self.WEAK_TARIFF_TERMS
        ]

        if len(
            set(
                weak_matches
            )
        ) >= 2:

            return True

        tariff_identifier_pattern = (
            r"\b6(?:\.\d+){2,}\b"
        )

        if re.search(
            tariff_identifier_pattern,
            normalized_question
        ):

            return True

        return False

    def _find_tariff_terms(
        self,
        normalized_question: str
    ) -> tuple[str, ...]:
        """
        Returns tariff-related terms found in the question.
        """

        matched_terms = []

        all_terms = (
            self.STRONG_TARIFF_TERMS
            + self.WEAK_TARIFF_TERMS
        )

        for term in all_terms:

            pattern = (
                r"\b"
                + re.escape(
                    term
                )
                + r"\b"
            )

            if re.search(
                pattern,
                normalized_question
            ):

                matched_terms.append(
                    term
                )

        return tuple(
            dict.fromkeys(
                matched_terms
            )
        )

    def _extract_years(
        self,
        normalized_question: str
    ) -> tuple[int, ...]:
        """
        Extracts unique four-digit years from the question.
        """

        year_values = re.findall(
            r"(?<!\d)(?:19|20)\d{2}(?!\d)",
            normalized_question
        )

        return tuple(
            dict.fromkeys(
                int(
                    year
                )
                for year in year_values
            )
        )

    def _matches_any(
        self,
        normalized_question: str,
        patterns: tuple[str, ...]
    ) -> bool:

        return any(
            re.search(
                pattern,
                normalized_question
            )
            is not None
            for pattern in patterns
        )

    def _validate_question(
        self,
        question: str
    ) -> str:

        if not isinstance(
            question,
            str
        ):

            raise TypeError(
                "question must be a string."
            )

        cleaned_question = " ".join(
            question.split()
        )

        if not cleaned_question:

            raise ValueError(
                "question cannot be empty."
            )

        return cleaned_question

    def _validate_top_k(
        self,
        top_k: int
    ) -> int:

        if (
            isinstance(
                top_k,
                bool
            )
            or not isinstance(
                top_k,
                int
            )
        ):

            raise TypeError(
                "requested_top_k must be an integer."
            )

        if top_k <= 0:

            raise ValueError(
                "requested_top_k must be greater "
                "than zero."
            )

        return min(
            top_k,
            self.MAX_RETRIEVAL_TOP_K
        )

    def _normalize_text(
        self,
        value: str
    ) -> str:

        normalized_value = (
            value.lower()
        )

        normalized_value = (
            normalized_value
            .replace(
                "\u2013",
                "-"
            )
            .replace(
                "\u2014",
                "-"
            )
        )

        normalized_value = re.sub(
            r"[^a-z0-9.$%/\-\s]",
            " ",
            normalized_value
        )

        normalized_value = re.sub(
            r"\s+",
            " ",
            normalized_value
        )

        return normalized_value.strip()