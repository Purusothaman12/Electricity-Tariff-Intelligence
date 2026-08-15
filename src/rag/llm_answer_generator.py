import re

from dataclasses import dataclass
from typing import Any

from src.rag.answer_generator import (
    AnswerEvidence,
    RAGAnswer,
    TariffAnswerGenerator
)
from src.rag.ollama_client import (
    OllamaLLMClient,
    OllamaLLMResponse
)
from src.rag.retriever import (
    RetrievalIntent
)


@dataclass(slots=True)
class LLMGroundedAnswer:
    """
    Represents a tariff answer enhanced by a local LLM.

    The deterministic answer remains available for auditing,
    validation and fallback.
    """

    query: str
    intent: RetrievalIntent
    answer_type: str
    answer: str
    deterministic_answer: str
    evidence: list[AnswerEvidence]
    is_grounded: bool
    generation_method: str
    model: str
    validation_passed: bool
    validation_notes: tuple[str, ...]
    prompt_tokens: int | None = None
    output_tokens: int | None = None
    total_duration_seconds: float | None = None

    def to_dict(
        self
    ) -> dict[str, Any]:

        return {
            "query": self.query,
            "intent": self.intent.value,
            "answer_type": self.answer_type,
            "answer": self.answer,
            "deterministic_answer": (
                self.deterministic_answer
            ),
            "is_grounded": self.is_grounded,
            "generation_method": (
                self.generation_method
            ),
            "model": self.model,
            "validation_passed": (
                self.validation_passed
            ),
            "validation_notes": list(
                self.validation_notes
            ),
            "prompt_tokens": self.prompt_tokens,
            "output_tokens": self.output_tokens,
            "total_duration_seconds": (
                self.total_duration_seconds
            ),
            "evidence": [
                evidence.to_dict()
                for evidence in self.evidence
            ]
        }


class OllamaTariffAnswerGenerator:
    """
    Enhances verified tariff answers using a local Ollama model.

    Processing flow:

    1. Generate a deterministic answer from structured data.
    2. Send that answer and retrieved evidence to Ollama.
    3. Validate that Ollama preserved the tariff facts.
    4. Return Ollama's answer when validation succeeds.
    5. Return the deterministic answer when validation fails.
    """

    SYSTEM_PROMPT = """
You are an electricity tariff analysis assistant.

Follow these rules strictly:

1. Use only the verified answer and tariff evidence supplied by
   the application.
2. Never invent, estimate, alter or recalculate a tariff value.
3. Never change currency amounts, percentages, effective dates,
   schedule names, Rider names or section identifiers.
4. Do not introduce facts absent from the supplied evidence.
5. Keep the answer clear, direct and professional.
6. Do not mention prompts, embeddings, vector databases,
   retrieval systems or validation.
7. Do not say that you are an AI.
8. Do not provide unsupported recommendations.
9. Preserve any statement that evidence is insufficient.
10. Preserve the meaning of the verified answer.

Return only the final answer.
""".strip()

    IMPORTANT_METADATA_FIELDS = (
        "utility",
        "source_file",
        "old_source_file",
        "new_source_file",
        "schedule_id",
        "schedule_title",
        "section_id",
        "section_title",
        "category",
        "charge_name",
        "normalized_charge_name",
        "value_text",
        "old_value_text",
        "new_value_text",
        "unit",
        "effective_date",
        "old_effective_date",
        "new_effective_date",
        "status",
        "absolute_change",
        "percent_change",
        "applicability_status"
    )

    PROTECTED_METADATA_FIELDS = (
        "schedule_title",
        "section_id",
        "section_title",
        "charge_name",
        "value_text",
        "old_value_text",
        "new_value_text",
        "effective_date",
        "old_effective_date",
        "new_effective_date"
    )

    GENERIC_FACT_WORDS = {
        "AND",
        "CHARGE",
        "FACTOR",
        "FOR",
        "OF",
        "RATE",
        "RIDER",
        "SCHEDULE",
        "SERVICE",
        "THE",
        "TO"
    }

    MONTH_NAMES = (
        "January",
        "February",
        "March",
        "April",
        "May",
        "June",
        "July",
        "August",
        "September",
        "October",
        "November",
        "December"
    )

    def __init__(
        self,
        deterministic_generator: (
            TariffAnswerGenerator
        ),
        llm_client: OllamaLLMClient,
        max_evidence_chunks: int = 6,
        max_output_tokens: int = 500,
        fallback_on_error: bool = True,
        fallback_on_validation_failure: bool = True
    ) -> None:

        if max_evidence_chunks <= 0:

            raise ValueError(
                "max_evidence_chunks must be "
                "greater than zero."
            )

        if max_output_tokens <= 0:

            raise ValueError(
                "max_output_tokens must be "
                "greater than zero."
            )

        self.deterministic_generator = (
            deterministic_generator
        )

        self.llm_client = llm_client

        self.max_evidence_chunks = (
            max_evidence_chunks
        )

        self.max_output_tokens = (
            max_output_tokens
        )

        self.fallback_on_error = (
            fallback_on_error
        )

        self.fallback_on_validation_failure = (
            fallback_on_validation_failure
        )

    def answer(
        self,
        query: str,
        top_k: int = 8
    ) -> LLMGroundedAnswer:
        """
        Produces one validated local-LLM tariff answer.
        """

        cleaned_query = self._clean_text(
            query
        )

        if not cleaned_query:

            raise ValueError(
                "The question cannot be empty."
            )

        if top_k <= 0:

            raise ValueError(
                "top_k must be greater than zero."
            )

        deterministic_answer = (
            self.deterministic_generator.answer(
                query=cleaned_query,
                top_k=top_k
            )
        )

        if not deterministic_answer.is_grounded:

            return self._build_fallback_result(
                deterministic_answer=(
                    deterministic_answer
                ),
                generation_method=(
                    "DETERMINISTIC_INSUFFICIENT_"
                    "EVIDENCE"
                ),
                validation_passed=True,
                validation_notes=(
                    "The structured answer was not "
                    "grounded, so LLM generation was "
                    "not attempted.",
                )
            )

        if not deterministic_answer.evidence:

            return self._build_fallback_result(
                deterministic_answer=(
                    deterministic_answer
                ),
                generation_method=(
                    "DETERMINISTIC_NO_EVIDENCE"
                ),
                validation_passed=True,
                validation_notes=(
                    "No evidence chunks were "
                    "available for LLM generation.",
                )
            )

        user_prompt = self._build_user_prompt(
            deterministic_answer
        )

        try:

            llm_response = (
                self.llm_client.generate(
                    system_prompt=(
                        self.SYSTEM_PROMPT
                    ),
                    user_prompt=user_prompt,
                    max_output_tokens=(
                        self.max_output_tokens
                    ),
                    temperature=0.0
                )
            )

        except Exception as error:

            if not self.fallback_on_error:
                raise

            return self._build_fallback_result(
                deterministic_answer=(
                    deterministic_answer
                ),
                generation_method=(
                    "DETERMINISTIC_LLM_ERROR"
                ),
                validation_passed=False,
                validation_notes=(
                    "Ollama generation failed: "
                    f"{error}",
                )
            )

        (
            validation_passed,
            validation_notes
        ) = self._validate_llm_answer(
            generated_answer=(
                llm_response.text
            ),
            deterministic_answer=(
                deterministic_answer
            )
        )

        if not validation_passed:

            if (
                not self
                .fallback_on_validation_failure
            ):

                raise RuntimeError(
                    "The Ollama answer failed "
                    "grounding validation: "
                    + "; ".join(
                        validation_notes
                    )
                )

            return self._build_fallback_result(
                deterministic_answer=(
                    deterministic_answer
                ),
                generation_method=(
                    "DETERMINISTIC_VALIDATION_"
                    "FALLBACK"
                ),
                validation_passed=False,
                validation_notes=tuple(
                    validation_notes
                ),
                llm_response=llm_response
            )

        return LLMGroundedAnswer(
            query=deterministic_answer.query,
            intent=deterministic_answer.intent,
            answer_type=(
                deterministic_answer.answer_type
            ),
            answer=self._clean_text(
                llm_response.text
            ),
            deterministic_answer=(
                deterministic_answer.answer
            ),
            evidence=list(
                deterministic_answer.evidence
            ),
            is_grounded=True,
            generation_method="OLLAMA",
            model=llm_response.model,
            validation_passed=True,
            validation_notes=tuple(
                validation_notes
            ),
            prompt_tokens=(
                llm_response.prompt_tokens
            ),
            output_tokens=(
                llm_response.output_tokens
            ),
            total_duration_seconds=(
                llm_response
                .total_duration_seconds
            )
        )

    def _build_user_prompt(
        self,
        deterministic_answer: RAGAnswer
    ) -> str:
        """
        Builds the evidence-grounded prompt sent to Ollama.
        """

        evidence_sections = []

        selected_evidence = (
            deterministic_answer.evidence[
                :self.max_evidence_chunks
            ]
        )

        for index, evidence in enumerate(
            selected_evidence,
            start=1
        ):

            metadata_text = (
                self._format_metadata(
                    evidence.metadata
                )
            )

            evidence_sections.append(
                "\n".join(
                    [
                        f"EVIDENCE {index}",
                        (
                            "Content: "
                            f"{evidence.content}"
                        ),
                        (
                            "Metadata: "
                            f"{metadata_text}"
                        )
                    ]
                )
            )

        return "\n".join(
            [
                "USER QUESTION:",
                deterministic_answer.query,
                "",
                "VERIFIED STRUCTURED ANSWER:",
                deterministic_answer.answer,
                "",
                "VERIFIED TARIFF EVIDENCE:",
                "\n\n".join(
                    evidence_sections
                ),
                "",
                "TASK:",
                (
                    "Rewrite the verified structured "
                    "answer as a clear natural-language "
                    "answer."
                ),
                (
                    "Preserve every rate value, date, "
                    "percentage, tariff identity and "
                    "comparison result."
                ),
                (
                    "Do not add information that is "
                    "not present above."
                )
            ]
        )

    def _format_metadata(
        self,
        metadata: dict[str, Any]
    ) -> str:

        metadata_parts = []

        for field_name in (
            self.IMPORTANT_METADATA_FIELDS
        ):

            value = self._clean_text(
                metadata.get(
                    field_name
                )
            )

            if not value:
                continue

            metadata_parts.append(
                f"{field_name}={value}"
            )

        return " | ".join(
            metadata_parts
        )

    def _validate_llm_answer(
        self,
        generated_answer: str,
        deterministic_answer: RAGAnswer
    ) -> tuple[bool, list[str]]:
        """
        Checks that the generated answer remains grounded.
        """

        notes = []

        cleaned_generated_answer = (
            self._clean_text(
                generated_answer
            )
        )

        if not cleaned_generated_answer:

            return (
                False,
                [
                    "The LLM returned an empty answer."
                ]
            )

        if len(
            cleaned_generated_answer
        ) < 10:

            return (
                False,
                [
                    "The LLM answer was too short."
                ]
            )

        protected_facts = (
            self._collect_protected_facts(
                deterministic_answer
            )
        )

        missing_facts = [
            fact
            for fact in protected_facts
            if not self._contains_fact(
                answer=cleaned_generated_answer,
                fact=fact
            )
        ]

        if missing_facts:

            notes.append(
                "Missing protected facts: "
                + ", ".join(
                    missing_facts
                )
            )

        allowed_text_parts = [
            deterministic_answer.query,
            deterministic_answer.answer
        ]

        allowed_text_parts.extend(
            evidence.content
            for evidence
            in deterministic_answer.evidence
        )

        for evidence in (
            deterministic_answer.evidence
        ):

            allowed_text_parts.extend(
                self._clean_text(
                    value
                )
                for value
                in evidence.metadata.values()
            )

        allowed_numeric_tokens = (
            self._extract_numeric_tokens(
                " ".join(
                    allowed_text_parts
                )
            )
        )

        generated_numeric_tokens = (
            self._extract_numeric_tokens(
                cleaned_generated_answer
            )
        )

        unsupported_numbers = sorted(
            generated_numeric_tokens
            - allowed_numeric_tokens
        )

        if unsupported_numbers:

            notes.append(
                "Unsupported numeric values: "
                + ", ".join(
                    unsupported_numbers
                )
            )

        forbidden_phrases = (
            "I THINK",
            "I BELIEVE",
            "PROBABLY",
            "POSSIBLY",
            "MAYBE",
            "BASED ON MY KNOWLEDGE",
            "AS AN AI"
        )

        normalized_generated_answer = (
            self._normalize_for_match(
                cleaned_generated_answer
            )
        )

        found_forbidden_phrases = [
            phrase
            for phrase in forbidden_phrases
            if phrase
            in normalized_generated_answer
        ]

        if found_forbidden_phrases:

            notes.append(
                "Uncertain or unsupported wording: "
                + ", ".join(
                    found_forbidden_phrases
                )
            )

        validation_passed = not notes

        if validation_passed:

            notes.append(
                "The Ollama answer preserved the "
                "verified tariff facts."
            )

        return (
            validation_passed,
            notes
        )

    def _collect_protected_facts(
        self,
        deterministic_answer: RAGAnswer
    ) -> list[str]:
        """
        Collects facts that must remain in the generated answer.
        """

        deterministic_text = (
            deterministic_answer.answer
        )

        protected_facts = []

        protected_facts.extend(
            self._extract_currency_values(
                deterministic_text
            )
        )

        protected_facts.extend(
            self._extract_percentage_values(
                deterministic_text
            )
        )

        protected_facts.extend(
            self._extract_dates(
                deterministic_text
            )
        )

        normalized_deterministic_text = (
            self._normalize_for_match(
                deterministic_text
            )
        )

        for evidence in (
            deterministic_answer.evidence
        ):

            for field_name in (
                self.PROTECTED_METADATA_FIELDS
            ):

                value = self._clean_text(
                    evidence.metadata.get(
                        field_name
                    )
                )

                if not value:
                    continue

                normalized_value = (
                    self._normalize_for_match(
                        value
                    )
                )

                if (
                    normalized_value
                    in normalized_deterministic_text
                ):

                    protected_facts.append(
                        value
                    )

        unique_facts = []
        seen_facts = set()

        for fact in protected_facts:

            normalized_fact = (
                self._normalize_for_match(
                    fact
                )
            )

            if not normalized_fact:
                continue

            if normalized_fact in seen_facts:
                continue

            seen_facts.add(
                normalized_fact
            )

            unique_facts.append(
                fact
            )

        return unique_facts

    def _contains_fact(
        self,
        answer: str,
        fact: str
    ) -> bool:
        """
        Checks exact values strictly and textual tariff labels
        using their meaningful tokens.

        Example:

            RESIDENTIAL SERVICE

        can be preserved as:

            Residential Customer Charge

        because the identifying token RESIDENTIAL remains.
        """

        normalized_answer = (
            self._normalize_for_match(
                answer
            )
        )

        normalized_fact = (
            self._normalize_for_match(
                fact
            )
        )

        if not normalized_fact:
            return False

        if normalized_fact in normalized_answer:
            return True

        if self._is_strict_fact(
            fact
        ):

            return False

        fact_tokens = (
            self._significant_fact_tokens(
                fact
            )
        )

        if not fact_tokens:
            return False

        answer_tokens = set(
            re.findall(
                r"[A-Z0-9]+",
                normalized_answer
            )
        )

        return fact_tokens.issubset(
            answer_tokens
        )

    def _is_strict_fact(
        self,
        fact: str
    ) -> bool:
        """
        Numeric values, dates and identifiers require exact
        preservation.
        """

        if re.search(
            r"\d",
            fact
        ):

            return True

        if "$" in fact or "%" in fact:
            return True

        return False

    def _significant_fact_tokens(
        self,
        fact: str
    ) -> set[str]:

        tokens = set(
            re.findall(
                r"[A-Z0-9]+",
                self._normalize_for_match(
                    fact
                )
            )
        )

        return {
            token
            for token in tokens
            if (
                token
                not in self.GENERIC_FACT_WORDS
                and len(token) > 1
            )
        }

    def _extract_currency_values(
        self,
        text: str
    ) -> list[str]:

        return re.findall(
            r"\$\s*-?\d[\d,]*"
            r"(?:\.\d+)?",
            text
        )

    def _extract_percentage_values(
        self,
        text: str
    ) -> list[str]:

        return re.findall(
            r"(?<![A-Z0-9])"
            r"-?\d+(?:\.\d+)?%",
            text,
            flags=re.IGNORECASE
        )

    def _extract_dates(
        self,
        text: str
    ) -> list[str]:

        month_pattern = "|".join(
            self.MONTH_NAMES
        )

        return re.findall(
            (
                rf"\b(?:{month_pattern})"
                r"\s+\d{1,2},\s+\d{4}\b"
            ),
            text,
            flags=re.IGNORECASE
        )

    def _extract_numeric_tokens(
        self,
        text: str
    ) -> set[str]:
        """
        Extracts comparable numeric values.

        Currency and percentage symbols are removed here because
        protected-fact validation separately checks their required
        presence.

        Therefore:

            0.53
            $0.53

        are treated as the same underlying numeric value for the
        unsupported-number check.
        """

        raw_tokens = re.findall(
            (
                r"(?<![A-Z0-9])"
                r"\$?"
                r"-?"
                r"\(?"
                r"\d+(?:,\d{3})*"
                r"(?:\.\d+)?"
                r"\)?"
                r"%?"
                r"(?![A-Z0-9])"
            ),
            text,
            flags=re.IGNORECASE
        )

        normalized_tokens = set()

        for token in raw_tokens:

            normalized_token = (
                self._normalize_numeric_token(
                    token
                )
            )

            if normalized_token:

                normalized_tokens.add(
                    normalized_token
                )

        return normalized_tokens

    def _normalize_numeric_token(
        self,
        token: str
    ) -> str:
        """
        Normalizes numeric formatting without changing its value.
        """

        cleaned_token = (
            token
            .strip()
            .replace(
                "$",
                ""
            )
            .replace(
                "%",
                ""
            )
            .replace(
                ",",
                ""
            )
            .replace(
                " ",
                ""
            )
        )

        is_parenthesized_negative = (
            cleaned_token.startswith(
                "("
            )
            and cleaned_token.endswith(
                ")"
            )
        )

        cleaned_token = (
            cleaned_token
            .replace(
                "(",
                ""
            )
            .replace(
                ")",
                ""
            )
        )

        if (
            is_parenthesized_negative
            and not cleaned_token.startswith(
                "-"
            )
        ):

            cleaned_token = (
                "-"
                + cleaned_token
            )

        return cleaned_token.upper()

    def _normalize_for_match(
        self,
        value: Any
    ) -> str:

        text = self._clean_text(
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

    def _build_fallback_result(
        self,
        deterministic_answer: RAGAnswer,
        generation_method: str,
        validation_passed: bool,
        validation_notes: tuple[str, ...],
        llm_response: (
            OllamaLLMResponse | None
        ) = None
    ) -> LLMGroundedAnswer:

        return LLMGroundedAnswer(
            query=deterministic_answer.query,
            intent=deterministic_answer.intent,
            answer_type=(
                deterministic_answer.answer_type
            ),
            answer=deterministic_answer.answer,
            deterministic_answer=(
                deterministic_answer.answer
            ),
            evidence=list(
                deterministic_answer.evidence
            ),
            is_grounded=(
                deterministic_answer.is_grounded
            ),
            generation_method=(
                generation_method
            ),
            model=(
                llm_response.model
                if llm_response is not None
                else self.llm_client.model
            ),
            validation_passed=(
                validation_passed
            ),
            validation_notes=(
                validation_notes
            ),
            prompt_tokens=(
                llm_response.prompt_tokens
                if llm_response is not None
                else None
            ),
            output_tokens=(
                llm_response.output_tokens
                if llm_response is not None
                else None
            ),
            total_duration_seconds=(
                llm_response
                .total_duration_seconds
                if llm_response is not None
                else None
            )
        )

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