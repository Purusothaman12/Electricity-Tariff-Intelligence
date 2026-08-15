import re

from dataclasses import dataclass
from decimal import Decimal
from decimal import InvalidOperation
from typing import Any

from src.rag.answer_generator import (
    TariffAnswerGenerator
)
from src.rag.evidence_selector import (
    EvidenceSelection,
    TariffEvidenceSelector
)
from src.rag.ollama_client import (
    OllamaLLMClient,
    OllamaLLMResponse
)
from src.rag.question_planner import (
    TariffQuestionPlan,
    TariffQuestionPlanner,
    TariffQuestionType
)
from src.rag.retriever import (
    RetrievalIntent,
    RetrievalResult,
    TariffRetriever
)


@dataclass(slots=True)
class DynamicLLMAnswer:
    """
    Represents an answer produced from a dynamically planned
    tariff question.

    The response contains:

    - the final answer
    - the verified deterministic baseline
    - the question plan
    - the selected evidence
    - Ollama generation details
    - grounding-validation results
    """

    question: str

    question_type: TariffQuestionType

    intent: RetrievalIntent

    answer: str

    deterministic_answer: str

    is_grounded: bool

    generation_method: str

    model: str

    validation_passed: bool

    validation_notes: tuple[str, ...]

    plan: TariffQuestionPlan

    evidence_selection: EvidenceSelection

    prompt_tokens: int | None = None

    output_tokens: int | None = None

    total_duration_seconds: float | None = None

    def to_dict(
        self
    ) -> dict[str, Any]:

        selection_dictionary = (
            self.evidence_selection.to_dict()
        )

        selection_dictionary.pop(
            "results",
            None
        )

        return {
            "question": self.question,
            "question_type": (
                self.question_type.value
            ),
            "intent": self.intent.value,
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
            "prompt_tokens": (
                self.prompt_tokens
            ),
            "output_tokens": (
                self.output_tokens
            ),
            "total_duration_seconds": (
                self.total_duration_seconds
            ),
            "plan": self.plan.to_dict(),
            "evidence_selection": (
                selection_dictionary
            ),
            "evidence": [
                result.to_dict()
                for result
                in self.evidence_selection.results
            ]
        }


class DynamicOllamaTariffAnswerGenerator:
    """
    Handles natural-language electricity-tariff questions using
    dynamic planning, evidence selection and local Ollama
    generation.

    Supported flow:

        User question
            -> question planning
            -> semantic retrieval
            -> structured evidence selection
            -> verified factual baseline
            -> Ollama rewriting
            -> grounding validation
            -> safe deterministic fallback

    Out-of-scope questions do not call Ollama.
    """

    OUT_OF_SCOPE_ANSWER = (
        "This question is outside the indexed electricity "
        "tariff data. Please ask about tariff rates, charges, "
        "Riders, schedules, effective dates, comparisons or "
        "section applicability."
    )

    NO_EVIDENCE_ANSWER = (
        "The indexed tariff evidence does not contain enough "
        "information to answer this question."
    )

    SYSTEM_PROMPT = """
You are an electricity tariff analysis assistant.

Follow these rules strictly:

1. Answer only from the verified factual baseline and selected
   tariff evidence supplied by the application.
2. Never use general knowledge or external knowledge.
3. Never invent, estimate, alter or recalculate a tariff value.
4. Never change a rate, effective date, schedule name, charge
   name, Rider name, section identifier or comparison status.
5. Do not introduce facts absent from the supplied evidence.
6. When a rate is not present in the newer tariff, clearly state
   that it is not present or was removed.
7. Do not claim that a list is exhaustive unless the supplied
   evidence explicitly proves completeness.
8. For list questions, use bullet points rather than numbered
   points.
9. Keep the answer clear, direct and professional.
10. Do not mention prompts, embeddings, vector databases,
    retrieval, validation or internal application logic.
11. Do not say that you are an AI.
12. When the evidence is insufficient, say so directly.

Return only the final user-facing answer.
""".strip()

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
        "rate_value",
        "value"
    )

    OLD_VALUE_FIELDS = (
        "old_value_text",
        "old_rate",
        "old_value"
    )

    NEW_VALUE_FIELDS = (
        "new_value_text",
        "new_rate",
        "new_value"
    )

    DATE_FIELDS = (
        "effective_date",
    )

    OLD_DATE_FIELDS = (
        "old_effective_date",
    )

    NEW_DATE_FIELDS = (
        "new_effective_date",
    )

    STATUS_FIELDS = (
        "status",
        "comparison_status"
    )

    UNIT_FIELDS = (
        "unit",
        "rate_unit"
    )

    SOURCE_FIELDS = (
        "source_file",
    )

    OLD_SOURCE_FIELDS = (
        "old_source_file",
    )

    NEW_SOURCE_FIELDS = (
        "new_source_file",
    )

    ABSOLUTE_CHANGE_FIELDS = (
        "absolute_change",
    )

    PERCENT_CHANGE_FIELDS = (
        "percent_change",
        "percentage_change"
    )

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
        retriever: TariffRetriever,
        deterministic_generator: (
            TariffAnswerGenerator
        ),
        llm_client: OllamaLLMClient,
        planner: TariffQuestionPlanner | None = None,
        evidence_selector: (
            TariffEvidenceSelector | None
        ) = None,
        max_output_tokens: int = 700,
        fallback_on_error: bool = True,
        fallback_on_validation_failure: bool = True
    ) -> None:

        if max_output_tokens <= 0:

            raise ValueError(
                "max_output_tokens must be "
                "greater than zero."
            )

        self.retriever = retriever

        self.deterministic_generator = (
            deterministic_generator
        )

        self.llm_client = llm_client

        self.planner = (
            planner
            or TariffQuestionPlanner()
        )

        self.evidence_selector = (
            evidence_selector
            or TariffEvidenceSelector()
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
        question: str,
        requested_top_k: int = 8
    ) -> DynamicLLMAnswer:
        """
        Answers one dynamically supplied user question.
        """

        plan = self.planner.plan(
            question=question,
            requested_top_k=(
                requested_top_k
            )
        )

        if not plan.allow_llm:

            empty_selection = (
                self.evidence_selector.select(
                    plan=plan,
                    results=[]
                )
            )

            return DynamicLLMAnswer(
                question=plan.question,
                question_type=(
                    plan.question_type
                ),
                intent=plan.retrieval_intent,
                answer=(
                    self.OUT_OF_SCOPE_ANSWER
                ),
                deterministic_answer=(
                    self.OUT_OF_SCOPE_ANSWER
                ),
                is_grounded=False,
                generation_method=(
                    "OUT_OF_SCOPE"
                ),
                model="",
                validation_passed=True,
                validation_notes=(
                    "The question was rejected "
                    "before retrieval and LLM "
                    "generation.",
                ),
                plan=plan,
                evidence_selection=(
                    empty_selection
                )
            )

        retrieval_results = (
            self.retriever.retrieve(
                query=plan.question,
                top_k=(
                    plan.retrieval_top_k
                ),
                intent=(
                    plan.retrieval_intent
                )
            )
        )

        evidence_selection = (
            self.evidence_selector.select(
                plan=plan,
                results=retrieval_results
            )
        )

        if not evidence_selection.results:

            return DynamicLLMAnswer(
                question=plan.question,
                question_type=(
                    plan.question_type
                ),
                intent=plan.retrieval_intent,
                answer=(
                    self.NO_EVIDENCE_ANSWER
                ),
                deterministic_answer=(
                    self.NO_EVIDENCE_ANSWER
                ),
                is_grounded=False,
                generation_method=(
                    "NO_EVIDENCE"
                ),
                model="",
                validation_passed=True,
                validation_notes=(
                    "No usable tariff evidence "
                    "was selected.",
                ),
                plan=plan,
                evidence_selection=(
                    evidence_selection
                )
            )

        deterministic_answer = (
            self._build_verified_baseline(
                plan=plan,
                selection=(
                    evidence_selection
                )
            )
        )

        user_prompt = self._build_user_prompt(
            plan=plan,
            deterministic_answer=(
                deterministic_answer
            ),
            selection=(
                evidence_selection
            )
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
                plan=plan,
                selection=(
                    evidence_selection
                ),
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
        ) = self._validate_generated_answer(
            generated_answer=(
                llm_response.text
            ),
            deterministic_answer=(
                deterministic_answer
            ),
            plan=plan,
            selection=(
                evidence_selection
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
                plan=plan,
                selection=(
                    evidence_selection
                ),
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

        return DynamicLLMAnswer(
            question=plan.question,
            question_type=(
                plan.question_type
            ),
            intent=plan.retrieval_intent,
            answer=self._clean_text(
                llm_response.text
            ),
            deterministic_answer=(
                deterministic_answer
            ),
            is_grounded=True,
            generation_method="OLLAMA",
            model=llm_response.model,
            validation_passed=True,
            validation_notes=tuple(
                validation_notes
            ),
            plan=plan,
            evidence_selection=(
                evidence_selection
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

    def _build_verified_baseline(
        self,
        plan: TariffQuestionPlan,
        selection: EvidenceSelection
    ) -> str:
        """
        Builds a deterministic factual answer before Ollama is
        called.

        Rider and section questions continue using the existing
        structured deterministic generator.

        Rate lookup, list and comparison questions use the
        evidence selected by the new evidence selector.
        """

        if plan.question_type in {
            TariffQuestionType.RIDER_CHANGE,
            TariffQuestionType.SECTION_COVERAGE
        }:

            deterministic_result = (
                self.deterministic_generator.answer(
                    query=plan.question,
                    top_k=(
                        plan.retrieval_top_k
                    )
                )
            )

            if (
                deterministic_result.is_grounded
                and deterministic_result.answer
            ):

                return self._clean_text(
                    deterministic_result.answer
                )

        if (
            plan.question_type
            == TariffQuestionType.RATE_LOOKUP
        ):

            return self._build_rate_lookup_baseline(
                selection.results[0]
            )

        if (
            plan.question_type
            == TariffQuestionType.RATE_COMPARISON
        ):

            return (
                self._build_comparison_baseline(
                    result=selection.results[0],
                    requested_years=(
                        plan.extracted_years
                    )
                )
            )

        if (
            plan.question_type
            == TariffQuestionType.RATE_LIST
        ):

            return self._build_rate_list_baseline(
                selection
            )

        return " ".join(
            self._get_content(
                result
            )
            for result
            in selection.results
        )

    def _build_rate_lookup_baseline(
        self,
        result: RetrievalResult
    ) -> str:

        metadata = self._get_metadata(
            result
        )

        schedule = self._first_value(
            metadata,
            self.SCHEDULE_FIELDS
        )

        charge = self._first_value(
            metadata,
            self.CHARGE_FIELDS
        )

        value = self._first_value(
            metadata,
            self.VALUE_FIELDS
        )

        unit = self._first_value(
            metadata,
            self.UNIT_FIELDS
        )

        effective_date = self._first_value(
            metadata,
            self.DATE_FIELDS
        )

        source_file = self._first_value(
            metadata,
            self.SOURCE_FIELDS
        )

        if not value:

            value = self._extract_label_value(
                self._get_content(
                    result
                ),
                "Rate value"
            )

        if not schedule:

            schedule = self._extract_label_value(
                self._get_content(
                    result
                ),
                "Schedule"
            )

        if not charge:

            charge = self._extract_label_value(
                self._get_content(
                    result
                ),
                "Charge"
            )

        if not effective_date:

            effective_date = (
                self._extract_label_value(
                    self._get_content(
                        result
                    ),
                    "Effective date"
                )
            )

        if not unit:

            unit = self._extract_label_value(
                self._get_content(
                    result
                ),
                "Unit"
            )

        if not source_file:

            source_file = (
                self._extract_source_file(
                    self._get_content(
                        result
                    )
                )
            )

        if not value:

            return self._get_content(
                result
            )

        schedule_text = (
            schedule
            or "the selected tariff schedule"
        )

        charge_text = (
            charge
            or "the selected charge"
        )

        answer_parts = [
            (
                f"For {schedule_text}, "
                f"{charge_text} is {value}"
            )
        ]

        if (
            unit
            and unit.upper()
            not in {
                "NOT SPECIFIED",
                "NOT AVAILABLE"
            }
        ):

            answer_parts[-1] += (
                f" {unit}"
            )

        if (
            effective_date
            and effective_date.upper()
            != "NOT AVAILABLE"
        ):

            answer_parts.append(
                (
                    "The effective date is "
                    f"{effective_date}"
                )
            )

        if source_file:

            answer_parts.append(
                (
                    "The source tariff document "
                    f"is {source_file}"
                )
            )

        return ". ".join(
            answer_parts
        ) + "."

    def _build_comparison_baseline(
        self,
        result: RetrievalResult,
        requested_years: tuple[int, ...]
    ) -> str:

        metadata = self._get_metadata(
            result
        )

        content = self._get_content(
            result
        )

        schedule = self._first_value(
            metadata,
            self.SCHEDULE_FIELDS
        )

        charge = self._first_value(
            metadata,
            self.CHARGE_FIELDS
        )

        status = self._first_value(
            metadata,
            self.STATUS_FIELDS
        )

        old_value = self._first_value(
            metadata,
            self.OLD_VALUE_FIELDS
        )

        new_value = self._first_value(
            metadata,
            self.NEW_VALUE_FIELDS
        )

        old_date = self._first_value(
            metadata,
            self.OLD_DATE_FIELDS
        )

        new_date = self._first_value(
            metadata,
            self.NEW_DATE_FIELDS
        )

        old_source = self._first_value(
            metadata,
            self.OLD_SOURCE_FIELDS
        )

        new_source = self._first_value(
            metadata,
            self.NEW_SOURCE_FIELDS
        )

        absolute_change = self._first_value(
            metadata,
            self.ABSOLUTE_CHANGE_FIELDS
        )

        percent_change = self._first_value(
            metadata,
            self.PERCENT_CHANGE_FIELDS
        )

        if not schedule:

            schedule = self._extract_label_value(
                content,
                "Schedule"
            )

        if not charge:

            charge = self._extract_label_value(
                content,
                "Charge"
            )

        if not status:

            status = self._extract_label_value(
                content,
                "Comparison status"
            )

        if not old_value:

            old_value = self._extract_label_value(
                content,
                "Old rate"
            )

        if not new_value:

            new_value = self._extract_label_value(
                content,
                "New rate"
            )

        if not old_date:

            old_date = (
                self._extract_effective_date_after(
                    content,
                    "Old rate"
                )
            )

        if not new_date:

            new_date = (
                self._extract_effective_date_after(
                    content,
                    "New rate"
                )
            )

        if not old_source:

            old_source = self._extract_label_value(
                content,
                "Old tariff document"
            )

        if not new_source:

            new_source = self._extract_label_value(
                content,
                "New tariff document"
            )

        if not absolute_change:

            absolute_change = (
                self._extract_label_value(
                    content,
                    "Absolute change"
                )
            )

        if not percent_change:

            percent_change = (
                self._extract_label_value(
                    content,
                    "Percentage change"
                )
            )

        schedule_text = (
            schedule
            or "the selected tariff schedule"
        )

        charge_text = (
            charge
            or "the selected charge"
        )

        old_present = self._value_is_present(
            old_value
        )

        new_present = self._value_is_present(
            new_value
        )

        old_year = self._extract_latest_year(
            old_source
        )

        new_year = self._extract_latest_year(
            new_source
        )

        if old_present and new_present:

            answer = (
                f"For {schedule_text}, "
                f"{charge_text} changed from "
                f"{old_value}"
            )

            if self._date_is_present(
                old_date
            ):

                answer += (
                    f", effective {old_date}"
                )

            answer += (
                f", to {new_value}"
            )

            if self._date_is_present(
                new_date
            ):

                answer += (
                    f", effective {new_date}"
                )

            answer += "."

        elif old_present and not new_present:

            answer = (
                f"For {schedule_text}, "
                f"{charge_text} was {old_value}"
            )

            if self._date_is_present(
                old_date
            ):

                answer += (
                    f", effective {old_date}"
                )

            answer += "."

            if new_year:

                answer += (
                    f" In the {new_year} tariff, "
                    "the corresponding rate is not "
                    "present."
                )

            else:

                answer += (
                    " The corresponding rate is not "
                    "present in the newer tariff."
                )

        elif not old_present and new_present:

            answer = (
                f"For {schedule_text}, "
                f"{charge_text} was added at "
                f"{new_value}"
            )

            if self._date_is_present(
                new_date
            ):

                answer += (
                    f", effective {new_date}"
                )

            answer += "."

        else:

            answer = content

        if status:

            answer += (
                f" Comparison status: "
                f"{status}."
            )

        if self._value_is_present(
            absolute_change
        ):

            answer += (
                f" Absolute change: "
                f"{absolute_change}."
            )

        formatted_percentage = (
            self._format_percentage(
                percent_change
            )
        )

        if formatted_percentage:

            answer += (
                f" Percentage change: "
                f"{formatted_percentage}."
            )

        for requested_year in (
            requested_years
        ):

            if str(
                requested_year
            ) not in answer:

                if (
                    requested_year == old_year
                    or requested_year == new_year
                ):

                    answer += (
                        f" Referenced tariff year: "
                        f"{requested_year}."
                    )

        return self._clean_text(
            answer
        )

    def _build_rate_list_baseline(
        self,
        selection: EvidenceSelection
    ) -> str:

        first_metadata = self._get_metadata(
            selection.results[0]
        )

        schedule = self._first_value(
            first_metadata,
            self.SCHEDULE_FIELDS
        )

        source_file = (
            selection.selected_source_file
            or self._first_value(
                first_metadata,
                self.SOURCE_FIELDS
            )
        )

        heading = (
            "Selected tariff charge records"
        )

        if schedule:

            heading += (
                f" for {schedule}"
            )

        if source_file:

            heading += (
                f" from {source_file}"
            )

        evidence_lines = []

        for result in selection.results:

            metadata = self._get_metadata(
                result
            )

            charge = self._first_value(
                metadata,
                self.CHARGE_FIELDS
            )

            value = self._first_value(
                metadata,
                self.VALUE_FIELDS
            )

            unit = self._first_value(
                metadata,
                self.UNIT_FIELDS
            )

            effective_date = self._first_value(
                metadata,
                self.DATE_FIELDS
            )

            content = self._get_content(
                result
            )

            if not charge:

                charge = self._extract_label_value(
                    content,
                    "Charge"
                )

            if not value:

                value = self._extract_label_value(
                    content,
                    "Rate value"
                )

            if not unit:

                unit = self._extract_label_value(
                    content,
                    "Unit"
                )

            if not effective_date:

                effective_date = (
                    self._extract_label_value(
                        content,
                        "Effective date"
                    )
                )

            if not charge or not value:

                evidence_lines.append(
                    f"- {content}"
                )

                continue

            line = (
                f"- {charge}: {value}"
            )

            if (
                unit
                and unit.upper()
                not in {
                    "NOT SPECIFIED",
                    "NOT AVAILABLE"
                }
            ):

                line += (
                    f" {unit}"
                )

            if self._date_is_present(
                effective_date
            ):

                line += (
                    f", effective "
                    f"{effective_date}"
                )

            evidence_lines.append(
                line
            )

        return "\n".join(
            [
                heading + ":",
                *evidence_lines
            ]
        )

    def _build_user_prompt(
        self,
        plan: TariffQuestionPlan,
        deterministic_answer: str,
        selection: EvidenceSelection
    ) -> str:

        evidence_sections = []

        for index, result in enumerate(
            selection.results,
            start=1
        ):

            result_dictionary = (
                result.to_dict()
            )

            metadata = self._get_metadata(
                result
            )

            evidence_sections.append(
                "\n".join(
                    [
                        f"EVIDENCE {index}",
                        (
                            "Content: "
                            f"{self._get_content(result)}"
                        ),
                        (
                            "Metadata: "
                            f"{self._format_metadata(metadata)}"
                        ),
                        (
                            "Chunk ID: "
                            f"{result_dictionary.get('chunk_id', '')}"
                        )
                    ]
                )
            )

        task_instruction = (
            "Answer the user question using only "
            "the verified answer and selected "
            "evidence."
        )

        if (
            plan.question_type
            == TariffQuestionType.RATE_LIST
        ):

            task_instruction = (
                "Provide a clear bullet-point list "
                "of the distinct charge records in "
                "the selected evidence. Include each "
                "rate value, unit and effective date "
                "when available. Do not use a "
                "numbered list and do not claim the "
                "list is exhaustive."
            )

        elif (
            plan.question_type
            == TariffQuestionType.RATE_COMPARISON
        ):

            task_instruction = (
                "Explain the old and new tariff "
                "positions clearly. When the new "
                "rate is not present, state that it "
                "is not present or was removed. Do "
                "not invent a replacement rate."
            )

        return "\n".join(
            [
                "USER QUESTION:",
                plan.question,
                "",
                "QUESTION TYPE:",
                plan.question_type.value,
                "",
                "VERIFIED FACTUAL ANSWER:",
                deterministic_answer,
                "",
                "SELECTED TARIFF EVIDENCE:",
                "\n\n".join(
                    evidence_sections
                ),
                "",
                "TASK:",
                task_instruction,
                (
                    " Preserve every tariff value, "
                    "date, schedule identity, charge "
                    "identity and comparison status."
                )
            ]
        )

    def _validate_generated_answer(
        self,
        generated_answer: str,
        deterministic_answer: str,
        plan: TariffQuestionPlan,
        selection: EvidenceSelection
    ) -> tuple[bool, list[str]]:

        notes = []

        cleaned_answer = self._clean_text(
            generated_answer
        )

        if not cleaned_answer:

            return (
                False,
                [
                    "Ollama returned an empty answer."
                ]
            )

        if len(
            cleaned_answer
        ) < 10:

            return (
                False,
                [
                    "The Ollama answer was too short."
                ]
            )

        allowed_text_parts = [
            plan.question,
            deterministic_answer
        ]

        for result in selection.results:

            allowed_text_parts.append(
                self._get_content(
                    result
                )
            )

            allowed_text_parts.extend(
                self._clean_text(
                    value
                )
                for value
                in self._get_metadata(
                    result
                ).values()
            )

        allowed_text = " ".join(
            allowed_text_parts
        )

        answer_without_numbering = (
            self._remove_list_numbering(
                generated_answer
            )
        )

        allowed_numbers = (
            self._extract_numeric_tokens(
                allowed_text
            )
        )

        generated_numbers = (
            self._extract_numeric_tokens(
                answer_without_numbering
            )
        )

        unsupported_numbers = sorted(
            generated_numbers
            - allowed_numbers
        )

        if unsupported_numbers:

            notes.append(
                "Unsupported numeric values: "
                + ", ".join(
                    unsupported_numbers
                )
            )

        if plan.question_type in {
            TariffQuestionType.RATE_LOOKUP,
            TariffQuestionType.RATE_COMPARISON
        }:

            required_facts = []

            required_facts.extend(
                self._extract_currency_values(
                    deterministic_answer
                )
            )

            required_facts.extend(
                self._extract_dates(
                    deterministic_answer
                )
            )

            for year in plan.extracted_years:

                year_text = str(
                    year
                )

                if year_text in deterministic_answer:

                    required_facts.append(
                        year_text
                    )

            unique_required_facts = list(
                dict.fromkeys(
                    required_facts
                )
            )

            missing_facts = [
                fact
                for fact in unique_required_facts
                if not self._contains_fact(
                    cleaned_answer,
                    fact
                )
            ]

            if missing_facts:

                notes.append(
                    "Missing protected facts: "
                    + ", ".join(
                        missing_facts
                    )
                )

        forbidden_phrases = (
            "I THINK",
            "I BELIEVE",
            "PROBABLY",
            "POSSIBLY",
            "MAYBE",
            "BASED ON MY KNOWLEDGE",
            "AS AN AI",
            "I DO NOT HAVE ACCESS"
        )

        normalized_answer = (
            cleaned_answer.upper()
        )

        found_forbidden_phrases = [
            phrase
            for phrase in forbidden_phrases
            if phrase in normalized_answer
        ]

        if found_forbidden_phrases:

            notes.append(
                "Unsupported or uncertain wording: "
                + ", ".join(
                    found_forbidden_phrases
                )
            )

        validation_passed = not notes

        if validation_passed:

            notes.append(
                "The Ollama answer preserved the "
                "selected tariff evidence."
            )

        return (
            validation_passed,
            notes
        )

    def _build_fallback_result(
        self,
        plan: TariffQuestionPlan,
        selection: EvidenceSelection,
        deterministic_answer: str,
        generation_method: str,
        validation_passed: bool,
        validation_notes: tuple[str, ...],
        llm_response: (
            OllamaLLMResponse | None
        ) = None
    ) -> DynamicLLMAnswer:

        return DynamicLLMAnswer(
            question=plan.question,
            question_type=(
                plan.question_type
            ),
            intent=plan.retrieval_intent,
            answer=deterministic_answer,
            deterministic_answer=(
                deterministic_answer
            ),
            is_grounded=bool(
                selection.results
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
            plan=plan,
            evidence_selection=selection,
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

    def _first_value(
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

    def _format_metadata(
        self,
        metadata: dict[str, Any]
    ) -> str:

        metadata_parts = []

        for field_name, value in (
            metadata.items()
        ):

            cleaned_value = self._clean_text(
                value
            )

            if not cleaned_value:
                continue

            metadata_parts.append(
                f"{field_name}={cleaned_value}"
            )

        return " | ".join(
            metadata_parts
        )

    def _extract_label_value(
        self,
        content: str,
        label: str
    ) -> str:

        pattern = (
            re.escape(
                label
            )
            + r":\s*(.+?)"
            + r"(?=\.\s+[A-Z][A-Za-z ]+:|$)"
        )

        match = re.search(
            pattern,
            content,
            flags=re.IGNORECASE
        )

        if not match:
            return ""

        return self._clean_text(
            match.group(1)
        ).rstrip(".")

    def _extract_effective_date_after(
        self,
        content: str,
        label: str
    ) -> str:

        month_pattern = "|".join(
            self.MONTH_NAMES
        )

        pattern = (
            re.escape(
                label
            )
            + r":.*?effective\s+"
            + rf"((?:{month_pattern})"
            + r"\s+\d{1,2},\s+\d{4})"
        )

        match = re.search(
            pattern,
            content,
            flags=re.IGNORECASE
        )

        if not match:
            return ""

        return self._clean_text(
            match.group(1)
        )

    def _extract_source_file(
        self,
        content: str
    ) -> str:

        match = re.search(
            (
                r"Source tariff document:\s*"
                r"([^.\s]+\.pdf)"
            ),
            content,
            flags=re.IGNORECASE
        )

        if not match:
            return ""

        return self._clean_text(
            match.group(1)
        )

    def _extract_currency_values(
        self,
        text: str
    ) -> list[str]:

        return re.findall(
            r"\$\s*-?\d[\d,]*"
            r"(?:\.\d+)?",
            text
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

        parenthesized_negative = (
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
            parenthesized_negative
            and not cleaned_token.startswith(
                "-"
            )
        ):

            cleaned_token = (
                "-"
                + cleaned_token
            )

        return cleaned_token.upper()

    def _remove_list_numbering(
        self,
        text: str
    ) -> str:

        return re.sub(
            r"(?m)^\s*\d+\s*[\.\)]\s+",
            "",
            text
        )

    def _contains_fact(
        self,
        answer: str,
        fact: str
    ) -> bool:

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

        return bool(
            normalized_fact
            and normalized_fact
            in normalized_answer
        )

    def _normalize_for_match(
        self,
        value: Any
    ) -> str:

        text = self._clean_text(
            value
        )

        text = (
            text
            .replace(
                "\u2013",
                "-"
            )
            .replace(
                "\u2014",
                "-"
            )
        )

        return text.upper()

    def _value_is_present(
        self,
        value: str
    ) -> bool:

        normalized_value = (
            self._clean_text(
                value
            ).upper()
        )

        return bool(
            normalized_value
            and normalized_value
            not in {
                "NOT PRESENT",
                "NOT AVAILABLE",
                "NOT APPLICABLE",
                "NONE",
                "NULL",
                "N/A"
            }
        )

    def _date_is_present(
        self,
        value: str
    ) -> bool:

        return self._value_is_present(
            value
        )

    def _format_percentage(
        self,
        value: str
    ) -> str:

        cleaned_value = self._clean_text(
            value
        )

        if not self._value_is_present(
            cleaned_value
        ):

            return ""

        has_percentage_symbol = (
            "%"
            in cleaned_value
        )

        numeric_value = (
            cleaned_value
            .replace(
                "%",
                ""
            )
            .strip()
        )

        try:

            decimal_value = Decimal(
                numeric_value
            )

        except (
            InvalidOperation,
            ValueError
        ):

            return cleaned_value

        formatted_value = (
            f"{decimal_value:.2f}"
        )

        return (
            formatted_value
            + "%"
            if (
                has_percentage_symbol
                or numeric_value
            )
            else formatted_value
        )

    def _extract_latest_year(
        self,
        value: str
    ) -> int:

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