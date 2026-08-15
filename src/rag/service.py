from dataclasses import dataclass
from typing import Any

from src.comparison.rate_comparator import (
    RateComparator,
    TariffComparisonResult
)
from src.loaders.rate_json_loader import (
    LoadedRateDocument,
    RateJSONLoader
)
from src.rag.answer_generator import (
    RAGAnswer,
    TariffAnswerGenerator
)
from src.rag.embedding_model import (
    LocalEmbeddingModel
)
from src.rag.llm_answer_generator import (
    LLMGroundedAnswer,
    OllamaTariffAnswerGenerator
)
from src.rag.ollama_client import (
    OllamaLLMClient
)
from src.rag.retriever import (
    RetrievalIntent,
    RetrievalResult,
    TariffRetriever
)
from src.rag.vector_store import (
    TariffVectorStore
)


@dataclass(slots=True)
class RAGServiceStatus:
    """
    Describes the current state of the tariff RAG service.
    """

    ready: bool
    indexed_chunk_count: int
    rate_document_count: int
    old_source_file: str
    new_source_file: str
    old_rate_count: int
    new_rate_count: int
    comparison_count: int
    embedding_model: str
    collection_name: str
    persist_directory: str

    def to_dict(
        self
    ) -> dict[str, Any]:

        return {
            "ready": self.ready,
            "indexed_chunk_count": (
                self.indexed_chunk_count
            ),
            "rate_document_count": (
                self.rate_document_count
            ),
            "old_source_file": (
                self.old_source_file
            ),
            "new_source_file": (
                self.new_source_file
            ),
            "old_rate_count": (
                self.old_rate_count
            ),
            "new_rate_count": (
                self.new_rate_count
            ),
            "comparison_count": (
                self.comparison_count
            ),
            "embedding_model": (
                self.embedding_model
            ),
            "collection_name": (
                self.collection_name
            ),
            "persist_directory": (
                self.persist_directory
            )
        }


class TariffRAGService:
    """
    Reusable application service for tariff question answering.

    Deterministic flow:

        Question
            -> retrieval
            -> metadata reranking
            -> structured answer

    Local-LLM flow:

        Question
            -> retrieval
            -> structured answer
            -> Ollama explanation
            -> grounding validation
            -> safe fallback when required

    Ollama is initialized lazily. Starting the deterministic API
    therefore does not automatically load the local LLM.
    """

    DEFAULT_RATES_DIRECTORY = (
        "output/rates"
    )

    DEFAULT_PERSIST_DIRECTORY = (
        "output/chroma"
    )

    DEFAULT_COLLECTION_NAME = (
        "electricity_tariff_intelligence"
    )

    DEFAULT_OLD_SOURCE_FILE = (
        "Oncor_November_27_2017.pdf"
    )

    DEFAULT_NEW_SOURCE_FILE = (
        "Oncor_May_1_2023.pdf"
    )

    def __init__(
        self,
        rates_directory: str = (
            DEFAULT_RATES_DIRECTORY
        ),
        persist_directory: str = (
            DEFAULT_PERSIST_DIRECTORY
        ),
        collection_name: str = (
            DEFAULT_COLLECTION_NAME
        ),
        old_source_file: str = (
            DEFAULT_OLD_SOURCE_FILE
        ),
        new_source_file: str = (
            DEFAULT_NEW_SOURCE_FILE
        ),
        embedding_model: (
            LocalEmbeddingModel | None
        ) = None,
        llm_enabled: bool = True,
        ollama_model: str | None = None,
        ollama_host: str | None = None,
        llm_timeout_seconds: float = 240.0,
        llm_max_output_tokens: int = 500,
        llm_max_evidence_chunks: int = 1
    ) -> None:

        self.rates_directory = (
            self._validate_text(
                rates_directory,
                "rates_directory"
            )
        )

        self.persist_directory = (
            self._validate_text(
                persist_directory,
                "persist_directory"
            )
        )

        self.collection_name = (
            self._validate_text(
                collection_name,
                "collection_name"
            )
        )

        self.old_source_file = (
            self._validate_text(
                old_source_file,
                "old_source_file"
            )
        )

        self.new_source_file = (
            self._validate_text(
                new_source_file,
                "new_source_file"
            )
        )

        if (
            self.old_source_file
            == self.new_source_file
        ):

            raise ValueError(
                "The old and new source files "
                "must be different."
            )

        if llm_timeout_seconds <= 0:

            raise ValueError(
                "llm_timeout_seconds must be "
                "greater than zero."
            )

        if llm_max_output_tokens <= 0:

            raise ValueError(
                "llm_max_output_tokens must be "
                "greater than zero."
            )

        if llm_max_evidence_chunks <= 0:

            raise ValueError(
                "llm_max_evidence_chunks must be "
                "greater than zero."
            )

        self.llm_enabled = bool(
            llm_enabled
        )

        self.ollama_model = (
            ollama_model.strip()
            if (
                isinstance(
                    ollama_model,
                    str
                )
                and ollama_model.strip()
            )
            else None
        )

        self.ollama_host = (
            ollama_host.strip()
            if (
                isinstance(
                    ollama_host,
                    str
                )
                and ollama_host.strip()
            )
            else None
        )

        self.llm_timeout_seconds = (
            llm_timeout_seconds
        )

        self.llm_max_output_tokens = (
            llm_max_output_tokens
        )

        self.llm_max_evidence_chunks = (
            llm_max_evidence_chunks
        )

        self.embedding_model = (
            embedding_model
            or LocalEmbeddingModel(
                batch_size=32,
                normalize_embeddings=True,
                show_progress_bar=False
            )
        )

        self.loader = RateJSONLoader()

        self.documents = (
            self.loader.load_directory(
                self.rates_directory
            )
        )

        if not self.documents:

            raise RuntimeError(
                "No structured rate JSON documents "
                f"were found in "
                f"{self.rates_directory}."
            )

        self.documents_by_source = {
            document.source_file: document
            for document in self.documents
        }

        self.old_document = (
            self._get_required_document(
                self.old_source_file
            )
        )

        self.new_document = (
            self._get_required_document(
                self.new_source_file
            )
        )

        self.comparator = RateComparator()

        self.comparison_result = (
            self.comparator.compare(
                old_document=(
                    self.old_document
                ),
                new_document=(
                    self.new_document
                )
            )
        )

        self.vector_store = TariffVectorStore(
            embedding_model=(
                self.embedding_model
            ),
            persist_directory=(
                self.persist_directory
            ),
            collection_name=(
                self.collection_name
            )
        )

        if self.vector_store.count == 0:

            raise RuntimeError(
                "The tariff vector collection is "
                "empty. Build the Chroma index "
                "before starting the RAG service."
            )

        self.retriever = TariffRetriever(
            vector_store=self.vector_store
        )

        self.answer_generator = (
            TariffAnswerGenerator(
                retriever=self.retriever,
                comparison_result=(
                    self.comparison_result
                )
            )
        )

        self._ollama_client: (
            OllamaLLMClient | None
        ) = None

        self._llm_answer_generator: (
            OllamaTariffAnswerGenerator | None
        ) = None

    def ask(
        self,
        question: str,
        top_k: int = 8
    ) -> RAGAnswer:
        """
        Answers one tariff question using the deterministic,
        structured answer generator.

        This method does not call Ollama.
        """

        cleaned_question = (
            self._validate_text(
                question,
                "question"
            )
        )

        if top_k <= 0:

            raise ValueError(
                "top_k must be greater than zero."
            )

        return self.answer_generator.answer(
            query=cleaned_question,
            top_k=top_k
        )

    def ask_with_llm(
        self,
        question: str,
        top_k: int = 8
    ) -> LLMGroundedAnswer:
        """
        Answers one tariff question using Ollama.

        Ollama is created only when this method is called for the
        first time. The same model instance is then reused by
        subsequent requests.
        """

        cleaned_question = (
            self._validate_text(
                question,
                "question"
            )
        )

        if top_k <= 0:

            raise ValueError(
                "top_k must be greater than zero."
            )

        llm_answer_generator = (
            self._get_llm_answer_generator()
        )

        return llm_answer_generator.answer(
            query=cleaned_question,
            top_k=top_k
        )

    def retrieve(
        self,
        question: str,
        top_k: int = 5,
        intent: RetrievalIntent | None = None
    ) -> list[RetrievalResult]:
        """
        Retrieves evidence without generating a final answer.
        """

        cleaned_question = (
            self._validate_text(
                question,
                "question"
            )
        )

        if top_k <= 0:

            raise ValueError(
                "top_k must be greater than zero."
            )

        return self.retriever.retrieve(
            query=cleaned_question,
            top_k=top_k,
            intent=intent
        )

    def get_status(
        self
    ) -> RAGServiceStatus:
        """
        Returns deterministic service and data-health
        information.
        """

        return RAGServiceStatus(
            ready=(
                self.vector_store.count > 0
            ),
            indexed_chunk_count=(
                self.vector_store.count
            ),
            rate_document_count=len(
                self.documents
            ),
            old_source_file=(
                self.old_document.source_file
            ),
            new_source_file=(
                self.new_document.source_file
            ),
            old_rate_count=(
                self.old_document.rate_count
            ),
            new_rate_count=(
                self.new_document.rate_count
            ),
            comparison_count=(
                self.comparison_result
                .comparison_count
            ),
            embedding_model=(
                self.embedding_model
                .model_name
            ),
            collection_name=(
                self.collection_name
            ),
            persist_directory=(
                self.persist_directory
            )
        )

    def get_llm_status(
        self
    ) -> dict[str, Any]:
        """
        Returns Ollama configuration and initialization status.

        This method does not initialize Ollama.
        """

        active_model = ""

        active_host = ""

        if self._ollama_client is not None:

            active_model = (
                self._ollama_client.model
            )

            active_host = (
                self._ollama_client.host
            )

        else:

            active_model = (
                self.ollama_model
                or "Resolved from OLLAMA_MODEL"
            )

            active_host = (
                self.ollama_host
                or (
                    OllamaLLMClient
                    .DEFAULT_HOST
                )
            )

        return {
            "enabled": self.llm_enabled,
            "initialized": (
                self._llm_answer_generator
                is not None
            ),
            "model": active_model,
            "host": active_host,
            "max_evidence_chunks": (
                self.llm_max_evidence_chunks
            ),
            "max_output_tokens": (
                self.llm_max_output_tokens
            ),
            "timeout_seconds": (
                self.llm_timeout_seconds
            )
        }

    def get_comparison_result(
        self
    ) -> TariffComparisonResult:
        """
        Returns the structured tariff comparison result.
        """

        return self.comparison_result

    def get_old_document(
        self
    ) -> LoadedRateDocument:

        return self.old_document

    def get_new_document(
        self
    ) -> LoadedRateDocument:

        return self.new_document

    def _get_llm_answer_generator(
        self
    ) -> OllamaTariffAnswerGenerator:
        """
        Lazily initializes and reuses the Ollama answer
        generator.
        """

        if not self.llm_enabled:

            raise RuntimeError(
                "Local LLM generation is disabled "
                "for this RAG service."
            )

        if self._llm_answer_generator is not None:

            return self._llm_answer_generator

        self._ollama_client = OllamaLLMClient(
            model=self.ollama_model,
            host=self.ollama_host,
            timeout_seconds=(
                self.llm_timeout_seconds
            ),
            max_output_tokens=(
                self.llm_max_output_tokens
            )
        )

        self._llm_answer_generator = (
            OllamaTariffAnswerGenerator(
                deterministic_generator=(
                    self.answer_generator
                ),
                llm_client=(
                    self._ollama_client
                ),
                max_evidence_chunks=(
                    self.llm_max_evidence_chunks
                ),
                max_output_tokens=(
                    self.llm_max_output_tokens
                ),
                fallback_on_error=True,
                fallback_on_validation_failure=True
            )
        )

        return self._llm_answer_generator

    def _get_required_document(
        self,
        source_file: str
    ) -> LoadedRateDocument:

        document = (
            self.documents_by_source.get(
                source_file
            )
        )

        if document is not None:
            return document

        available_documents = sorted(
            self.documents_by_source
        )

        available_text = (
            ", ".join(
                available_documents
            )
            or "None"
        )

        raise FileNotFoundError(
            f"Structured rate data for "
            f"{source_file} was not found. "
            f"Available source documents: "
            f"{available_text}."
        )

    def _validate_text(
        self,
        value: str,
        field_name: str
    ) -> str:

        if not isinstance(
            value,
            str
        ):

            raise TypeError(
                f"{field_name} must be a string."
            )

        cleaned_value = " ".join(
            value.split()
        )

        if not cleaned_value:

            raise ValueError(
                f"{field_name} cannot be empty."
            )

        return cleaned_value