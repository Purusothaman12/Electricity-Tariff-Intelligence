import os

from collections.abc import Callable
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI
from fastapi import HTTPException
from fastapi import Request
from pydantic import BaseModel
from pydantic import Field

from src.rag.retriever import (
    RetrievalIntent
)
from src.rag.service import (
    TariffRAGService
)


ServiceFactory = Callable[
    [],
    TariffRAGService
]


class AskRequest(BaseModel):
    """
    Request body for tariff question answering.
    """

    question: str = Field(
        min_length=1,
        max_length=2000,
        description=(
            "Natural-language electricity "
            "tariff question."
        ),
        examples=[
            (
                "How did the Residential "
                "Customer Charge change between "
                "the old and new tariffs?"
            )
        ]
    )

    top_k: int = Field(
        default=8,
        ge=1,
        le=50,
        description=(
            "Maximum number of retrieved "
            "evidence chunks."
        )
    )


class RetrieveRequest(BaseModel):
    """
    Request body for evidence retrieval without answer
    generation.
    """

    question: str = Field(
        min_length=1,
        max_length=2000,
        description=(
            "Natural-language retrieval query."
        )
    )

    top_k: int = Field(
        default=5,
        ge=1,
        le=50
    )

    intent: RetrievalIntent | None = Field(
        default=None,
        description=(
            "Optional explicit retrieval intent. "
            "When omitted, the intent is detected "
            "automatically."
        )
    )


def build_service_from_environment(
) -> TariffRAGService:
    """
    Builds the production RAG service from environment
    variables.

    Data configuration:

        RAG_RATES_DIRECTORY
        RAG_PERSIST_DIRECTORY
        RAG_COLLECTION_NAME
        RAG_OLD_SOURCE_FILE
        RAG_NEW_SOURCE_FILE

    Local-LLM configuration:

        RAG_LLM_ENABLED
        OLLAMA_MODEL
        OLLAMA_HOST
        RAG_LLM_TIMEOUT_SECONDS
        RAG_LLM_MAX_OUTPUT_TOKENS
        RAG_LLM_MAX_EVIDENCE_CHUNKS
    """

    return TariffRAGService(
        rates_directory=os.getenv(
            "RAG_RATES_DIRECTORY",
            TariffRAGService
            .DEFAULT_RATES_DIRECTORY
        ),
        persist_directory=os.getenv(
            "RAG_PERSIST_DIRECTORY",
            TariffRAGService
            .DEFAULT_PERSIST_DIRECTORY
        ),
        collection_name=os.getenv(
            "RAG_COLLECTION_NAME",
            TariffRAGService
            .DEFAULT_COLLECTION_NAME
        ),
        old_source_file=os.getenv(
            "RAG_OLD_SOURCE_FILE",
            TariffRAGService
            .DEFAULT_OLD_SOURCE_FILE
        ),
        new_source_file=os.getenv(
            "RAG_NEW_SOURCE_FILE",
            TariffRAGService
            .DEFAULT_NEW_SOURCE_FILE
        ),
        llm_enabled=_environment_boolean(
            name="RAG_LLM_ENABLED",
            default=True
        ),
        ollama_model=_optional_environment_text(
            "OLLAMA_MODEL"
        ),
        ollama_host=_optional_environment_text(
            "OLLAMA_HOST"
        ),
        llm_timeout_seconds=(
            _environment_float(
                name=(
                    "RAG_LLM_TIMEOUT_SECONDS"
                ),
                default=240.0
            )
        ),
        llm_max_output_tokens=(
            _environment_integer(
                name=(
                    "RAG_LLM_MAX_OUTPUT_TOKENS"
                ),
                default=500
            )
        ),
        llm_max_evidence_chunks=(
            _environment_integer(
                name=(
                    "RAG_LLM_MAX_EVIDENCE_CHUNKS"
                ),
                default=1
            )
        )
    )


def create_app(
    service_factory: ServiceFactory | None = None
) -> FastAPI:
    """
    Creates the FastAPI application.

    A custom service factory can be supplied for tests.
    """

    resolved_service_factory = (
        service_factory
        or build_service_from_environment
    )

    @asynccontextmanager
    async def lifespan(
        app: FastAPI
    ):
        """
        Loads structured tariff data, the embedding model and
        Chroma during application startup.

        Ollama remains lazy and is loaded only when /ask-llm is
        called for the first time.
        """

        app.state.rag_service = None
        app.state.startup_error = None

        try:

            app.state.rag_service = (
                resolved_service_factory()
            )

        except Exception as error:

            app.state.startup_error = str(
                error
            )

        yield

        app.state.rag_service = None

    application = FastAPI(
        title=(
            "Electricity Tariff "
            "Intelligence API"
        ),
        description=(
            "Grounded electricity tariff RAG API "
            "with deterministic answering and "
            "local Ollama LLM generation."
        ),
        version="1.1.0",
        lifespan=lifespan
    )

    @application.get(
        "/",
        tags=["System"]
    )
    def root() -> dict[str, Any]:

        return {
            "name": (
                "Electricity Tariff "
                "Intelligence API"
            ),
            "version": "1.1.0",
            "health_endpoint": "/health",
            "llm_status_endpoint": (
                "/llm/status"
            ),
            "ask_endpoint": "/ask",
            "ask_llm_endpoint": "/ask-llm",
            "retrieve_endpoint": "/retrieve",
            "documentation": "/docs"
        }

    @application.get(
        "/health",
        tags=["System"]
    )
    def health(
        request: Request
    ) -> dict[str, Any]:

        service = getattr(
            request.app.state,
            "rag_service",
            None
        )

        startup_error = getattr(
            request.app.state,
            "startup_error",
            None
        )

        if service is None:

            return {
                "ready": False,
                "error": (
                    startup_error
                    or (
                        "The RAG service has not "
                        "been initialized."
                    )
                )
            }

        status = service.get_status().to_dict()

        status[
            "llm"
        ] = service.get_llm_status()

        return status

    @application.get(
        "/llm/status",
        tags=["System"]
    )
    def llm_status(
        request: Request
    ) -> dict[str, Any]:
        """
        Returns local Ollama configuration and lazy-loading
        status.

        Calling this endpoint does not initialize Ollama.
        """

        service = get_service(
            request
        )

        return service.get_llm_status()

    @application.post(
        "/ask",
        tags=["RAG"]
    )
    def ask(
        payload: AskRequest,
        request: Request
    ) -> dict[str, Any]:
        """
        Generates a deterministic grounded tariff answer.

        This endpoint does not call Ollama.
        """

        service = get_service(
            request
        )

        try:

            answer = service.ask(
                question=payload.question,
                top_k=payload.top_k
            )

        except ValueError as error:

            raise HTTPException(
                status_code=400,
                detail=str(
                    error
                )
            ) from error

        except Exception as error:

            raise HTTPException(
                status_code=500,
                detail=(
                    "Deterministic tariff question "
                    f"answering failed: {error}"
                )
            ) from error

        return answer.to_dict()

    @application.post(
        "/ask-llm",
        tags=["RAG"]
    )
    def ask_with_llm(
        payload: AskRequest,
        request: Request
    ) -> dict[str, Any]:
        """
        Generates a locally produced Ollama answer.

        Flow:

            question
                -> retrieval
                -> deterministic answer
                -> Ollama generation
                -> grounding validation
                -> safe fallback when necessary
        """

        service = get_service(
            request
        )

        try:

            answer = service.ask_with_llm(
                question=payload.question,
                top_k=payload.top_k
            )

        except ValueError as error:

            raise HTTPException(
                status_code=400,
                detail=str(
                    error
                )
            ) from error

        except RuntimeError as error:

            raise HTTPException(
                status_code=503,
                detail=(
                    "The local LLM service is "
                    f"unavailable: {error}"
                )
            ) from error

        except Exception as error:

            raise HTTPException(
                status_code=500,
                detail=(
                    "Ollama tariff question "
                    f"answering failed: {error}"
                )
            ) from error

        return answer.to_dict()

    @application.post(
        "/retrieve",
        tags=["RAG"]
    )
    def retrieve(
        payload: RetrieveRequest,
        request: Request
    ) -> dict[str, Any]:
        """
        Returns reranked tariff evidence without generating
        a final answer.
        """

        service = get_service(
            request
        )

        try:

            resolved_intent = (
                payload.intent
                or service.retriever
                .detect_intent(
                    payload.question
                )
            )

            results = service.retrieve(
                question=payload.question,
                top_k=payload.top_k,
                intent=resolved_intent
            )

        except ValueError as error:

            raise HTTPException(
                status_code=400,
                detail=str(
                    error
                )
            ) from error

        except Exception as error:

            raise HTTPException(
                status_code=500,
                detail=(
                    "Tariff evidence retrieval "
                    f"failed: {error}"
                )
            ) from error

        return {
            "question": payload.question,
            "intent": (
                resolved_intent.value
            ),
            "result_count": len(
                results
            ),
            "results": [
                result.to_dict()
                for result in results
            ]
        }

    return application


def get_service(
    request: Request
) -> TariffRAGService:
    """
    Returns the application-level RAG service or raises an
    HTTP 503 response.
    """

    service = getattr(
        request.app.state,
        "rag_service",
        None
    )

    if service is not None:
        return service

    startup_error = getattr(
        request.app.state,
        "startup_error",
        None
    )

    detail = (
        "The tariff RAG service is not ready."
    )

    if startup_error:

        detail += (
            f" Startup error: "
            f"{startup_error}"
        )

    raise HTTPException(
        status_code=503,
        detail=detail
    )


def _optional_environment_text(
    name: str
) -> str | None:

    value = os.getenv(
        name
    )

    if value is None:
        return None

    cleaned_value = value.strip()

    return (
        cleaned_value
        or None
    )


def _environment_boolean(
    name: str,
    default: bool
) -> bool:

    raw_value = os.getenv(
        name
    )

    if raw_value is None:
        return default

    normalized_value = (
        raw_value.strip().lower()
    )

    true_values = {
        "1",
        "true",
        "yes",
        "on"
    }

    false_values = {
        "0",
        "false",
        "no",
        "off"
    }

    if normalized_value in true_values:
        return True

    if normalized_value in false_values:
        return False

    raise ValueError(
        f"Environment variable {name} "
        "must contain one of: "
        "true, false, yes, no, on, off, 1 or 0."
    )


def _environment_integer(
    name: str,
    default: int
) -> int:

    raw_value = os.getenv(
        name
    )

    if raw_value is None:
        return default

    try:

        value = int(
            raw_value
        )

    except ValueError as error:

        raise ValueError(
            f"Environment variable {name} "
            "must be an integer."
        ) from error

    if value <= 0:

        raise ValueError(
            f"Environment variable {name} "
            "must be greater than zero."
        )

    return value


def _environment_float(
    name: str,
    default: float
) -> float:

    raw_value = os.getenv(
        name
    )

    if raw_value is None:
        return default

    try:

        value = float(
            raw_value
        )

    except ValueError as error:

        raise ValueError(
            f"Environment variable {name} "
            "must be numeric."
        ) from error

    if value <= 0:

        raise ValueError(
            f"Environment variable {name} "
            "must be greater than zero."
        )

    return value


app = create_app()