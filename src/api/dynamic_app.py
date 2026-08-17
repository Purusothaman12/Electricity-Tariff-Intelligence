import os

from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import FastAPI
from fastapi import HTTPException
from fastapi.responses import FileResponse

from pydantic import BaseModel
from pydantic import Field

from src.rag.dynamic_answer_generator import (
    DynamicOllamaTariffAnswerGenerator,
)
from src.rag.ollama_client import (
    OllamaLLMClient,
)
from src.rag.service import (
    TariffRAGService,
)


# =============================================================================
# PROJECT PATHS
# =============================================================================

PROJECT_ROOT = (
    Path(__file__)
    .resolve()
    .parents[2]
)

FRONTEND_FILE = (
    PROJECT_ROOT
    / "frontend"
    / "index.html"
)


# =============================================================================
# CONFIGURATION
# =============================================================================

RATES_DIRECTORY = os.getenv(
    "RAG_RATES_DIRECTORY",
    str(
        PROJECT_ROOT
        / "output"
        / "rates"
    ),
)

PERSIST_DIRECTORY = os.getenv(
    "RAG_PERSIST_DIRECTORY",
    str(
        PROJECT_ROOT
        / "output"
        / "chroma_test"
    ),
)

COLLECTION_NAME = os.getenv(
    "RAG_COLLECTION_NAME",
    "electricity_tariff_test",
)

OLLAMA_MODEL = os.getenv(
    "OLLAMA_MODEL",
    "llama3.2:latest",
)

OLLAMA_TIMEOUT_SECONDS = float(
    os.getenv(
        "RAG_LLM_TIMEOUT_SECONDS",
        "240",
    )
)

MAX_OUTPUT_TOKENS = int(
    os.getenv(
        "RAG_LLM_MAX_OUTPUT_TOKENS",
        "700",
    )
)


# =============================================================================
# GLOBAL APPLICATION STATE
# =============================================================================

rag_service: TariffRAGService | None = None

dynamic_answer_generator: (
    DynamicOllamaTariffAnswerGenerator | None
) = None

startup_error: str | None = None


# =============================================================================
# REQUEST / RESPONSE MODELS
# =============================================================================

class AskRequest(BaseModel):
    """
    Natural-language tariff question.
    """

    question: str = Field(
        ...,
        min_length=2,
        description=(
            "Natural-language electricity "
            "tariff question."
        ),
        examples=[
            (
                "What is the Residential Service "
                "Customer Charge?"
            )
        ],
    )

    top_k: int = Field(
        default=8,
        ge=1,
        le=50,
        description=(
            "Requested retrieval depth. "
            "The question planner may adjust "
            "this automatically."
        ),
    )


class HealthResponse(BaseModel):

    ready: bool

    ollama_ready: bool

    model: str

    indexed_chunks: int

    comparison_records: int

    rate_documents: int

    embedding_model: str

    error: str | None = None


# =============================================================================
# INITIALIZATION
# =============================================================================

def initialize_application() -> None:
    """
    Initialize the tariff RAG system,
    ChromaDB retrieval layer and Ollama LLM.
    """

    global rag_service
    global dynamic_answer_generator
    global startup_error

    startup_error = None

    try:

        print(
            "\n"
            "========================================"
        )

        print(
            "Initializing Electricity "
            "Tariff Intelligence..."
        )

        print(
            "========================================"
        )

        print(
            f"Rates directory: "
            f"{RATES_DIRECTORY}"
        )

        print(
            f"Vector store: "
            f"{PERSIST_DIRECTORY}"
        )

        print(
            f"Ollama model: "
            f"{OLLAMA_MODEL}"
        )

        print(
            "\nLoading RAG service..."
        )

        rag_service = TariffRAGService(
            rates_directory=(
                RATES_DIRECTORY
            ),
            persist_directory=(
                PERSIST_DIRECTORY
            ),
            collection_name=(
                COLLECTION_NAME
            ),
            llm_enabled=False,
        )

        status = (
            rag_service
            .get_status()
        )

        if not status.ready:

            raise RuntimeError(
                "The tariff RAG service "
                "is not ready."
            )

        print(
            "RAG service loaded."
        )

        print(
            f"Indexed chunks: "
            f"{status.indexed_chunk_count}"
        )

        print(
            f"Comparison records: "
            f"{status.comparison_count}"
        )

        print(
            f"Tariff documents: "
            f"{status.rate_document_count}"
        )

        print(
            "\nConnecting to Ollama..."
        )

        llm_client = OllamaLLMClient(
            model=OLLAMA_MODEL,
            timeout_seconds=(
                OLLAMA_TIMEOUT_SECONDS
            ),
            max_output_tokens=(
                MAX_OUTPUT_TOKENS
            ),
        )

        if not llm_client.health_check():

            raise RuntimeError(
                "Ollama is not available. "
                "Start Ollama and make sure "
                f"{OLLAMA_MODEL} is installed."
            )

        print(
            "Ollama connected."
        )

        dynamic_answer_generator = (
            DynamicOllamaTariffAnswerGenerator(
                retriever=(
                    rag_service.retriever
                ),
                deterministic_generator=(
                    rag_service.answer_generator
                ),
                llm_client=(
                    llm_client
                ),
                max_output_tokens=(
                    MAX_OUTPUT_TOKENS
                ),
                fallback_on_error=True,
                fallback_on_validation_failure=True,
            )
        )

        print(
            "\nApplication initialized "
            "successfully."
        )

        print(
            "========================================\n"
        )

    except Exception as error:

        startup_error = str(
            error
        )

        dynamic_answer_generator = None

        print(
            "\nAPPLICATION STARTUP ERROR:"
        )

        print(
            startup_error
        )

        print()


# =============================================================================
# FASTAPI LIFESPAN
# =============================================================================

@asynccontextmanager
async def lifespan(
    app: FastAPI,
):

    initialize_application()

    yield


# =============================================================================
# FASTAPI APPLICATION
# =============================================================================

app = FastAPI(
    title=(
        "Electricity Tariff "
        "Intelligence API"
    ),
    description=(
        "Grounded RAG application for "
        "electricity tariff analysis."
    ),
    version="1.0.0",
    lifespan=lifespan,
)


# =============================================================================
# CHATBOT UI
# =============================================================================

@app.get(
    "/ui",
    include_in_schema=False,
)
def chatbot_ui():

    if not FRONTEND_FILE.exists():

        raise HTTPException(
            status_code=404,
            detail=(
                "frontend/index.html "
                "was not found."
            ),
        )

    return FileResponse(
        path=str(
            FRONTEND_FILE
        ),
        media_type="text/html",
    )


# =============================================================================
# ROOT
# =============================================================================

@app.get("/")
def root() -> dict[str, Any]:

    return {
        "application": (
            "Electricity Tariff Intelligence"
        ),
        "version": "1.0.0",
        "status": (
            "ready"
            if dynamic_answer_generator
            is not None
            else "not_ready"
        ),
        "model": OLLAMA_MODEL,
        "chatbot": "/ui",
        "documentation": "/docs",
        "endpoints": {
            "health": "/health",
            "ask": "/ask",
            "examples": "/examples",
        },
    }


# =============================================================================
# HEALTH
# =============================================================================

@app.get(
    "/health",
    response_model=HealthResponse,
)
def health() -> HealthResponse:

    if rag_service is None:

        return HealthResponse(
            ready=False,
            ollama_ready=False,
            model=OLLAMA_MODEL,
            indexed_chunks=0,
            comparison_records=0,
            rate_documents=0,
            embedding_model="",
            error=(
                startup_error
                or "Service not initialized."
            ),
        )

    status = (
        rag_service
        .get_status()
    )

    return HealthResponse(
        ready=(
            status.ready
            and dynamic_answer_generator
            is not None
        ),
        ollama_ready=(
            dynamic_answer_generator
            is not None
        ),
        model=OLLAMA_MODEL,
        indexed_chunks=(
            status.indexed_chunk_count
        ),
        comparison_records=(
            status.comparison_count
        ),
        rate_documents=(
            status.rate_document_count
        ),
        embedding_model=(
            status.embedding_model
        ),
        error=startup_error,
    )


# =============================================================================
# EXAMPLE QUESTIONS
# =============================================================================

@app.get("/examples")
def examples() -> dict[str, Any]:

    return {
        "examples": [
            {
                "type": "rate_lookup",
                "question": (
                    "What is the Residential "
                    "Service Customer Charge?"
                ),
            },
            {
                "type": "rate_list",
                "question": (
                    "What charges are available "
                    "under Lighting Service?"
                ),
            },
            {
                "type": "comparison",
                "question": (
                    "Compare the Residential "
                    "Transmission System Charge "
                    "effective in 2018 and 2023."
                ),
            },
            {
                "type": "rider_change",
                "question": (
                    "Which Riders were added "
                    "in the new tariff?"
                ),
            },
            {
                "type": "section_coverage",
                "question": (
                    "Which tariff sections are "
                    "marked not applicable?"
                ),
            },
        ]
    }


# =============================================================================
# ASK
# =============================================================================

@app.post("/ask")
def ask(
    request: AskRequest,
) -> dict[str, Any]:
    """
    Process a natural-language tariff question.

    Flow:

        Question Planner
            ↓
        Retriever
            ↓
        ChromaDB
            ↓
        Evidence Selector
            ↓
        Deterministic Grounding
            ↓
        Ollama / Llama 3.2
            ↓
        Validation
            ↓
        Safe Fallback
    """

    if (
        dynamic_answer_generator
        is None
    ):

        raise HTTPException(
            status_code=503,
            detail=(
                startup_error
                or (
                    "Tariff intelligence "
                    "service is not ready."
                )
            ),
        )

    try:

        result = (
            dynamic_answer_generator
            .answer(
                question=(
                    request.question
                ),
                requested_top_k=(
                    request.top_k
                ),
            )
        )

    except ValueError as error:

        raise HTTPException(
            status_code=400,
            detail=str(
                error
            ),
        ) from error

    except Exception as error:

        raise HTTPException(
            status_code=500,
            detail=(
                "Unable to process tariff "
                f"question: {error}"
            ),
        ) from error

    return result.to_dict()