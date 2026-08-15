# Electricity-Tariff-Intelligence
# Electricity Tariff Intelligence

An end-to-end **GenAI + RAG application** for extracting, comparing, and querying historical electricity tariff documents using Python, semantic search, structured data extraction, and a local LLM.

The project currently analyzes historical **Oncor electricity tariff PDFs** and allows users to ask natural-language questions about rates, charges, Riders, effective dates, tariff sections, and historical changes.

---

## Problem Statement

Electric utility tariff documents are often hundreds of pages long and contain complex rate schedules, tables, Rider charges, effective dates, and regulatory information.

Manually finding and comparing tariff changes across historical documents is time-consuming and error-prone.

This project automates the process by:

* extracting structured tariff data from PDFs,
* identifying schedules and charges,
* resolving effective dates,
* comparing historical tariff versions,
* indexing tariff knowledge in a vector database,
* retrieving relevant evidence for user questions,
* generating grounded answers using a local LLM.

---

## Key Features

### PDF Tariff Processing

* Loads historical electricity tariff PDFs.
* Extracts page-level text and tables.
* Uses both **pdfplumber** and **Docling** for table extraction.
* Parses tariff sections and schedules.

### Structured Rate Extraction

Extracts information such as:

* Schedule name
* Schedule ID
* Charge name
* Rate value
* Unit
* Effective date
* Rate category
* Source page
* Extraction method

### Effective-Date Resolution

Automatically resolves tariff effective dates using:

* explicit dates,
* schedule-level consensus,
* section-level context.

### Historical Tariff Comparison

Compares rates across tariff versions and classifies changes as:

* `INCREASED`
* `DECREASED`
* `UNCHANGED`
* `ADDED`
* `REMOVED`

The comparison layer also performs charge-name normalization to reduce false mismatches.

---

## Retrieval-Augmented Generation

Structured tariff information is converted into RAG chunks and indexed using:

* **Sentence Transformers**
* `sentence-transformers/all-MiniLM-L6-v2`
* **ChromaDB**

The current vector index contains approximately:

* **1,557 tariff knowledge chunks**
* **358 comparison records**
* **2 historical tariff documents**

---

## Intelligent Question Processing

The application includes a custom question-planning layer that identifies different tariff question types.

Supported examples include:

### Rate Lookup

```text
What is the Residential Service Customer Charge?
```

### Rate List

```text
What charges are available under Lighting Service?
```

### Historical Comparison

```text
Compare the Residential Transmission System Charge effective in 2018 and 2023.
```

### Rider Changes

```text
Which Riders were added in the new tariff?
```

### Section Coverage

```text
Which tariff sections are marked not applicable?
```

### Out-of-Scope Detection

```text
What is tomorrow's weather?
```

Out-of-scope questions are rejected instead of being answered using unsupported LLM knowledge.

---

## Grounded LLM Generation

The project uses:

* **Ollama**
* **Llama 3.2**
* local inference

The LLM does not answer directly from general knowledge.

The pipeline first retrieves structured tariff evidence and creates a verified factual baseline.

The LLM is then used to convert the verified information into a natural-language response.

---

## Hallucination Protection

A validation layer checks generated responses against retrieved tariff evidence.

The application verifies information such as:

* Rate values
* Effective dates
* Percentages
* Schedule names
* Comparison results

If the LLM introduces unsupported information, the response is rejected and the application returns the deterministic grounded answer instead.

Example:

```text
Generation method:
DETERMINISTIC_VALIDATION_FALLBACK
```

This provides a safer response than trusting the LLM output directly.

---

## System Architecture

```text
Electricity Tariff PDFs
        |
        v
PDF Ingestion
        |
        v
Text + Table Extraction
        |
        v
Section / Schedule Parsing
        |
        v
Rate Extraction
        |
        v
Effective Date Resolution
        |
        v
Structured JSON
        |
        +-------------------+
        |                   |
        v                   v
Historical Comparison    RAG Chunk Builder
                            |
                            v
                    Sentence Transformers
                            |
                            v
                         ChromaDB
                            |
                            v
                     User Question
                            |
                            v
                    Question Planner
                            |
                            v
                    Semantic Retrieval
                            |
                            v
                     Metadata Reranking
                            |
                            v
                    Evidence Selection
                            |
                            v
                 Deterministic Grounding
                            |
                            v
                     Llama 3.2 / Ollama
                            |
                            v
                  Grounding Validation
                            |
                            v
                    Final User Answer
```

---

## Technology Stack

### Programming

* Python

### PDF Processing

* pdfplumber
* Docling

### Data Processing

* Pandas
* NumPy

### Machine Learning / NLP

* Sentence Transformers
* Hugging Face Transformers

### Vector Database

* ChromaDB

### Generative AI

* Ollama
* Llama 3.2

### Backend

* FastAPI
* Uvicorn
* Pydantic

---

## Project Structure

```text
Electricity_Tariff_Intelligence/
|
|-- data/
|   |-- Oncor_May_1_2023.pdf
|   `-- Oncor_November_27_2017.pdf
|
|-- output/
|   |-- extracted/
|   `-- rates/
|
|-- src/
|   |
|   |-- api/
|   |   |-- app.py
|   |   `-- dynamic_app.py
|   |
|   |-- comparison/
|   |   |-- charge_identity_normalizer.py
|   |   `-- rate_comparator.py
|   |
|   |-- exporters/
|   |
|   |-- ingestion/
|   |
|   |-- loaders/
|   |
|   |-- models/
|   |
|   |-- parsing/
|   |
|   |-- pipelines/
|   |
|   |-- rag/
|   |   |-- answer_generator.py
|   |   |-- chunk_builder.py
|   |   |-- dynamic_answer_generator.py
|   |   |-- embedding_model.py
|   |   |-- evidence_selector.py
|   |   |-- llm_answer_generator.py
|   |   |-- ollama_client.py
|   |   |-- question_planner.py
|   |   |-- retriever.py
|   |   |-- service.py
|   |   `-- vector_store.py
|   |
|   `-- table_extraction/
|
|-- chat.py
|-- requirements.txt
|-- .gitignore
`-- README.md
```

---

## Installation

### 1. Clone the repository

```bash
git clone <YOUR_GITHUB_REPOSITORY_URL>
cd Electricity_Tariff_Intelligence
```

### 2. Create a virtual environment

Windows:

```powershell
python -m venv .venv
.venv\Scripts\activate
```

### 3. Install dependencies

```powershell
pip install -r requirements.txt
```

---

## Install Ollama

Install Ollama and download the Llama 3.2 model.

```powershell
ollama pull llama3.2
```

Verify:

```powershell
ollama list
```

Set the model:

```powershell
$env:OLLAMA_MODEL="llama3.2:latest"
```

---

## Run the CLI Application

```powershell
python chat.py
```

Example:

```text
You: What is the Residential Service Customer Charge?

The Residential Service Customer Charge is $1.43
per Retail Customer. The effective date is May 1, 2023.
```

---

## Run the FastAPI Application

```powershell
uvicorn src.api.dynamic_app:app --reload
```

Open Swagger UI:

```text
http://127.0.0.1:8000/docs
```

---

## API Health Check

```text
GET /health
```

Example:

```json
{
  "ready": true,
  "ollama_ready": true,
  "model": "llama3.2:latest",
  "indexed_chunks": 1557,
  "comparison_records": 358,
  "rate_documents": 2,
  "embedding_model": "sentence-transformers/all-MiniLM-L6-v2",
  "error": null
}
```

---

## Ask API

```text
POST /ask
```

Example request:

```json
{
  "question": "What is the Residential Service Customer Charge?",
  "top_k": 8
}
```

Example response:

```json
{
  "question_type": "RATE_LOOKUP",
  "answer": "The Residential Service Customer Charge is $1.43 per Retail Customer. The effective date is May 1, 2023.",
  "is_grounded": true,
  "generation_method": "OLLAMA",
  "model": "llama3.2:latest",
  "validation_passed": true
}
```

---

## Example Historical Comparison

User:

```text
Compare the Residential Transmission System Charge effective in 2018 and 2023.
```

The system retrieves the structured historical comparison evidence and determines whether the charge:

* increased,
* decreased,
* remained unchanged,
* was added,
* or was removed.

When no corresponding rate exists in the newer tariff, the application reports that the charge is not present instead of generating an artificial value.

---

## Project Highlights

* Built an end-to-end document intelligence pipeline for complex electricity tariff PDFs.
* Combined text extraction and table extraction to generate structured rate data.
* Implemented historical tariff comparison with normalized charge identities.
* Built semantic search using Sentence Transformers and ChromaDB.
* Developed metadata-aware reranking and evidence-selection logic.
* Created dynamic tariff question classification and retrieval planning.
* Integrated local Llama 3.2 through Ollama.
* Added hallucination detection and deterministic fallback.
* Exposed the system through CLI and FastAPI interfaces.

---

## Current Limitations

* Current dataset focuses on Oncor tariff documents.
* Local LLM response latency depends on available hardware.
* Complex tariff tables may still require additional extraction normalization.
* The current project is designed as a local RAG application rather than a production cloud deployment.

---

## Future Improvements

Potential extensions include:

* Support for additional US electricity utilities.
* Automatic tariff-document discovery and ingestion.
* Cross-utility tariff comparison.
* Hybrid keyword + semantic retrieval.
* Reranking using a cross-encoder.
* Streaming LLM responses.
* Web-based frontend.
* Containerization with Docker.
* Cloud deployment.

---

## Author

**Purusothaman S**

Data Analyst | Python | SQL | Machine Learning | NLP | Generative AI | RAG
