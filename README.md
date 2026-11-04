# DocuMind 🧠📄

**DocuMind** is a portfolio-grade, local Retrieval-Augmented Generation (RAG) system built in Python. It allows users to upload technical research papers and complex PDF documents, extract and chunk information, generate dense vector embeddings, index them with FAISS, and query them with an LLM to receive grounded answers with precise source citations.

---

## 🎯 Current Status: Phase 6.1 (Conversational RAG & Follow-up Question Resolution)

This repository implements a full-stack, local-first conversational RAG application:
- **Phase 1**: PDF Ingestion using PyMuPDF (`src/pdf_loader.py`).
- **Phase 2**: Overlapping character-based text chunking with 1-based page metadata (`src/chunker.py`).
- **Phase 3**: Dense vector embeddings with Sentence Transformers (`all-MiniLM-L6-v2`, 384 dimensions) (`src/embeddings.py`).
- **Phase 4**: Vector similarity search using FAISS (`IndexFlatIP`, exact inner product) (`src/retriever.py`).
- **Phase 5.1**: Groq API connectivity and client management (`src/llm.py`).
- **Phase 5.2**: Context construction and prompt engineering (`src/context_builder.py`, `src/prompt_builder.py`).
- **Phase 5.3**: End-to-end RAG pipeline orchestration and answer generation (`src/rag_pipeline.py`).
- **Phase 5.4**: Deterministic citation extraction, validation, and answer grounding checks (`src/citation_validator.py`, `src/answer_validator.py`).
- **Phase 5.5**: Interactive Streamlit web application with chat interface, citation verification badges, expandable evidence, and document session management (`app.py`).
- **Phase 6.1**: Conversational context and follow-up question resolution (`src/conversation.py`).

---

## 📐 Complete RAG Architecture & Pipeline Flow

The DocuMind pipeline processes conversational queries through ten decoupled, verifiable stages:

```
User PDF Document
       │
       ▼ (Phase 1)
[ PyMuPDF Loader ] ──> List[PageRecord] (1-based page numbers)
       │
       ▼ (Phase 2)
[ Text Chunker ]   ──> List[TextChunk] (chunk_id, page_number, start/end offsets)
       │
       ▼ (Phase 3)
[ Embedding Model ] ─> ChunkEmbeddings ((N, 384) unit float32 matrix)
       │
       ▼ (Phase 4)
[ FAISS Index ]    ──> VectorStore (IndexFlatIP exact cosine similarity)
       │
       │  <── User Follow-up Question (e.g. "What optimizer was used with it?")
       │  <── Recent Conversation History (last N messages)
       ▼ (Phase 6.1)
[ Query Contextualizer ] ──> Standalone Retrieval Query (e.g. "What optimizer was used with learning rate 0.001?")
       │                     (Bypassed if history is empty; falls back gracefully on transient errors)
       ▼
[ FAISS Retriever ] ─────> List[SearchResult] (top-k nearest neighbors + metadata)
       │
       ▼ (Phase 5.2)
[ Context Builder ] ─────> Formatted evidence block with [Source N | Page X | Chunk Y]
       │
       ▼ (Phase 5.2)
[ Prompt Builder ]  ─────> [{"role": "system", ...}, {"role": "user", ...}]
       │                   (Grounding rules + Document Security + Isolated Evidence)
       ▼ (Phase 5.3)
[ Groq LLM API ]    ─────> Model completion (openai/gpt-oss-120b)
       │
       ▼ (Phase 5.3)
[ RAGResponse ]     ─────> Structured dataclass (answer, sources, model, question)
       │
       ▼ (Phase 5.4)
[ Answer Validator ] ────> ValidationReport (is_valid, cited_pages, invalid_citations, warnings)
       │
       ▼ (Phase 6.1)
[ ConversationalRAGResponse ] ──> (original_question, standalone_question, rag_response, warning)
       │
       ▼ (Phase 5.5)
[ Streamlit Web UI ] ────> User Chat Interface, Citation Badges & Expandable Sources
```

---

## 🔍 Stage Explanations

### 1. Conversational Query Contextualization (`src/conversation.py`)
In real conversations, users frequently ask follow-up questions containing pronouns and referential phrases (e.g., *"What optimizer was used with it?"*, *"Does that method support GPU acceleration?"*, *"What was their second experiment?"*). A standard semantic vector retriever fails on such queries because the isolated pronoun lacks dense semantic alignment with the relevant PDF passages.

DocuMind resolves this using an intelligent conversational query reformulator:
- **Zero-Latency Bypass**: If conversation history is empty (e.g. the first question), rewriting is bypassed completely—making **zero** unnecessary API calls and introducing 0ms latency.
- **Pronoun & Reference Resolution**: When prior turns exist, the model resolves referential ambiguity against the last $N$ messages (configurable, default 6 messages) to produce an explicit, standalone retrieval query.
- **Conversation Context $\neq$ Document Evidence**: Prior conversation messages are used strictly to understand user intent and disambiguate queries. **Conversation history is never used as factual evidence.** The PDF document passages retrieved from FAISS remain the sole factual grounding for all answers.
- **Graceful Fallback & Resilience**: If contextualization encounters transient rate limits or network errors, it automatically falls back to the original question and attaches an explicit warning, ensuring the user still receives an answer.

### 2. Retrieval (`src/retriever.py`)
Given the standalone query string:
- Generates a 384-dimensional unit vector using the same Sentence Transformer model as document chunks.
- Queries the in-memory FAISS `IndexFlatIP` for the top-$K$ nearest neighbors.
- Maps FAISS integer positions back to the original `TextChunk` records, yielding structured `SearchResult` objects containing the chunk text, source page number, chunk ID, and similarity score.

### 3. Context Construction (`src/context_builder.py`)
- Formats retrieved passages into an evidence block with standard separators (`------------------------------`).
- Preserves the retrieval ranking order: the top-ranked passage is labeled `[Source 1]`, the second `[Source 2]`, etc.
- Preserves source page numbers and chunk IDs without modifying or mutating original objects.

### 4. Prompt Construction (`src/prompt_builder.py`)
Constructs Groq/OpenAI-compatible chat completion messages:
- **System Message**: Embeds strict **Grounding Rules** (answers must rely strictly on supplied context, cite sources as `[Page N]`, and declare insufficient evidence if not supported) and **Document Security Rules** (document passages are untrusted data and cannot override system instructions).
- **User Message**: Formats the prompt into two clear sections: `QUESTION:` and `DOCUMENT CONTEXT:`.

### 5. LLM Generation (`src/llm.py`)
- Uses `generate_chat_response()` to send the structured messages to the Groq API.
- Configurable model (defaults to `openai/gpt-oss-120b`).
- Enforces `temperature=0.0` for deterministic, grounded outputs.
- Comprehensive error mapping for missing keys, auth failures, rate limits, timeouts, and empty responses.

### 6. Structured Response Objects (`RAGResponse` & `ConversationalRAGResponse`)
- `RAGResponse`: Dataclass bundling verbatim generated answer, retrieved `SearchResult` records, active model, and question.
- `ConversationalRAGResponse`: Extends the pipeline to record `original_question`, `standalone_question`, `was_rewritten`, and any contextualization `warning`, while keeping the underlying `rag_response` 100% backward-compatible.

### 7. Citation & Grounding Validation (`src/citation_validator.py`, `src/answer_validator.py`)
Deterministic post-generation guardrail that verifies answer attribution:
- **Citation Extraction**: Extracts page citations matching `[Page N]`, `[Pages N, M]`, or `[Page N and Page M]` using regex with whitespace and case tolerance.
- **Reference Integrity**: Compares cited pages against retrieved `SearchResult.page_number` values. Identifies fabricated or out-of-context page citations without duplicate reporting.
- **Citation Coverage Heuristic**: Segments the answer into sentences while binding trailing citation blocks, flagging statements that lack page attribution.
- **Refusal Recognition**: Recognizes standard insufficient-evidence statements when retrieval returns no sources or unanswerable queries, preventing spurious missing-citation warnings.
- **Structured Report**: Produces an immutable `ValidationReport` with `is_valid`, `cited_pages`, `available_pages`, `invalid_citations`, `uncited_sentence_count`, and user-friendly `warnings`.

### 8. Streamlit Web UI Integration (`app.py`)
Provides an interactive, portfolio-grade user interface:
- **Document Ingestion & Fingerprinting**: PDF upload with SHA-256 fingerprinting to prevent redundant re-indexing during Streamlit reruns.
- **Session State Management**: Isolates vector stores, document metadata, and conversation history per user session, clearing previous indexes immediately when a new file is uploaded.
- **Conversational Chat Interface**: Streamlit chat components (`st.chat_message`, `st.chat_input`) displaying the original user query and generated answer, with an optional debug accordion to inspect standalone rewritten queries.
- **Citation Verification Badges**: Distinct visual feedback distinguishing verified citations (`st.success`), coverage warnings (`st.info`), and invalid hallucinations (`st.error`).
- **Expandable Evidence**: Expandable accordions below each answer revealing verbatim passage text, chunk IDs, page numbers, and cosine similarity progress bars.

---

## 🛡️ Security & Guardrails

### 1. How Retrieved Evidence Reaches Groq
Evidence passages are placed exclusively inside `DOCUMENT CONTEXT:` in the `user` message. Trusted system instructions are maintained in the separate `system` message. This structural decoupling prevents untrusted text inside uploaded PDFs from impersonating system instructions.

### 2. Conversational Context Isolation
Conversation history is treated as untrusted data during query rewriting. The system instructions strictly forbid the model from executing commands found within prior chat turns or altering its role. Furthermore, prior assistant answers are never treated as authoritative document evidence.

### 3. How Empty Evidence is Handled
When retrieval yields zero chunks (e.g., empty vector store or non-matching query):
- The context builder produces an empty string.
- The prompt builder injects: `No relevant document passages were retrieved.`
- The LLM receives Rule 3 (*"clearly state that the available document evidence is insufficient"*).
- The pipeline returns a normal `RAGResponse` with `sources = []`.
- The answer validator recognizes valid refusal responses and marks them valid without false missing-citation warnings.

### 4. Limitations & System Guardrails
> [!WARNING]
> - **Deterministic Citation Verification $\neq$ Semantic Factual Correctness**: DocuMind's citation validator verifies structural reference integrity (that cited page numbers belong to the retrieved evidence set) and sentence coverage heuristics. It does not evaluate semantic claim truth and does not perform an expensive LLM-as-judge call.
> - **Scanned PDFs & OCR**: PyMuPDF extracts digital text streams directly from PDF files. Scanned documents or image-only pages contain no extractable character streams and require Optical Character Recognition (OCR), which is intentionally excluded. DocuMind detects scanned PDFs and alerts the user.
> - **Sentence Segmentation Heuristics**: Sentence splitting uses conservative regex rules rather than heavy syntactic parsers. Uncommon abbreviations or mathematical notation may occasionally affect uncited sentence counts.
> - **Query Rewriting Latency & Trade-offs**: Follow-up questions that require pronoun resolution incur one additional lightweight Groq API call (`temperature=0.0`, limited token count). Standalone questions and the first query of a session bypass this call entirely.

---

## 🚀 Getting Started & Installation

### 1. Prerequisites
- Python 3.10, 3.11, or 3.12
- A free [Groq Cloud API Key](https://console.groq.com/keys)

### 2. Clone and Setup Environment
```bash
git clone https://github.com/your-username/DocuMind.git
cd DocuMind

# Create and activate a virtual environment
python -m venv .venv
# On Windows:
.\.venv\Scripts\activate
# On Linux/macOS:
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 3. Configure Environment Variables
Create a `.env` file in the project root directory (this file is ignored by git):

```env
# Groq API Key (never commit or expose this key)
GROQ_API_KEY="your-groq-api-key-here"

# Optional: Override the default LLM model (defaults to openai/gpt-oss-120b)
GROQ_MODEL="openai/gpt-oss-120b"
```

---

## 🖥️ Running the Streamlit Web Application

Launch the full interactive Q&A interface:

```bash
streamlit run app.py
```

### How to Upload & Query a Document:
1. **Upload a PDF**: In the left sidebar, use the file uploader to select any research paper or technical report in PDF format.
2. **Automatic Indexing**: DocuMind extracts the digital text, slices pages into overlapping chunks, generates dense 384-dimensional embeddings, and builds an in-memory FAISS `IndexFlatIP` index.
3. **Ask Questions**: Type your question in the chat input at the bottom of the main screen (e.g., *"What optimizer and learning rate were used for training?"*).
4. **Inspect Answers & Citations**:
   - The LLM generates a grounded answer citing sources as `[Page N]`.
   - The **Citation Verification Banner** shows whether all cited pages correspond to retrieved evidence.
   - Expand the **Retrieved Sources** accordion to inspect exact page numbers, chunk IDs, similarity scores, and verbatim passage text.
5. **Reset / Change Document**: Click **"Reset Document & Chat"** in the sidebar at any time to clear the index and conversation history.

---

## 🏗️ Project Structure

```text
DocuMind/
├── app.py                      # Interactive Streamlit Web UI (Ingestion, Search, Chat, Citations)
├── requirements.txt            # Pinned dependencies (PyMuPDF, FAISS, Sentence Transformers, Groq, Streamlit)
├── pytest.ini                  # Pytest configuration (real_model, real_api markers)
├── README.md                   # Documentation and execution guide
├── .env                        # Local secrets (GROQ_API_KEY) - git-ignored
├── .gitignore                  # Ignores venv, caches, local configs, and PDFs
├── evaluation/
│   ├── dataset.json            # IR benchmark dataset with queries & relevant chunk IDs
│   └── dataset.schema.json     # JSON schema for evaluation benchmark annotations
├── scripts/
│   ├── smoke_test_llm.py       # Standalone Groq API live connectivity verification script
│   ├── evaluate_retrieval.py   # CLI runner for retrieval benchmarking (P@K, R@K, MRR)
│   └── evaluate_answers.py     # CLI runner for LLM-as-a-Judge answer quality evaluation
├── src/
│   ├── __init__.py             # Package root
│   ├── pdf_loader.py           # PyMuPDF ingestion and PageRecord extraction
│   ├── chunker.py              # Sliding window chunking & TextChunk data models
│   ├── embeddings.py           # Sentence Transformers embedding engine (all-MiniLM-L6-v2)
│   ├── retriever.py            # FAISS IndexFlatIP similarity index & metadata mapping
│   ├── context_builder.py      # Structured passage formatting with page attribution
│   ├── prompt_builder.py       # RAG chat messages with grounding and security rules
│   ├── llm.py                  # Groq API client interface (openai/gpt-oss-120b)
│   ├── rag_pipeline.py         # End-to-end RAG pipeline (answer_question & RAGResponse)
│   ├── citation_validator.py   # Regex-based [Page N] citation extraction and verification
│   ├── answer_validator.py     # Deterministic answer validation & sentence coverage heuristics
│   ├── conversation.py         # Conversational RAG, contextualization & pronoun resolution
│   ├── evaluation.py           # IR evaluation metrics (Precision@K, Recall@K, MRR)
│   └── answer_evaluator.py     # LLM-as-a-Judge answer evaluation (Relevance, Faithfulness, Correctness)
└── tests/
    ├── test_pdf_loader.py        # Unit tests for Phase 1 PDF extraction (8 tests)
    ├── test_chunker.py           # Unit tests for Phase 2 text chunking (13 tests)
    ├── test_embeddings.py        # Unit tests for Phase 3 embeddings (12 unit + 1 smoke test)
    ├── test_retriever.py         # Unit tests for Phase 4 FAISS retrieval (26 tests)
    ├── test_llm.py               # Unit tests for Phase 5.1 Groq client (19 unit + 1 smoke test)
    ├── test_context_builder.py   # Unit tests for Phase 5.2 context builder (8 tests)
    ├── test_prompt_builder.py    # Unit tests for Phase 5.2 prompt builder (10 tests)
    ├── test_rag_pipeline.py      # Unit tests for Phase 5.3 RAG pipeline (13 unit + 1 smoke test)
    ├── test_citation_validator.py# Unit tests for Phase 5.4 citation validator (15 tests)
    ├── test_answer_validator.py  # Unit tests for Phase 5.4 answer validator (13 tests)
    ├── test_conversation.py      # Unit tests for Phase 6.1 Conversational RAG (18 tests)
    ├── test_evaluation.py        # Unit tests for Phase 6.2 Retrieval Evaluation (36 tests)
    ├── test_answer_evaluator.py  # Unit tests for Phase 6.3 Answer Quality Evaluation (27 unit + 1 smoke test)
    └── test_app.py               # Unit and AppTest integration tests for Streamlit UI (25 tests)
```

---

## 💻 Python API Usage Example

```python
from src.pdf_loader import extract_text_from_pdf
from src.chunker import chunk_pages
from src.embeddings import embed_chunks
from src.retriever import build_index
from src.conversation import answer_with_history
from src.answer_validator import validate_answer

# 1. Ingest PDF and index chunks into FAISS
pages = extract_text_from_pdf(open("research_paper.pdf", "rb").read())
chunks = chunk_pages(pages, chunk_size=1000, chunk_overlap=150)
chunk_embeddings = embed_chunks(chunks)
vector_store = build_index(chunk_embeddings)

# 2. Ask an initial question
history = []
resp1 = answer_with_history(
    question="What optimizer and learning rate were used for training?",
    vector_store=vector_store,
    conversation_history=history,
    top_k=3,
)
print(f"Q: {resp1.original_question}")
print(f"A:\n{resp1.rag_response.answer}\n")

# Update conversation history
history.append({"role": "user", "content": resp1.original_question})
history.append({"role": "assistant", "content": resp1.rag_response.answer})

# 3. Ask a follow-up question requiring conversational resolution
resp2 = answer_with_history(
    question="Why did they choose it over Adam?",
    vector_store=vector_store,
    conversation_history=history,
    top_k=3,
)
print(f"Original Q:   {resp2.original_question}")
print(f"Rewritten Q:  {resp2.standalone_question}")
print(f"A:\n{resp2.rag_response.answer}\n")

# 4. Deterministically validate citations and grounding
report = validate_answer(resp2.rag_response)
print(f"Valid citations: {report.is_valid}")
print(f"Cited pages:     {report.cited_pages}")
print(f"Available pages: {report.available_pages}")
for warning in report.warnings:
    print(f"  [Warning] {warning}")
```

---

## 📊 Retrieval Quality Evaluation & Benchmarking

Phase 6.2 provides an offline evaluation framework to benchmark retrieval quality independently of LLM generation using standard Information Retrieval metrics:
- **Precision@K**: Proportion of top-K retrieved chunks that are relevant ($\text{hits} / K$, fixed-K denominator).
- **Recall@K**: Proportion of all known relevant chunks retrieved in top-K ($\text{hits} / \text{total relevant}$).
- **Reciprocal Rank (RR) & Mean Reciprocal Rank (MRR)**: Evaluates the rank of the first relevant result ($1 / \text{rank}$).

### Run Retrieval Evaluation CLI

```bash
# Run against the sample/placeholder benchmark dataset:
python scripts/evaluate_retrieval.py

# Run against a specific PDF document and custom evaluation dataset:
python scripts/evaluate_retrieval.py --pdf path/to/document.pdf --dataset evaluation/dataset.json --k 1,3,5

# Export evaluation metrics to JSON:
python scripts/evaluate_retrieval.py --output-json evaluation_report.json
```

---

## 🎯 Answer Quality & Groundedness Evaluation (LLM-as-a-Judge)

Phase 6.3 implements an independent LLM-as-a-Judge framework to evaluate generated RAG answers across three core dimensions using a strict 1–5 integer rubric:

1. **Answer Relevance (1–5)**: Evaluates whether the generated answer directly addresses the core user inquiry without off-topic or tangential information.
2. **Faithfulness / Groundedness (1–5)**: Evaluates whether every factual claim in the answer is strictly supported by the retrieved document passages. Valid refusals when evidence is insufficient receive a score of 5.
3. **Answer Correctness (1–5)**: Evaluates agreement with an authoritative human-written reference answer on key facts and findings.

### Run Answer Evaluation CLI

```bash
# Run against the sample/benchmark dataset using in-memory sample document:
python scripts/evaluate_answers.py

# Run against a real PDF document and custom evaluation dataset:
python scripts/evaluate_answers.py --pdf path/to/paper.pdf --dataset evaluation/dataset.json --top-k 5

# Export evaluation results to JSON:
python scripts/evaluate_answers.py --output-json answer_eval_report.json
```

### Security & Prompt Injection Defense
- **Untrusted Context Demarcation**: The judge treats retrieved text and generated answers as raw untrusted data enclosed inside `<retrieved_context>` and `<generated_answer>` delimiters. System instructions instruct the judge to ignore instructions inside context blocks. Delimiters provide structural separation to reduce injection risks, but do not completely eliminate prompt injection vulnerabilities.
- **JSON Mode & Schema Enforcement**: Groq's JSON mode (`response_format={"type": "json_object"}`) provides syntactical formatting support, while application-side validation (`parse_judge_response`) strictly validates the schema, verifies all required keys, enforces integer types, and rejects out-of-bound scores outside the 1–5 range.
- **Independent Pipeline**: Answer evaluation runs strictly out-of-band via CLI scripts or benchmarks. Normal user chat turns in Streamlit do not make judge calls.

### Limitations & Guardrails
> [!WARNING]
> - **LLM Judge $\neq$ Absolute Truth**: An LLM judge provides scalable, heuristic quality signals. It cannot replace human expert review.
> - **Synthetic / Placeholder Dataset**: Scores obtained from template or placeholder datasets (such as `evaluation/dataset.json` with sample text) demonstrate pipeline execution and do **not** demonstrate real-world performance or constitute empirical benchmark evidence.
> - **Failure vs. Low Score Distinction**: Errored evaluations (e.g. API rate limits, network timeouts, or malformed JSON) are recorded with status `FAILED` and excluded from mean score averages, rather than defaulting to low scores or 5s.

---

## 🧪 Running the Tests

### Run the Complete Offline Test Suite (242 Tests)
```bash
pytest -v -m "not real_model and not real_api"
```

### Run Answer Quality Evaluation Tests Only (27 Tests)
```bash
pytest -v tests/test_answer_evaluator.py
```

### Run Retrieval Evaluation Tests Only (36 Tests)
```bash
pytest -v tests/test_evaluation.py
```

### Run Conversational RAG Tests Only (18 Tests)
```bash
pytest -v tests/test_conversation.py
```

### Run Streamlit UI & Integration Tests Only (25 Tests)
```bash
pytest -v tests/test_app.py
```

### Run All 246 Tests (Including Optional Live Model & API Smoke Tests)
```bash
pytest -v
```

All 242 offline tests execute in ~15 seconds with zero external network dependencies.

---

## 🗺️ Roadmap & Project Milestones

1. **Phase 1: Project Setup & PDF Ingestion** 🟢 *(Completed)*
2. **Phase 2: Semantic Text Chunking** 🟢 *(Completed)*
3. **Phase 3: Dense Vector Embeddings** 🟢 *(Completed)*
4. **Phase 4: FAISS Vector Similarity Search** 🟢 *(Completed)*
5. **Phase 5: Grounded LLM Generation & Citations** 🟢 *(Completed)*
   - **Phase 5.1: Groq API Connectivity** 🟢 *(Completed)*
   - **Phase 5.2: Context Construction & Prompt Engineering** 🟢 *(Completed)*
   - **Phase 5.3: End-to-End RAG Pipeline Integration** 🟢 *(Completed)*
   - **Phase 5.4: Citation Validation & Answer Grounding Checks** 🟢 *(Completed)*
   - **Phase 5.5: Streamlit Q&A UI Integration** 🟢 *(Completed)*
6. **Phase 6: Conversational Context, Multi-turn Interactions & Evaluation**
   - **Phase 6.1: Conversational RAG & Follow-up Question Resolution** 🟢 *(Completed)*
   - **Phase 6.2: Retrieval Evaluation and Benchmarking** 🟢 *(Completed)*
   - **Phase 6.3: Answer Quality & Groundedness Evaluation** 🟢 *(Completed)*
