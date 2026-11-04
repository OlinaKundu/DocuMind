"""DocuMind — Conversational RAG with Verifiable Page Citations.

Phase 5.5: Streamlit user interface integrating the complete RAG backend:
- Document upload and digital text extraction (PyMuPDF).
- Text chunking and dense vector embedding (Sentence Transformers).
- Sub-millisecond exact vector similarity indexing (FAISS IndexFlatIP).
- Grounded answer generation via Groq API (openai/gpt-oss-120b).
- Deterministic citation verification and coverage analysis.
- Expandable retrieved source evidence display.
"""

import hashlib
import time
from typing import Any, List, Optional, Tuple

import streamlit as st

from src.pdf_loader import (
    EmptyPDFError,
    InvalidPDFError,
    PDFLoadingError,
    PageRecord,
    extract_text_from_pdf,
)
from src.chunker import TextChunk, chunk_pages
from src.embeddings import (
    DEFAULT_MODEL_NAME,
    EMBEDDING_DIMENSION,
    ChunkEmbeddings,
    ModelLoadingError,
    embed_chunks,
)
from src.retriever import (
    RetrievalError,
    SearchResult,
    VectorStore,
    build_index,
)
from src.rag_pipeline import (
    RAGResponse,
    answer_question,
)
from src.answer_validator import (
    ValidationReport,
    validate_answer,
)
from src.llm import (
    DEFAULT_MODEL,
    LLMAPIError,
    LLMAuthenticationError,
    LLMConfigurationError,
    LLMEmptyResponseError,
    LLMError,
    LLMRateLimitError,
    get_model_name,
)


class ScannedPDFError(Exception):
    """Raised when an uploaded PDF contains no extractable digital text streams."""
    pass


def compute_file_fingerprint(file_bytes: bytes, filename: str) -> str:
    """Computes a deterministic SHA-256 fingerprint for an uploaded file.

    Args:
        file_bytes: Raw binary content of the file.
        filename: Original name of the file.

    Returns:
        Hexadecimal SHA-256 hash string.
    """
    hasher = hashlib.sha256()
    hasher.update(filename.encode("utf-8", errors="replace"))
    hasher.update(b"\x00")
    hasher.update(file_bytes)
    return hasher.hexdigest()


def compute_document_signature(
    fingerprint: str,
    chunk_size: int,
    chunk_overlap: int,
) -> Tuple[str, int, int]:
    """Generates a composite signature representing the document and chunk settings.

    Args:
        fingerprint: SHA-256 fingerprint of the file.
        chunk_size: Character length of chunks.
        chunk_overlap: Character overlap between consecutive chunks.

    Returns:
        Tuple representing the document's active indexing signature.
    """
    return (fingerprint, int(chunk_size), int(chunk_overlap))


def process_document(
    file_bytes: bytes,
    filename: str,
    chunk_size: int = 1000,
    chunk_overlap: int = 150,
) -> Tuple[dict[str, Any], VectorStore]:
    """Extracts, chunks, embeds, and indexes a PDF document into a FAISS VectorStore.

    Args:
        file_bytes: Raw bytes of the PDF file.
        filename: Name of the uploaded file.
        chunk_size: Maximum character length per text chunk.
        chunk_overlap: Overlap character count between consecutive chunks.

    Returns:
        Tuple of (metadata_dict, VectorStore).

    Raises:
        EmptyPDFError: If file has 0 bytes or no pages.
        InvalidPDFError: If file is not a valid PDF.
        ScannedPDFError: If all pages contain zero digital text characters.
        ValueError: If chunking parameters are invalid or produce no chunks.
    """
    records: List[PageRecord] = extract_text_from_pdf(file_bytes)
    total_pages = len(records)
    total_chars = sum(len(record.text.strip()) for record in records)
    empty_pages = sum(1 for record in records if not record.text.strip())

    if total_chars == 0:
        raise ScannedPDFError(
            "The uploaded PDF contains no extractable digital text. It appears to be a scanned "
            "image document or contains only graphics. DocuMind does not perform Optical Character "
            "Recognition (OCR)."
        )

    chunks: List[TextChunk] = chunk_pages(
        pages=records,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        doc_id="doc",
    )

    if not chunks:
        raise ValueError("No text chunks could be produced from the extracted pages.")

    chunk_embeddings: ChunkEmbeddings = embed_chunks(chunks)
    vector_store: VectorStore = build_index(chunk_embeddings)

    metadata: dict[str, Any] = {
        "filename": filename,
        "page_count": total_pages,
        "char_count": total_chars,
        "chunk_count": len(chunks),
        "empty_pages": empty_pages,
        "file_size": len(file_bytes),
        "chunks": chunks,
    }

    return metadata, vector_store


def init_session_state() -> None:
    """Initializes required keys in Streamlit session state."""
    if "messages" not in st.session_state:
        st.session_state["messages"] = []
    if "doc_fingerprint" not in st.session_state:
        st.session_state["doc_fingerprint"] = None
    if "doc_signature" not in st.session_state:
        st.session_state["doc_signature"] = None
    if "doc_metadata" not in st.session_state:
        st.session_state["doc_metadata"] = None
    if "vector_store" not in st.session_state:
        st.session_state["vector_store"] = None
    if "uploader_had_file" not in st.session_state:
        st.session_state["uploader_had_file"] = False


def reset_document_state() -> None:
    """Clears all active document state, FAISS vector store, and conversation history."""
    st.session_state["messages"] = []
    st.session_state["doc_fingerprint"] = None
    st.session_state["doc_signature"] = None
    st.session_state["doc_metadata"] = None
    st.session_state["vector_store"] = None
    st.session_state["uploader_had_file"] = False


def render_assistant_response(
    answer: str,
    response: Optional[RAGResponse] = None,
    validation: Optional[ValidationReport] = None,
) -> None:
    """Renders an LLM-generated answer, its citation validation status, and expandable sources.

    Args:
        answer: Text answer generated by the model.
        response: Optional RAGResponse object containing source chunks and metadata.
        validation: Optional ValidationReport verifying citation integrity.
    """
    # 1. Answer text display
    st.markdown(answer)

    # 2. Citation validation status
    if validation is not None and response is not None:
        if not response.has_sources:
            st.info(
                "ℹ️ **No Retrieved Evidence**: The model acknowledged that no relevant document "
                "passages were found for this query.",
                icon="ℹ️",
            )
        elif validation.is_valid:
            if validation.uncited_sentence_count == 0:
                pages_str = ", ".join(str(p) for p in validation.cited_pages)
                st.success(
                    f"✅ **Citations Verified**: All statements cite retrieved sources "
                    f"(Cited pages: [{pages_str}]).",
                    icon="✅",
                )
            else:
                pages_str = ", ".join(str(p) for p in validation.cited_pages)
                st.info(
                    f"ℹ️ **Citations Verified (Coverage Note)**: Cited pages [{pages_str}] belong "
                    f"to retrieved sources, but {validation.uncited_sentence_count} statement(s) "
                    "lack explicit page citations.",
                    icon="ℹ️",
                )
        else:
            if validation.has_invalid_citations:
                invalid_str = ", ".join(str(p) for p in validation.invalid_citations)
                avail_str = ", ".join(str(p) for p in validation.available_pages)
                st.error(
                    f"❌ **Citation Warning — Invalid Page Citation**: The answer cited page(s) "
                    f"[{invalid_str}], which do not appear in the retrieved sources ([{avail_str}]).",
                    icon="❌",
                )
            elif not validation.cited_pages:
                st.warning(
                    f"⚠️ **Citation Warning — Missing Citations**: The answer contains no `[Page N]` citations "
                    f"despite {response.source_count} retrieved source passages being available.",
                    icon="⚠️",
                )
            else:
                st.warning(
                    "⚠️ **Citation Validation Issue**: The answer did not pass deterministic citation validation.",
                    icon="⚠️",
                )

        # Show detailed warning messages if present
        if validation.warnings:
            with st.expander("⚠️ View Citation Warning Details", expanded=False):
                for warn in validation.warnings:
                    st.markdown(f"- {warn}")

        # Grounding disclaimer
        st.caption(
            "🛡️ *Citation validation checks deterministic reference integrity against retrieved source chunks, "
            "not semantic factual correctness.*"
        )

    # 3. Expandable source passages
    if response is not None and response.has_sources:
        with st.expander(f"📚 Retrieved Sources ({response.source_count} passages)", expanded=False):
            for rank, src in enumerate(response.sources, start=1):
                st.markdown(
                    f"**[Source {rank}] — Page {src.page_number}** &nbsp;|&nbsp; "
                    f"Chunk: `{src.chunk_id}` &nbsp;|&nbsp; "
                    f"Cosine Similarity: `{src.score:.4f}`"
                )
                norm_score = max(0.0, min(1.0, float(src.score)))
                st.progress(norm_score)
                st.markdown(f"> {src.text.strip()}")
                st.divider()


def main() -> None:
    """Main Streamlit application entrypoint."""
    st.set_page_config(
        page_title="DocuMind — Conversational RAG with Citations",
        page_icon="🧠",
        layout="wide",
    )

    init_session_state()

    # Sidebar: Document Ingestion, Metadata & Configuration
    with st.sidebar:
        st.header("🧠 DocuMind")
        st.caption("Local Grounded RAG with FAISS & Groq")
        st.divider()

        st.subheader("1. Document Ingestion")
        uploaded_file = st.file_uploader(
            label="Upload a research paper or report (PDF):",
            type=["pdf"],
            help="Upload a digital PDF. Scanned image documents requiring OCR are not supported.",
            key="pdf_uploader",
        )

        st.subheader("⚙️ Settings")
        top_k = st.slider(
            label="Top-K Retrieved Passages:",
            min_value=1,
            max_value=10,
            value=5,
            step=1,
            help="Number of nearest-neighbor chunks retrieved from FAISS for evidence.",
        )

        with st.expander("🔧 Chunking & Indexing Configuration", expanded=False):
            chunk_size = st.number_input(
                label="Chunk Size (chars):",
                min_value=100,
                max_value=3000,
                value=1000,
                step=50,
                help="Maximum characters in each text chunk.",
            )
            chunk_overlap = st.number_input(
                label="Chunk Overlap (chars):",
                min_value=0,
                max_value=min(500, chunk_size - 1),
                value=min(150, chunk_size - 1),
                step=10,
                help="Characters shared between consecutive chunks.",
            )
            active_model = get_model_name()
            st.markdown(f"**LLM Model:** `{active_model}`")
            st.markdown(f"**Embedding Model:** `{DEFAULT_MODEL_NAME}`")
            st.markdown(f"**Vector Dimension:** `{EMBEDDING_DIMENSION}` float32")
            st.markdown("**FAISS Index:** `IndexFlatIP (Cosine)`")

        # Process uploaded document
        if uploaded_file is not None:
            st.session_state["uploader_had_file"] = True
            file_bytes = uploaded_file.getvalue()
            fp = compute_file_fingerprint(file_bytes, uploaded_file.name)
            current_sig = compute_document_signature(fp, chunk_size, chunk_overlap)

            # Invalidate and re-index if document or chunking parameters changed
            if st.session_state["doc_signature"] != current_sig:
                reset_document_state()
                st.session_state["uploader_had_file"] = True
                try:
                    with st.spinner(f"Ingesting and indexing '{uploaded_file.name}' into FAISS..."):
                        t0 = time.perf_counter()
                        meta, vstore = process_document(
                            file_bytes=file_bytes,
                            filename=uploaded_file.name,
                            chunk_size=int(chunk_size),
                            chunk_overlap=int(chunk_overlap),
                        )
                        elapsed_s = time.perf_counter() - t0
                        st.session_state["doc_fingerprint"] = fp
                        st.session_state["doc_signature"] = current_sig
                        st.session_state["doc_metadata"] = meta
                        st.session_state["vector_store"] = vstore
                        st.success(f"Indexed {meta['chunk_count']} chunks in {elapsed_s:.2f}s!")
                except EmptyPDFError as exc:
                    st.error(f"❌ **Empty PDF Error**: {exc}")
                except InvalidPDFError as exc:
                    st.error(f"❌ **Invalid PDF Error**: {exc}")
                except ScannedPDFError as exc:
                    st.error(f"⚠️ **Scanned PDF Detected**: {exc}")
                except PDFLoadingError as exc:
                    st.error(f"❌ **PDF Extraction Error**: {exc}")
                except Exception as exc:
                    st.error(f"❌ **Processing Error**: {exc}")
        else:
            # File uploader was cleared or empty
            if st.session_state.get("uploader_had_file", False):
                reset_document_state()

        # Display active document information if successfully indexed
        doc_meta = st.session_state.get("doc_metadata")
        if doc_meta is not None:
            st.divider()
            st.subheader("📄 Active Document")
            st.markdown(f"**File:** `{doc_meta['filename']}`")
            file_kb = doc_meta["file_size"] / 1024
            st.markdown(f"**Size:** `{file_kb:.1f} KB`")
            st.markdown(f"**Pages:** `{doc_meta['page_count']}`")
            st.markdown(f"**Extracted Chars:** `{doc_meta['char_count']:,}`")
            st.markdown(f"**Indexed Chunks:** `{doc_meta['chunk_count']:,}`")

            if doc_meta["empty_pages"] > 0:
                st.caption(
                    f"ℹ️ {doc_meta['empty_pages']} page(s) contain no digital text."
                )

            if st.button("🗑️ Reset Document & Chat", use_container_width=True):
                reset_document_state()
                st.rerun()

            # Optional chunk inspector
            with st.expander("🔍 Inspect Document Chunks", expanded=False):
                chunks = doc_meta.get("chunks", [])
                if chunks:
                    selected_idx = st.selectbox(
                        label="Select chunk:",
                        options=range(len(chunks)),
                        format_func=lambda i: f"Chunk {i + 1}/{len(chunks)} [{chunks[i].chunk_id}] (p.{chunks[i].page_number})",
                        key="sidebar_chunk_select",
                    )
                    target_chunk = chunks[selected_idx]
                    st.markdown(f"**Page:** {target_chunk.page_number} | **Chars:** {target_chunk.char_count}")
                    st.text_area("Chunk Content", target_chunk.text, height=120, disabled=True)

    # Main Area: Q&A Interface
    st.title("🧠 DocuMind — Grounded Document Q&A")
    st.caption("Ask questions about your uploaded PDF with verifiable page citations.")

    vector_store: Optional[VectorStore] = st.session_state.get("vector_store")
    doc_metadata: Optional[dict[str, Any]] = st.session_state.get("doc_metadata")

    if vector_store is None:
        st.info(
            "👋 **Welcome to DocuMind!**\n\n"
            "To begin querying your document:\n"
            "1. **Upload a PDF** using the file uploader in the sidebar.\n"
            "2. DocuMind will automatically extract text, generate dense embeddings, and build an in-memory FAISS index.\n"
            "3. Ask questions in the chat box below to receive answers grounded strictly in your document with verified `[Page N]` citations.\n\n"
            "*Note: Scanned documents requiring OCR are not supported.*",
            icon="📄",
        )
    else:
        st.caption(
            f"📄 Active Document: **{doc_metadata['filename']}** "
            f"({doc_metadata['page_count']} pages, {doc_metadata['chunk_count']} indexed chunks) &nbsp;|&nbsp; "
            f"⚡ Independent stateless retrieval per question."
        )

        # Render conversation history
        for msg in st.session_state["messages"]:
            with st.chat_message(msg["role"]):
                if msg["role"] == "user":
                    st.markdown(msg["content"])
                else:
                    render_assistant_response(
                        answer=msg["content"],
                        response=msg.get("response"),
                        validation=msg.get("validation"),
                    )

        # Chat question input
        if prompt := st.chat_input("Ask a question about this document..."):
            # Display user question
            with st.chat_message("user"):
                st.markdown(prompt)
            st.session_state["messages"].append({"role": "user", "content": prompt})

            # Generate RAG response
            with st.chat_message("assistant"):
                with st.spinner("Retrieving document context and generating answer with Groq..."):
                    try:
                        rag_response = answer_question(
                            question=prompt,
                            vector_store=vector_store,
                            top_k=int(top_k),
                        )
                        validation_report = validate_answer(rag_response)
                        render_assistant_response(
                            answer=rag_response.answer,
                            response=rag_response,
                            validation=validation_report,
                        )
                        st.session_state["messages"].append({
                            "role": "assistant",
                            "content": rag_response.answer,
                            "response": rag_response,
                            "validation": validation_report,
                        })
                    except LLMConfigurationError as exc:
                        st.error(f"⚠️ **Configuration Error**: {exc}")
                    except LLMAuthenticationError as exc:
                        st.error(f"🔒 **Authentication Error**: {exc}")
                    except LLMRateLimitError as exc:
                        st.error(f"⏳ **Rate Limit Exceeded**: {exc}")
                    except LLMAPIError as exc:
                        st.error(f"🌐 **Groq API / Network Error**: {exc}")
                    except LLMEmptyResponseError as exc:
                        st.error(f"⚠️ **Empty Response**: {exc}")
                    except RetrievalError as exc:
                        st.error(f"🔍 **Retrieval Error**: {exc}")
                    except Exception as exc:
                        st.error(f"❌ **Unexpected Error**: {exc}")


if __name__ == "__main__":
    main()
