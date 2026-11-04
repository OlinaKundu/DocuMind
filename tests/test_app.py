"""Unit and integration tests for the DocuMind Streamlit UI application.

Phase 5.5: Verifies document fingerprinting, session state management,
document processing, RAG pipeline integration, citation validation display,
and error handling.
"""

from pathlib import Path
from unittest.mock import MagicMock, patch
import pytest
from streamlit.testing.v1 import AppTest

APP_PATH = str(Path(__file__).parent.parent / "app.py")

from app import (
    ScannedPDFError,
    compute_document_signature,
    compute_file_fingerprint,
    init_session_state,
    process_document,
    render_assistant_response,
    reset_document_state,
)
from src.pdf_loader import EmptyPDFError, InvalidPDFError, PageRecord
from src.chunker import TextChunk
from src.embeddings import ChunkEmbeddings
from src.retriever import RetrievalError, SearchResult, VectorStore
from src.rag_pipeline import RAGResponse
from src.conversation import ConversationalRAGResponse
from src.answer_validator import ValidationReport
from src.llm import (
    LLMAPIError,
    LLMAuthenticationError,
    LLMConfigurationError,
    LLMEmptyResponseError,
    LLMRateLimitError,
)


# ---------------------------------------------------------------------------
# 1. Fingerprinting & Signature Tests
# ---------------------------------------------------------------------------

def test_compute_file_fingerprint_deterministic():
    """Identical file bytes and name must produce the exact same fingerprint."""
    data = b"%PDF-1.4 sample content"
    name = "document.pdf"
    fp1 = compute_file_fingerprint(data, name)
    fp2 = compute_file_fingerprint(data, name)
    assert fp1 == fp2
    assert isinstance(fp1, str)
    assert len(fp1) == 64  # SHA-256 hex length


def test_compute_file_fingerprint_changes_with_content():
    """Different file contents must produce different fingerprints."""
    data1 = b"%PDF-1.4 content A"
    data2 = b"%PDF-1.4 content B"
    fp1 = compute_file_fingerprint(data1, "test.pdf")
    fp2 = compute_file_fingerprint(data2, "test.pdf")
    assert fp1 != fp2


def test_compute_file_fingerprint_changes_with_filename():
    """Same content with different filenames must produce different fingerprints."""
    data = b"%PDF-1.4 identical content"
    fp1 = compute_file_fingerprint(data, "paper1.pdf")
    fp2 = compute_file_fingerprint(data, "paper2.pdf")
    assert fp1 != fp2


def test_compute_document_signature():
    """Signature must combine fingerprint, chunk_size, and chunk_overlap."""
    fp = "abcd1234efgh5678"
    sig1 = compute_document_signature(fp, chunk_size=1000, chunk_overlap=150)
    sig2 = compute_document_signature(fp, chunk_size=1000, chunk_overlap=150)
    sig3 = compute_document_signature(fp, chunk_size=800, chunk_overlap=150)
    sig4 = compute_document_signature(fp, chunk_size=1000, chunk_overlap=200)

    assert sig1 == (fp, 1000, 150)
    assert sig1 == sig2
    assert sig1 != sig3
    assert sig1 != sig4


# ---------------------------------------------------------------------------
# 2. Document Processing Tests
# ---------------------------------------------------------------------------

@patch("app.build_index")
@patch("app.embed_chunks")
@patch("app.chunk_pages")
@patch("app.extract_text_from_pdf")
def test_process_document_success(
    mock_extract, mock_chunk, mock_embed, mock_build
):
    """Verifies end-to-end document processing returning metadata and VectorStore."""
    # Mock extracted pages
    page1 = PageRecord(page_number=1, text="Introduction to transformers.")
    page2 = PageRecord(page_number=2, text="Attention is all you need.")
    mock_extract.return_value = [page1, page2]

    # Mock chunking
    chunk1 = TextChunk(text=page1.text, page_number=1, chunk_id="doc_p1_c1", start_char=0, end_char=29)
    chunk2 = TextChunk(text=page2.text, page_number=2, chunk_id="doc_p2_c1", start_char=0, end_char=26)
    mock_chunk.return_value = [chunk1, chunk2]

    # Mock embeddings & vector store
    mock_embed.return_value = MagicMock(spec=ChunkEmbeddings)
    fake_store = MagicMock(spec=VectorStore)
    fake_store.count = 2
    mock_build.return_value = fake_store

    metadata, vstore = process_document(
        file_bytes=b"%PDF-1.4 test",
        filename="test.pdf",
        chunk_size=1000,
        chunk_overlap=150,
    )

    assert metadata["filename"] == "test.pdf"
    assert metadata["page_count"] == 2
    assert metadata["char_count"] == 55
    assert metadata["chunk_count"] == 2
    assert metadata["empty_pages"] == 0
    assert metadata["file_size"] == len(b"%PDF-1.4 test")
    assert vstore == fake_store
    mock_build.assert_called_once()


@patch("app.extract_text_from_pdf")
def test_process_document_scanned_pdf_error(mock_extract):
    """Scanned PDF with 0 extractable characters must raise ScannedPDFError."""
    empty_p1 = PageRecord(page_number=1, text="")
    empty_p2 = PageRecord(page_number=2, text="   \n")
    mock_extract.return_value = [empty_p1, empty_p2]

    with pytest.raises(ScannedPDFError, match="no extractable digital text"):
        process_document(file_bytes=b"%PDF-1.4 scanned", filename="scanned.pdf")


@patch("app.extract_text_from_pdf")
def test_process_document_empty_pdf_error(mock_extract):
    """Empty PDF error from loader must propagate cleanly."""
    mock_extract.side_effect = EmptyPDFError("The provided PDF byte string is empty (0 bytes).")
    with pytest.raises(EmptyPDFError):
        process_document(file_bytes=b"", filename="empty.pdf")


@patch("app.extract_text_from_pdf")
def test_process_document_invalid_pdf_error(mock_extract):
    """Invalid PDF error from loader must propagate cleanly."""
    mock_extract.side_effect = InvalidPDFError("Not a PDF file")
    with pytest.raises(InvalidPDFError):
        process_document(file_bytes=b"not a pdf", filename="corrupt.pdf")


@patch("app.chunk_pages")
@patch("app.extract_text_from_pdf")
def test_process_document_no_chunks_error(mock_extract, mock_chunk):
    """If chunking produces 0 chunks, process_document must raise ValueError."""
    mock_extract.return_value = [PageRecord(page_number=1, text="Hello world")]
    mock_chunk.return_value = []

    with pytest.raises(ValueError, match="No text chunks could be produced"):
        process_document(file_bytes=b"%PDF-1.4 valid", filename="test.pdf")


# ---------------------------------------------------------------------------
# 3. State Management Unit Tests
# ---------------------------------------------------------------------------

def test_init_session_state_and_reset():
    """Verifies session state keys initialization and resetting behavior."""
    fake_state = {}
    with patch("streamlit.session_state", fake_state):
        init_session_state()
        assert fake_state["messages"] == []
        assert fake_state["doc_fingerprint"] is None
        assert fake_state["doc_signature"] is None
        assert fake_state["doc_metadata"] is None
        assert fake_state["vector_store"] is None

        # Populate state with active data
        fake_state["messages"].append({"role": "user", "content": "test?"})
        fake_state["doc_fingerprint"] = "abc123"
        fake_state["doc_signature"] = ("abc123", 1000, 150)
        fake_state["doc_metadata"] = {"filename": "test.pdf"}
        fake_state["vector_store"] = MagicMock()

        # Reset state
        reset_document_state()
        assert fake_state["messages"] == []
        assert fake_state["doc_fingerprint"] is None
        assert fake_state["doc_signature"] is None
        assert fake_state["doc_metadata"] is None
        assert fake_state["vector_store"] is None


# ---------------------------------------------------------------------------
# 4. Assistant Response Rendering Unit Tests
# ---------------------------------------------------------------------------

@patch("streamlit.caption")
@patch("streamlit.expander")
@patch("streamlit.success")
@patch("streamlit.markdown")
def test_render_assistant_response_valid_citation(
    mock_markdown, mock_success, mock_expander, mock_caption
):
    """Valid citation report renders success banner, caption disclaimer, and expandable sources."""
    src = SearchResult(
        chunk=TextChunk(text="AdamW optimizer", page_number=3, chunk_id="c1", start_char=0, end_char=15),
        score=0.91,
        index=0,
    )
    response = RAGResponse(
        answer="The model uses AdamW. [Page 3]",
        sources=[src],
        model="openai/gpt-oss-120b",
        question="What optimizer?",
    )
    validation = ValidationReport(
        is_valid=True,
        cited_pages=[3],
        available_pages=[3],
        invalid_citations=[],
        uncited_sentence_count=0,
        warnings=[],
    )

    render_assistant_response("The model uses AdamW. [Page 3]", response, validation)

    mock_markdown.assert_any_call("The model uses AdamW. [Page 3]")
    mock_success.assert_called_once()
    assert "Citations Verified" in mock_success.call_args[0][0]
    mock_caption.assert_called_once()
    mock_expander.assert_called_once()


@patch("streamlit.caption")
@patch("streamlit.error")
@patch("streamlit.markdown")
def test_render_assistant_response_invalid_citation(
    mock_markdown, mock_error, mock_caption
):
    """Invalid citation renders error banner flagging the invalid page."""
    src = SearchResult(
        chunk=TextChunk(text="Source text", page_number=2, chunk_id="c1", start_char=0, end_char=11),
        score=0.85,
        index=0,
    )
    response = RAGResponse(
        answer="Claims from future. [Page 9]",
        sources=[src],
        model="openai/gpt-oss-120b",
        question="What?",
    )
    validation = ValidationReport(
        is_valid=False,
        cited_pages=[9],
        available_pages=[2],
        invalid_citations=[9],
        uncited_sentence_count=0,
        warnings=["Cited page [9] absent from sources [2]."],
    )

    render_assistant_response(response.answer, response, validation)

    mock_error.assert_called_once()
    assert "Invalid Page Citation" in mock_error.call_args[0][0]
    assert "9" in mock_error.call_args[0][0]


@patch("streamlit.caption")
@patch("streamlit.warning")
@patch("streamlit.markdown")
def test_render_assistant_response_missing_citation(
    mock_markdown, mock_warning, mock_caption
):
    """Missing citation with available sources renders warning banner."""
    src = SearchResult(
        chunk=TextChunk(text="Source text", page_number=2, chunk_id="c1", start_char=0, end_char=11),
        score=0.85,
        index=0,
    )
    response = RAGResponse(
        answer="Uncited claim.",
        sources=[src],
        model="openai/gpt-oss-120b",
        question="What?",
    )
    validation = ValidationReport(
        is_valid=False,
        cited_pages=[],
        available_pages=[2],
        invalid_citations=[],
        uncited_sentence_count=1,
        warnings=["Answer contains no page citations."],
    )

    render_assistant_response(response.answer, response, validation)

    mock_warning.assert_called()
    assert "Missing Citations" in mock_warning.call_args[0][0]


@patch("streamlit.info")
@patch("streamlit.markdown")
def test_render_assistant_response_no_sources_refusal(
    mock_markdown, mock_info
):
    """Empty retrieved sources renders no retrieved evidence info banner."""
    response = RAGResponse(
        answer="The document does not provide enough information.",
        sources=[],
        model="openai/gpt-oss-120b",
        question="What?",
    )
    validation = ValidationReport(
        is_valid=True,
        cited_pages=[],
        available_pages=[],
        invalid_citations=[],
        uncited_sentence_count=0,
        warnings=[],
    )

    render_assistant_response(response.answer, response, validation)

    mock_info.assert_called_once()
    assert "No Retrieved Evidence" in mock_info.call_args[0][0]


@patch("streamlit.expander")
@patch("streamlit.markdown")
def test_render_assistant_response_with_rewritten_query(
    mock_markdown, mock_expander
):
    """When a question was reformulated, query rewriting details are rendered."""
    src = SearchResult(
        chunk=TextChunk(text="Source text", page_number=2, chunk_id="c1", start_char=0, end_char=11),
        score=0.85,
        index=0,
    )
    rag_resp = RAGResponse(
        answer="The learning rate was 0.001 [Page 2].",
        sources=[src],
        model="openai/gpt-oss-120b",
        question="What was the learning rate?",
    )
    conv_resp = ConversationalRAGResponse(
        original_question="What was it?",
        standalone_question="What was the learning rate?",
        rag_response=rag_resp,
        warning=None,
    )

    render_assistant_response(rag_resp.answer, response=rag_resp, validation=None, conv_response=conv_resp)

    # Check expander was opened for query rewriting
    mock_expander.assert_any_call("🔍 Query Rewriting (Conversational Context)", expanded=False)
    mock_markdown.assert_any_call("**Original Question:** What was it?")
    mock_markdown.assert_any_call("**Standalone Retrieval Query:** What was the learning rate?")


@patch("streamlit.warning")
@patch("streamlit.markdown")
def test_render_assistant_response_with_fallback_warning(
    mock_markdown, mock_warning
):
    """When conversational contextualization produces a warning, it is rendered in UI."""
    rag_resp = RAGResponse(
        answer="Fallback answer.",
        sources=[],
        model="openai/gpt-oss-120b",
        question="What was it?",
    )
    conv_resp = ConversationalRAGResponse(
        original_question="What was it?",
        standalone_question="What was it?",
        rag_response=rag_resp,
        warning="Rate limit reached during contextualization; fell back to original query.",
    )

    render_assistant_response(rag_resp.answer, response=rag_resp, validation=None, conv_response=conv_resp)

    mock_warning.assert_any_call("⚠️ Rate limit reached during contextualization; fell back to original query.")


# ---------------------------------------------------------------------------
# 5. Streamlit AppTest Integration Tests
# ---------------------------------------------------------------------------

def test_apptest_initial_render_empty_state():
    """Initial load of app.py displays title, welcome info, and sidebar uploader."""
    at = AppTest.from_file(APP_PATH)
    at.run()

    assert not at.exception
    assert len(at.title) >= 1
    assert "DocuMind" in at.title[0].value
    # Welcome banner is displayed when vector_store is None
    assert len(at.info) >= 1
    assert "Welcome to DocuMind" in at.info[0].value


def test_apptest_with_mocked_document_session():
    """AppTest displays active document metadata and previous messages from session state."""
    at = AppTest.from_file(APP_PATH)

    # Pre-populate session state as if a document was already indexed
    dummy_vstore = MagicMock(spec=VectorStore)
    dummy_vstore.count = 5
    dummy_vstore.dimension = 384

    doc_meta = {
        "filename": "attention_paper.pdf",
        "file_size": 256000,
        "page_count": 12,
        "char_count": 35000,
        "chunk_count": 35,
        "empty_pages": 0,
        "chunks": [],
    }

    at.session_state["vector_store"] = dummy_vstore
    at.session_state["doc_fingerprint"] = "dummy_fp"
    at.session_state["doc_signature"] = ("dummy_fp", 1000, 150)
    at.session_state["doc_metadata"] = doc_meta
    at.session_state["messages"] = [
        {"role": "user", "content": "What is self-attention?"},
        {
            "role": "assistant",
            "content": "Self-attention relates different positions. [Page 4]",
            "response": RAGResponse(
                answer="Self-attention relates different positions. [Page 4]",
                sources=[
                    SearchResult(
                        chunk=TextChunk(text="Self-attention relates positions.", page_number=4, chunk_id="c4", start_char=0, end_char=32),
                        score=0.92,
                        index=0,
                    )
                ],
                model="openai/gpt-oss-120b",
                question="What is self-attention?",
            ),
            "validation": ValidationReport(
                is_valid=True,
                cited_pages=[4],
                available_pages=[4],
                invalid_citations=[],
                uncited_sentence_count=0,
                warnings=[],
            ),
        },
    ]

    at.run()

    assert not at.exception
    # Check that chat messages are rendered
    assert len(at.chat_message) == 2
    assert at.chat_message[0].markdown[0].value == "What is self-attention?"
    assert "Self-attention relates different positions" in at.chat_message[1].markdown[0].value


@patch("src.answer_validator.validate_answer")
@patch("src.conversation.answer_with_history")
def test_apptest_chat_input_flow(mock_answer, mock_validate):
    """Simulates user submitting a prompt via chat_input and receiving a validated response."""
    src = SearchResult(
        chunk=TextChunk(text="AdamW was used.", page_number=3, chunk_id="c1", start_char=0, end_char=15),
        score=0.88,
        index=0,
    )
    mock_rag_response = RAGResponse(
        answer="The optimizer used was AdamW. [Page 3]",
        sources=[src],
        model="openai/gpt-oss-120b",
        question="Which optimizer?",
    )
    mock_conv_response = ConversationalRAGResponse(
        original_question="Which optimizer?",
        standalone_question="Which optimizer was used?",
        rag_response=mock_rag_response,
        warning=None,
    )
    mock_validation = ValidationReport(
        is_valid=True,
        cited_pages=[3],
        available_pages=[3],
        invalid_citations=[],
        uncited_sentence_count=0,
        warnings=[],
    )
    mock_answer.return_value = mock_conv_response
    mock_validate.return_value = mock_validation

    at = AppTest.from_file(APP_PATH)
    dummy_vstore = MagicMock(spec=VectorStore)
    at.session_state["vector_store"] = dummy_vstore
    at.session_state["doc_fingerprint"] = "fp123"
    at.session_state["doc_signature"] = ("fp123", 1000, 150)
    at.session_state["doc_metadata"] = {
        "filename": "paper.pdf",
        "file_size": 1000,
        "page_count": 3,
        "char_count": 2500,
        "chunk_count": 4,
        "empty_pages": 0,
        "chunks": [],
    }

    at.run()
    # Enter text into chat_input and submit
    at.chat_input[0].set_value("Which optimizer?").run()

    assert not at.exception
    mock_answer.assert_called_once()
    # Verify that conversation history was passed to the conversational pipeline
    call_kwargs = mock_answer.call_args[1]
    assert "conversation_history" in call_kwargs
    mock_validate.assert_called_once()

    # Session state should now contain user and assistant messages
    messages = at.session_state["messages"]
    assert len(messages) == 2
    assert messages[0]["role"] == "user"
    assert messages[0]["content"] == "Which optimizer?"
    assert messages[1]["role"] == "assistant"
    assert messages[1]["content"] == "The optimizer used was AdamW. [Page 3]"


@patch("src.conversation.answer_with_history")
def test_apptest_chat_auth_error_handling(mock_answer):
    """Authentication errors from Groq must be captured and displayed without crashing."""
    mock_answer.side_effect = LLMAuthenticationError("Invalid Groq API key")

    at = AppTest.from_file(APP_PATH)
    dummy_vstore = MagicMock(spec=VectorStore)
    at.session_state["vector_store"] = dummy_vstore
    at.session_state["doc_fingerprint"] = "fp123"
    at.session_state["doc_signature"] = ("fp123", 1000, 150)
    at.session_state["doc_metadata"] = {
        "filename": "paper.pdf",
        "file_size": 1000,
        "page_count": 1,
        "char_count": 500,
        "chunk_count": 1,
        "empty_pages": 0,
        "chunks": [],
    }

    at.run()
    at.chat_input[0].set_value("Test query").run()

    assert not at.exception
    # Error message should be rendered in UI
    assert len(at.error) >= 1
    assert "Authentication Error" in at.error[0].value


@patch("src.conversation.answer_with_history")
def test_apptest_chat_rate_limit_error_handling(mock_answer):
    """Rate limit errors from Groq must be caught and rendered clearly."""
    mock_answer.side_effect = LLMRateLimitError("Rate limit reached. Please wait.")

    at = AppTest.from_file(APP_PATH)
    dummy_vstore = MagicMock(spec=VectorStore)
    at.session_state["vector_store"] = dummy_vstore
    at.session_state["doc_fingerprint"] = "fp123"
    at.session_state["doc_signature"] = ("fp123", 1000, 150)
    at.session_state["doc_metadata"] = {
        "filename": "paper.pdf",
        "file_size": 1000,
        "page_count": 1,
        "char_count": 500,
        "chunk_count": 1,
        "empty_pages": 0,
        "chunks": [],
    }

    at.run()
    at.chat_input[0].set_value("Rate limit test").run()

    assert not at.exception
    assert len(at.error) >= 1
    assert "Rate Limit Exceeded" in at.error[0].value


@patch("src.conversation.answer_with_history")
def test_apptest_chat_empty_response_error_handling(mock_answer):
    """Empty response errors from LLM must be rendered without crashing."""
    mock_answer.side_effect = LLMEmptyResponseError("The model returned an empty completion.")

    at = AppTest.from_file(APP_PATH)
    dummy_vstore = MagicMock(spec=VectorStore)
    at.session_state["vector_store"] = dummy_vstore
    at.session_state["doc_fingerprint"] = "fp123"
    at.session_state["doc_signature"] = ("fp123", 1000, 150)
    at.session_state["doc_metadata"] = {
        "filename": "paper.pdf",
        "file_size": 1000,
        "page_count": 1,
        "char_count": 500,
        "chunk_count": 1,
        "empty_pages": 0,
        "chunks": [],
    }

    at.run()
    at.chat_input[0].set_value("Empty test").run()

    assert not at.exception
    assert len(at.error) >= 1
    assert "Empty Response" in at.error[0].value


@patch("src.conversation.answer_with_history")
def test_apptest_chat_network_api_error_handling(mock_answer):
    """Network and API errors from Groq must be rendered clearly."""
    mock_answer.side_effect = LLMAPIError("Connection timeout while contacting Groq.")

    at = AppTest.from_file(APP_PATH)
    dummy_vstore = MagicMock(spec=VectorStore)
    at.session_state["vector_store"] = dummy_vstore
    at.session_state["doc_fingerprint"] = "fp123"
    at.session_state["doc_signature"] = ("fp123", 1000, 150)
    at.session_state["doc_metadata"] = {
        "filename": "paper.pdf",
        "file_size": 1000,
        "page_count": 1,
        "char_count": 500,
        "chunk_count": 1,
        "empty_pages": 0,
        "chunks": [],
    }

    at.run()
    at.chat_input[0].set_value("Network error test").run()

    assert not at.exception
    assert len(at.error) >= 1
    assert "Groq API / Network Error" in at.error[0].value


@patch("src.conversation.answer_with_history")
def test_apptest_chat_retrieval_error_handling(mock_answer):
    """Retrieval errors from FAISS must be captured and displayed."""
    mock_answer.side_effect = RetrievalError("FAISS index corrupted or unavailable.")

    at = AppTest.from_file(APP_PATH)
    dummy_vstore = MagicMock(spec=VectorStore)
    at.session_state["vector_store"] = dummy_vstore
    at.session_state["doc_fingerprint"] = "fp123"
    at.session_state["doc_signature"] = ("fp123", 1000, 150)
    at.session_state["doc_metadata"] = {
        "filename": "paper.pdf",
        "file_size": 1000,
        "page_count": 1,
        "char_count": 500,
        "chunk_count": 1,
        "empty_pages": 0,
        "chunks": [],
    }

    at.run()
    at.chat_input[0].set_value("Retrieval error test").run()

    assert not at.exception
    assert len(at.error) >= 1
    assert "Retrieval Error" in at.error[0].value


def test_apptest_reset_button_clears_state():
    """Clicking 'Reset Document & Chat' clears all session state and returns to welcome screen."""
    at = AppTest.from_file(APP_PATH)
    dummy_vstore = MagicMock(spec=VectorStore)
    at.session_state["vector_store"] = dummy_vstore
    at.session_state["doc_fingerprint"] = "fp_reset"
    at.session_state["doc_signature"] = ("fp_reset", 1000, 150)
    at.session_state["doc_metadata"] = {
        "filename": "paper_to_reset.pdf",
        "file_size": 1000,
        "page_count": 1,
        "char_count": 500,
        "chunk_count": 1,
        "empty_pages": 0,
        "chunks": [],
    }
    at.session_state["messages"] = [{"role": "user", "content": "Question before reset"}]

    at.run()
    # Find the reset button
    reset_buttons = [b for b in at.button if "Reset Document & Chat" in b.label]
    assert len(reset_buttons) == 1
    reset_buttons[0].click().run()

    assert not at.exception
    assert at.session_state["vector_store"] is None
    assert at.session_state["doc_fingerprint"] is None
    assert at.session_state["doc_metadata"] is None
    assert at.session_state["messages"] == []
    assert len(at.info) >= 1
    assert "Welcome to DocuMind" in at.info[0].value
