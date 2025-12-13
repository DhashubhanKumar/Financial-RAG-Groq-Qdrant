import streamlit as st
import os
import tempfile
from pathlib import Path

import nltk
import qdrant_client

from llama_index.core import (
    VectorStoreIndex,
    SimpleDirectoryReader,
    Settings,
)
from llama_index.llms.groq import Groq
from llama_index.vector_stores.qdrant import QdrantVectorStore
from llama_index.core.storage import StorageContext
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.prompts import ChatPromptTemplate, MessageRole
from llama_index.core.llms import ChatMessage


# ===============================
# FIX NLTK PERMISSION ERROR
# ===============================
@st.cache_resource
def setup_nltk():
    nltk.data.path.append("/tmp/nltk_data")
    nltk.download("stopwords", download_dir="/tmp/nltk_data", quiet=True)

setup_nltk()


# ===============================
# SECRETS
# ===============================
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
QDRANT_ENDPOINT = os.environ.get("QDRANT_ENDPOINT")
QDRANT_API_KEY = os.environ.get("QDRANT_API_KEY")

COLLECTION_NAME = "financial_rag_streamlit"


# ===============================
# INIT MODELS
# ===============================
@st.cache_resource
def initialize_models():
    if not GROQ_API_KEY:
        st.error("GROQ_API_KEY not found")
        st.stop()

    # LLM
    Settings.llm = Groq(
        model="llama-3.1-8b-instant",
        api_key=GROQ_API_KEY,
    )

    # IMPORTANT: use default embeddings (cloud-safe)
    Settings.embed_model = None

    SYSTEM_PROMPT = (
        "You are a financial analyst.\n"
        "Answer ONLY using the document context.\n"
        "If the answer is not present, say so clearly."
    )

    st.session_state.chat_template = ChatPromptTemplate(
        message_templates=[
            ChatMessage(role=MessageRole.SYSTEM, content=SYSTEM_PROMPT),
            ChatMessage(
                role=MessageRole.USER,
                content="Context:\n{context_str}\n\nQuestion:\n{query_str}",
            ),
        ]
    )


# ===============================
# INDEX BUILDING
# ===============================
def build_index(pdf_path: str):
    client = qdrant_client.QdrantClient(
        url=f"https://{QDRANT_ENDPOINT}",
        api_key=QDRANT_API_KEY,
        prefer_grpc=False,
        timeout=60,
    )

    vector_store = QdrantVectorStore(
        client=client,
        collection_name=COLLECTION_NAME,
    )

    storage_context = StorageContext.from_defaults(
        vector_store=vector_store
    )

    docs = SimpleDirectoryReader(
        input_files=[Path(pdf_path)]
    ).load_data()

    return VectorStoreIndex.from_documents(
        docs,
        storage_context=storage_context,
        show_progress=True,
    )


def get_query_engine(index):
    retriever = VectorIndexRetriever(
        index=index,
        similarity_top_k=8,
    )

    query_engine = RetrieverQueryEngine(
        retriever=retriever,
    )

    query_engine.update_prompts(
        {"response_synthesizer:text_qa_template": st.session_state.chat_template}
    )

    return query_engine


# ===============================
# STREAMLIT UI
# ===============================
st.set_page_config(
    page_title="Financial RAG Analyst (Groq + Qdrant)",
    layout="wide",
)

st.title("📊 Financial RAG Analyst (Groq + Qdrant)")
st.markdown("---")

initialize_models()

with st.sidebar:
    st.header("Upload Financial Report")
    uploaded_file = st.file_uploader(
        "Upload a 10-K / Annual Report (PDF)",
        type="pdf",
    )

query_enabled = False

if uploaded_file:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(uploaded_file.read())
        pdf_path = tmp.name

    if "indexed" not in st.session_state:
        with st.spinner("Indexing document (first time only)..."):
            st.session_state.index = build_index(pdf_path)
            st.session_state.query_engine = get_query_engine(
                st.session_state.index
            )
            st.session_state.indexed = True

        st.success("✅ Document indexed successfully")

    query_enabled = True

    os.unlink(pdf_path)

else:
    st.info("Upload a PDF to begin")

user_query = st.text_input(
    "Ask a question about the report:",
    disabled=not query_enabled,
)

if user_query and query_enabled:
    with st.spinner("Analyzing..."):
        response = st.session_state.query_engine.query(user_query)

    st.markdown(
        f"""
        <div style="background:#1a1a1a;padding:20px;border-radius:12px">
            <h3>Answer</h3>
            <p>{response.response}</p>
