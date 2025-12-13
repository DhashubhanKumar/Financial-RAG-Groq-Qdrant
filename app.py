import os
os.environ["NLTK_DATA"] = "/tmp/nltk_data"

import nltk
try:
    nltk.data.find("corpora/stopwords")
except LookupError:
    nltk.download("stopwords", download_dir="/tmp/nltk_data", quiet=True)

import streamlit as st
import tempfile
from pathlib import Path

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

# =====================
# CONFIG
# =====================

GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
QDRANT_ENDPOINT = os.environ.get("QDRANT_ENDPOINT")
QDRANT_API_KEY = os.environ.get("QDRANT_API_KEY")

COLLECTION_NAME = "financial-rag-final"

# =====================
# INIT MODELS
# =====================

@st.cache_resource
def initialize_models():
    Settings.llm = Groq(
        model="llama-3.1-8b-instant",
        api_key=GROQ_API_KEY,
    )

    SYSTEM_PROMPT = (
        "You are a financial analyst.\n"
        "Rules:\n"
        "1. Answer ONLY using document context.\n"
        "2. If answer is missing, say so.\n"
        "3. Cite page numbers using 'page_label'."
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

initialize_models()

# =====================
# INDEXING
# =====================

def build_index(pdf_path: str):
    client = qdrant_client.QdrantClient(
        url=f"https://{QDRANT_ENDPOINT}",
        api_key=QDRANT_API_KEY,
        timeout=60,
    )

    try:
        client.delete_collection(COLLECTION_NAME)
    except Exception:
        pass

    vector_store = QdrantVectorStore(
        client=client,
        collection_name=COLLECTION_NAME,
        enable_hybrid=True,
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
    )

def get_query_engine(index):
    retriever = VectorIndexRetriever(
        index=index,
        similarity_top_k=8,
    )

    query_engine = RetrieverQueryEngine(
        retriever=retriever
    )

    query_engine.update_prompts(
        {"response_synthesizer:text_qa_template": st.session_state.chat_template}
    )

    return query_engine

# =====================
# UI
# =====================

st.set_page_config(page_title="Financial RAG Analyst", layout="wide")
st.title("📊 Financial RAG Analyst (Groq + Qdrant)")
st.markdown("---")

with st.sidebar:
    uploaded_file = st.file_uploader("Upload financial PDF", type="pdf")

query_enabled = False

if uploaded_file:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(uploaded_file.read())
        pdf_path = tmp.name

    with st.spinner("Indexing document..."):
        st.session_state.index = build_index(pdf_path)
        st.session_state.query_engine = get_query_engine(
            st.session_state.index
        )

    query_enabled = True
    st.success("✅ Document indexed!")

    os.unlink(pdf_path)

else:
    st.info("Upload a PDF to begin.")

query = st.text_input(
    "Ask a question:",
    disabled=not query_enabled
)

if query and query_enabled:
    with st.spinner("Analyzing..."):
        response = st.session_state.query_engine.query(query)

    st.markdown(
        f"""
        <div style="background:#1a1a1a;padding:20px;border-radius:10px">
            <h3>Answer</h3>
            <p>{response.response}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
