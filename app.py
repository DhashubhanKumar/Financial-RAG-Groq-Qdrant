# =====================
# 🔥 CRITICAL CLOUD FIX (MUST BE FIRST)
# =====================
import os
os.environ["LLAMA_INDEX_DISABLE_NLTK"] = "1"
os.environ["NLTK_DATA"] = "/tmp/nltk_data"

# =====================
# STANDARD IMPORTS
# =====================
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
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.vector_stores.qdrant import QdrantVectorStore
from llama_index.core.storage import StorageContext
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.postprocessor import SentenceTransformerRerank
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.prompts import ChatPromptTemplate, MessageRole
from llama_index.core.llms import ChatMessage

# =====================
# CONFIGURATION & SECRETS
# =====================
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
QDRANT_ENDPOINT = os.environ.get("QDRANT_ENDPOINT")
QDRANT_API_KEY = os.environ.get("QDRANT_API_KEY")

COLLECTION_NAME = "financial-rag-final"

# =====================
# INITIALIZE MODELS
# =====================
@st.cache_resource
def initialize_models():
    if not GROQ_API_KEY:
        st.error("❌ GROQ_API_KEY missing in Streamlit secrets")
        st.stop()

    Settings.llm = Groq(
        model="llama-3.1-8b-instant",
        api_key=GROQ_API_KEY,
    )

    Settings.embed_model = HuggingFaceEmbedding(
        model_name="BAAI/bge-small-en-v1.5"
    )

    SYSTEM_PROMPT = (
        "You are a highly skilled financial analyst.\n"
        "Rules:\n"
        "1. Answer ONLY using the retrieved document context.\n"
        "2. If information is missing, say so clearly.\n"
        "3. Cite page numbers using metadata field 'page_label'."
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

# =====================
# INDEX BUILDING
# =====================
def build_index(pdf_path: str):
    client = qdrant_client.QdrantClient(
        url=f"https://{QDRANT_ENDPOINT}",
        api_key=QDRANT_API_KEY,
        timeout=60,
    )

    vector_store = QdrantVectorStore(
        client=client,
        collection_name=COLLECTION_NAME,
        enable_hybrid=True,
    )

    storage_context = StorageContext.from_defaults(
        vector_store=vector_store
    )

    documents = SimpleDirectoryReader(
        input_files=[Path(pdf_path)]
    ).load_data()

    return VectorStoreIndex.from_documents(
        documents,
        storage_context=storage_context,
        show_progress=True,
    )

def get_query_engine(index: VectorStoreIndex):
    retriever = VectorIndexRetriever(
        index=index,
        similarity_top_k=10,
    )

    reranker = SentenceTransformerRerank(
        model="cross-encoder/ms-marco-MiniLM-L-6-v2",
        top_n=3,
    )

    query_engine = RetrieverQueryEngine(
        retriever=retriever,
        node_postprocessors=[reranker],
    )

    query_engine.update_prompts(
        {"response_synthesizer:text_qa_template": st.session_state.chat_template}
    )

    return query_engine

# =====================
# STREAMLIT UI
# =====================
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
        "Upload a 10-K, 20-F, or Annual Report PDF",
        type="pdf",
    )

query_enabled = False

if uploaded_file:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(uploaded_file.read())
        pdf_path = tmp.name

    if st.session_state.get("current_file") != uploaded_file.name:
        st.session_state.current_file = uploaded_file.name

        with st.spinner("Indexing document (first time only)..."):
            st.session_state.index = build_index(pdf_path)
            st.session_state.query_engine = get_query_engine(
                st.session_state.index
            )

        st.success("✅ Document indexed and ready!")

    query_enabled = True
    os.unlink(pdf_path)

else:
    st.info("Upload a PDF to begin.")

user_query = st.text_input(
    "Ask a question about the report:",
    placeholder="e.g. What was the total revenue in 2023?",
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
        </div>
        """,
        unsafe_allow_html=True,
    )
