import os
import tempfile
from pathlib import Path
import streamlit as st
import qdrant_client

from llama_index.core import (
    VectorStoreIndex,
    SimpleDirectoryReader,
    Settings,
)
from llama_index.llms.groq import Groq
from llama_index.embeddings.fastembed import FastEmbedEmbedding
from llama_index.vector_stores.qdrant import QdrantVectorStore
from llama_index.core.storage.storage_context import StorageContext
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.postprocessor import SentenceTransformerRerank
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.prompts import ChatPromptTemplate, MessageRole
from llama_index.core.llms import ChatMessage


# =========================
# SECRETS
# =========================
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
QDRANT_ENDPOINT = os.environ.get("QDRANT_ENDPOINT")
QDRANT_API_KEY = os.environ.get("QDRANT_API_KEY")

COLLECTION_NAME = "financial_rag_prod"


# =========================
# INIT MODELS
# =========================
@st.cache_resource
def init_models():
    Settings.llm = Groq(
        model="llama-3.1-8b-instant",
        api_key=GROQ_API_KEY,
    )

    Settings.embed_model = FastEmbedEmbedding(
        model_name="BAAI/bge-small-en-v1.5"
    )

    SYSTEM_PROMPT = (
        "You are a financial analyst.\n"
        "Rules:\n"
        "1. Answer ONLY from the provided context.\n"
        "2. If the answer is missing, say so.\n"
        "3. Cite page numbers using 'page_label'."
    )

    st.session_state.prompt = ChatPromptTemplate(
        message_templates=[
            ChatMessage(role=MessageRole.SYSTEM, content=SYSTEM_PROMPT),
            ChatMessage(
                role=MessageRole.USER,
                content="Context:\n{context_str}\n\nQuestion:\n{query_str}",
            ),
        ]
    )


# =========================
# BUILD INDEX
# =========================
def build_index(pdf_path: str):
    client = qdrant_client.QdrantClient(
        url=f"https://{QDRANT_ENDPOINT}",
        api_key=QDRANT_API_KEY,
        timeout=60,
    )

    vector_store = QdrantVectorStore(
        client=client,
        collection_name=COLLECTION_NAME,
        embed_dim=384,
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


def get_query_engine(index: VectorStoreIndex):
    retriever = VectorIndexRetriever(index=index, similarity_top_k=10)

    reranker = SentenceTransformerRerank(
        model="cross-encoder/ms-marco-MiniLM-L-6-v2",
        top_n=3,
    )

    engine = RetrieverQueryEngine(
        retriever=retriever,
        node_postprocessors=[reranker],
    )

    engine.update_prompts(
        {"response_synthesizer:text_qa_template": st.session_state.prompt}
    )

    return engine


# =========================
# STREAMLIT UI
# =========================
st.set_page_config(
    page_title="Financial RAG Analyst (Groq + Qdrant)",
    layout="wide",
)

st.title("📊 Financial RAG Analyst (Groq + Qdrant)")
st.markdown("---")

init_models()

with st.sidebar:
    uploaded = st.file_uploader("Upload financial PDF", type="pdf")

if uploaded:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(uploaded.read())
        pdf_path = tmp.name

    with st.spinner("Indexing document…"):
        index = build_index(pdf_path)
        st.session_state.engine = get_query_engine(index)

    os.unlink(pdf_path)
    st.success("Document indexed successfully!")

query = st.text_input(
    "Ask a question about the report",
    disabled="engine" not in st.session_state,
)

if query and "engine" in st.session_state:
    with st.spinner("Analyzing…"):
        response = st.session_state.engine.query(query)

    st.markdown(
        f"""
        <div style="background:#1a1a1a;padding:20px;border-radius:12px">
            <h3>Answer</h3>
            <p>{response.response}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
