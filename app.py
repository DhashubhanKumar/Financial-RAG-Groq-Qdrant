# -----------------------------
# MUST BE AT THE VERY TOP
# -----------------------------
import os
os.environ["NLTK_DATA"] = "/tmp/nltk_data"

import nltk
nltk.data.path.append("/tmp/nltk_data")
nltk.download("punkt", quiet=True)
nltk.download("stopwords", quiet=True)

# -----------------------------
# NOW SAFE TO IMPORT EVERYTHING
# -----------------------------
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


# -----------------------------
# SECRETS
# -----------------------------
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
QDRANT_ENDPOINT = os.environ.get("QDRANT_ENDPOINT")
QDRANT_API_KEY = os.environ.get("QDRANT_API_KEY")

COLLECTION_NAME = "financial_rag_streamlit"


# -----------------------------
# INIT MODELS
# -----------------------------
@st.cache_resource
def init_models():
    if not GROQ_API_KEY:
        st.error("Missing GROQ_API_KEY")
        st.stop()

    Settings.llm = Groq(
        model="llama-3.1-8b-instant",
        api_key=GROQ_API_KEY,
    )

    Settings.embed_model = None

    st.session_state.prompt = ChatPromptTemplate(
        message_templates=[
            ChatMessage(
                role=MessageRole.SYSTEM,
                content="You are a financial analyst. Answer ONLY from the document context.",
            ),
            ChatMessage(
                role=MessageRole.USER,
                content="Context:\n{context_str}\n\nQuestion:\n{query_str}",
            ),
        ]
    )


# -----------------------------
# BUILD INDEX
# -----------------------------
def build_index(pdf_path: str):
    client = qdrant_client.QdrantClient(
        url=f"https://{QDRANT_ENDPOINT}",
        api_key=QDRANT_API_KEY,
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
    )


def get_query_engine(index):
    retriever = VectorIndexRetriever(index=index, similarity_top_k=8)
    engine = RetrieverQueryEngine(retriever=retriever)
    engine.update_prompts(
        {"response_synthesizer:text_qa_template": st.session_state.prompt}
    )
    return engine


# -----------------------------
# UI
# -----------------------------
st.set_page_config(page_title="Financial RAG Analyst", layout="wide")
st.title("📊 Financial RAG Analyst (Groq + Qdrant)")
st.markdown("---")

init_models()

with st.sidebar:
    uploaded = st.file_uploader("Upload financial PDF", type="pdf")

if uploaded:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(uploaded.read())
        pdf_path = tmp.name

    if "engine" not in st.session_state:
        with st.spinner("Indexing document..."):
            index = build_index(pdf_path)
            st.session_state.engine = get_query_engine(index)
        st.success("Document indexed successfully")

    os.unlink(pdf_path)

    q = st.text_input("Ask a question")

    if q:
        with st.spinner("Analyzing..."):
            res = st.session_state.engine.query(q)
        st.markdown(f"### Answer\n{res.response}")
else:
    st.info("Upload a PDF to begin")
