import os
os.environ["NLTK_DATA"] = "/tmp/nltk"

import nltk
try:
    nltk.data.find("corpora/stopwords")
except LookupError:
    nltk.download("stopwords", download_dir="/tmp/nltk", quiet=True)

import streamlit as st
import tempfile
from pathlib import Path
import qdrant_client

from llama_index import (
    VectorStoreIndex,
    SimpleDirectoryReader,
    Settings,
)
from llama_index.llms.groq import Groq
from llama_index.vector_stores.qdrant import QdrantVectorStore
from llama_index.storage.storage_context import StorageContext
from llama_index.retrievers import VectorIndexRetriever
from llama_index.query_engine import RetrieverQueryEngine
from llama_index.prompts import ChatPromptTemplate
from llama_index.llms import ChatMessage, MessageRole


# =====================
# SECRETS
# =====================

GROQ_API_KEY = os.environ["GROQ_API_KEY"]
QDRANT_ENDPOINT = os.environ["QDRANT_ENDPOINT"]
QDRANT_API_KEY = os.environ["QDRANT_API_KEY"]

COLLECTION_NAME = "financial-rag-final"


# =====================
# INIT
# =====================

@st.cache_resource
def init_models():
    Settings.llm = Groq(
        model="llama-3.1-8b-instant",
        api_key=GROQ_API_KEY,
    )

    st.session_state.prompt = ChatPromptTemplate(
        message_templates=[
            ChatMessage(
                role=MessageRole.SYSTEM,
                content=(
                    "You are a financial analyst. "
                    "Answer ONLY from the provided document context. "
                    "If not found, say so. Cite page numbers using page_label."
                ),
            ),
            ChatMessage(
                role=MessageRole.USER,
                content="Context:\n{context_str}\n\nQuestion:\n{query_str}",
            ),
        ]
    )

init_models()


# =====================
# INDEX
# =====================

def build_index(pdf_path: str):
    client = qdrant_client.QdrantClient(
        url=f"https://{QDRANT_ENDPOINT}",
        api_key=QDRANT_API_KEY,
    )

    try:
        client.delete_collection(COLLECTION_NAME)
    except Exception:
        pass

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


def make_query_engine(index):
    retriever = VectorIndexRetriever(index=index, similarity_top_k=8)

    engine = RetrieverQueryEngine(retriever=retriever)
    engine.update_prompts(
        {"response_synthesizer:text_qa_template": st.session_state.prompt}
    )
    return engine


# =====================
# UI
# =====================

st.set_page_config("Financial RAG Analyst", layout="wide")
st.title("📊 Financial RAG Analyst (Groq + Qdrant)")
st.markdown("---")

uploaded = st.file_uploader("Upload a financial PDF", type="pdf")

if uploaded:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as f:
        f.write(uploaded.read())
        path = f.name

    with st.spinner("Indexing document…"):
        st.session_state.index = build_index(path)
        st.session_state.engine = make_query_engine(st.session_state.index)

    os.unlink(path)
    st.success("✅ Document indexed!")

if "engine" in st.session_state:
    q = st.text_input("Ask a question")
    if q:
        with st.spinner("Thinking…"):
            res = st.session_state.engine.query(q)
        st.markdown(res.response)
