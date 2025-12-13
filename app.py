import nltk # <-- Ensure this import is here

# =====================
# NLTK FIX FOR STREAMLIT CLOUD
# =====================
@st.cache_resource
def download_nltk_data():
    """Forces NLTK to download necessary data files."""
    try:
        nltk.download('stopwords', quiet=True)
    except Exception as e:
        # Fail silently if download still restricted, hoping another part of the code handles it
        print(f"NLTK Download failed: {e}")
        
# Call the fix immediately after imports, before any other logic
download_nltk_data()

# =====================
# CONFIGURATION AND SECRETS
# =====================
# ... rest of your code ...



import streamlit as st
import os
import tempfile
from pathlib import Path

# --- Core RAG Imports ---
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
from llama_index.core.embeddings import resolve_embed_model
import qdrant_client

# =====================
# CONFIGURATION AND SECRETS
# =====================

# CRITICAL: These retrieve the keys set in your PowerShell environment ($env:KEY_NAME)
# Ensure you set these in your terminal before running: $env:GROQ_API_KEY="..." etc.
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
QDRANT_ENDPOINT = os.environ.get("QDRANT_ENDPOINT")
QDRANT_API_KEY = os.environ.get("QDRANT_API_KEY")

COLLECTION_NAME = "financial-rag-final"
EMBED_MODEL_NAME = "local:BAAI/bge-small-en-v1.5"
# Note: Using SimpleDirectoryReader since Camelot/OpenCV were removed

# =====================
# INIT MODELS & RAG SETUP
# =====================

@st.cache_resource
def initialize_models():
    """Initializes LLM and Embedding Model and sets up the System Prompt."""
    if not GROQ_API_KEY:
        st.error("GROQ_API_KEY not found in environment variables.")
        return None

    # 1. Initialize Settings
    Settings.llm = Groq(model="llama-3.1-8b-instant", api_key=GROQ_API_KEY)
    Settings.embed_model = resolve_embed_model(EMBED_MODEL_NAME)
    
    # 2. Setup Strict System Prompt for Guardrails and Citation
    SYSTEM_PROMPT = (
        "You are a highly skilled financial analyst. Your primary goal is to provide precise, "
        "data-driven answers to questions about the loaded financial report. "
        "You must adhere to the following strict rules:\n"
        "1. Answer ONLY using the context provided in the retrieved documents.\n"
        "2. If the context does not contain the answer, state, 'The necessary financial data was not found in the report context.'\n"
        "3. You must CITE the page number from the document's metadata (labeled 'page_label') for every fact used. "
        "Format the citation as:."
    )
    st.session_state.chat_template = ChatPromptTemplate(
        message_templates=[
            ChatMessage(role=MessageRole.SYSTEM, content=SYSTEM_PROMPT),
            ChatMessage(role=MessageRole.USER, content="The context is: {context_str}. The user's question is: {query_str}"),
        ]
    )
    return Settings.llm

def build_index(pdf_path: str):
    if not QDRANT_ENDPOINT or not QDRANT_API_KEY:
        raise ValueError("QDRANT secrets not configured. Check environment variables.")

    # CRITICAL FIX: Increased timeout to 60 seconds to prevent upload failures
    client = qdrant_client.QdrantClient(
        url=f"https://{QDRANT_ENDPOINT}",
        api_key=QDRANT_API_KEY,
        prefer_grpc=False,
        timeout=60  # Increased timeout
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

def get_optimized_query_engine(index):
    """Creates the final, optimized query engine with Reranker and strict prompt."""
    retriever = VectorIndexRetriever(
        index=index,
        similarity_top_k=10, 
    )

    # Use a Cross-Encoder Reranker for higher retrieval accuracy
    reranker = SentenceTransformerRerank(
        model="cross-encoder/ms-marco-MiniLM-L-6-v2", 
        top_n=3
    )
    
    query_engine = RetrieverQueryEngine(
        retriever=retriever,
        node_postprocessors=[reranker],
    )
    
    # Apply the strict System Prompt
    if 'chat_template' in st.session_state:
        query_engine.update_prompts({"response_synthesizer:text_qa_template": st.session_state.chat_template})
    
    return query_engine

# =====================
# STREAMLIT UI (X/Twitter Style)
# =====================

st.set_page_config(
    page_title="Financial RAG Analyst (Groq + Qdrant)",
    layout="wide",
)

# Apply a dark, X-like theme using CSS
st.markdown("""
<style>
/* X/Twitter Dark Theme */
body {
    background-color: #000000;
    color: #FFFFFF;
}
.st-emotion-cache-1c0j03h { /* Main content area */
    background-color: #000000; 
}
.st-emotion-cache-1avcm0s { /* Text input background */
    background-color: #1a1a1a;
    border-radius: 12px;
    border: 1px solid #38444d; /* Subtle X border color */
}
.st-emotion-cache-7ym5gk { /* Button styling */
    background-color: #1da1f2; /* X brand blue */
    color: white;
    font-weight: bold;
    border-radius: 9999px; 
    border: none;
    padding: 10px 20px;
}
h1 {
    color: #FFFFFF; 
    border-bottom: 2px solid #1da1f2; /* Blue underline for title */
    padding-bottom: 10px;
    margin-top: 0;
}
.st-emotion-cache-16niy5c { /* Text in input box */
    color: #FFFFFF; 
}
</style>
""", unsafe_allow_html=True)


st.title("Financial RAG Analyst 📊 (Groq + Qdrant)")
st.markdown("---")

# Initialize models
initialize_models()

# Sidebar for PDF Upload
with st.sidebar:
    st.header("Upload 10-K Report")
    uploaded_file = st.file_uploader(
        "Upload a 10-K, 20-F, or Annual Report PDF", 
        type="pdf"
    )
    st.markdown("---")
    st.markdown("Status: Ready to load.")


# --- MAIN APPLICATION LOGIC ---

query_enabled = False

if uploaded_file is None:
    st.info("Please upload a PDF file on the left sidebar to begin processing and enable the query box.")
    
elif 'index' not in st.session_state or st.session_state.uploaded_name != uploaded_file.name:
    # New file uploaded or first load
    st.session_state.uploaded_name = uploaded_file.name
    
    # 1. Save uploaded file to temp path
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(uploaded_file.read())
        pdf_path = tmp_file.name
        st.session_state.pdf_path = pdf_path
    
    # 2. Build Index (The slow part)
    with st.spinner(f"Indexing document: {uploaded_file.name}... (This is the long part)"):
        try:
            st.session_state.index = build_index(pdf_path)
            st.session_state.query_engine = get_optimized_query_engine(st.session_state.index)
            st.success(f"✅ Document: {uploaded_file.name} is ready for analysis!")
            query_enabled = True
        except Exception as e:
            st.error(f"An indexing error occurred (Timeout or connection): {e}")
            st.session_state.index = None
            st.session_state.query_engine = None
        finally:
            # 3. Clean up the temporary file immediately after indexing
            if 'pdf_path' in st.session_state and os.path.exists(st.session_state.pdf_path):
                os.unlink(st.session_state.pdf_path)
    
if 'query_engine' in st.session_state and st.session_state.query_engine:
    query_enabled = True


# --- QUERY INTERFACE (CONDITIONAL) ---

user_query = st.text_input(
    "Ask a question about the financial report:",
    placeholder="e.g., What was the Total Revenue for the year ended Dec 31, 2023?",
    disabled=not query_enabled # This disables the box until the index is built
)

if user_query and query_enabled:
    with st.spinner("Analyzing document... (Powered by Groq)"):
        try:
            response = st.session_state.query_engine.query(user_query)
            
            # Display response in a clean, X-like quote box
            st.markdown(f"""
            <div style="background-color: #1a1a1a; padding: 20px; border-radius: 12px; margin-top: 20px;">
                <h3 style="color: #1da1f2; margin-top: 0;">Analyst Response:</h3>
                <p style="white-space: pre-wrap; font-size: 1.1em;">{response.response}</p>
            </div>
            """, unsafe_allow_html=True)

        except Exception as e:
            st.error(f"An error occurred during query execution. Check your Groq API key: {e}")