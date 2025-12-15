# 📊 Financial RAG Analyst (Groq + Qdrant)

A high-performance, full-stack **Retrieval-Augmented Generation (RAG)** application designed to analyze complex financial and regulatory documents such as **10-K filings** (PDFs).

The system delivers **verifiable, sub-second responses** by combining high-speed inference, an advanced retrieval pipeline, and cloud-native vector storage.

---

## 🚀 Key Features & Architecture

This system is purpose-built to overcome common RAG failures related to latency, hallucination, and poor retrieval quality.

| Feature | Libraries Used | Technical Achievement |
|:------|:---------------|:----------------------|
| **Document Ingestion** | `LlamaIndex`, `SimpleDirectoryReader` | Efficiently loads and processes **unstructured text** from PDF documents. |
| **High-Speed Inference** | `Groq` | Uses **Llama-3.1-8B-Instant** for industry-leading sub-second inference. |
| **Advanced Retrieval** | `FastEmbed`, `SentenceTransformers` | Implements a **Cross-Encoder Re-ranker** (`ms-marco-MiniLM-L-6-v2`) to select the top-3 most relevant document chunks from the initial top-10 candidates. |
| **Trust Layer & Guardrails** | Custom System Prompt | Enforces **mandatory source citation** and includes a **context-only hallucination guardrail**. |
| **Persistent Vector Store** | `Qdrant Cloud` | Cloud-hosted vector database ensuring scalable, durable embeddings using the **BAAI/bge-small-en-v1.5** model. |

---

## 🛠️ Setup & Local Installation

### Prerequisites

- **Python 3.11** (important for dependency stability)
- **Git**

---

### Clone the Repository

```bash
git clone [https://github.com/DhashubhanKumar/Financial-RAG-Groq-Qdrant.git](https://github.com/DhashubhanKumar/Financial-RAG-Groq-Qdrant.git)
cd Financial-RAG-Groq-Qdrant

python -m venv venv

# Activate the environment
# Windows
venv\Scripts\activate

# macOS / Linux
source venv/bin/activate

# Install pinned dependencies
pip install -r requirements.txt

GROQ_API_KEY = "your_groq_api_key"
QDRANT_ENDPOINT = "your_qdrant_cloud_endpoint"
QDRANT_API_KEY = "your_qdrant_api_key"


streamlit run app.py
