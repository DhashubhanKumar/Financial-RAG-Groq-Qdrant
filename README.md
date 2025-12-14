# 📊 Financial RAG Analyst (Groq + Qdrant)

A high-performance, full-stack **Retrieval-Augmented Generation (RAG)** application designed to analyze complex financial and regulatory documents such as **10-K filings**.  
The system delivers **verifiable, sub-second responses** by combining high-speed inference, advanced retrieval pipelines, and cloud-native vector storage.

---

## 🚀 Key Features & Architecture

This system is purpose-built to overcome common RAG failures related to latency, hallucination, and poor retrieval quality.

| Feature | Libraries Used | Technical Achievement |
|------|---------------|----------------------|
| **Multi-Modal Ingestion** | `Camelot`, `OpenCV` | Extracts both **structured tables** and **unstructured text** from PDFs into a unified knowledge base |
| **High-Speed Inference** | `Groq` | Uses **Llama-3.1-8B-Instant** for industry-leading sub-second inference |
| **Advanced Retrieval** | `LlamaIndex`, `SentenceTransformers` | Implements a **Cross-Encoder Re-ranker** (`ms-marco-MiniLM-L-6-v2`) to select the top-3 most relevant chunks |
| **Trust Layer & Guardrails** | Custom System Prompt | Enforces **mandatory source citation** and includes a **hallucination guardrail** |
| **Persistent Vector Store** | `Qdrant Cloud` | Cloud-hosted vector database ensuring scalable, durable embeddings |

---

## 🛠️ Setup & Local Installation

### 1️⃣ Prerequisites

- **Python 3.11** (important for dependency stability)
- **Git**

---

### 2️⃣ Clone the Repository

```bash
git clone https://github.com/DhashubhanKumar/Financial-RAG-Groq-Qdrant.git
cd Financial-RAG-Groq-Qdrant


python -m venv venv

# Activate the environment
# Windows
venv\Scripts\activate

# macOS / Linux
source venv/bin/activate

# Install pinned dependencies
pip install -r requirements.txt
export GROQ_API_KEY="your_groq_api_key"
export QDRANT_ENDPOINT="your_qdrant_cloud_endpoint"
export QDRANT_API_KEY="your_qdrant_api_key"


.streamlit/secrets.toml


GROQ_API_KEY = "your_groq_api_key"
QDRANT_ENDPOINT = "your_qdrant_cloud_endpoint"
QDRANT_API_KEY = "your_qdrant_api_key"


streamlit run app.py
