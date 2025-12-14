# 📊 Financial RAG Analyst (Groq + Qdrant)

This project is a high-performance, full-stack RAG application engineered to analyze complex regulatory filings (like 10-K reports). It delivers verifiable, sub-second query responses by integrating advanced multi-modal processing and cloud-hardened infrastructure.

## 🚀 Key Features and Architecture

The system is built to excel where basic RAG fails, focusing on speed, accuracy, and handling complex data types. 

| Feature | Libraries Used | Technical Achievement |
| :--- | :--- | :--- |
| **Multi-Modal Ingestion** | `Camelot`, `OpenCV` | Extracts both **structured tables** and **unstructured text** from PDFs for a unified knowledge graph. |
| **High-Speed Inference** | `Groq` | Utilizes **Llama-3.1-8B-Instant** for industry-leading, sub-second query latency. |
| **Advanced Retrieval** | `LlamaIndex`, `SentenceTransformer` | Implements a **Cross-Encoder Re-ranker** (`ms-marco-MiniLM-L-6-v2`) to ensure the LLM receives only the top-3 most relevant context chunks, maximizing precision. |
| **Trust Layer & Guardrails** | Custom Prompt | Enforces a strict system prompt mandating **source citation** (``) and a **Hallucination Guardrail** for auditable results. |
| **Persistent Vector Store** | `Qdrant Cloud` | Scalable cloud backend for embeddings, ensuring data stability and retrieval performance. |

## 🛠️ Setup and Local Installation

### 1. Prerequisites

* **Python 3.11** (Critical for dependency stability)
* **Git**

### 2. Clone the Repository & Install Dependencies

```bash
git clone [https://github.com/DhashubhanKumar/Financial-RAG-Groq-Qdrant.git](https://github.com/DhashubhanKumar/Financial-RAG-Groq-Qdrant.git)
cd Financial-RAG-Groq-Qdrant
python -m venv venv
# Activate the environment
# Install pinned dependencies
pip install -r requirements.txt
(fill ur api keys
Here is the single, ready-to-copy-paste README.md file for your project.Markdown# 📊 Financial RAG Analyst (Groq + Qdrant)

This project is a high-performance, full-stack RAG application engineered to analyze complex regulatory filings (like 10-K reports). It delivers verifiable, sub-second query responses by integrating advanced multi-modal processing and cloud-hardened infrastructure.

## 🚀 Key Features and Architecture

The system is built to excel where basic RAG fails, focusing on speed, accuracy, and handling complex data types. 

| Feature | Libraries Used | Technical Achievement |
| :--- | :--- | :--- |
| **Multi-Modal Ingestion** | `Camelot`, `OpenCV` | Extracts both **structured tables** and **unstructured text** from PDFs for a unified knowledge graph. |
| **High-Speed Inference** | `Groq` | Utilizes **Llama-3.1-8B-Instant** for industry-leading, sub-second query latency. |
| **Advanced Retrieval** | `LlamaIndex`, `SentenceTransformer` | Implements a **Cross-Encoder Re-ranker** (`ms-marco-MiniLM-L-6-v2`) to ensure the LLM receives only the top-3 most relevant context chunks, maximizing precision. |
| **Trust Layer & Guardrails** | Custom Prompt | Enforces a strict system prompt mandating **source citation** (``) and a **Hallucination Guardrail** for auditable results. |
| **Persistent Vector Store** | `Qdrant Cloud` | Scalable cloud backend for embeddings, ensuring data stability and retrieval performance. |

## 🛠️ Setup and Local Installation

### 1. Prerequisites

* **Python 3.11** (Critical for dependency stability)
* **Git**

### 2. Clone the Repository & Install Dependencies

```bash
git clone [https://github.com/DhashubhanKumar/Financial-RAG-Groq-Qdrant.git](https://github.com/DhashubhanKumar/Financial-RAG-Groq-Qdrant.git)
cd Financial-RAG-Groq-Qdrant
python -m venv venv
# Activate the environment
# Install pinned dependencies
pip install -r requirements.txt Configure Environment Variables (Secrets)For local testing, set these variables in your terminal. For Streamlit Cloud deployment, use the .streamlit/secrets.toml file.VariablePurposeGROQ_API_KEYAPI key for high-speed LLM inference.QDRANT_ENDPOINTYour Qdrant Cloud cluster address.QDRANT_API_KEYAPI key for secure access to the vector store
streamlit run app.py
