# 📚 PDF RAG Chatbot (Groq + ChromaDB)

A production-grade Retrieval-Augmented Generation (RAG) Streamlit application for interactive Q&A over PDF documents. Powered by local **SentenceTransformers** embeddings, persistent **ChromaDB** vector storage, hybrid retrieval with lexical term-boosting, optional **Groq LLM reranking**, and fast context-grounded response generation.

---

## 🌟 Key Features

* **📄 Page-Aware PDF Ingestion:** Extracts text page-by-page using `pypdf` and creates sentence-boundary-aware chunks that preserve exact document page ranges.


* **⚡ Local & Offline Embeddings:** Generates normalized embeddings locally via `SentenceTransformers` (`all-MiniLM-L6-v2` / `all-MiniLM-L12-v2`) without sending document text to external embedding APIs.


* **💾 Persistent Vector Storage:** Uses `ChromaDB` to persist vector indices locally across application restarts, featuring single-document deletion and collection reset controls.


* **🔀 Hybrid Retrieval & Dynamic Reranking:** Combines dense cosine similarity with sparse lexical token overlap and technical-term boosting. Dynamically lowers thresholds if context pool yields low results.


* **🧠 LLM Context Reranking:** Optional Groq-powered LLM pass to evaluate candidate chunks and select the top 8–12 most relevant context passages before final answer synthesis.


* **💬 Transparent Chat Interface:** Provides complete source citations (file names and page numbers), exact chunk score breakdowns (semantic, lexical, combined), and JSON chat export capabilities.



---

## 🏗️ System Architecture

```text
[ Raw PDF Upload ] ──► [ Page-Level Text Extraction ]
                                │
                                ▼
                   [ Sentence-Aware Chunking ]
                                │
                                ▼
               [ SentenceTransformers Embeddings ]
                                │
                                ▼
               [ Persistent Local ChromaDB Store ]
                                │
                                ▼ (User Query)
             [ Hybrid Search (Semantic + Lexical) ]
                                │
                                ▼
                 [ Optional Groq LLM Reranker ]
                                │
                                ▼
            [ Augmented Prompt + Groq Generation ] ──► [ Cited Answer ]

```

---

## 📂 Repository Structure

```text
.
├── app.py                 # Main Streamlit RAG application
├── requirements.txt       # Python dependency definitions
├── venv/                  # Python virtual environment directory
├── .gitignore             # Excludes database artifacts, cache folders, and venv
└── README.md              # Project documentation

```

---

## 🚀 Getting Started

### Prerequisites

* Python **3.10+**
* A **Groq API Key** (Get one at [console.groq.com](https://console.groq.com/?utm_source=gemini))



### 1. Clone the Repository

```bash
git clone https://github.com/YOUR_USERNAME/YOUR_REPO_NAME.git
cd YOUR_REPO_NAME

```

### 2. Set Up a Virtual Environment

```bash
# On macOS / Linux
python3 -m venv venv
source venv/bin/activate

# On Windows (Command Prompt / PowerShell)
python -m venv venv
venv\Scripts\activate

```

### 3. Install Dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt

```

### 4. Set Up Environment Variables (Optional)

You can pass your Groq API key directly via an environment variable or input it in the UI under **⚙️ Configuration**:

```bash
export GROQ_API_KEY="your_groq_api_key_here"

```

### 5. Run the Streamlit App

```bash
streamlit run app.py

```

The application will open automatically in your browser at `http://localhost:8501`.

---

## ⚙️ Configuration Options

Adjustable directly inside the **⚙️ Configuration** tab:

| Parameter | Default | Description |
| --- | --- | --- |
| **Groq Model** | `openai/gpt-oss-20b`<br> | LLM used for reranking and final synthesis. Options include `llama-3.1-8b-instant`, `llama-3.3-70b-versatile`, `mixtral-8x7b-32768`, `gemma2-9b-it`, and `llama-4-scout-17b-16e-instruct`.

 |
| **Chunk Size** | `1000` chars

 | Target size for text chunks.

 |
| **Chunk Overlap** | `200` chars

 | Character overlap between consecutive chunks.

 |
| **Fetch K** | `120`<br> | Initial candidate pool size retrieved from ChromaDB during semantic search.

 |
| **Hybrid Threshold** | `0.65`<br> | Score threshold for hybrid ranking (dynamically drops to `0.55` if context is sparse).

 |
| **LLM Rerank** | `Enabled`<br> | Uses Groq to rank and filter retrieved chunks down to the best 8–12 context blocks.

 |

---

## 🛡️ Data Privacy & Git Security

The local vector database (`chroma_db/`), virtual environment (`venv/`), and compiled bytecode (`__pycache__/`) are intentionally excluded from Git control to keep the repository lightweight and prevent private document embeddings from leaking.

Ensure your `.gitignore` contains:

```gitignore
chroma_db/
__pycache__/
venv/
.env

```
