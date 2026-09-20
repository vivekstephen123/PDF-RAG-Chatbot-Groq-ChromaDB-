import os
import re
import json
import time
import math
import uuid
import shutil
import hashlib
import datetime
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import streamlit as st
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
from pypdf import PdfReader
from groq import Groq

try:
    import psutil  # optional but included in requirements for stats
except Exception:
    psutil = None


# =========================
# App Constants / Defaults
# =========================
APP_TITLE = "📚 PDF RAG Chatbot (Groq + ChromaDB)"
DEFAULT_COLLECTION_NAME = "pdf_documents"
DEFAULT_CHROMA_DIR = "./chroma_db"
DEFAULT_FETCH_K = 120  # candidate set size for broad questions; rerank will cut down to <= 12
MIN_DYNAMIC_THRESHOLD = 0.55
DEFAULT_THRESHOLD = 0.65
MAX_CONTEXT_CHUNKS_FOR_ANSWER = 12
LLM_RERANK_MIN = 8
LLM_RERANK_MAX = 12
DEFAULT_GROQ_MODEL = "openai/gpt-oss-20b"
GROQ_API_KEY = "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"
GROQ_MODEL_OPTIONS = [
    "llama-3.1-8b-instant",
    "llama-3.3-70b-versatile",
    "openai/gpt-oss-20b",
    "mixtral-8x7b-32768",
    "gemma2-9b-it",
    "meta-llama/llama-4-scout-17b-16e-instruct",
]


def get_default_api_key() -> str:
    env_key = (os.getenv("GROQ_API_KEY") or "").strip()
    if env_key:
        return env_key
    return GROQ_API_KEY.strip()

# You asked for all-MiniLM-L6-v2, but also noted a local folder named all-MiniLM-L12-v2 next to app.py.
# We will try local folder first if present (offline-friendly), otherwise fall back to model name.
LOCAL_MODEL_DIR_PREFERRED = "./all-MiniLM-L12-v2"
MODEL_NAME_FALLBACK = "all-MiniLM-L6-v2"

# Light stopword list to improve lexical overlap scoring
STOPWORDS = {
    "a", "an", "the", "and", "or", "but", "if", "then", "else", "when", "while",
    "is", "are", "was", "were", "be", "been", "being", "to", "of", "in", "on",
    "for", "with", "as", "by", "at", "from", "into", "about", "over", "under",
    "this", "that", "these", "those", "it", "its", "we", "you", "your", "our",
    "they", "their", "i", "me", "my", "can", "could", "should", "would", "may",
    "might", "will", "just", "not", "no", "yes", "do", "does", "did", "done",
}


# =========================
# Data Structures
# =========================
@dataclass
class ChunkRecord:
    chunk_id: str
    text: str
    embedding: List[float]
    metadata: Dict[str, Any]


@dataclass
class RetrievedChunk:
    chunk_id: str
    text: str
    metadata: Dict[str, Any]
    semantic_similarity: float
    lexical_score: float
    combined_score: float


# =========================
# Utility Functions
# =========================
def now_iso() -> str:
    return datetime.datetime.now().isoformat(timespec="seconds")


def safe_makedirs(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def folder_size_bytes(path: str) -> int:
    total = 0
    if not os.path.exists(path):
        return 0
    for root, _, files in os.walk(path):
        for f in files:
            fp = os.path.join(root, f)
            try:
                total += os.path.getsize(fp)
            except Exception:
                pass
    return total


def human_bytes(num: int) -> str:
    if num <= 0:
        return "0 B"
    units = ["B", "KB", "MB", "GB", "TB"]
    i = int(math.floor(math.log(num, 1024)))
    i = min(i, len(units) - 1)
    val = num / (1024 ** i)
    return f"{val:.2f} {units[i]}"


def hash_file_bytes(file_bytes: bytes) -> str:
    return hashlib.sha256(file_bytes).hexdigest()[:16]


def tokenize(text: str) -> List[str]:
    tokens = re.findall(r"[A-Za-z0-9_]+", text.lower())
    return [t for t in tokens if t and t not in STOPWORDS]


def lexical_overlap_score(query: str, doc: str) -> float:
    """
    Simple, robust lexical score:
    - token overlap ratio
    - + mild boosting for exact matches of longer tokens (technical terms)
    """
    q_tokens = tokenize(query)
    if not q_tokens:
        return 0.0

    d_tokens = set(tokenize(doc))
    overlap = [t for t in q_tokens if t in d_tokens]
    overlap_ratio = len(overlap) / max(1, len(set(q_tokens)))

    # Technical term boosting: longer query tokens that appear verbatim
    long_terms = [t for t in set(q_tokens) if len(t) >= 8]
    boosts = sum(1 for t in long_terms if t in d_tokens)
    boost_score = min(1.0, boosts / max(1, len(long_terms))) if long_terms else 0.0

    # combine (bounded)
    score = 0.75 * overlap_ratio + 0.25 * boost_score
    return float(max(0.0, min(1.0, score)))


def combine_scores(semantic_similarity: float, lexical: float) -> float:
    """
    Hybrid rerank score:
    prioritize semantic relevance but allow lexical rescue for exact technical terms.
    """
    semantic_similarity = max(0.0, min(1.0, semantic_similarity))
    lexical = max(0.0, min(1.0, lexical))
    return float(0.75 * semantic_similarity + 0.25 * lexical)


def chunk_id_for(source_file: str, page_start: int, page_end: int, chunk_index: int) -> str:
    # Use uuid suffix so re-uploads don't collide; we also delete by source_file before re-adding.
    suffix = uuid.uuid4().hex[:10]
    return f"{source_file}::p{page_start}-{page_end}::c{chunk_index}::{suffix}"


def validate_api_key(key: str) -> Tuple[bool, str]:
    if not key or not key.strip():
        return False, "API key is empty."
    k = key.strip()
    # Not strict; just basic sanity checks
    if len(k) < 20:
        return False, "API key looks too short."
    return True, "API key format looks OK."


# =========================
# PDF Extraction
# =========================
def extract_pdf_text_by_page(pdf_bytes: bytes) -> Tuple[List[str], Optional[str]]:
    """
    Returns: (page_texts, error_message)
    Handles encrypted PDFs (tries empty password) and extraction errors gracefully.
    """
    try:
        reader = PdfReader(io_bytes(pdf_bytes))
    except Exception as e:
        return [], f"Failed to read PDF: {e}"

    try:
        if getattr(reader, "is_encrypted", False):
            # Attempt empty password decrypt (common for "protected" but accessible PDFs)
            try:
                ok = reader.decrypt("")  # returns int in many versions; non-zero usually success
                if not ok:
                    return [], "PDF is password-protected and could not be decrypted (empty password failed)."
            except Exception:
                return [], "PDF is password-protected and could not be decrypted."
        page_texts: List[str] = []
        for i, page in enumerate(reader.pages):
            try:
                txt = page.extract_text() or ""
            except Exception:
                txt = ""
            page_texts.append(txt)
        # If all pages empty, still return (caller will handle)
        return page_texts, None
    except Exception as e:
        return [], f"Error extracting text: {e}"


def io_bytes(b: bytes):
    # Small helper to avoid importing io at top (but still clean/explicit)
    import io
    return io.BytesIO(b)


# =========================
# Chunking (Sentence-aware, page-range preserving)
# =========================
def split_into_sentences(text: str) -> List[str]:
    """
    Lightweight sentence splitter that:
    - prefers sentence boundaries
    - avoids producing empty strings
    """
    text = re.sub(r"\s+", " ", text).strip()
    if not text:
        return []
    # Split on ., !, ? followed by space/capital-ish; keep it simple and robust
    parts = re.split(r"(?<=[.!?])\s+", text)
    return [p.strip() for p in parts if p and p.strip()]


def build_sentence_stream(page_texts: List[str]) -> List[Tuple[str, int]]:
    """
    Returns list of (sentence_text, page_number_1based).
    """
    stream: List[Tuple[str, int]] = []
    for idx, page_txt in enumerate(page_texts):
        page_num = idx + 1
        sentences = split_into_sentences(page_txt or "")
        for s in sentences:
            s2 = s.strip()
            if s2:
                stream.append((s2, page_num))
    return stream


def chunk_sentence_stream(
    sentence_stream: List[Tuple[str, int]],
    chunk_size: int,
    chunk_overlap: int,
) -> List[Tuple[str, int, int]]:
    """
    Create chunks from a sentence stream, preserving page ranges.
    Output items: (chunk_text, page_start, page_end)
    Overlap is approximate by character count, implemented as a rolling buffer.
    Ensures chunks are non-empty and not whitespace-only.
    """
    if chunk_size < 200:
        chunk_size = 200
    chunk_overlap = max(0, min(chunk_overlap, chunk_size - 1))

    chunks: List[Tuple[str, int, int]] = []
    cur_sentences: List[Tuple[str, int]] = []
    cur_len = 0

    def flush_chunk(buf: List[Tuple[str, int]]) -> Optional[Tuple[str, int, int]]:
        if not buf:
            return None
        text = " ".join(s for s, _ in buf).strip()
        if not text or not text.strip():
            return None
        pages = [p for _, p in buf]
        return text, min(pages), max(pages)

    # Build chunks
    for sent, page in sentence_stream:
        sent = sent.strip()
        if not sent:
            continue

        # If sentence itself is huge, we still include it as a chunk (trimmed to avoid infinite loops)
        if len(sent) > chunk_size:
            # flush existing
            flushed = flush_chunk(cur_sentences)
            if flushed:
                chunks.append(flushed)
            cur_sentences = []
            cur_len = 0

            # hard-split the long sentence by character
            start = 0
            while start < len(sent):
                end = min(len(sent), start + chunk_size)
                piece = sent[start:end].strip()
                if piece:
                    chunks.append((piece, page, page))
                start = max(end - chunk_overlap, end) if chunk_overlap > 0 else end
            continue

        # Check if adding sentence exceeds chunk_size
        add_len = len(sent) + (1 if cur_sentences else 0)
        if cur_len + add_len <= chunk_size:
            cur_sentences.append((sent, page))
            cur_len += add_len
        else:
            flushed = flush_chunk(cur_sentences)
            if flushed:
                chunks.append(flushed)

            # Start new chunk with overlap
            if chunk_overlap > 0 and cur_sentences:
                # keep tail sentences to approximate overlap chars
                overlap_buf: List[Tuple[str, int]] = []
                overlap_chars = 0
                for s, p in reversed(cur_sentences):
                    sc = len(s) + (1 if overlap_buf else 0)
                    if overlap_chars + sc <= chunk_overlap:
                        overlap_buf.append((s, p))
                        overlap_chars += sc
                    else:
                        break
                overlap_buf.reverse()
                cur_sentences = overlap_buf.copy()
                cur_len = sum(len(s) + 1 for s, _ in cur_sentences) - (1 if cur_sentences else 0)
            else:
                cur_sentences = []
                cur_len = 0

            # add current sentence
            cur_sentences.append((sent, page))
            cur_len += len(sent) + (1 if cur_len > 0 else 0)

    flushed = flush_chunk(cur_sentences)
    if flushed:
        chunks.append(flushed)

    # Safety: drop any empty/whitespace chunks
    cleaned = [(t, ps, pe) for (t, ps, pe) in chunks if t and t.strip()]
    return cleaned


# =========================
# Embeddings + Chroma
# =========================
@st.cache_resource(show_spinner=False)
def load_embedding_model() -> Tuple[SentenceTransformer, str]:
    """
    Loads a local model if present; otherwise pulls by name.
    Returns (model, loaded_from_str).
    """
    if os.path.isdir(LOCAL_MODEL_DIR_PREFERRED):
        model = SentenceTransformer(LOCAL_MODEL_DIR_PREFERRED)
        return model, f"local:{LOCAL_MODEL_DIR_PREFERRED}"
    model = SentenceTransformer(MODEL_NAME_FALLBACK)
    return model, f"hf:{MODEL_NAME_FALLBACK}"


@st.cache_resource(show_spinner=False)
def get_chroma_client(persist_dir: str) -> chromadb.PersistentClient:
    safe_makedirs(persist_dir)
    # Settings: allow reset by deleting directory or collection.
    return chromadb.PersistentClient(
        path=persist_dir,
        settings=Settings(anonymized_telemetry=False),
    )


def get_or_create_collection(client: chromadb.PersistentClient, collection_name: str):
    # Ensure cosine space
    try:
        return client.get_collection(collection_name)
    except Exception:
        return client.create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )


def embed_texts_with_progress(
    model: SentenceTransformer,
    texts: List[str],
    progress_label: str = "Generating embeddings...",
    batch_size: int = 32,
) -> List[List[float]]:
    if not texts:
        return []
    # Streamlit progress
    prog = st.progress(0, text=progress_label)
    embeddings: List[List[float]] = []

    total = len(texts)
    for start in range(0, total, batch_size):
        end = min(total, start + batch_size)
        batch = texts[start:end]
        embs = model.encode(
            batch,
            batch_size=batch_size,
            show_progress_bar=False,
            normalize_embeddings=True,  # important for cosine similarity stability
        )
        if isinstance(embs, np.ndarray):
            embs = embs.tolist()
        embeddings.extend(embs)
        prog.progress(int(100 * end / total), text=f"{progress_label} ({end}/{total})")

    prog.empty()
    return embeddings


def chroma_count(collection) -> int:
    try:
        return int(collection.count())
    except Exception:
        # Some versions may return None on empty; fallback
        try:
            res = collection.get(include=[])
            return len(res.get("ids", []))
        except Exception:
            return 0


def chroma_unique_sources(collection) -> List[str]:
    try:
        res = collection.get(include=["metadatas"])
        metas = res.get("metadatas", []) or []
        sources = sorted({m.get("source_file", "unknown") for m in metas if isinstance(m, dict)})
        return [s for s in sources if s and s != "unknown"]
    except Exception:
        return []


def delete_by_source(collection, source_file: str) -> None:
    # Delete all chunks for a source file
    try:
        collection.delete(where={"source_file": source_file})
    except Exception:
        # Fallback: fetch ids then delete explicitly
        res = collection.get(where={"source_file": source_file}, include=[])
        ids = res.get("ids", []) or []
        if ids:
            collection.delete(ids=ids)


# =========================
# Groq Calls (with retry)
# =========================
def groq_generate_with_retry(
    model_name: str,
    api_key: str,
    prompt: str,
    temperature: float = 0.2,
    max_retries: int = 3,
) -> str:
    client = Groq(api_key=api_key)
    last_err = None

    for attempt in range(max_retries):
        try:
            resp = client.chat.completions.create(
                model=model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature,
                max_tokens=1024,
            )
            text = resp.choices[0].message.content
            if text is None:
                text = ""
            return (text or "").strip()
        except Exception as e:
            last_err = e
            time.sleep(0.7 * (attempt + 1) ** 2)

    raise RuntimeError(f"Groq call failed after {max_retries} retries: {last_err}")


def groq_llm_rerank(
    api_key: str,
    model_name: str,
    question: str,
    candidates: List[RetrievedChunk],
    min_pick: int = LLM_RERANK_MIN,
    max_pick: int = LLM_RERANK_MAX,
) -> List[int]:
    """
    Ask Groq to pick best chunks.
    Returns list of indices into `candidates` (not chunk_ids).
    Safe parse with fallbacks.
    """
    lines = []
    for i, c in enumerate(candidates):
        src = c.metadata.get("source_file", "unknown")
        pn = c.metadata.get("page_number", "")
        snippet = c.text[:450].replace("\n", " ").strip()
        lines.append(f"[{i}] ({src}, p.{pn}) {snippet}")

    prompt = f"""
You are ranking document chunks for answering a user question.

QUESTION:
{question}

CHUNKS:
{chr(10).join(lines)}

TASK:
Select the best chunks to answer the question.
- Pick between {min_pick} and {max_pick} chunks.
- Prefer chunks with exact technical term matches when relevant.
- Return ONLY a JSON array of integers (chunk indices), like: [0, 3, 5]
""".strip()

    out = groq_generate_with_retry(
        model_name=model_name,
        api_key=api_key,
        prompt=prompt,
        temperature=0.1,
        max_retries=3,
    )
    try:
        match = re.search(r"\[[\s\S]*\]", out)
        if not match:
            return []
        arr = json.loads(match.group(0))
        if not isinstance(arr, list):
            return []
        idxs = []
        for x in arr:
            if isinstance(x, int) and 0 <= x < len(candidates):
                idxs.append(x)
        seen = set()
        idxs2 = []
        for i in idxs:
            if i not in seen:
                seen.add(i)
                idxs2.append(i)
        return idxs2
    except Exception:
        return []


# =========================
# Retrieval Pipeline
# =========================
def semantic_retrieve(
    collection,
    query_embedding: List[float],
    fetch_k: int,
) -> Tuple[List[str], List[str], List[Dict[str, Any]], List[float]]:
    """
    Returns: (ids, documents, metadatas, distances)
    Chroma distances for cosine are typically (1 - cosine_similarity) when normalized,
    so similarity = 1 - distance.
    """
    res = collection.query(
        query_embeddings=[query_embedding],
        n_results=fetch_k,
        include=["documents", "metadatas", "distances"],
    )
    ids = (res.get("ids", [[]]) or [[]])[0] or []
    docs = (res.get("documents", [[]]) or [[]])[0] or []
    metas = (res.get("metadatas", [[]]) or [[]])[0] or []
    dists = (res.get("distances", [[]]) or [[]])[0] or []
    return ids, docs, metas, dists


def hybrid_rerank(
    question: str,
    ids: List[str],
    docs: List[str],
    metas: List[Dict[str, Any]],
    dists: List[float],
    threshold: float,
    min_threshold: float,
    desired_k: int,
) -> List[RetrievedChunk]:
    candidates: List[RetrievedChunk] = []
    for cid, doc, meta, dist in zip(ids, docs, metas, dists):
        doc = (doc or "").strip()
        if not doc:
            continue
        dist = float(dist) if dist is not None else 1.0
        semantic_sim = max(0.0, min(1.0, 1.0 - dist))
        lex = lexical_overlap_score(question, doc)
        comb = combine_scores(semantic_sim, lex)
        candidates.append(
            RetrievedChunk(
                chunk_id=cid,
                text=doc,
                metadata=meta or {},
                semantic_similarity=semantic_sim,
                lexical_score=lex,
                combined_score=comb,
            )
        )

    # Sort by combined score
    candidates.sort(key=lambda c: c.combined_score, reverse=True)

    # Dynamic thresholding: if too few pass, lower threshold down to min_threshold
    cur_thresh = threshold
    filtered: List[RetrievedChunk] = [c for c in candidates if c.combined_score >= cur_thresh]

    while len(filtered) < min(desired_k, len(candidates)) and cur_thresh > min_threshold:
        cur_thresh = max(min_threshold, cur_thresh - 0.05)
        filtered = [c for c in candidates if c.combined_score >= cur_thresh]

    # If still too few, take top-N
    if len(filtered) < desired_k:
        filtered = candidates[: max(desired_k, min(LLM_RERANK_MAX, len(candidates)))]

    return filtered


def build_context_block(chunks: List[RetrievedChunk]) -> str:
    """
    Format chunks + source info for the final prompt.
    """
    lines = []
    for i, c in enumerate(chunks, start=1):
        src = c.metadata.get("source_file", "unknown")
        pn = c.metadata.get("page_number", "")
        ps = c.metadata.get("page_start", "")
        pe = c.metadata.get("page_end", "")
        if ps and pe and str(ps) != str(pe):
            page_info = f"pages {ps}-{pe}"
        else:
            page_info = f"page {pn}" if pn else "page ?"
        lines.append(f"[{i}] Source: {src}, {page_info}\n{c.text}")
    return "\n\n".join(lines).strip()


def build_answer_prompt(context_block: str, question: str) -> str:
    return f"""
You are a helpful assistant that answers questions based on the provided document context.

CONTEXT FROM DOCUMENTS:
{context_block}

USER QUESTION: {question}

INSTRUCTIONS:
- Answer based ONLY on the provided context
- If the context doesn't contain enough information, say so
- Cite your sources by mentioning the document name and page number
- Be concise but comprehensive
""".strip()


# =========================
# Streamlit State
# =========================
def init_state() -> None:
    if "config" not in st.session_state:
        st.session_state.config = {
            "api_key": get_default_api_key(),
            "model_name": DEFAULT_GROQ_MODEL,
            "chunk_size": 1000,
            "chunk_overlap": 200,
            "top_k": 5,
            "collection_name": DEFAULT_COLLECTION_NAME,
            "persist_dir": DEFAULT_CHROMA_DIR,
            "use_llm_rerank": True,
            "fetch_k": DEFAULT_FETCH_K,
            "hybrid_threshold": DEFAULT_THRESHOLD,
        }
    if "chat" not in st.session_state:
        st.session_state.chat = []  # list of dicts: {role, content, meta}
    if "upload_report" not in st.session_state:
        st.session_state.upload_report = []
    if "last_stats" not in st.session_state:
        st.session_state.last_stats = {}


# =========================
# UI Helpers
# =========================
def toast_success(msg: str) -> None:
    try:
        st.toast(msg, icon="✅")
    except Exception:
        st.success(msg)


def toast_error(msg: str) -> None:
    try:
        st.toast(msg, icon="❌")
    except Exception:
        st.error(msg)


def render_kb_stats(collection, persist_dir: str) -> None:
    total_chunks = chroma_count(collection)
    sources = chroma_unique_sources(collection)
    chroma_bytes = folder_size_bytes(persist_dir)

    col1, col2, col3 = st.columns(3)
    col1.metric("📦 Total chunks", f"{total_chunks}")
    col2.metric("📄 Documents", f"{len(sources)}")
    col3.metric("💾 Vector DB size", human_bytes(chroma_bytes))

    # Optional process memory stats
    if psutil is not None:
        try:
            p = psutil.Process(os.getpid())
            rss = p.memory_info().rss
            st.caption(f"🧠 App memory (RSS): {human_bytes(int(rss))}")
        except Exception:
            pass


def render_chat_export() -> None:
    if not st.session_state.chat:
        st.info("No chat history to export yet.")
        return

    export_obj = {
        "exported_at": now_iso(),
        "messages": st.session_state.chat,
    }
    json_bytes = json.dumps(export_obj, ensure_ascii=False, indent=2).encode("utf-8")
    st.download_button(
        "⬇️ Export chat history (JSON)",
        data=json_bytes,
        file_name=f"chat_history_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
        mime="application/json",
        use_container_width=True,
    )


# =========================
# Main App
# =========================
def main():
    st.set_page_config(page_title=APP_TITLE, page_icon="📚", layout="wide")
    init_state()

    st.title(APP_TITLE)
    st.caption("Local embeddings (Sentence-Transformers) + Persistent ChromaDB + Groq LLM RAG")

    # Load resources early (fast due to caching)
    with st.spinner("Loading embedding model and vector store..."):
        embed_model, model_loaded_from = load_embedding_model()
        client = get_chroma_client(st.session_state.config["persist_dir"])
        collection = get_or_create_collection(client, st.session_state.config["collection_name"])

    tabs = st.tabs(["⚙️ Configuration", "📄 Document Upload", "💬 Chat Interface"])

    # =========================
    # Tab 1: Configuration
    # =========================
    with tabs[0]:
        st.subheader("⚙️ Configuration")
        st.write("Configure retrieval and chunking parameters. Changes apply immediately.")

        left, right = st.columns([1.2, 1.0])

        with left:
            st.session_state.config["api_key"] = get_default_api_key()

            model_name = st.selectbox(
                "Groq model to use",
                options=GROQ_MODEL_OPTIONS,
                index=GROQ_MODEL_OPTIONS.index(st.session_state.config.get("model_name", DEFAULT_GROQ_MODEL)),
                help="Choose the Groq model for both the rerank step and final answer generation.",
            )
            st.session_state.config["model_name"] = model_name

            collection_name = st.text_input(
                "ChromaDB Collection Name",
                value=st.session_state.config["collection_name"],
                help=f"Default: {DEFAULT_COLLECTION_NAME}. Changing this uses a different collection.",
            ).strip() or DEFAULT_COLLECTION_NAME

            persist_dir = st.text_input(
                "ChromaDB Persist Directory",
                value=st.session_state.config["persist_dir"],
                help="Local directory where vectors are stored persistently (survives restarts).",
            ).strip() or DEFAULT_CHROMA_DIR

            # If user changed these, rebuild client/collection safely by clearing caches in-session
            changed = (collection_name != st.session_state.config["collection_name"]) or (persist_dir != st.session_state.config["persist_dir"])
            st.session_state.config["collection_name"] = collection_name
            st.session_state.config["persist_dir"] = persist_dir

            if changed:
                # Recreate resources (not via cache invalidation; just re-call with new args)
                client = get_chroma_client(persist_dir)
                collection = get_or_create_collection(client, collection_name)
                toast_success("Switched to the new ChromaDB collection/persist directory.")

            st.divider()

            chunk_size = st.slider(
                "Chunk size (characters)",
                min_value=200,
                max_value=2000,
                value=int(st.session_state.config["chunk_size"]),
                step=50,
                help="Larger chunks preserve context; smaller chunks improve retrieval granularity.",
            )
            chunk_overlap = st.slider(
                "Chunk overlap (characters)",
                min_value=0,
                max_value=500,
                value=int(st.session_state.config["chunk_overlap"]),
                step=25,
                help="Overlap helps avoid losing context at chunk boundaries.",
            )
            top_k = st.slider(
                "Number of relevant chunks to retrieve (top-k)",
                min_value=1,
                max_value=10,
                value=int(st.session_state.config["top_k"]),
                step=1,
                help="Final chunks used for answering after reranking (capped to 12 in prompt).",
            )

            st.session_state.config["chunk_size"] = int(chunk_size)
            st.session_state.config["chunk_overlap"] = int(chunk_overlap)
            st.session_state.config["top_k"] = int(top_k)

        with right:
            st.markdown("#### 🧩 Retrieval & Reranking")
            use_llm_rerank = st.checkbox(
                "Use optional LLM rerank (Groq) to select best 8–12 chunks",
                value=bool(st.session_state.config["use_llm_rerank"]),
                help="Recommended for broad questions; costs an extra Groq call per user question.",
            )
            st.session_state.config["use_llm_rerank"] = bool(use_llm_rerank)

            fetch_k = st.slider(
                "Candidate set size for semantic retrieval (fetch_k)",
                min_value=80,
                max_value=150,
                value=int(st.session_state.config["fetch_k"]),
                step=5,
                help="Larger fetch_k improves recall for broad queries; reranking reduces to <= 12 chunks.",
            )
            st.session_state.config["fetch_k"] = int(fetch_k)

            threshold = st.slider(
                "Hybrid score threshold (dynamic lowering to 0.55 if needed)",
                min_value=0.55,
                max_value=0.90,
                value=float(st.session_state.config["hybrid_threshold"]),
                step=0.01,
                help="Higher = stricter filtering; app will lower gradually if too few chunks pass.",
            )
            st.session_state.config["hybrid_threshold"] = float(threshold)

            st.divider()
            st.markdown("#### 🗃️ Knowledge Base")
            render_kb_stats(collection, st.session_state.config["persist_dir"])

            # Clear/reset vector database
            c1, c2 = st.columns(2)
            with c1:
                if st.button("🧹 Clear / Reset Vector Database", use_container_width=True):
                    try:
                        # delete collection then recreate
                        try:
                            client.delete_collection(st.session_state.config["collection_name"])
                        except Exception:
                            pass
                        # Also safe to remove persistent dir to truly reset
                        # (only if it exists and user is using default path or a chosen path)
                        if os.path.isdir(st.session_state.config["persist_dir"]):
                            shutil.rmtree(st.session_state.config["persist_dir"], ignore_errors=True)
                        safe_makedirs(st.session_state.config["persist_dir"])
                        client2 = get_chroma_client(st.session_state.config["persist_dir"])
                        collection2 = get_or_create_collection(client2, st.session_state.config["collection_name"])
                        # swap local refs
                        collection = collection2
                        toast_success("Vector database reset successfully.")
                    except Exception as e:
                        toast_error(f"Failed to reset vector database: {e}")

            with c2:
                sources = chroma_unique_sources(collection)
                if sources:
                    to_remove = st.multiselect(
                        "Remove specific documents",
                        options=sources,
                        help="Deletes all chunks belonging to selected source files from ChromaDB.",
                    )
                    if st.button("🗑️ Remove Selected Documents", use_container_width=True, disabled=not bool(to_remove)):
                        removed = 0
                        for s in to_remove:
                            try:
                                delete_by_source(collection, s)
                                removed += 1
                            except Exception:
                                pass
                        toast_success(f"Removed {removed} document(s) from the knowledge base.")
                else:
                    st.info("No documents stored yet. Upload PDFs in the next tab.")

            st.divider()
            st.markdown("#### ✅ Configuration Status")
            ok, msg = validate_api_key(st.session_state.config["api_key"])
            st.write(f"- **API key**: {'✅ Set' if ok else '❌ Not ready'} ({msg})")
            st.write(f"- **Groq model**: `{st.session_state.config['model_name']}`")
            st.write(f"- **Embedding model loaded from**: `{model_loaded_from}`")
            st.write(f"- **Collection**: `{st.session_state.config['collection_name']}`")
            st.write(f"- **Persist dir**: `{st.session_state.config['persist_dir']}`")

    # =========================
    # Tab 2: Document Upload
    # =========================
    with tabs[1]:
        st.subheader("📄 Document Upload")
        st.write("Upload one or more PDFs. Text is extracted per-page, chunked with page ranges, embedded locally, and stored persistently in ChromaDB.")

        files = st.file_uploader(
            "Upload PDF files",
            type=["pdf"],
            accept_multiple_files=True,
            help="You can upload multiple PDFs at once.",
        )

        if st.button("🚀 Process Uploaded PDFs", use_container_width=True, disabled=not bool(files)):
            if not files:
                toast_error("Please upload at least one PDF.")
            else:
                st.session_state.upload_report = []
                total_new_chunks = 0

                for f in files:
                    fname = f.name
                    file_bytes = f.read()
                    file_sig = hash_file_bytes(file_bytes)

                    status_box = st.container()
                    with status_box:
                        st.markdown(f"**Processing:** `{fname}`")
                        pbar = st.progress(0, text="Reading PDF...")

                    # Extract pages
                    page_texts, err = extract_pdf_text_by_page(file_bytes)
                    if err:
                        pbar.empty()
                        toast_error(f"{fname}: {err}")
                        st.session_state.upload_report.append({
                            "source_file": fname,
                            "file_hash": file_sig,
                            "pages": 0,
                            "chunks": 0,
                            "status": "failure",
                            "error": err,
                        })
                        continue

                    page_count = len(page_texts)
                    pbar.progress(15, text=f"Extracted {page_count} page(s). Building sentences...")

                    sentence_stream = build_sentence_stream(page_texts)
                    if not sentence_stream:
                        pbar.empty()
                        toast_error(f"{fname}: No extractable text found (scanned or empty PDF).")
                        st.session_state.upload_report.append({
                            "source_file": fname,
                            "file_hash": file_sig,
                            "pages": page_count,
                            "chunks": 0,
                            "status": "failure",
                            "error": "No extractable text found.",
                        })
                        continue

                    pbar.progress(25, text="Chunking text...")
                    chunks = chunk_sentence_stream(
                        sentence_stream=sentence_stream,
                        chunk_size=int(st.session_state.config["chunk_size"]),
                        chunk_overlap=int(st.session_state.config["chunk_overlap"]),
                    )

                    # Remove empties just in case
                    chunks = [(t, ps, pe) for (t, ps, pe) in chunks if t and t.strip()]
                    if not chunks:
                        pbar.empty()
                        toast_error(f"{fname}: Chunking produced no valid chunks.")
                        st.session_state.upload_report.append({
                            "source_file": fname,
                            "file_hash": file_sig,
                            "pages": page_count,
                            "chunks": 0,
                            "status": "failure",
                            "error": "Chunking produced no valid chunks.",
                        })
                        continue

                    # Delete prior chunks for same filename to avoid duplicates on re-upload
                    try:
                        delete_by_source(collection, fname)
                    except Exception:
                        pass

                    pbar.progress(35, text=f"Generating embeddings for {len(chunks)} chunks...")

                    texts = [t for (t, _, _) in chunks]
                    with st.spinner("Embedding locally with Sentence-Transformers..."):
                        embeddings = embed_texts_with_progress(
                            model=embed_model,
                            texts=texts,
                            progress_label=f"Embedding {fname}",
                            batch_size=32,
                        )

                    # Build records + add to Chroma
                    upload_ts = now_iso()
                    records: List[ChunkRecord] = []
                    for idx, ((t, ps, pe), emb) in enumerate(zip(chunks, embeddings)):
                        page_number = str(ps) if ps == pe else f"{ps}-{pe}"
                        rid = chunk_id_for(fname, ps, pe, idx)
                        md = {
                            "source_file": fname,
                            "page_number": page_number,
                            "page_start": int(ps),
                            "page_end": int(pe),
                            "chunk_index": int(idx),
                            "upload_timestamp": upload_ts,
                            "file_hash": file_sig,
                        }
                        records.append(ChunkRecord(chunk_id=rid, text=t, embedding=emb, metadata=md))

                    try:
                        pbar.progress(90, text="Storing vectors in ChromaDB...")
                        collection.add(
                            ids=[r.chunk_id for r in records],
                            documents=[r.text for r in records],
                            embeddings=[r.embedding for r in records],
                            metadatas=[r.metadata for r in records],
                        )
                        pbar.progress(100, text="Done.")
                        pbar.empty()
                        toast_success(f"{fname}: processed successfully ({page_count} pages, {len(records)} chunks).")
                        total_new_chunks += len(records)
                        st.session_state.upload_report.append({
                            "source_file": fname,
                            "file_hash": file_sig,
                            "pages": page_count,
                            "chunks": len(records),
                            "status": "success",
                            "error": "",
                        })
                    except Exception as e:
                        pbar.empty()
                        toast_error(f"{fname}: Failed to store in ChromaDB: {e}")
                        st.session_state.upload_report.append({
                            "source_file": fname,
                            "file_hash": file_sig,
                            "pages": page_count,
                            "chunks": 0,
                            "status": "failure",
                            "error": f"ChromaDB add failed: {e}",
                        })

                if total_new_chunks > 0:
                    toast_success(f"Added {total_new_chunks} chunk(s) to the knowledge base.")

        st.divider()

        # Report
        if st.session_state.upload_report:
            st.markdown("### 📋 Upload Report")
            for r in st.session_state.upload_report:
                icon = "✅" if r["status"] == "success" else "❌"
                st.write(
                    f"{icon} **{r['source_file']}** — pages: {r['pages']}, chunks: {r['chunks']} "
                    + (f"— error: {r['error']}" if r["status"] != "success" else "")
                )

            # Summary
            total_docs = len([r for r in st.session_state.upload_report if r["status"] == "success"])
            total_chunks = sum(r["chunks"] for r in st.session_state.upload_report if r["status"] == "success")
            st.info(f"Processed **{total_docs}** document(s) successfully. Total new chunks: **{total_chunks}**.")
        else:
            st.info("Upload PDFs and click **Process Uploaded PDFs** to build the knowledge base.")

        st.divider()
        st.markdown("### 🗃️ Knowledge Base Overview")
        render_kb_stats(collection, st.session_state.config["persist_dir"])

        # Show all docs in KB
        sources = chroma_unique_sources(collection)
        if sources:
            with st.expander("📚 View all documents in the knowledge base", expanded=False):
                for s in sources:
                    st.write(f"- {s}")
        else:
            st.caption("No documents in the knowledge base yet.")

    # =========================
    # Tab 3: Chat
    # =========================
    with tabs[2]:
        st.subheader("💬 Chat Interface")
        render_kb_stats(collection, st.session_state.config["persist_dir"])

        # Gate checks
        api_ok, api_msg = validate_api_key(st.session_state.config["api_key"])
        total_chunks = chroma_count(collection)

        if not api_ok:
            st.warning(f"Set a valid API key in **⚙️ Configuration** to chat. ({api_msg})")
        if total_chunks <= 0:
            st.warning("Upload and process at least one PDF in **📄 Document Upload** before chatting.")

        # Chat history
        st.markdown("#### 🧾 Conversation")
        for msg in st.session_state.chat:
            role = msg.get("role", "assistant")
            content = msg.get("content", "")
            meta = msg.get("meta", {})

            with st.chat_message(role):
                st.write(content)

                if role == "assistant" and meta:
                    citations = meta.get("citations", [])
                    retrieved = meta.get("retrieved_chunks", [])

                    if citations:
                        st.markdown("**Sources:**")
                        for c in citations:
                            st.write(f"- {c}")

                    if retrieved:
                        with st.expander("🔎 Retrieved chunks (with scores)", expanded=False):
                            for i, rc in enumerate(retrieved, start=1):
                                st.markdown(
                                    f"**Chunk {i}** — "
                                    f"semantic: `{rc['semantic_similarity']:.3f}`, "
                                    f"lexical: `{rc['lexical_score']:.3f}`, "
                                    f"combined: `{rc['combined_score']:.3f}`"
                                )
                                st.caption(
                                    f"Source: {rc['source_file']} | page(s): {rc['page_number']} | chunk_index: {rc['chunk_index']}"
                                )
                                st.write(rc["text"])
                                st.divider()

        st.divider()

        # Export + Clear chat
        c1, c2 = st.columns([1, 1])
        with c1:
            render_chat_export()
        with c2:
            if st.button("🧽 Clear chat history", use_container_width=True):
                st.session_state.chat = []
                toast_success("Chat history cleared.")

        st.divider()

        # Chat input
        user_q = st.chat_input("Ask a question about your uploaded PDFs...")
        if user_q is not None:
            question = (user_q or "").strip()
            if not question:
                toast_error("Please enter a non-empty question.")
                st.stop()

            if not api_ok:
                toast_error(f"API key not ready: {api_msg}")
                st.stop()

            if total_chunks <= 0:
                toast_error("No documents found. Please upload PDFs first.")
                st.stop()

            # Add user message
            st.session_state.chat.append({"role": "user", "content": question, "meta": {}})

            with st.chat_message("assistant"):
                with st.spinner("Retrieving relevant context..."):
                    # Embed query
                    q_emb = embed_model.encode([question], normalize_embeddings=True, show_progress_bar=False)
                    if isinstance(q_emb, np.ndarray):
                        q_emb = q_emb.tolist()
                    query_embedding = q_emb[0]

                    # Semantic retrieval (broad candidates)
                    ids, docs, metas, dists = semantic_retrieve(
                        collection=collection,
                        query_embedding=query_embedding,
                        fetch_k=int(st.session_state.config["fetch_k"]),
                    )

                    # Hybrid rerank + dynamic thresholding
                    desired_k = int(st.session_state.config["top_k"])
                    threshold = float(st.session_state.config["hybrid_threshold"])

                    hybrid_candidates = hybrid_rerank(
                        question=question,
                        ids=ids,
                        docs=docs,
                        metas=metas,
                        dists=dists,
                        threshold=threshold,
                        min_threshold=MIN_DYNAMIC_THRESHOLD,
                        desired_k=max(desired_k, LLM_RERANK_MAX),  # prepare enough for optional LLM rerank
                    )

                    # Candidate pool for LLM rerank (top 20 is usually plenty)
                    candidate_pool = hybrid_candidates[: min(20, len(hybrid_candidates))]

                    final_chunks: List[RetrievedChunk] = []

                    # Optional LLM rerank to pick best 8–12
                    if st.session_state.config["use_llm_rerank"] and candidate_pool:
                        with st.spinner("Optional LLM rerank (Groq) selecting best chunks..."):
                            idxs = groq_llm_rerank(
                                api_key=st.session_state.config["api_key"],
                                model_name=st.session_state.config.get("model_name", DEFAULT_GROQ_MODEL),
                                question=question,
                                candidates=candidate_pool,
                                min_pick=LLM_RERANK_MIN,
                                max_pick=LLM_RERANK_MAX,
                            )
                            if idxs:
                                final_chunks = [candidate_pool[i] for i in idxs]
                            else:
                                # fallback: just take top max( desired_k, 8 ) but cap to 12
                                final_chunks = candidate_pool[: max(desired_k, min(LLM_RERANK_MIN, len(candidate_pool)))]

                    else:
                        final_chunks = candidate_pool[: max(desired_k, min(LLM_RERANK_MIN, len(candidate_pool)))]

                    # Always cap to <= 12 in final answer prompt
                    final_chunks = final_chunks[:MAX_CONTEXT_CHUNKS_FOR_ANSWER]

                    if not final_chunks:
                        answer_text = (
                            "I couldn't find any relevant context in the uploaded documents to answer that. "
                            "Try rephrasing your question or uploading a document that contains the needed information."
                        )
                        st.write(answer_text)
                        st.session_state.chat.append(
                            {"role": "assistant", "content": answer_text, "meta": {"citations": [], "retrieved_chunks": []}}
                        )
                        st.stop()

                    # Build prompt
                    context_block = build_context_block(final_chunks)
                    prompt = build_answer_prompt(context_block=context_block, question=question)

                # Call Groq for final answer
                with st.spinner("Generating answer with Groq..."):
                    try:
                        answer = groq_generate_with_retry(
                            model_name=st.session_state.config.get("model_name", DEFAULT_GROQ_MODEL),
                            api_key=st.session_state.config["api_key"],
                            prompt=prompt,
                            temperature=0.2,
                            max_retries=3,
                        )
                    except Exception as e:
                        toast_error(f"LLM call failed: {e}")
                        answer = (
                            "I ran into an error while generating the answer. "
                            "Please try again (or check your API key / network)."
                        )

                # Render answer
                st.write(answer)

                # Build citations display
                citations = []
                for c in final_chunks:
                    src = c.metadata.get("source_file", "unknown")
                    pn = c.metadata.get("page_number", "")
                    citations.append(f"{src} (page(s) {pn})")

                # Retrieved chunk display payload (scores)
                retrieved_payload = []
                for c in final_chunks:
                    retrieved_payload.append({
                        "chunk_id": c.chunk_id,
                        "text": c.text,
                        "source_file": c.metadata.get("source_file", "unknown"),
                        "page_number": c.metadata.get("page_number", ""),
                        "chunk_index": c.metadata.get("chunk_index", ""),
                        "semantic_similarity": float(c.semantic_similarity),
                        "lexical_score": float(c.lexical_score),
                        "combined_score": float(c.combined_score),
                    })

                st.markdown("**Sources:**")
                for c in citations:
                    st.write(f"- {c}")

                with st.expander("🔎 Retrieved chunks (with relevance scores)", expanded=False):
                    for i, rc in enumerate(retrieved_payload, start=1):
                        st.markdown(
                            f"**Chunk {i}** — "
                            f"semantic: `{rc['semantic_similarity']:.3f}`, "
                            f"lexical: `{rc['lexical_score']:.3f}`, "
                            f"combined: `{rc['combined_score']:.3f}`"
                        )
                        st.caption(
                            f"Source: {rc['source_file']} | page(s): {rc['page_number']} | chunk_index: {rc['chunk_index']}"
                        )
                        st.write(rc["text"])
                        st.divider()

            # Save assistant message (with meta)
            st.session_state.chat.append(
                {
                    "role": "assistant",
                    "content": answer,
                    "meta": {
                        "citations": citations,
                        "retrieved_chunks": retrieved_payload,
                    },
                }
            )


if __name__ == "__main__":
    main()
