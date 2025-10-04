# build_index.py
# Simple, robust indexer for PDFs (and TXTs) → FAISS + SentenceTransformers
# Uses: sentence-transformers, faiss-cpu, pypdf, numpy

import os, json, glob
from pathlib import Path
from typing import List, Tuple, Dict
import numpy as np

DATA_DIR  = Path("data_policies")         # your existing PDFs live here
INDEX_DIR = Path("index")                  # index output folder
INDEX_DIR.mkdir(parents=True, exist_ok=True)
DATA_DIR.mkdir(parents=True, exist_ok=True)

# ---- Embeddings
from sentence_transformers import SentenceTransformer
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
DIM = 384  # embedding size for the model above

# ---- Try FAISS (fast). If unavailable, we still save embeddings for brute-force.
try:
    import faiss  # pip install faiss-cpu
    HAVE_FAISS = True
except Exception:
    HAVE_FAISS = False

# ---- PDF loader
from pypdf import PdfReader


def read_pdf(path: Path) -> List[Tuple[str, int]]:
    """Return list of (text, page_number starting at 1) for a PDF."""
    out = []
    try:
        reader = PdfReader(str(path))
        for i, page in enumerate(reader.pages):
            text = page.extract_text() or ""
            text = text.strip()
            if text:
                out.append((text, i + 1))
    except Exception as e:
        print(f"Skipping {path.name}: {e}")
    return out


def read_txt(path: Path) -> List[Tuple[str, int]]:
    """Treat the whole .txt file as one page."""
    try:
        text = path.read_text(encoding="utf-8", errors="ignore").strip()
        return [(text, 1)] if text else []
    except Exception as e:
        print(f"Skipping {path.name}: {e}")
        return []


def chunk_text(text: str, chunk_size=800, overlap=200) -> List[str]:
    """Simple fixed-size chunking with overlap; whitespace-normalized."""
    text = " ".join(text.split())
    if not text:
        return []
    chunks = []
    start = 0
    n = len(text)
    while start < n:
        end = min(start + chunk_size, n)
        chunks.append(text[start:end])
        if end == n:
            break
        start = max(end - overlap, start + 1)
    return chunks


def l2_normalize(X: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(X, axis=1, keepdims=True) + 1e-12
    return X / norms


def build():
    files = list(DATA_DIR.glob("*.pdf")) + list(DATA_DIR.glob("*.txt"))
    if not files:
        print(f"No PDFs/TXTs found in {DATA_DIR}. Add your policy PDFs there.")
        return

    # Collect per-chunk text + metadata
    texts: List[str] = []
    meta:  List[Dict] = []

    for f in files:
        pages = read_pdf(f) if f.suffix.lower() == ".pdf" else read_txt(f)
        for content, page_num in pages:
            for ch in chunk_text(content, chunk_size=800, overlap=200):
                texts.append(ch)
                meta.append({"source": f.name, "page": page_num})

    if not texts:
        print("No extractable text found in your files.")
        return

    print(f"Found {len(texts)} chunks from {len(files)} file(s). Embedding with {MODEL_NAME} ...")
    model = SentenceTransformer(MODEL_NAME)
    embs = model.encode(texts, batch_size=64, show_progress_bar=True)
    embs = embs.astype("float32")
    embs = l2_normalize(embs)  # so inner product ≈ cosine similarity

    # Persist index artifacts
    np.save(INDEX_DIR / "embeddings.npy", embs)
    (INDEX_DIR / "texts.json").write_text(json.dumps(texts, ensure_ascii=False), encoding="utf-8")
    (INDEX_DIR / "meta.json").write_text(json.dumps(meta, ensure_ascii=False), encoding="utf-8")
    (INDEX_DIR / "config.json").write_text(
        json.dumps({"model": MODEL_NAME, "dim": DIM, "normalized": True}, ensure_ascii=False),
        encoding="utf-8"
    )

    if HAVE_FAISS:
        index = faiss.IndexFlatIP(DIM)  # inner product on normalized vectors
        index.add(embs)
        faiss.write_index(index, str(INDEX_DIR / "faiss.index"))
        print(f"FAISS index written with {index.ntotal} vectors.")
    else:
        print("FAISS not available — saved raw embeddings; app will use brute-force cosine.")

    print("Index build complete.")


if __name__ == "__main__":
    build()
