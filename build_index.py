"""Index builder for HR Policy Copilot.

This module creates vector embeddings for HR policy documents (PDFs and TXTs)
using SentenceTransformers, and builds a FAISS index for efficient similarity search.
The resulting index enables fast retrieval of relevant policy passages during query processing.

Dependencies: sentence-transformers, faiss-cpu, pypdf, numpy
"""

import os, json, glob, argparse
from pathlib import Path
from typing import List, Tuple, Dict
import numpy as np

# Global constants
DATA_DIR = Path("data_policies")  # Directory containing policy PDFs and TXT files
INDEX_DIR = Path("index")  # Directory where index artifacts will be stored
INDEX_DIR.mkdir(parents=True, exist_ok=True)
DATA_DIR.mkdir(parents=True, exist_ok=True)

# Embedding configuration
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"  # Pre-trained sentence transformer model
DIM = 384  # Embedding dimension for the model

# FAISS availability check for accelerated similarity search
try:
    import faiss  # pip install faiss-cpu
    HAVE_FAISS = True
except Exception:
    HAVE_FAISS = False

# PDF processing
from pypdf import PdfReader


def read_pdf(path: Path) -> List[Tuple[str, int]]:
    """Extract text content from a PDF file.

    Args:
        path: Path to the PDF file to read.

    Returns:
        List of (text_content, page_number) tuples. Page numbers start at 1.
        Returns empty list if PDF cannot be read or has no extractable text.

    Note:
        Uses pypdf's PdfReader to extract text from each page. Pages with no
        extractable text are skipped.
    """
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
    """Read the entire content of a TXT file as a single page.

    Args:
        path: Path to the TXT file to read.

    Returns:
        List containing a single (text_content, 1) tuple if file can be read,
        or empty list if file cannot be read or is empty.

    Note:
        Unlike PDFs, TXT files are treated as single-page documents.
        Errors during reading are ignored except for warnings.
    """
    try:
        text = path.read_text(encoding="utf-8", errors="ignore").strip()
        return [(text, 1)] if text else []
    except Exception as e:
        print(f"Skipping {path.name}: {e}")
        return []


def chunk_text(text: str, chunk_size=600, overlap=120) -> List[str]:
    """Split text into overlapping chunks of specified size.

    Args:
        text: Input text to be chunked.
        chunk_size: Maximum number of characters per chunk.
        overlap: Number of characters to overlap between consecutive chunks.

    Returns:
        List of text chunks. Returns empty list if input text is empty.

    Note:
        Text is first normalized by collapsing whitespace. Chunks are created
        with overlap to preserve context across chunk boundaries.
    """
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
    """Normalize vectors using L2 normalization.

    Args:
        X: Input array of shape (n_samples, n_features).

    Returns:
        L2-normalized array where each row has unit norm.

    Note:
        Adds a small epsilon (1e-12) to norms to avoid division by zero.
        Normalization enables cosine similarity to be computed as inner product.
    """
    norms = np.linalg.norm(X, axis=1, keepdims=True) + 1e-12
    return X / norms


def build(chunk_size=800, overlap=200):
    """Build the FAISS index from policy documents.

    Args:
        chunk_size: Number of characters per text chunk.
        overlap: Number of characters to overlap between chunks.

    Process:
    1. Discover PDF and TXT files in DATA_DIR
    2. Extract text content from each file and split into chunks
    3. Generate vector embeddings for all text chunks
    4. Save embeddings and metadata to disk
    5. Build and save FAISS index (if available)

    Note:
        Embeddings are L2-normalized, allowing cosine similarity via inner product.
        If FAISS is unavailable, falls back to brute-force similarity in the app.
    """
    files = list(DATA_DIR.glob("*.pdf")) + list(DATA_DIR.glob("*.txt"))
    if not files:
        print(f"No PDFs/TXTs found in {DATA_DIR}. Add your policy PDFs there.")
        return

    # Step 1: Extract and chunk text from all documents
    texts: List[str] = []  # List of text chunks
    meta: List[Dict] = []  # Metadata for each chunk: {"source": filename, "page": page_num}

    for f in files:
        pages = read_pdf(f) if f.suffix.lower() == ".pdf" else read_txt(f)
        for content, page_num in pages:
            for ch in chunk_text(content, chunk_size=chunk_size, overlap=overlap):
                texts.append(ch)
                meta.append({"source": f.name, "page": page_num})

    if not texts:
        print("No extractable text found in your files.")
        return

    # Step 2: Generate embeddings
    print(f"Found {len(texts)} chunks from {len(files)} file(s). Embedding with {MODEL_NAME} ...")
    model = SentenceTransformer(MODEL_NAME)
    embs = model.encode(texts, batch_size=64, show_progress_bar=True)
    embs = embs.astype("float32")
    embs = l2_normalize(embs)  # L2 normalization for cosine similarity

    # Step 3: Persist index artifacts to disk
    np.save(INDEX_DIR / "embeddings.npy", embs)
    (INDEX_DIR / "texts.json").write_text(json.dumps(texts, ensure_ascii=False), encoding="utf-8")
    (INDEX_DIR / "meta.json").write_text(json.dumps(meta, ensure_ascii=False), encoding="utf-8")
    (INDEX_DIR / "config.json").write_text(
        json.dumps({"model": MODEL_NAME, "dim": DIM, "normalized": True}, ensure_ascii=False),
        encoding="utf-8"
    )

    # Step 4: Build FAISS index if available
    if HAVE_FAISS:
        index = faiss.IndexFlatIP(DIM)  # Inner product index for normalized vectors
        index.add(embs)
        faiss.write_index(index, str(INDEX_DIR / "faiss.index"))
        print(f"FAISS index written with {index.ntotal} vectors.")
    else:
        print("FAISS not available — saved raw embeddings; app will use brute-force cosine.")

    print("Index build complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build FAISS index from PDFs/TXTs")
    parser.add_argument("--chunk-size", type=int, default=800, help="Chunk size for text splitting")
    parser.add_argument("--overlap", type=int, default=200, help="Overlap for chunking")
    args = parser.parse_args()
    build(chunk_size=args.chunk_size, overlap=args.overlap)
