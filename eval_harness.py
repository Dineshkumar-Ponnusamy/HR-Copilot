"""Evaluation harness for HR Policy Copilot.

This module provides evaluation capabilities to assess the retrieval and generation
quality of the HR Policy Copilot system. It supports both standard retrieval and
cross-encoder reranking for more accurate passage selection.

Supports evaluation on question sets stored in JSON format with expected sources
and answer content for automated scoring.
"""

import os, json, argparse, time, csv
from pathlib import Path
from typing import List, Dict, Tuple
import numpy as np
from sentence_transformers import SentenceTransformer, CrossEncoder
from groq import Groq
from prompts import SYSTEM_PROMPT

# ---- Optional cross-encoder reranker configuration
RERANK_MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"
_reranker = None

def get_reranker():
    """Lazy-load the cross-encoder reranker model."""
    global _reranker
    if _reranker is None:
        _reranker = CrossEncoder(RERANK_MODEL_NAME)
    return _reranker

# ---- Configuration
os.environ["TOKENIZERS_PARALLELISM"] = "false"  # Disable parallelism warning
INDEX_DIR = Path("index")
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
MODEL = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")

# ---- FAISS availability check
try:
    import faiss
    HAVE_FAISS = True
except Exception:
    HAVE_FAISS = False

def l2_normalize(X: np.ndarray) -> np.ndarray:
    """Normalize vectors using L2 normalization for cosine similarity computation.

    Args:
        X: Input array of shape (n_samples, n_features).

    Returns:
        L2-normalized array where each row has unit norm.

    Note:
        Enables cosine similarity to be computed efficiently via inner product.
    """
    norms = np.linalg.norm(X, axis=1, keepdims=True) + 1e-12
    return X / norms

def load_index():
    try:
        embs = np.load(INDEX_DIR / "embeddings.npy")
    except FileNotFoundError as e:
        raise FileNotFoundError(f"Embeddings file missing: {e}")

    try:
        texts = json.loads((INDEX_DIR / "texts.json").read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError) as e:
        raise FileNotFoundError(f"Texts index file missing or invalid: {e}")

    try:
        meta = json.loads((INDEX_DIR / "meta.json").read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError) as e:
        raise FileNotFoundError(f"Meta index file missing or invalid: {e}")

    index = None
    if HAVE_FAISS:
        fp = INDEX_DIR / "faiss.index"
        if fp.exists() and fp.stat().st_size > 0:
            try:
                index = faiss.read_index(str(fp))
            except Exception:
                index = None
    emb_model = SentenceTransformer(MODEL_NAME)
    return emb_model, embs, texts, meta, index

def retrieve_topk_with_rerank(emb_model, embs, texts, meta, query, want_k, index=None):
    # 1) get a candidate pool via embeddings (fast)
    pool_k = min(len(texts), max(want_k * 6, 30))
    idxs, sims = rank(emb_model, embs, query, pool_k, index=index)
    candidates = [{"i": i, "sim": float(sims[n]), "text": texts[i], "meta": meta[i]} for n, i in enumerate(idxs)]

    # 2) re-rank with cross-encoder (accurate)
    reranker = get_reranker()
    pairs = [(query, c["text"]) for c in candidates]
    scores = reranker.predict(pairs)  # higher is better
    for c, sc in zip(candidates, scores):
        c["rerank"] = float(sc)

    # 3) sort by rerank desc, keep top-k
    candidates.sort(key=lambda x: x["rerank"], reverse=True)
    chosen = [{"text": c["text"], "meta": c["meta"], "score": c["rerank"]} for c in candidates[:want_k]]
    return chosen


def rank(emb_model, embs, query: str, k: int, index=None) -> Tuple[List[int], List[float]]:
    q = emb_model.encode([query]).astype("float32")
    q = l2_normalize(q)
    if index is not None:
        D, I = index.search(q, k)
        return I[0].tolist(), D[0].tolist()
    sims = (embs @ q[0])
    idxs = np.argsort(-sims)[:k].tolist()
    scores = sims[idxs].tolist()
    return idxs, scores

def format_context(chosen: List[Dict]) -> str:
    return "\n\n".join([f"[{c['meta']['source']} p.{c['meta']['page']}] {c['text']}" for c in chosen])

def sources_list(chosen: List[Dict]) -> List[str]:
    seen=set(); out=[]
    for c in chosen:
        lab=f"{c['meta']['source']} (p.{c['meta']['page']})"
        if lab not in seen: out.append(lab); seen.add(lab)
    return out

def call_groq(key: str, model: str, system_prompt: str, user_prompt: str):
    client = Groq(api_key=key)
    t0 = time.time()
    resp = client.chat.completions.create(
        model=model, temperature=0, max_tokens=800,
        messages=[{"role":"system","content":system_prompt},
                  {"role":"user","content":user_prompt}]
    )
    latency_ms = int((time.time()-t0)*1000)
    content = resp.choices[0].message.content
    return content, latency_ms

from difflib import SequenceMatcher
def fuzzy_match(a, b):
    return SequenceMatcher(None, a.lower(), b.lower()).ratio()

def evaluate(questions_path="eval_questions.json", top_k=5, refuse_threshold=0.10, out_csv="eval_results.csv", use_reranker=False):
    key = (os.getenv("GROQ_API_KEY") or "").strip()
    if not key: raise RuntimeError("Set GROQ_API_KEY")
    qset = json.loads(Path(questions_path).read_text(encoding="utf-8"))
    emb_model, embs, texts, meta, index = load_index()

    results=[]
    passed=0
    for item in qset:
        q = item["q"]
        if use_reranker:
            chosen = retrieve_topk_with_rerank(emb_model, embs, texts, meta, q, top_k, index=index)
        else:
            idxs, scores = rank(emb_model, embs, q, max(top_k*6, 30), index=index)
            # take top_k
            chosen=[]
            for i,s in zip(idxs, scores):
                chosen.append({"text":texts[i], "meta":meta[i], "score":float(s)})
                if len(chosen)>=top_k: break

        max_score = max([c["score"] for c in chosen]) if chosen else 0.0
        if max_score < refuse_threshold:
            ans = "I don’t have enough policy evidence to answer. Please check with HR."
            latency_ms = 0
        else:
            context = format_context(chosen)
            user_prompt = f"Question: {q}\n\nContext:\n{context}\n\nAnswer and cite sources."
            ans, latency_ms = call_groq(key, MODEL, SYSTEM_PROMPT, user_prompt)

        srcs = "; ".join(sources_list(chosen))
        grounded = (item["expect_source_contains"].lower() in srcs.lower()) if item["expect_source_contains"] else True
        
        # Answer acceptance logic
        if item["expect_text_contains"]:
            answer_contains_expected_text = fuzzy_match(item["expect_text_contains"], ans) > 0.5
        else:
            answer_contains_expected_text = "I don’t have enough policy evidence" in ans
        
        print(f"[fuzzy] score={fuzzy_match(item['expect_text_contains'], ans):.2f} | Q: {q}")

        ok = grounded and answer_contains_expected_text
        passed += int(ok)

        results.append({
            "question": q,
            "expected_source_contains": item["expect_source_contains"],
            "expected_text_contains": item["expect_text_contains"],
            "sources": srcs,
            "max_retrieval_score": round(max_score,3),
            "latency_ms": latency_ms,
            "grounded_ok": grounded,
            "answer_contains_expected_text": answer_contains_expected_text,
            "pass": ok,
            "answer": ans
        })

    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(results[0].keys()))
        w.writeheader(); w.writerows(results)

    total=len(qset)
    print(f"PASS: {passed}/{total}  ({passed/total*100:.1f}%)  → details in {out_csv}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate HR Copilot retrieval and generation")
    parser.add_argument("--use-reranker", action="store_true", help="Enable cross-encoder reranking for accuracy")
    args = parser.parse_args()
    evaluate(use_reranker=args.use_reranker)
