import os, json, time, csv
from pathlib import Path
from typing import List, Dict, Tuple
import numpy as np
from sentence_transformers import SentenceTransformer
from groq import Groq
import os

os.environ["TOKENIZERS_PARALLELISM"] = "false"
INDEX_DIR = Path("index")
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
MODEL = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")

try:
    import faiss
    HAVE_FAISS = True
except Exception:
    HAVE_FAISS = False

SYSTEM_PROMPT = (
    "You are an HR Policy Assistant. Answer ONLY using the provided context. "
    "If the answer is not clearly in the context, say: "
    "\"I don’t have enough policy evidence to answer. Please check with HR.\" "
    "Keep answers concise and practical."
)

def l2_normalize(X: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(X, axis=1, keepdims=True) + 1e-12
    return X / norms

def load_index():
    embs = np.load(INDEX_DIR / "embeddings.npy")
    texts = json.loads((INDEX_DIR / "texts.json").read_text(encoding="utf-8"))
    meta  = json.loads((INDEX_DIR / "meta.json").read_text(encoding="utf-8"))
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

def evaluate(questions_path="eval_questions.json", top_k=5, refuse_threshold=0.20, out_csv="eval_results.csv"):
    key = (os.getenv("GROQ_API_KEY") or "").strip()
    if not key: raise RuntimeError("Set GROQ_API_KEY")
    qset = json.loads(Path(questions_path).read_text(encoding="utf-8"))
    emb_model, embs, texts, meta, index = load_index()

    results=[]
    passed=0
    for item in qset:
        q = item["q"]
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
        contains = (item["expect_text_contains"].lower() in ans.lower()) if item["expect_text_contains"] else ("I don’t have enough policy evidence" in ans)

        ok = grounded and contains
        passed += int(ok)

        results.append({
            "question": q,
            "expected_source_contains": item["expect_source_contains"],
            "expected_text_contains": item["expect_text_contains"],
            "sources": srcs,
            "max_retrieval_score": round(max_score,3),
            "latency_ms": latency_ms,
            "grounded_ok": grounded,
            "answer_contains_expected_text": contains,
            "pass": ok,
            "answer": ans
        })

    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(results[0].keys()))
        w.writeheader(); w.writerows(results)

    total=len(qset)
    print(f"PASS: {passed}/{total}  ({passed/total*100:.1f}%)  → details in {out_csv}")

if __name__ == "__main__":
    evaluate()
