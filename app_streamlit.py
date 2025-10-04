# app_streamlit.py
# HR Policy Copilot — Groq + FAISS (role-aware + logging), no LangChain

import os, json, time, csv
from pathlib import Path
from typing import List, Dict
import numpy as np
import streamlit as st
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from groq import Groq
from prompts import SYSTEM_PROMPT

os.environ["TOKENIZERS_PARALLELISM"] = "false"  # Disable parallelism warning
INDEX_DIR = Path("index")
LOGS_DIR = Path("logs"); LOGS_DIR.mkdir(parents=True, exist_ok=True)
ROLES_PATH = Path("roles.json")
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

# Optional FAISS (brute-force fallback if missing)
try:
    import faiss
    HAVE_FAISS = True
except Exception:
    HAVE_FAISS = False

def l2_normalize(X: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(X, axis=1, keepdims=True) + 1e-12
    return X / norms

@st.cache_resource(show_spinner=False)
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

    # Only load FAISS if the file exists and is non-empty; otherwise fall back.
    index = None
    if HAVE_FAISS:
        faiss_path = (INDEX_DIR / "faiss.index")
        if faiss_path.exists() and faiss_path.stat().st_size > 0:
            try:
                index = faiss.read_index(str(faiss_path))
            except Exception:
                index = None

    emb_model = SentenceTransformer(MODEL_NAME)
    return emb_model, embs, texts, meta, index

@st.cache_resource(show_spinner=False)
def load_roles():
    # roles.json example (optional):
    # {
    #   "default_allow": ["Employee","Manager"],
    #   "file_rules": {
    #     "People_Leader_Guide.pdf": ["Manager"],
    #     "Leave_Policy.pdf": ["Employee","Manager"]
    #   }
    # }
    if ROLES_PATH.exists():
        try:
            cfg = json.loads(ROLES_PATH.read_text(encoding="utf-8"))
        except Exception:
            cfg = {}
    else:
        cfg = {}
    default_allow = cfg.get("default_allow", ["Employee","Manager"])
    file_rules = cfg.get("file_rules", {})
    return default_allow, file_rules

def allowed_for_role(filename: str, role: str, default_allow: List[str], file_rules: Dict[str, List[str]]) -> bool:
    allow = file_rules.get(filename, default_allow)
    return role in allow

def rank_search(emb_model, embs, query: str, top_k: int, index=None):
    q = emb_model.encode([query]).astype("float32")
    q = l2_normalize(q)
    if index is not None:
        D, I = index.search(q, top_k)
        return I[0].tolist(), D[0].tolist()
    sims = (embs @ q[0])
    idxs = np.argsort(-sims)[:top_k].tolist()
    scores = sims[idxs].tolist()
    return idxs, scores

def retrieve_with_role(emb_model, embs, texts, meta, query: str, role: str, want_k: int, index=None):
    # get a larger pool, then filter by role
    pool_k = min(len(texts), max(want_k * 6, 30))
    all_idxs, all_scores = rank_search(emb_model, embs, query, pool_k, index=index)
    default_allow, file_rules = load_roles()

    chosen = []
    for rank, i in enumerate(all_idxs):
        filename = meta[i]["source"]
        if allowed_for_role(filename, role, default_allow, file_rules):
            chosen.append({"text": texts[i], "meta": meta[i], "score": float(all_scores[rank])})
            if len(chosen) >= want_k:
                break
    return chosen

def format_context(chosen: List[Dict]) -> str:
    blocks = []
    for c in chosen:
        src = c["meta"]["source"]; page = c["meta"]["page"]; txt = c["text"]
        blocks.append(f"[{src} p.{page}] {txt}")
    return "\n\n".join(blocks)

def sources_list(chosen: List[Dict]) -> List[str]:
    seen = set(); out=[]
    for c in chosen:
        label = f"{c['meta']['source']} (p.{c['meta']['page']})"
        if label not in seen:
            out.append(label); seen.add(label)
    return out

def call_groq(api_key: str, model: str, system_prompt: str, user_prompt: str, max_tokens: int = 800):
    client = Groq(api_key=api_key)
    t0 = time.time()
    resp = client.chat.completions.create(
        model=model, temperature=0, max_tokens=max_tokens,
        messages=[{"role":"system","content":system_prompt},
                  {"role":"user","content":user_prompt}]
    )
    latency_ms = int((time.time() - t0) * 1000)
    content = resp.choices[0].message.content
    usage = getattr(resp, "usage", None)
    return content, latency_ms, usage

def log_event(row: Dict):
    path = LOGS_DIR / "query_log.csv"
    file_exists = path.exists()
    with path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(["ts","model","role","question","top_sources","scores","latency_ms","input_tokens","output_tokens","total_tokens"])
        writer.writerow([
            int(time.time()),
            row["model"],
            row["role"],
            row["question"],
            row["sources"],
            row["scores"],
            row["latency_ms"],
            row.get("input_tokens"),
            row.get("output_tokens"),
            row.get("total_tokens"),
        ])

def main():
    load_dotenv()
    st.title("HR Policy Copilot — Groq + FAISS (no LangChain)")
    st.caption("Role-aware retrieval + CSV logging. Answers cite local PDFs/TXTs.")

    with st.sidebar:
        st.header("Chat History")
        if "messages" not in st.session_state:
            st.session_state.messages = []
        for msg in reversed(st.session_state.messages[-5:]):
            with st.expander(f"Q: {msg['question'][:50]}..."):
                st.write(msg["answer"])
                st.write("Sources:", msg["sources"])

    groq_key = os.getenv("GROQ_API_KEY") or st.text_input("GROQ_API_KEY", type="password")
    if not groq_key:
        st.warning("API key is stored in session state for the duration of the app session. For secure deployment, use environment variables.")
        st.info("Enter your GROQ_API_KEY.")
        st.stop()

    model_choice = st.selectbox("Groq model",
        ["llama-3.1-8b-instant", "llama3-70b-8192", "llama-3.1-70b-versatile"])
    role = st.radio("Role", ["Employee","Manager"], horizontal=True)

    try:
        emb_model, embs, texts, meta, index = load_index()
    except Exception as e:
        st.error(f"Couldn't load index. Did you run `python build_index.py`? ({e})")
        st.stop()

    question = st.text_input("Your question (e.g., What is parental leave?)")
    top_k = st.slider("Top-K passages (after role filter)", 3, 8, 5)
    refuse_threshold = st.slider("Refusal threshold (max similarity)", 0.0, 1.0, 0.20, 0.01)

    if st.button("Ask") and question:
        chosen = retrieve_with_role(emb_model, embs, texts, meta, question, role, top_k, index=index)
        if not chosen:
            st.write("I don’t have enough policy evidence to answer. Please check with HR.")
            return

        scores = [c["score"] for c in chosen]
        if max(scores) < refuse_threshold:
            st.write("I don’t have enough policy evidence to answer. Please check with HR.")
            return

        context = format_context(chosen)
        user_prompt = f"Question: {question}\n\nContext:\n{context}\n\nAnswer with specific guidance and cite sources."
        answer, latency_ms, usage = call_groq(groq_key, model_choice, SYSTEM_PROMPT, user_prompt)

        st.write(answer)
        st.markdown("**Sources:**")
        src_list = sources_list(chosen)
        for s in src_list:
            st.write(f"- {s}")

        # Append to history
        st.session_state.messages.append({
            "question": question,
            "answer": answer,
            "sources": src_list
        })

        log_event({
            "model": model_choice,
            "role": role,
            "question": question,
            "sources": "; ".join(src_list),
            "scores": "; ".join([f"{s:.3f}" for s in scores]),
            "latency_ms": latency_ms,
            "input_tokens": getattr(usage, 'input_tokens', None) if usage else None,
            "output_tokens": getattr(usage, 'output_tokens', None) if usage else None,
            "total_tokens": getattr(usage, 'total_tokens', None) if usage else None,
        })

        with st.expander("Debug (retrieval scores)"):
            st.write([round(s, 3) for s in scores])

if __name__ == "__main__":
    main()
