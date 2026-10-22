"""FastAPI service for the retrieval pipeline.

Endpoints:
    GET  /health   service state and index size
    POST /ingest   build the index from data/corpus.jsonl (or load the
                   existing one) and make the chain ready to answer
    POST /query    run the real chain: hybrid retrieval, cross-encoder
                   reranking, then an extractive answer, or a generated
                   answer when mode is "generate" and OPENAI_API_KEY is
                   set

Run it with:
    uvicorn api.main:app --port 8000
"""

import sys
import time
from pathlib import Path

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.chain import RAGChain  # noqa: E402
from src.embeddings import Embedder  # noqa: E402
from src.reranker import Reranker  # noqa: E402
from src.store import load_store  # noqa: E402

app = FastAPI(
    title="rag-enterprise-document-intelligence",
    description="Hybrid retrieval question answering over a document "
                "corpus. Extractive answers by default; generated "
                "answers only when OPENAI_API_KEY is configured.",
)

_state = {"chain": None, "meta": None}


def _load_chain():
    index_dir = ROOT / "index"
    if not (index_dir / "faiss.index").exists():
        raise HTTPException(
            status_code=503,
            detail="no index found; call POST /ingest first")
    store = load_store(index_dir)
    embedder = Embedder(store.meta["embedding_model"])
    _state["chain"] = RAGChain(store, embedder, reranker=Reranker())
    _state["meta"] = store.meta


class IngestRequest(BaseModel):
    rebuild: bool = False


class QueryRequest(BaseModel):
    question: str
    top_k: int = 5
    mode: str = "extractive"


@app.get("/health")
def health():
    return {
        "status": "ok",
        "index_loaded": _state["chain"] is not None,
        "index_meta": _state["meta"],
    }


@app.post("/ingest")
def ingest(request: IngestRequest):
    index_dir = ROOT / "index"
    if request.rebuild or not (index_dir / "faiss.index").exists():
        from src.chunker import chunk_corpus
        from src.loader import load_corpus
        from src.store import build_store, save_store

        documents = load_corpus(ROOT / "data/corpus.jsonl")
        chunks = chunk_corpus(documents)
        embedder = Embedder()
        store = build_store(chunks, embedder, meta_extra={
            "num_documents": len(documents),
            "chunk_max_words": 180,
            "chunk_overlap_words": 40,
        })
        save_store(store, index_dir)
    _load_chain()
    return {"status": "ready", "index_meta": _state["meta"]}


@app.post("/query")
def query(request: QueryRequest):
    if _state["chain"] is None:
        _load_chain()
    if request.mode not in ("extractive", "generate"):
        raise HTTPException(status_code=400,
                            detail="mode must be 'extractive' or 'generate'")
    started = time.perf_counter()
    try:
        result = _state["chain"].query(
            request.question, top_k=request.top_k, mode=request.mode)
    except RuntimeError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    result["latency_ms"] = round((time.perf_counter() - started) * 1000, 1)
    return result
