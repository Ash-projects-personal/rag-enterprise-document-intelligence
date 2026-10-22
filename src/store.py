"""Index build, save, load, and search for the two retrievers.

Dense side: FAISS IndexFlatIP over normalised embeddings (cosine).
Sparse side: BM25Okapi from rank_bm25 over lowercase alphanumeric tokens.

An index directory holds:
    faiss.index   the FAISS index
    chunks.jsonl  chunk records in index order (row i of the index is line i)
    bm25.pkl      the fitted BM25Okapi object
    meta.json     model name, chunking settings, and counts from build time
"""

import json
import pickle
import re
from pathlib import Path

import faiss
from rank_bm25 import BM25Okapi

TOKEN_RE = re.compile(r"[a-z0-9]+")


def tokenize(text):
    return TOKEN_RE.findall(text.lower())


class VectorStore:
    def __init__(self, chunks, faiss_index, bm25, meta):
        self.chunks = chunks
        self.faiss_index = faiss_index
        self.bm25 = bm25
        self.meta = meta

    def __len__(self):
        return len(self.chunks)

    def search_dense(self, query_vector, k):
        scores, idx = self.faiss_index.search(
            query_vector.reshape(1, -1), k)
        return [(int(i), float(s)) for i, s in zip(idx[0], scores[0]) if i >= 0]

    def search_bm25(self, query, k):
        scores = self.bm25.get_scores(tokenize(query))
        ranked = sorted(range(len(scores)), key=lambda i: scores[i],
                        reverse=True)[:k]
        return [(int(i), float(scores[i])) for i in ranked]


def build_store(chunks, embedder, meta_extra=None):
    texts = [c["text"] for c in chunks]
    vectors = embedder.encode(texts)
    index = faiss.IndexFlatIP(vectors.shape[1])
    index.add(vectors)
    bm25 = BM25Okapi([tokenize(t) for t in texts])
    meta = {
        "embedding_model": embedder.model_name,
        "embedding_dim": int(vectors.shape[1]),
        "num_chunks": len(chunks),
    }
    if meta_extra:
        meta.update(meta_extra)
    return VectorStore(chunks, index, bm25, meta)


def save_store(store, index_dir):
    index_dir = Path(index_dir)
    index_dir.mkdir(parents=True, exist_ok=True)
    faiss.write_index(store.faiss_index, str(index_dir / "faiss.index"))
    with (index_dir / "chunks.jsonl").open("w", encoding="utf-8") as f:
        for chunk in store.chunks:
            f.write(json.dumps(chunk, ensure_ascii=False) + "\n")
    with (index_dir / "bm25.pkl").open("wb") as f:
        pickle.dump(store.bm25, f)
    with (index_dir / "meta.json").open("w", encoding="utf-8") as f:
        json.dump(store.meta, f, indent=2)


def load_store(index_dir):
    from .loader import load_jsonl

    index_dir = Path(index_dir)
    chunks = load_jsonl(index_dir / "chunks.jsonl")
    faiss_index = faiss.read_index(str(index_dir / "faiss.index"))
    with (index_dir / "bm25.pkl").open("rb") as f:
        bm25 = pickle.load(f)
    with (index_dir / "meta.json").open(encoding="utf-8") as f:
        meta = json.load(f)
    return VectorStore(chunks, faiss_index, bm25, meta)
