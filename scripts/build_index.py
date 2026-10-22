"""Chunk the corpus, embed the chunks, and build the FAISS and BM25
indexes. Writes everything under index/ and prints what it built.
"""

import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.chunker import chunk_corpus  # noqa: E402
from src.embeddings import Embedder  # noqa: E402
from src.loader import load_corpus  # noqa: E402
from src.store import build_store, save_store  # noqa: E402


def main():
    started = time.time()
    documents = load_corpus(ROOT / "data/corpus.jsonl")
    chunks = chunk_corpus(documents)
    print(f"documents: {len(documents)}")
    print(f"chunks: {len(chunks)}")
    embedder = Embedder()
    store = build_store(chunks, embedder, meta_extra={
        "num_documents": len(documents),
        "chunk_max_words": 180,
        "chunk_overlap_words": 40,
    })
    save_store(store, ROOT / "index")
    print(f"index written to {ROOT / 'index'} in {time.time() - started:.1f}s")


if __name__ == "__main__":
    main()
