"""Hybrid retrieval: dense and BM25 candidate lists fused with reciprocal
rank fusion. Reranking lives in src/reranker.py.
"""

RRF_K = 60


class HybridRetriever:
    def __init__(self, store, embedder, dense_weight=1.0, bm25_weight=1.0):
        self.store = store
        self.embedder = embedder
        self.dense_weight = dense_weight
        self.bm25_weight = bm25_weight

    def retrieve(self, query, dense_k=30, bm25_k=30, fused_k=20):
        query_vector = self.embedder.encode_one(query)
        dense_hits = self.store.search_dense(query_vector, dense_k)
        bm25_hits = self.store.search_bm25(query, bm25_k)
        fused_scores = {}
        for weight, hits in ((self.dense_weight, dense_hits),
                             (self.bm25_weight, bm25_hits)):
            for rank, (chunk_idx, _score) in enumerate(hits):
                fused_scores[chunk_idx] = fused_scores.get(chunk_idx, 0.0) \
                    + weight / (RRF_K + rank + 1)
        ranked = sorted(fused_scores.items(), key=lambda kv: kv[1],
                        reverse=True)[:fused_k]
        results = []
        for chunk_idx, score in ranked:
            chunk = dict(self.store.chunks[chunk_idx])
            chunk["score"] = float(score)
            chunk["chunk_idx"] = int(chunk_idx)
            results.append(chunk)
        return results
