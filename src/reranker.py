"""Cross-encoder reranking with a local model.

The reranker scores (question, passage) pairs and reorders the fused
retrieval candidates. The model runs on this machine; no API key.
"""

from sentence_transformers import CrossEncoder

DEFAULT_RERANKER = "cross-encoder/ms-marco-MiniLM-L-6-v2"


class Reranker:
    def __init__(self, model_name=DEFAULT_RERANKER):
        self.model_name = model_name
        self.model = CrossEncoder(model_name)

    def rerank(self, query, candidates, top_n=5):
        if not candidates:
            return []
        pairs = [(query, c["text"]) for c in candidates]
        scores = self.model.predict(pairs, show_progress_bar=False)
        scored = []
        for candidate, score in zip(candidates, scores):
            item = dict(candidate)
            item["rerank_score"] = float(score)
            scored.append(item)
        scored.sort(key=lambda c: c["rerank_score"], reverse=True)
        return scored[:top_n]
