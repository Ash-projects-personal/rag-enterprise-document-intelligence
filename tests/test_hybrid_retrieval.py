"""
Hybrid retrieval + reranker tests for ``src.retrieval.hybrid_retriever``.

Pins the contracts that make hybrid search (dense + BM25 via reciprocal-rank
fusion) and cross-encoder reranking work:

  - Fusion weight & ranking: heavier-weighted modality should drag a
    document up the fused list, RRF score is monotonic in rank.
  - Reranker ordering: the cross-encoder scores *override* the fused
    ranking — the post-rerank list is sorted by reranker score, descending.
  - ``rerank_top_n`` truncation is applied after reranking, not before.
  - Determinism: same inputs → same output ordering, twice in a row.
  - Edge cases: empty BM25 corpus, missing reranker (pass-through), and
    no-overlap dense/sparse results still produce a sane fused list.

No network, no Pinecone, no actual cross-encoder — the dense backend and
reranker are tiny in-test fakes that match the real interfaces.
"""
from __future__ import annotations

from typing import Any, Dict, List, Tuple

import pytest


# Skip if rank_bm25 isn't installed — the HybridRetriever only initialises
# its BM25 backend when the lib is importable.
rank_bm25 = pytest.importorskip("rank_bm25")

from src.retrieval.hybrid_retriever import HybridRetriever  # noqa: E402


# ── fakes ────────────────────────────────────────────────────────────────


class FakePineconeIndex:
    """Pinecone-shaped stub: ``index.query(vector=..., top_k=..., include_metadata=...)``
    returns ``{"matches": [{"metadata": {"text": ...}, "score": float}, ...]}``.

    The fake just hands back the seeded matches in their existing order so
    the test can pin "this doc was at dense rank 0" assertions.
    """

    def __init__(self, matches: List[Tuple[str, float]]) -> None:
        self._matches = matches

    def query(self, *, vector: Any, top_k: int,
              include_metadata: bool = True) -> Dict[str, Any]:
        return {
            "matches": [
                {"metadata": {"text": text}, "score": score}
                for text, score in self._matches[:top_k]
            ]
        }


def identity_embed(query: str) -> List[float]:
    return [float(len(query))]  # not actually used by the fake — just shape


class ListReranker:
    """Cross-encoder stub: ``predict(pairs)`` returns the score for each pair.

    The score for a pair (query, text) is looked up by text from a dict the
    test provides; missing texts get score 0.0. That lets a test seed a
    specific reordering and assert on the post-rerank list.
    """

    def __init__(self, score_by_text: Dict[str, float]) -> None:
        self._scores = score_by_text
        self.calls: List[List[Tuple[str, str]]] = []

    def predict(self, pairs: List[Tuple[str, str]]) -> List[float]:
        self.calls.append(list(pairs))
        return [self._scores.get(text, 0.0) for _query, text in pairs]


# ── fixtures ─────────────────────────────────────────────────────────────


@pytest.fixture()
def corpus() -> List[str]:
    return [
        "kafka streams ksqldb tutorial",
        "postgres replication lag monitoring",
        "rag pipeline reranker cross encoder",
        "kubernetes pod autoscaling guide",
        "fastapi async dependency injection",
        "vector database hnsw index parameters",
        "bm25 tokenization stopword removal",
        "openai embedding ada cost per million",
    ]


# ── reciprocal-rank-fusion ───────────────────────────────────────────────


class TestReciprocalRankFusion:
    def test_dense_only_passes_through_in_dense_order(
        self, corpus: List[str]
    ) -> None:
        retr = HybridRetriever(
            pinecone_index=FakePineconeIndex([(corpus[0], 0.9), (corpus[1], 0.8)]),
            bm25_corpus=corpus,
            embed_fn=identity_embed,
            reranker=None,
            top_k=2,
            rerank_top_n=5,
            dense_weight=1.0,
            sparse_weight=0.0,
        )
        # All-dense weighting + an empty BM25 contribution → dense order wins.
        # Force BM25 to return zero scores by querying with a token that does
        # not appear in any corpus document.
        results = retr.retrieve("zzzzzzz")
        assert [text for text, _ in results][:2] == [corpus[0], corpus[1]]

    def test_fusion_score_is_monotonic_in_rank(self) -> None:
        # Two distinct dense docs at rank 0 and rank 1 → fused score for the
        # rank-0 doc must be strictly larger.
        dense = [("rank0", 0.9), ("rank1", 0.5)]
        retr = HybridRetriever(dense_weight=1.0, sparse_weight=0.0)
        fused = retr._reciprocal_rank_fusion(dense, [])
        assert fused[0][0] == "rank0"
        assert fused[0][1] > fused[1][1]

    def test_sparse_only_returns_bm25_ranking(self, corpus: List[str]) -> None:
        # No Pinecone index → dense returns []; we should still get fused
        # results coming entirely from BM25.
        retr = HybridRetriever(
            pinecone_index=None,
            bm25_corpus=corpus,
            embed_fn=identity_embed,
            reranker=None,
            top_k=4,
            rerank_top_n=4,
            dense_weight=0.0,
            sparse_weight=1.0,
        )
        results = retr.retrieve("bm25 tokenization")
        texts = [text for text, _ in results]
        assert "bm25 tokenization stopword removal" in texts[:2]

    def test_weight_changes_fused_winner(self) -> None:
        """The same dense+sparse inputs should produce a different top-1 doc
        depending on which weight dominates."""
        dense = [("DOC_A", 0.9), ("DOC_B", 0.5)]
        sparse = [("DOC_B", 5.0), ("DOC_A", 0.0)]

        heavy_dense = HybridRetriever(dense_weight=0.99, sparse_weight=0.01)
        heavy_sparse = HybridRetriever(dense_weight=0.01, sparse_weight=0.99)

        top_when_dense = heavy_dense._reciprocal_rank_fusion(dense, sparse)[0][0]
        top_when_sparse = heavy_sparse._reciprocal_rank_fusion(dense, sparse)[0][0]
        assert top_when_dense == "DOC_A"
        assert top_when_sparse == "DOC_B"

    def test_overlap_documents_get_summed_score(self) -> None:
        """A doc appearing in BOTH dense and sparse should outrank a doc that
        only appears in one (given equal weights)."""
        dense = [("BOTH", 0.5), ("DENSE_ONLY", 0.4)]
        sparse = [("BOTH", 3.0), ("SPARSE_ONLY", 1.0)]
        retr = HybridRetriever(dense_weight=0.5, sparse_weight=0.5)
        fused = retr._reciprocal_rank_fusion(dense, sparse)
        # BOTH must be first; the other two follow in some order.
        assert fused[0][0] == "BOTH"
        assert {t for t, _ in fused[1:]} == {"DENSE_ONLY", "SPARSE_ONLY"}


# ── reranker output ordering ─────────────────────────────────────────────


class TestRerankerOrdering:
    def test_reranker_overrides_fused_order(self, corpus: List[str]) -> None:
        """If the cross-encoder strongly prefers a doc the fused list put
        last, the final result must put it first."""
        # Seed dense + sparse so DOC_FIRST is rank-0 (fused winner).
        dense_index = FakePineconeIndex(
            [("DOC_FIRST", 0.9), ("DOC_MID", 0.5), ("DOC_LAST", 0.1)]
        )
        # BM25 corpus matches all three so they appear in fused list.
        bm25_corpus = ["DOC_FIRST tokens here", "DOC_MID tokens", "DOC_LAST tokens"]
        # Reranker flips the order — DOC_LAST is the actual best answer.
        reranker = ListReranker({"DOC_LAST tokens": 10.0,
                                  "DOC_MID tokens": 5.0,
                                  "DOC_FIRST tokens here": 1.0})

        retr = HybridRetriever(
            pinecone_index=dense_index,
            bm25_corpus=bm25_corpus,
            embed_fn=identity_embed,
            reranker=reranker,
            top_k=3,
            rerank_top_n=3,
        )
        results = retr.retrieve("tokens")
        texts = [text for text, _ in results]
        assert texts == ["DOC_LAST tokens", "DOC_MID tokens", "DOC_FIRST tokens here"]

    def test_reranker_returns_descending_scores(self, corpus: List[str]) -> None:
        reranker = ListReranker({c: float(len(c)) for c in corpus})
        candidates = [(c, 0.0) for c in corpus]
        retr = HybridRetriever(reranker=reranker, rerank_top_n=len(corpus))
        ranked = retr._rerank("ignored-query", candidates)
        scores = [s for _, s in ranked]
        assert scores == sorted(scores, reverse=True)

    def test_rerank_top_n_truncates_after_scoring(self) -> None:
        reranker = ListReranker({"A": 1.0, "B": 9.0, "C": 5.0, "D": 7.0})
        candidates = [("A", 0.0), ("B", 0.0), ("C", 0.0), ("D", 0.0)]
        retr = HybridRetriever(reranker=reranker, rerank_top_n=2)
        ranked = retr._rerank("q", candidates)
        assert [text for text, _ in ranked] == ["B", "D"]
        assert len(ranked) == 2

    def test_no_reranker_truncates_fused_to_rerank_top_n(self) -> None:
        candidates = [(f"doc{i}", 0.0) for i in range(10)]
        retr = HybridRetriever(reranker=None, rerank_top_n=3)
        # Pass-through behaviour: same order, first N items.
        ranked = retr._rerank("q", candidates)
        assert [text for text, _ in ranked] == ["doc0", "doc1", "doc2"]

    def test_reranker_is_called_with_query_and_candidate_pairs(self) -> None:
        reranker = ListReranker({"x": 1.0, "y": 2.0})
        retr = HybridRetriever(reranker=reranker, rerank_top_n=2)
        retr._rerank("my query", [("x", 0.0), ("y", 0.0)])
        assert reranker.calls == [[("my query", "x"), ("my query", "y")]]


# ── determinism & integration ────────────────────────────────────────────


class TestDeterminismAndIntegration:
    def test_same_inputs_produce_same_results_twice(self, corpus: List[str]) -> None:
        dense_index = FakePineconeIndex([(corpus[2], 0.9), (corpus[6], 0.7)])
        reranker = ListReranker({c: 1.0 / (i + 1) for i, c in enumerate(corpus)})
        retr = HybridRetriever(
            pinecone_index=dense_index,
            bm25_corpus=corpus,
            embed_fn=identity_embed,
            reranker=reranker,
            top_k=5,
            rerank_top_n=3,
        )
        first = retr.retrieve("rag reranker cross encoder")
        second = retr.retrieve("rag reranker cross encoder")
        assert first == second

    def test_end_to_end_topk_and_rerank_top_n_respected(
        self, corpus: List[str]
    ) -> None:
        dense_index = FakePineconeIndex([(c, 1.0 - 0.1 * i) for i, c in enumerate(corpus)])
        reranker = ListReranker({c: float(i) for i, c in enumerate(corpus)})
        retr = HybridRetriever(
            pinecone_index=dense_index,
            bm25_corpus=corpus,
            embed_fn=identity_embed,
            reranker=reranker,
            top_k=5,         # fused candidate pool of 5
            rerank_top_n=2,  # but only 2 returned
        )
        results = retr.retrieve("anything")
        assert len(results) == 2
