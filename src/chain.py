"""The question answering chain.

Default mode is extractive and needs no API key:
1. Hybrid retrieval returns fused candidates.
2. The cross-encoder reranker orders them.
3. The answer is the single sentence from the top passage with the
   highest token overlap F1 against the question's content words,
   returned together with the passage it came from.

Optional generate mode: if OPENAI_API_KEY is set in the environment and
the caller asks for mode="generate", the retrieved passages are sent to
the OpenAI chat API and the model's answer is returned instead. Nothing
in the evaluation uses this mode; all measured numbers come from the
extractive path.
"""

import os
import re

from .reranker import Reranker
from .retriever import HybridRetriever
from .store import tokenize

SENTENCE_RE = re.compile(r"[^.!?]+[.!?]")
STOPWORDS = set(
    "a an the is are was were what who whom whose when where which why how "
    "did do does in on of for to from by with as at it its this that these "
    "those be been being have has had".split())


def split_sentences(text):
    sentences = [s.strip() for s in SENTENCE_RE.findall(text)]
    return [s for s in sentences if s]


def content_tokens(text):
    return [t for t in tokenize(text) if t not in STOPWORDS]


def token_f1(candidate_tokens, reference_tokens):
    if not candidate_tokens or not reference_tokens:
        return 0.0
    cand_counts = {}
    for t in candidate_tokens:
        cand_counts[t] = cand_counts.get(t, 0) + 1
    overlap = 0
    ref_counts = {}
    for t in reference_tokens:
        ref_counts[t] = ref_counts.get(t, 0) + 1
    for t, c in cand_counts.items():
        overlap += min(c, ref_counts.get(t, 0))
    if overlap == 0:
        return 0.0
    precision = overlap / len(candidate_tokens)
    recall = overlap / len(reference_tokens)
    return 2 * precision * recall / (precision + recall)


def extract_answer_sentence(question, passage_text):
    """Return the sentence in the passage that best matches the question.

    Matching is token overlap F1 between the question's content words and
    each sentence. This is a transparent heuristic, and the evaluation
    scores its output against the gold answer so its quality is measured,
    not asserted.
    """
    sentences = split_sentences(passage_text)
    if not sentences:
        return passage_text.strip()
    q_tokens = content_tokens(question)
    best_sentence = max(
        sentences, key=lambda s: token_f1(tokenize(s), q_tokens))
    return best_sentence


class RAGChain:
    def __init__(self, store, embedder, reranker=None):
        self.retriever = HybridRetriever(store, embedder)
        self.reranker = reranker or Reranker()

    def retrieve(self, question, fused_k=20):
        return self.retriever.retrieve(question, fused_k=fused_k)

    def query(self, question, top_k=5, fused_k=20, mode="extractive"):
        fused = self.retrieve(question, fused_k=fused_k)
        reranked = self.reranker.rerank(question, fused, top_n=top_k)
        if not reranked:
            return {"question": question, "answer": None, "sources": []}
        top = reranked[0]
        if mode == "generate":
            answer = self._generate(question, reranked)
        else:
            answer = extract_answer_sentence(question, top["text"])
        sources = [{
            "chunk_id": c["chunk_id"],
            "doc_id": c["doc_id"],
            "title": c["title"],
            "rerank_score": c["rerank_score"],
            "text": c["text"],
        } for c in reranked]
        return {"question": question, "answer": answer, "sources": sources}

    def _generate(self, question, passages):
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError(
                "mode='generate' needs OPENAI_API_KEY in the environment; "
                "use mode='extractive' otherwise")
        from openai import OpenAI

        context = "\n\n".join(
            f"[{p['title']}] {p['text']}" for p in passages)
        client = OpenAI(api_key=api_key)
        response = client.chat.completions.create(
            model=os.environ.get("OPENAI_MODEL", "gpt-4o-mini"),
            messages=[
                {"role": "system", "content":
                 "Answer using only the context below. If the context does "
                 "not contain the answer, say you do not have that "
                 "information rather than guessing."},
                {"role": "user", "content":
                 f"Context:\n{context}\n\nQuestion: {question}"},
            ],
        )
        return response.choices[0].message.content
