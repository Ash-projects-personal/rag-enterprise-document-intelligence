"""Run the evaluation on the SQuAD sample and write the results.

For every evaluation question this measures:
- retrieval: rank of the first chunk that covers the gold answer span,
  in the fused hybrid list and in the reranked list; Recall@k and MRR
  computed from those ranks.
- answering (extractive mode): whether the gold answer text appears in
  the top reranked passage, and the token F1 and exact match of the
  extracted answer sentence against the gold answer.
- latency: wall clock time of the full query path (retrieve, rerank,
  extract) per question; mean and p95 are reported.

Outputs, all written by this run:
    outputs/metrics.json        every number reported anywhere
    outputs/per_question.jsonl one row per question with its ranks
    outputs/run_log.txt         human readable log of the run
"""

import json
import platform
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.chain import RAGChain, extract_answer_sentence, token_f1  # noqa: E402
from src.embeddings import Embedder  # noqa: E402
from src.loader import load_eval_questions  # noqa: E402
from src.reranker import Reranker  # noqa: E402
from src.store import load_store, tokenize  # noqa: E402


def normalize(text):
    return " ".join(tokenize(text))


def gold_chunk_indices(store, question):
    hits = []
    for idx, chunk in enumerate(store.chunks):
        if chunk["doc_id"] != question["doc_id"]:
            continue
        if (chunk["char_start"] < question["answer_end"]
                and question["answer_start"] < chunk["char_end"]):
            hits.append(idx)
    return set(hits)


def first_rank(results, gold):
    for rank, item in enumerate(results, start=1):
        if item["chunk_idx"] in gold:
            return rank
    return None


def recall_at_k(ranks, k):
    if not ranks:
        return 0.0
    return sum(1 for r in ranks if r is not None and r <= k) / len(ranks)


def main():
    log_lines = []

    def log(msg):
        print(msg, flush=True)
        log_lines.append(msg)

    started = time.time()
    log("rag-enterprise-document-intelligence evaluation run")
    log(f"python: {platform.python_version()}  platform: {platform.platform()}")
    log(f"machine cpus: {__import__('os').cpu_count()}")

    store = load_store(ROOT / "index")
    log(f"index: {store.meta}")
    questions = load_eval_questions(ROOT / "data/eval_questions.jsonl")
    log(f"evaluation questions: {len(questions)}")

    embedder = Embedder(store.meta["embedding_model"])
    chain = RAGChain(store, embedder, reranker=Reranker())

    fused_ranks = []
    reranked_ranks = []
    latencies = []
    containment = 0
    sentence_f1_scores = []
    sentence_exact = 0
    per_question = []

    for i, q in enumerate(questions):
        gold = gold_chunk_indices(store, q)
        t0 = time.perf_counter()
        fused = chain.retrieve(q["question"], fused_k=20)
        fused_rank = first_rank(fused, gold)
        reranked = chain.reranker.rerank(q["question"], fused, top_n=5)
        reranked_rank = first_rank(reranked, gold)
        top = reranked[0] if reranked else None
        answer_sentence = None
        if top is not None:
            answer_sentence = extract_answer_sentence(q["question"], top["text"])
        latency = time.perf_counter() - t0

        fused_ranks.append(fused_rank)
        reranked_ranks.append(reranked_rank)
        latencies.append(latency)

        gold_norm = normalize(q["answer_text"])
        if top is not None and gold_norm in normalize(top["text"]):
            containment += 1
        if answer_sentence is not None:
            sentence_f1_scores.append(
                token_f1(tokenize(answer_sentence), tokenize(q["answer_text"])))
            if normalize(answer_sentence) == gold_norm:
                sentence_exact += 1
        per_question.append({
            "qid": q["qid"],
            "question": q["question"],
            "gold_answer": q["answer_text"],
            "fused_rank": fused_rank,
            "reranked_rank": reranked_rank,
            "answer_sentence": answer_sentence,
            "latency_seconds": round(latency, 4),
        })
        if (i + 1) % 100 == 0:
            log(f"  {i + 1}/{len(questions)} questions done")

    n = len(questions)
    lat_sorted = sorted(latencies)
    p95 = lat_sorted[min(int(0.95 * n), n - 1)] if n else 0.0
    metrics = {
        "dataset": json.loads(
            (ROOT / "data/dataset_info.json").read_text(encoding="utf-8")),
        "index": store.meta,
        "num_questions": n,
        "retrieval_fused": {
            "recall@1": round(recall_at_k(fused_ranks, 1), 4),
            "recall@5": round(recall_at_k(fused_ranks, 5), 4),
            "recall@10": round(recall_at_k(fused_ranks, 10), 4),
            "recall@20": round(recall_at_k(fused_ranks, 20), 4),
            "mrr": round(sum(1.0 / r for r in fused_ranks if r) / n, 4),
        },
        "retrieval_reranked": {
            "recall@1": round(recall_at_k(reranked_ranks, 1), 4),
            "recall@5": round(recall_at_k(reranked_ranks, 5), 4),
            "mrr": round(sum(1.0 / r for r in reranked_ranks if r) / n, 4),
        },
        "answering_extractive": {
            "method": "answer sentence picked from the top reranked passage "
                      "by token overlap F1 with the question content words; "
                      "scored against the SQuAD gold answer",
            "gold_answer_in_top_passage_rate": round(containment / n, 4),
            "answer_sentence_token_f1_mean": round(
                statistics.mean(sentence_f1_scores), 4)
            if sentence_f1_scores else 0.0,
            "answer_sentence_exact_match_rate": round(sentence_exact / n, 4),
        },
        "latency_seconds_full_query": {
            "mean": round(statistics.mean(latencies), 4),
            "p95": round(p95, 4),
            "max": round(max(latencies), 4),
        },
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "cpus": __import__("os").cpu_count(),
        },
        "total_run_seconds": round(time.time() - started, 1),
    }

    outputs = ROOT / "outputs"
    outputs.mkdir(exist_ok=True)
    with (outputs / "metrics.json").open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    with (outputs / "per_question.jsonl").open("w", encoding="utf-8") as f:
        for row in per_question:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    log(json.dumps(metrics, indent=2))
    with (outputs / "run_log.txt").open("w", encoding="utf-8") as f:
        f.write("\n".join(log_lines) + "\n")
    print(f"wrote outputs to {outputs}")


if __name__ == "__main__":
    main()
