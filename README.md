# rag-enterprise-document-intelligence

I built this to answer questions over a document corpus with retrieval I can measure. It loads documents, splits them into chunks, indexes the chunks two ways, and answers a question by retrieving the most relevant chunks, reranking them, and returning the best sentence from the top chunk together with the passages it came from. Every number in this README comes from the evaluation run saved in `outputs/`. If a number is not in `outputs/metrics.json`, I do not claim it.

## How it works

1. **Load.** Documents come from a JSONL corpus file, one document per line with a title and full text.
2. **Chunk.** Each document is split on paragraph boundaries, and each paragraph is cut into sliding windows of at most 180 words with a 40 word overlap. Every chunk records its character offsets in the source document.
3. **Embed.** Chunks are embedded with `sentence-transformers/all-MiniLM-L6-v2`, a local model, and stored in a FAISS index (flat inner product over normalised vectors, so it is cosine similarity).
4. **Retrieve.** A question is answered from two candidate lists: dense search over the FAISS index and BM25 search over the chunk text. The lists are fused with reciprocal rank fusion, 30 candidates from each side, top 20 kept.
5. **Rerank.** A local cross-encoder, `cross-encoder/ms-marco-MiniLM-L-6-v2`, scores the 20 fused candidates against the question and reorders them.
6. **Answer.** The answer is one sentence picked from the top passage. I pick it by token overlap F1 between the question content words and each sentence in the passage. It is a simple heuristic and the evaluation below scores it honestly. An optional generate mode exists behind an API key (see below), and none of the measured results use it.

## Data

The corpus and the questions come from **SQuAD v1.1** (Stanford Question Answering Dataset), training split. SQuAD is Wikipedia articles with human written questions and gold answer spans. License: CC BY-SA 4.0. Source file: https://rajpurkar.github.io/SQuAD-explorer/dataset/train-v1.1.json

The full training split has 442 articles and 87,599 questions. For this run I sampled 60 whole articles with a fixed seed (13). Every paragraph of those articles went into the corpus, so the corpus holds the paragraphs the questions were written about plus the rest of the same articles as distractors. The 60 articles became 60 documents and 2,601 chunks. From the 10,997 questions that belong to the sampled articles, I sampled 300 questions, same seed, as the evaluation set. I kept the run at this size because it has to finish on a 2 vCPU machine. `data/dataset_info.json` records the seed and the exact counts, and `scripts/download_data.py` rebuilds the same files from the raw download.

## Run it

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
# torch in requirements.txt is the CPU build. If pip tries to replace it
# with a CUDA build, install it first from the CPU index:
# .venv/bin/pip install torch --index-url https://download.pytorch.org/whl/cpu

curl -L -o data/raw/train-v1.1.json https://rajpurkar.github.io/SQuAD-explorer/dataset/train-v1.1.json
.venv/bin/python scripts/download_data.py --eval-questions 300
.venv/bin/python scripts/build_index.py
.venv/bin/python scripts/run_eval.py
```

Serve the API:

```bash
.venv/bin/python -m uvicorn api.main:app --port 8000
curl -X POST localhost:8000/ingest -H 'Content-Type: application/json' -d '{"rebuild": false}'
curl -X POST localhost:8000/query -H 'Content-Type: application/json' -d '{"question": "What is the capital of Tuvalu?", "top_k": 5}'
```

`POST /ingest` builds the index from the corpus if none exists and loads the chain. `POST /query` runs the full chain and returns the answer sentence, the top passages with their titles and rerank scores, and the measured latency of that request. `GET /health` reports whether an index is loaded.

Optional generate mode: if `OPENAI_API_KEY` is set in the environment, `POST /query` with `"mode": "generate"` sends the retrieved passages to the OpenAI chat API and returns a generated answer instead of the extracted sentence. The model name comes from `OPENAI_MODEL` and defaults to `gpt-4o-mini`. I did not use this mode for any measurement in this repo.

## Measured results

One run, recorded in `outputs/metrics.json` with the per question detail in `outputs/per_question.jsonl` and the log in `outputs/run_log.txt`. Conditions: 60 documents, 2,601 chunks, 300 evaluation questions, embedding model `all-MiniLM-L6-v2`, reranker `ms-marco-MiniLM-L-6-v2`, on a virtual machine with 2 vCPUs and 7 GB of memory. The machine was also running other jobs during the run, which is part of why the latency figures look the way they do.

Retrieval, where a hit means a returned chunk covers the gold answer span:

| Stage | Recall@1 | Recall@5 | Recall@10 | Recall@20 | MRR |
|---|---|---|---|---|---|
| Hybrid fusion (dense + BM25) | 0.8033 | 0.9467 | 0.9767 | 0.9867 | 0.8670 |
| After cross-encoder reranking | 0.9033 | 0.9800 | n/a | n/a | 0.9355 |

The reranker earns its place: Recall@1 goes from 0.8033 to 0.9033.

Answering, extractive mode:

- The gold answer text appears in the top reranked passage for 0.9367 of the questions.
- The sentence I extract contains the gold answer for 0.7167 of the questions (computed from `outputs/per_question.jsonl`).
- The extracted sentence scores a mean token F1 of 0.1543 against the gold answer span, and an exact match rate of 0.0. This is the honest weak point. The gold answers in SQuAD are short spans, often two or three words, and I return a full sentence, so span level scores stay low even when the right sentence is picked. A span extraction model is the clear next step; I have not built one yet, and I am not going to dress the sentence numbers up as span numbers.

Latency, full query path (retrieve, rerank, extract), timed inside the evaluation run: mean 9.0 seconds, p95 13.0 seconds, max 16.7 seconds per question on the 2 vCPU machine described above. The whole 300 question evaluation took 2,711 seconds of wall time. Most of the per query time is the cross-encoder scoring 20 candidates on a small shared CPU. I have not load tested this service and I make no claim about concurrency or uptime. It runs locally under uvicorn; that is the extent of the deployment story.

## Layout

- `src/loader.py` corpus and evaluation file loading
- `src/chunker.py` paragraph aware word window chunking with character offsets
- `src/embeddings.py` the local embedding model wrapper
- `src/store.py` FAISS and BM25 build, save, load, and search
- `src/retriever.py` hybrid retrieval with reciprocal rank fusion
- `src/reranker.py` cross-encoder reranking
- `src/chain.py` the question answering chain, extractive by default
- `api/main.py` the FastAPI service (`/health`, `/ingest`, `/query`)
- `scripts/download_data.py` builds the corpus and evaluation set from raw SQuAD
- `scripts/build_index.py` chunks, embeds, and writes the index under `index/`
- `scripts/run_eval.py` runs the evaluation and writes `outputs/`
- `tests/test_pipeline.py` smoke tests for chunking, tokenising, and sentence picking

## What this is not

It is not a hosted product. There is no cloud deployment in this repo, no user accounts, and no benchmark against any external system. The corpus is a public research dataset, not private enterprise documents; the pipeline is document agnostic, but I have only run it on SQuAD, so SQuAD is all I claim.
