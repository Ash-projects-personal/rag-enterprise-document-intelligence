"""Load the processed corpus and evaluation set built by scripts/download_data.py.

The corpus file is JSONL with one document per line:
    {"doc_id": str, "title": str, "text": str}

The evaluation file is JSONL with one question per line:
    {"qid": str, "doc_id": str, "title": str, "question": str,
     "answer_text": str, "answer_start": int, "answer_end": int}

answer_start and answer_end are character offsets into the document text.
"""

import json
from pathlib import Path


def load_jsonl(path):
    path = Path(path)
    rows = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def load_corpus(path):
    return load_jsonl(path)


def load_eval_questions(path):
    return load_jsonl(path)
