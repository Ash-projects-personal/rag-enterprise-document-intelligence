"""Paragraph aware fixed size chunking.

Strategy, exactly as implemented here:
- A document is split into paragraphs on blank lines.
- Each paragraph is split into sliding windows of at most ``max_words``
  words, advancing by ``max_words - overlap_words`` words each step.
- A paragraph shorter than the window becomes a single chunk.
- Every chunk records the character offsets of its first and last word
  inside the original document text, so an evaluation can tell whether a
  chunk covers a given character span (for example a gold answer).
"""

import re

WORD_RE = re.compile(r"\S+")

DEFAULT_MAX_WORDS = 180
DEFAULT_OVERLAP_WORDS = 40


def _word_spans(text):
    return [(m.group(0), m.start(), m.end()) for m in WORD_RE.finditer(text)]


def chunk_document(doc_id, title, text, max_words=DEFAULT_MAX_WORDS,
                   overlap_words=DEFAULT_OVERLAP_WORDS):
    if max_words <= overlap_words:
        raise ValueError("max_words must be greater than overlap_words")
    step = max_words - overlap_words
    chunks = []
    para_offset = 0
    for para in text.split("\n\n"):
        words = _word_spans(para)
        if words:
            start_idx = 0
            while start_idx < len(words):
                window = words[start_idx:start_idx + max_words]
                char_start = para_offset + window[0][1]
                char_end = para_offset + window[-1][2]
                chunk_text = para[window[0][1]:window[-1][2]]
                chunks.append({
                    "chunk_id": f"{doc_id}::chunk-{len(chunks)}",
                    "doc_id": doc_id,
                    "title": title,
                    "text": chunk_text,
                    "char_start": char_start,
                    "char_end": char_end,
                })
                if start_idx + max_words >= len(words):
                    break
                start_idx += step
        para_offset += len(para) + 2  # account for the "\n\n" separator
    return chunks


def chunk_corpus(documents, max_words=DEFAULT_MAX_WORDS,
                 overlap_words=DEFAULT_OVERLAP_WORDS):
    all_chunks = []
    for doc in documents:
        all_chunks.extend(chunk_document(
            doc["doc_id"], doc["title"], doc["text"],
            max_words=max_words, overlap_words=overlap_words))
    return all_chunks
