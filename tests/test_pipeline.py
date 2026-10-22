"""Smoke tests for the parts of the pipeline that need no models.

Run with: python -m pytest tests/ -q
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.chunker import chunk_document  # noqa: E402
from src.chain import extract_answer_sentence, token_f1  # noqa: E402
from src.store import tokenize  # noqa: E402


def test_chunk_offsets_cover_answer_span():
    text = ("First paragraph has some words in it.\n\n"
            "Second paragraph says the capital of France is Paris today.")
    chunks = chunk_document("doc-x", "Test", text, max_words=8,
                            overlap_words=2)
    assert chunks
    for chunk in chunks:
        assert text[chunk["char_start"]:chunk["char_end"]] == chunk["text"]
    answer_start = text.index("Paris")
    covering = [c for c in chunks
                if c["char_start"] <= answer_start < c["char_end"]]
    assert covering


def test_tokenize_lowercases_and_splits():
    assert tokenize("Hello, World 42!") == ["hello", "world", "42"]


def test_token_f1_identical_is_one():
    tokens = tokenize("the answer is paris")
    assert token_f1(tokens, tokens) == 1.0


def test_extract_answer_picks_relevant_sentence():
    passage = ("The weather in Oslo was cold all week. "
               "The capital of Norway is Oslo. "
               "Trains run on time in winter.")
    picked = extract_answer_sentence("What is the capital of Norway?",
                                     passage)
    assert "capital of Norway" in picked
