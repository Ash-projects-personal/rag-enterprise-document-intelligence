"""
pytest configuration for rag-enterprise-document-intelligence tests.

Makes the repo's src/ tree importable as a flat ``src.<pkg>`` namespace so
tests can write ``from src.retrieval.hybrid_retriever import HybridRetriever``
without needing a wheel install.
"""
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
