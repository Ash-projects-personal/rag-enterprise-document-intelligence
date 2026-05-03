"""
Semantic chunking with configurable overlap.
Splits documents into chunks optimised for embedding and retrieval.
"""
from typing import List
from dataclasses import dataclass, field


@dataclass
class Chunk:
    text: str
    doc_id: str
    chunk_index: int
    metadata: dict = field(default_factory=dict)


class SemanticChunker:
    """
    Splits documents into overlapping chunks.
    Default: 512 tokens per chunk, 64-token overlap.
    """

    def __init__(self, chunk_size: int = 512, overlap: int = 64):
        self.chunk_size = chunk_size
        self.overlap = overlap

    def _word_chunks(self, text: str) -> List[str]:
        words = text.split()
        chunks = []
        start = 0
        while start < len(words):
            end = min(start + self.chunk_size, len(words))
            chunks.append(" ".join(words[start:end]))
            start += self.chunk_size - self.overlap
        return chunks

    def chunk_document(self, doc_id: str, text: str, metadata: dict = None) -> List[Chunk]:
        raw_chunks = self._word_chunks(text)
        return [
            Chunk(
                text=chunk,
                doc_id=doc_id,
                chunk_index=i,
                metadata={**(metadata or {}), "chunk_index": i, "total_chunks": len(raw_chunks)},
            )
            for i, chunk in enumerate(raw_chunks)
            if chunk.strip()
        ]

    def chunk_documents(self, documents) -> List[Chunk]:
        all_chunks = []
        for doc in documents:
            all_chunks.extend(self.chunk_document(doc.doc_id, doc.content, doc.metadata))
        print(f"[INFO] Generated {len(all_chunks)} chunks from {len(documents)} documents")
        return all_chunks
