"""Embedding wrapper around a local sentence-transformers model.

Vectors are L2 normalised, so inner product in FAISS equals cosine
similarity. No API key is involved; the model runs on this machine.
"""

from sentence_transformers import SentenceTransformer

DEFAULT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


class Embedder:
    def __init__(self, model_name=DEFAULT_MODEL):
        self.model_name = model_name
        self.model = SentenceTransformer(model_name)

    @property
    def dim(self):
        return self.model.get_sentence_embedding_dimension()

    def encode(self, texts, batch_size=64):
        return self.model.encode(
            texts,
            batch_size=batch_size,
            normalize_embeddings=True,
            show_progress_bar=False,
        ).astype("float32")

    def encode_one(self, text):
        return self.encode([text])[0]
