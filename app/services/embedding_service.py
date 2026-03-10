from __future__ import annotations

from threading import Lock


class EmbeddingService:
    def __init__(self, model_name: str) -> None:
        self._model_name = model_name
        self._model = None
        self._lock = Lock()

    @property
    def is_loaded(self) -> bool:
        return self._model is not None

    def load(self):
        if self._model is not None:
            return self._model
        with self._lock:
            if self._model is None:
                from sentence_transformers import SentenceTransformer

                self._model = SentenceTransformer(self._model_name)
        return self._model

    def encode(self, texts: list[str]) -> list[list[float]]:
        model = self.load()
        vectors = model.encode(texts, normalize_embeddings=True)
        return vectors.tolist()

