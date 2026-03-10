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

    def encode(self, texts: list[str], batch_size: int | None = None) -> list[list[float]]:
        model = self.load()
        kwargs = {"normalize_embeddings": True}
        if batch_size:
            kwargs["batch_size"] = batch_size
        vectors = model.encode(texts, **kwargs)
        return vectors.tolist()
