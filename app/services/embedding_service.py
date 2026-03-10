from __future__ import annotations

from threading import Lock, Thread


class EmbeddingService:
    def __init__(self, model_name: str) -> None:
        self._model_name = model_name
        self._model = None
        self._lock = Lock()
        self._loading = False
        self._load_error: str | None = None

    @property
    def is_loaded(self) -> bool:
        return self._model is not None

    @property
    def is_loading(self) -> bool:
        return self._loading

    @property
    def load_error(self) -> str | None:
        return self._load_error

    def load(self):
        if self._model is not None:
            return self._model
        with self._lock:
            if self._model is None:
                self._loading = True
                self._load_error = None
                try:
                    from sentence_transformers import SentenceTransformer

                    self._model = SentenceTransformer(self._model_name)
                except Exception as exc:
                    self._load_error = str(exc)
                    raise
                finally:
                    self._loading = False
        return self._model

    def ensure_loading(self) -> None:
        if self.is_loaded or self.is_loading:
            return
        Thread(target=self._load_safely, daemon=True).start()

    def _load_safely(self) -> None:
        try:
            self.load()
        except Exception as exc:
            print(f"Embedding model warmup failed: {exc}")

    def encode(self, texts: list[str], batch_size: int | None = None) -> list[list[float]]:
        model = self.load()
        kwargs = {"normalize_embeddings": True}
        if batch_size:
            kwargs["batch_size"] = batch_size
        vectors = model.encode(texts, **kwargs)
        return vectors.tolist()
