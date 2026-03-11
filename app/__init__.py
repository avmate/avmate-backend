from __future__ import annotations

__all__ = ["app"]


def __getattr__(name: str):
    if name == "app":
        from .main import app as fastapi_app

        return fastapi_app
    raise AttributeError(name)
