from __future__ import annotations
import threading
from pathlib import Path
import numpy as np
from ._qwen_embedder import QwenEmbedder, EmbedderConfig
_EMBEDDER_LOCK = threading.Lock()
_EMBEDDER_CACHE: dict[str, QwenEmbedder] = {}

def get_persistent_embedder(model_path: str | Path, cache_dir: Path | None=None, batch_size: int=32, max_length: int=8192) -> QwenEmbedder:
    key = str(Path(model_path).resolve())
    with _EMBEDDER_LOCK:
        if key not in _EMBEDDER_CACHE:
            cfg_kwargs = {'model_path': Path(model_path), 'batch_size': batch_size, 'max_length': max_length}
            if cache_dir is not None:
                cfg_kwargs['cache_dir'] = Path(cache_dir)
            cfg = EmbedderConfig(**cfg_kwargs)
            _EMBEDDER_CACHE[key] = QwenEmbedder(cfg)
        return _EMBEDDER_CACHE[key]

def embed_texts(texts: list[str], model_path: str | Path, cache_dir: Path | None=None, batch_size: int=32) -> np.ndarray:
    embedder = get_persistent_embedder(model_path, cache_dir, batch_size)
    return embedder.embed_texts(texts)
