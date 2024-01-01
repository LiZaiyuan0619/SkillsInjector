from __future__ import annotations
import hashlib
import threading
from pathlib import Path
import numpy as np
from ..embedding import get_persistent_embedder
from ..paths import ENCODER_0_6B, STATE_DIR
from .base import Selector, Selection

class CosineSelector(Selector):
    name = 'cosine'

    def __init__(self, pool, k: int=3, model_path: str | Path=ENCODER_0_6B):
        super().__init__(pool)
        self.k = k
        self.model_path = model_path
        self._embedder = get_persistent_embedder(model_path, cache_dir=STATE_DIR / 'embeddings/skills_0.6b')
        self._skill_emb: np.ndarray | None = None
        self._skill_ids: list[str] = []
        self._query_cache: dict[str, np.ndarray] = {}
        self._cache_lock = threading.Lock()
        self._warmup_lock = threading.Lock()

    def warmup(self):
        with self._warmup_lock:
            if self._skill_emb is not None:
                return
            skill_texts = [s.body for s in self.pool.skills]
            self._skill_ids = [s.skill_id for s in self.pool.skills]
            emb = self._embedder.embed_texts(skill_texts)
            self._skill_emb = self._normalize(emb)

    @staticmethod
    def _normalize(x: np.ndarray) -> np.ndarray:
        return x / (np.linalg.norm(x, axis=-1, keepdims=True) + 1e-09)

    def _embed_query(self, query: str) -> np.ndarray:
        h = hashlib.sha256(query.encode()).hexdigest()[:16]
        cached = self._query_cache.get(h)
        if cached is not None:
            return cached
        with self._cache_lock:
            cached = self._query_cache.get(h)
            if cached is not None:
                return cached
            e = self._embedder.embed_texts([query])[0]
            e = self._normalize(e[None])[0]
            self._query_cache[h] = e
            return e

    def _scores(self, query: str) -> np.ndarray:
        q = self._embed_query(query)
        return self._skill_emb @ q

    def select(self, task: dict, query: str, trial_idx: int, seed: int) -> Selection:
        self.warmup()
        scores = self._scores(query)
        order = np.argsort(-scores)[:self.k].tolist()
        chosen = [self.pool.skills[i] for i in order]
        return Selection(skill_ids=[s.skill_id for s in chosen], method=self.name, task_id=task['task_id'], trial_idx=trial_idx, selected_skills=chosen, extra={'top_scores': [float(scores[i]) for i in order]})
