from __future__ import annotations
from pathlib import Path
import numpy as np
from ..embedding import QwenEmbedder
from ..manifest import Manifest
from ..skill_pool import SkillPool
from .base import Selection, Selector

class CosineSelector(Selector):
    name = 'cosine'

    def __init__(self, k: int=3, embedder: QwenEmbedder | None=None):
        self.k = k
        self._embedder = embedder or QwenEmbedder()
        self._skill_ids: list[str] = []
        self._skill_emb: np.ndarray | None = None
        self._task_emb_cache: dict[str, np.ndarray] = {}

    def warmup(self, manifest: Manifest, pool: SkillPool) -> None:
        new_ids = sorted(pool.skills.keys())
        if self._skill_emb is not None and self._skill_ids == new_ids:
            return
        self._skill_ids = new_ids
        skill_texts = [pool.skills[sid].body for sid in self._skill_ids]
        self._skill_emb = self._embedder.embed_texts(skill_texts)
        task_texts = []
        task_ids = [t.task_id for t in manifest.tasks]
        for t in manifest.tasks:
            task_texts.append((Path(t.task_dir) / 'instruction.md').read_text(errors='replace'))
        task_emb = self._embedder.embed_texts(task_texts)
        for (tid, e) in zip(task_ids, task_emb):
            self._task_emb_cache[tid] = e

    def _ensure(self, manifest: Manifest, pool: SkillPool) -> None:
        if self._skill_emb is None:
            self.warmup(manifest, pool)

    def select(self, task_id: str, manifest: Manifest, pool: SkillPool, trial_idx: int, seed: int) -> Selection:
        self._ensure(manifest, pool)
        if not self._skill_ids:
            return Selection(skill_ids=[], method=self.name, task_id=task_id, trial_idx=trial_idx)
        if task_id not in self._task_emb_cache:
            for t in manifest.tasks:
                if t.task_id == task_id:
                    text = (Path(t.task_dir) / 'instruction.md').read_text(errors='replace')
                    self._task_emb_cache[task_id] = self._embedder.embed_texts([text])[0]
                    break
            else:
                raise KeyError(f'task_id {task_id!r} not in manifest')
        q = self._task_emb_cache[task_id]
        assert self._skill_emb is not None
        sims = self._skill_emb @ q
        order = np.argsort(-sims)
        k = min(self.k, len(self._skill_ids))
        top_idx = order[:k]
        top_ids = [self._skill_ids[i] for i in top_idx]
        return Selection(skill_ids=top_ids, method=self.name, task_id=task_id, trial_idx=trial_idx, extra={'k': k, 'pool_size': len(self._skill_ids), 'top_scores': [round(float(sims[i]), 4) for i in top_idx]})

    def get_skill_emb_view(self) -> tuple[list[str], np.ndarray]:
        if self._skill_emb is None:
            raise RuntimeError('call warmup(...) before get_skill_emb_view()')
        return (list(self._skill_ids), self._skill_emb)

    def get_task_emb(self, task_id: str, manifest: Manifest) -> np.ndarray:
        self._ensure_task(task_id, manifest)
        return self._task_emb_cache[task_id]

    def _ensure_task(self, task_id: str, manifest: Manifest) -> None:
        if task_id in self._task_emb_cache:
            return
        for t in manifest.tasks:
            if t.task_id == task_id:
                text = (Path(t.task_dir) / 'instruction.md').read_text(errors='replace')
                self._task_emb_cache[task_id] = self._embedder.embed_texts([text])[0]
                return
        raise KeyError(f'task_id {task_id!r} not in manifest')
