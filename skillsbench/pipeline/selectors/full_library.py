from __future__ import annotations
from ..manifest import Manifest
from ..skill_pool import SkillPool
from .base import Selection, Selector
from .cosine import CosineSelector
DEFAULT_TOKEN_BUDGET = 96000
CHARS_PER_TOKEN = 4

class FullLibrarySelector(Selector):
    name = 'full-library'

    def __init__(self, token_budget: int=DEFAULT_TOKEN_BUDGET, cosine_selector: CosineSelector | None=None):
        self.token_budget = token_budget
        self.char_budget = token_budget * CHARS_PER_TOKEN
        self._cosine = cosine_selector or CosineSelector(k=1)

    def select(self, task_id: str, manifest: Manifest, pool: SkillPool, trial_idx: int, seed: int) -> Selection:
        all_ids = sorted(pool.skills.keys())
        if not all_ids:
            return Selection(skill_ids=[], method=self.name, task_id=task_id, trial_idx=trial_idx)
        self._cosine.warmup(manifest, pool)
        (skill_ids, skill_emb) = self._cosine.get_skill_emb_view()
        q = self._cosine.get_task_emb(task_id, manifest)
        sims = skill_emb @ q
        order = sims.argsort()[::-1]
        kept_ids: list[str] = []
        kept_chars = 0
        for i in order:
            sid = skill_ids[i]
            n = pool.skills[sid].n_chars
            if kept_chars + n > self.char_budget and kept_ids:
                break
            kept_ids.append(sid)
            kept_chars += n
        truncated_to = len(kept_ids) if len(kept_ids) < len(skill_ids) else None
        return Selection(skill_ids=kept_ids, method=self.name, task_id=task_id, trial_idx=trial_idx, truncated_to=truncated_to, extra={'pool_size': len(skill_ids), 'kept': len(kept_ids), 'kept_chars': kept_chars, 'char_budget': self.char_budget, 'token_budget': self.token_budget})
