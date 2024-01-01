from __future__ import annotations
import numpy as np
from ..manifest import Manifest
from ..skill_pool import SkillPool
from .base import Selection, Selector
from .cosine import CosineSelector
DEFAULT_TAU_STAR = 0.75
DEFAULT_MIN_BUDGET = 1
DEFAULT_MAX_BUDGET = 15
SCORE_RANGE_EPS = 1e-06

class OursSelector(Selector):
    name = 'ours'

    def __init__(self, tau_star: float=DEFAULT_TAU_STAR, min_budget: int=DEFAULT_MIN_BUDGET, max_budget: int=DEFAULT_MAX_BUDGET, cosine_selector: CosineSelector | None=None):
        self.tau_star = tau_star
        self.min_budget = min_budget
        self.max_budget = max_budget
        self._cosine = cosine_selector or CosineSelector(k=1)

    def select(self, task_id: str, manifest: Manifest, pool: SkillPool, trial_idx: int, seed: int) -> Selection:
        self._cosine.warmup(manifest, pool)
        (skill_ids, skill_emb) = self._cosine.get_skill_emb_view()
        if not skill_ids:
            return Selection(skill_ids=[], method=self.name, task_id=task_id, trial_idx=trial_idx)
        q = self._cosine.get_task_emb(task_id, manifest)
        sims = skill_emb @ q
        s_min = float(sims.min())
        s_max = float(sims.max())
        score_range = s_max - s_min
        cap_hit = False
        floor_hit = False
        degenerate = score_range < SCORE_RANGE_EPS
        order = np.argsort(-sims, kind='stable')
        if degenerate:
            k = min(3, len(skill_ids))
            chosen = order[:k]
            extra = {'tau_star': self.tau_star, 'score_min': s_min, 'score_max': s_max, 'degenerate': True, 'budget': k, 'cap_hit': False, 'floor_hit': False}
        else:
            norm = (sims - s_min) / score_range
            mask = norm >= self.tau_star
            chosen = order[mask[order]]
            if len(chosen) < self.min_budget:
                floor_hit = True
                chosen = order[:self.min_budget]
            elif len(chosen) > self.max_budget:
                cap_hit = True
                chosen = chosen[:self.max_budget]
            extra = {'tau_star': self.tau_star, 'score_min': s_min, 'score_max': s_max, 'degenerate': False, 'budget': int(len(chosen)), 'cap_hit': cap_hit, 'floor_hit': floor_hit, 'top_norm_scores': [round(float(norm[i]), 4) for i in chosen[:5]]}
        chosen_ids = [skill_ids[i] for i in chosen]
        return Selection(skill_ids=chosen_ids, method=self.name, task_id=task_id, trial_idx=trial_idx, extra=extra)
