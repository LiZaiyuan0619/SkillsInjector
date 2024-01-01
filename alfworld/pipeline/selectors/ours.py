from __future__ import annotations
import numpy as np
from .base import Selector, Selection
from .cosine import CosineSelector
from ..tau_star import load_tau_star
from ..paths import ENCODER_0_6B

class OursSelector(Selector):
    name = 'ours'

    def __init__(self, pool, model_path=ENCODER_0_6B, max_budget: int=30):
        super().__init__(pool)
        self._cosine = CosineSelector(pool, k=len(pool.skills), model_path=model_path)
        self._tau = load_tau_star()
        self.max_budget = max_budget

    def warmup(self):
        self._cosine.warmup()

    def select(self, task: dict, query: str, trial_idx: int, seed: int) -> Selection:
        self.warmup()
        scores = self._cosine._scores(query)
        (s_min, s_max) = (float(scores.min()), float(scores.max()))
        if s_max - s_min < 1e-09:
            norm = np.zeros_like(scores)
        else:
            norm = (scores - s_min) / (s_max - s_min)
        keep = np.where(norm >= self._tau)[0]
        if len(keep) == 0:
            keep = np.array([int(np.argmax(scores))])
        elif len(keep) > self.max_budget:
            keep = keep[np.argsort(-scores[keep])[:self.max_budget]]
        keep = keep[np.argsort(-scores[keep])]
        chosen = [self.pool.skills[i] for i in keep]
        return Selection(skill_ids=[s.skill_id for s in chosen], method=self.name, task_id=task['task_id'], trial_idx=trial_idx, selected_skills=chosen, extra={'tau_star': self._tau, 'budget': len(chosen), 'score_range': [s_min, s_max], 'max_budget': self.max_budget})
