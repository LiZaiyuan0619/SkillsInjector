from __future__ import annotations
from ..manifest import Manifest
from ..skill_pool import SkillPool
from .base import Selection, Selector

class NoSkillSelector(Selector):
    name = 'no-skill'

    def select(self, task_id: str, manifest: Manifest, pool: SkillPool, trial_idx: int, seed: int) -> Selection:
        return Selection(skill_ids=[], method=self.name, task_id=task_id, trial_idx=trial_idx)
