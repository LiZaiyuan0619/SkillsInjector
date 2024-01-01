from __future__ import annotations
import hashlib
import random as _random
from ..manifest import Manifest
from ..skill_pool import SkillPool
from .base import Selection, Selector

def _per_task_seed(global_seed: int, task_id: str, trial_idx: int) -> int:
    h = hashlib.sha256(f'{global_seed}|{task_id}|{trial_idx}'.encode()).hexdigest()
    return int(h[:16], 16)

class RandomSelector(Selector):
    name = 'random'

    def __init__(self, k: int=3):
        self.k = k

    def select(self, task_id: str, manifest: Manifest, pool: SkillPool, trial_idx: int, seed: int) -> Selection:
        all_ids = sorted(pool.skills.keys())
        if not all_ids:
            return Selection(skill_ids=[], method=self.name, task_id=task_id, trial_idx=trial_idx)
        rng = _random.Random(_per_task_seed(seed, task_id, trial_idx))
        k = min(self.k, len(all_ids))
        sampled = rng.sample(all_ids, k)
        return Selection(skill_ids=sampled, method=self.name, task_id=task_id, trial_idx=trial_idx, extra={'k': k, 'pool_size': len(all_ids)})
