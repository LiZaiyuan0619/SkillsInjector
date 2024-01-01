import random
from .base import Selector, Selection

class RandomSelector(Selector):
    name = 'random'

    def __init__(self, pool, k: int=3):
        super().__init__(pool)
        self.k = k

    def select(self, task: dict, query: str, trial_idx: int, seed: int) -> Selection:
        rng = random.Random(seed)
        bucket = self.pool.by_type(task['task_type'])
        if len(bucket) >= self.k:
            chosen = rng.sample(bucket, self.k)
        else:
            other = [s for s in self.pool.skills if s not in bucket]
            need = self.k - len(bucket)
            top_up = rng.sample(other, min(need, len(other)))
            chosen = list(bucket) + top_up
        return Selection(skill_ids=[s.skill_id for s in chosen], method=self.name, task_id=task['task_id'], trial_idx=trial_idx, selected_skills=chosen, extra={'bucket_size': len(bucket), 'task_type': task['task_type']})
