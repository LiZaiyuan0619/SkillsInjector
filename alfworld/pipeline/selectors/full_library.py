from .base import Selector, Selection

class FullLibrarySelector(Selector):
    name = 'full-library'

    def select(self, task: dict, query: str, trial_idx: int, seed: int) -> Selection:
        chosen = list(self.pool.skills)
        return Selection(skill_ids=[s.skill_id for s in chosen], method=self.name, task_id=task['task_id'], trial_idx=trial_idx, selected_skills=chosen, truncated_to=None, extra={'pool_size': len(chosen), 'pool_token_count': self.pool.meta.get('full_library_token_count')})
