from .base import Selector, Selection

class NoSkillSelector(Selector):
    name = 'no-skill'

    def select(self, task: dict, query: str, trial_idx: int, seed: int) -> Selection:
        return Selection(skill_ids=[], method=self.name, task_id=task['task_id'], trial_idx=trial_idx, selected_skills=[])
