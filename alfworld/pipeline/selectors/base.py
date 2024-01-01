from __future__ import annotations
import abc
from dataclasses import dataclass, field
from ..skill_pool import Skill, SkillPool

@dataclass
class Selection:
    skill_ids: list[str]
    method: str
    task_id: str
    trial_idx: int
    selected_skills: list[Skill] = field(default_factory=list)
    truncated_to: int | None = None
    extra: dict = field(default_factory=dict)

class Selector(abc.ABC):
    name: str

    def __init__(self, pool: SkillPool):
        self.pool = pool

    def warmup(self) -> None:
        pass

    @abc.abstractmethod
    def select(self, task: dict, query: str, trial_idx: int, seed: int) -> Selection:
        ...
