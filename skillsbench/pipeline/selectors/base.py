from __future__ import annotations
import abc
from dataclasses import dataclass, field
from ..manifest import Manifest
from ..skill_pool import SkillPool

@dataclass
class Selection:
    skill_ids: list[str]
    method: str
    task_id: str
    trial_idx: int
    truncated_to: int | None = None
    extra: dict = field(default_factory=dict)

class Selector(abc.ABC):
    name: str

    @abc.abstractmethod
    def select(self, task_id: str, manifest: Manifest, pool: SkillPool, trial_idx: int, seed: int) -> Selection:
        pass
