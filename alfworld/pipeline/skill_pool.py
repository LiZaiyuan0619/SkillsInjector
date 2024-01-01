from __future__ import annotations
import hashlib
import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from .paths import STATE_DIR
log = logging.getLogger(__name__)

@dataclass
class Skill:
    skill_id: str
    display_name: str
    source_task_type: str
    sha8: str
    char_count: int
    body: str = ''

@dataclass
class SkillPool:
    skills: list[Skill]
    meta: dict
    fingerprint: str

    def by_id(self, skill_id: str) -> Skill | None:
        return next((s for s in self.skills if s.skill_id == skill_id), None)

    def by_type(self, task_type: str) -> list[Skill]:
        return [s for s in self.skills if s.source_task_type == task_type]

def load_skill_pool(state_dir: Path | None=None) -> SkillPool:
    base = state_dir or STATE_DIR
    pool_file = base / 'skill_pool.json'
    if not pool_file.exists():
        raise FileNotFoundError(f'{pool_file} missing — run distill (Phase 1) first')
    data = json.loads(pool_file.read_text())
    skills = []
    for s in data['skills']:
        body_path = base / 'skills' / s['skill_id'] / 'SKILL.md'
        body = body_path.read_text() if body_path.exists() else ''
        skills.append(Skill(skill_id=s['skill_id'], display_name=s['display_name'], source_task_type=s['source_task_type'], sha8=s['sha8'], char_count=s['char_count'], body=body))
    fp_blob = json.dumps({'meta': data['meta'], 'skill_sha8s': sorted((s.sha8 for s in skills))}, sort_keys=True)
    fp = hashlib.sha256(fp_blob.encode()).hexdigest()[:16]
    return SkillPool(skills=skills, meta=data['meta'], fingerprint=fp)
