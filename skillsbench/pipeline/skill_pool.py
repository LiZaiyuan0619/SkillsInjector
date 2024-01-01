from __future__ import annotations
import datetime as _dt
import hashlib
import json
import logging
from dataclasses import asdict, dataclass, field
from pathlib import Path
from .manifest import Manifest, load_manifest
from .paths import SKILL_POOL_PATH, STATE_DIR
logger = logging.getLogger(__name__)

def _sha8(text: str) -> str:
    return hashlib.sha256(text.encode('utf-8')).hexdigest()[:8]

@dataclass
class SkillEntry:
    skill_id: str
    display_name: str
    sha8: str
    n_chars: int
    n_tokens_approx: int
    source_tasks: list[str]
    source_paths: list[str]
    body: str

@dataclass
class SkillPool:
    schema_version: int
    built_at: str
    n_files_seen: int
    n_unique_skills: int
    skills: dict[str, SkillEntry]

def build_skill_pool(manifest: Manifest) -> SkillPool:
    seen_files = 0
    by_id: dict[str, SkillEntry] = {}
    for t in manifest.tasks:
        task_dir = Path(t.task_dir)
        for sname in t.skill_dir_names:
            skill_md = task_dir / 'environment' / 'skills' / sname / 'SKILL.md'
            if not skill_md.is_file():
                continue
            content = skill_md.read_text(errors='replace')
            digest = _sha8(content)
            sid = f'{sname}__{digest}'
            seen_files += 1
            if sid in by_id:
                e = by_id[sid]
                if t.task_id not in e.source_tasks:
                    e.source_tasks.append(t.task_id)
                e.source_paths.append(str(skill_md))
            else:
                by_id[sid] = SkillEntry(skill_id=sid, display_name=sname, sha8=digest, n_chars=len(content), n_tokens_approx=max(1, len(content) // 4), source_tasks=[t.task_id], source_paths=[str(skill_md)], body=content)
    pool = SkillPool(schema_version=1, built_at=_dt.datetime.now(_dt.timezone.utc).isoformat(timespec='seconds'), n_files_seen=seen_files, n_unique_skills=len(by_id), skills=dict(sorted(by_id.items())))
    return pool

def save_skill_pool(pool: SkillPool, path: Path=SKILL_POOL_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {'schema_version': pool.schema_version, 'built_at': pool.built_at, 'n_files_seen': pool.n_files_seen, 'n_unique_skills': pool.n_unique_skills, 'skills': {sid: asdict(e) for (sid, e) in pool.skills.items()}}
    path.write_text(json.dumps(payload, indent=2))
    logger.info('wrote %s (%d unique skills from %d files)', path, pool.n_unique_skills, pool.n_files_seen)

def load_skill_pool(path: Path=SKILL_POOL_PATH) -> SkillPool:
    raw = json.loads(path.read_text())
    skills = {sid: SkillEntry(**e) for (sid, e) in raw.pop('skills').items()}
    return SkillPool(skills=skills, **raw)
