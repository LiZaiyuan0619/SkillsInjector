from __future__ import annotations
import datetime as _dt
import json
import logging
import shutil
from dataclasses import asdict, dataclass
from pathlib import Path
from .paths import EXCLUDED_TASKS, MANIFEST_PATH, STATE_DIR, TASKS_DIR
logger = logging.getLogger(__name__)
REQUIRED_FILES = ('instruction.md', 'task.toml', 'environment/Dockerfile', 'tests/test.sh')

@dataclass
class TaskEntry:
    task_id: str
    task_dir: str
    has_skills: bool
    skill_dir_names: list[str]

@dataclass
class Manifest:
    schema_version: int
    built_at: str
    tasks_dir: str
    n_tasks_total: int
    n_excluded: int
    n_kept: int
    excluded: list[dict]
    tasks: list[TaskEntry]

def _check_required(task_dir: Path) -> str | None:
    for rel in REQUIRED_FILES:
        if not (task_dir / rel).is_file():
            return f'missing {rel}'
    return None

def _list_skill_dirs(task_dir: Path) -> list[str]:
    skills_dir = task_dir / 'environment' / 'skills'
    if not skills_dir.is_dir():
        return []
    return sorted((d.name for d in skills_dir.iterdir() if d.is_dir() and (d / 'SKILL.md').is_file()))

def build_manifest(tasks_dir: Path=TASKS_DIR) -> Manifest:
    if not tasks_dir.is_dir():
        raise FileNotFoundError(f'tasks_dir does not exist: {tasks_dir}')
    all_dirs = sorted((d for d in tasks_dir.iterdir() if d.is_dir()))
    excluded: list[dict] = []
    kept: list[TaskEntry] = []
    for d in all_dirs:
        tid = d.name
        if tid in EXCLUDED_TASKS:
            excluded.append({'task_id': tid, 'reason': 'configured exclusion (not self-contained)'})
            continue
        missing = _check_required(d)
        if missing:
            excluded.append({'task_id': tid, 'reason': f'structural: {missing}'})
            continue
        skill_dirs = _list_skill_dirs(d)
        kept.append(TaskEntry(task_id=tid, task_dir=str(d.resolve()), has_skills=bool(skill_dirs), skill_dir_names=skill_dirs))
    manifest = Manifest(schema_version=1, built_at=_dt.datetime.now(_dt.timezone.utc).isoformat(timespec='seconds'), tasks_dir=str(tasks_dir.resolve()), n_tasks_total=len(all_dirs), n_excluded=len(excluded), n_kept=len(kept), excluded=excluded, tasks=kept)
    return manifest

def save_manifest(manifest: Manifest, path: Path=MANIFEST_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        backup = path.with_suffix(path.suffix + f".bak.{_dt.datetime.now().strftime('%Y%m%d-%H%M%S')}")
        shutil.copy2(path, backup)
        logger.info('backed up old manifest to %s', backup)
    payload = asdict(manifest)
    payload['tasks'] = [asdict(t) for t in manifest.tasks]
    path.write_text(json.dumps(payload, indent=2))
    logger.info('wrote %s (%d tasks kept, %d excluded)', path, manifest.n_kept, manifest.n_excluded)

def load_manifest(path: Path=MANIFEST_PATH) -> Manifest:
    raw = json.loads(path.read_text())
    raw['tasks'] = [TaskEntry(**t) for t in raw['tasks']]
    return Manifest(**raw)
