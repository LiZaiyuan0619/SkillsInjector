from __future__ import annotations
import importlib
import json
import logging
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict
from pathlib import Path
from typing import Iterable
from .manifest import Manifest, load_manifest
from .methods import METHODS, MethodSpec
from .paths import JOBS_DIR, MANIFEST_PATH, SKILL_POOL_PATH
from .selectors.base import Selection, Selector
from .selectors.cosine import CosineSelector
from .skill_pool import SkillPool, load_skill_pool
logger = logging.getLogger(__name__)

def _import_class(dotted: str) -> type:
    (mod_path, _, cls_name) = dotted.rpartition('.')
    module = importlib.import_module(mod_path)
    return getattr(module, cls_name)

def _build_selectors(methods: list[MethodSpec], shared_cosine: CosineSelector) -> dict[str, Selector]:
    out: dict[str, Selector] = {}
    for spec in methods:
        cls = _import_class(spec.selector_class)
        if spec.name == 'cosine':
            out[spec.name] = shared_cosine
        elif spec.name in ('full-library', 'ours'):
            out[spec.name] = cls(cosine_selector=shared_cosine)
        elif spec.name == 'random':
            out[spec.name] = cls(k=spec.k or 3)
        elif spec.name == 'bm25':
            out[spec.name] = cls(k=spec.k or 3)
        else:
            out[spec.name] = cls()
    return out

def _bodies_for(skill_ids: list[str], pool: SkillPool) -> list[tuple[str, str]]:
    out: list[tuple[str, str]] = []
    for sid in skill_ids:
        e = pool.skills.get(sid)
        if e is None:
            logger.warning('selection contained unknown skill_id %s — skipping', sid)
            continue
        out.append((e.display_name, e.body))
    return out

def _task_dir_for(task_id: str, manifest: Manifest) -> Path:
    for t in manifest.tasks:
        if t.task_id == task_id:
            return Path(t.task_dir)
    raise KeyError(f'task_id {task_id!r} not in manifest')

def run_one_trial(method_name: str, task_id: str, trial_idx: int, *, manifest: Manifest, pool: SkillPool, selectors: dict[str, Selector], output_dir: Path, seed: int=42, max_steps: int=30) -> dict:
    from runner.runner import TrialSpec, run_trial
    if method_name not in selectors:
        raise KeyError(f'unknown method {method_name!r}; available: {list(selectors)}')
    selector = selectors[method_name]
    selection = selector.select(task_id, manifest, pool, trial_idx, seed)
    bodies = _bodies_for(selection.skill_ids, pool)
    task_dir = _task_dir_for(task_id, manifest)
    spec = TrialSpec(task_id=task_id, task_dir=task_dir, method=method_name, trial_idx=trial_idx, max_steps=max_steps, skill_bodies=bodies, skill_ids=selection.skill_ids, truncated_to=selection.truncated_to)
    method_subdir = output_dir / method_name
    record = run_trial(spec, method_subdir)
    record['selection_meta'] = {'method': selection.method, 'skill_ids': selection.skill_ids, 'truncated_to': selection.truncated_to, 'extra': selection.extra}
    out_json = method_subdir / f'{task_id}__{method_name}__t{trial_idx}.json'
    if out_json.is_file():
        out_json.write_text(json.dumps(record, indent=2))
    return record

def run_grid(methods: list[str], task_ids: list[str], trial_indices: Iterable[int], output_dir: Path, *, manifest: Manifest | None=None, pool: SkillPool | None=None, seed: int=42, max_steps: int=50, fail_fast: bool=False, concurrency: int=4) -> list[dict]:
    manifest = manifest or load_manifest()
    pool = pool or load_skill_pool()
    method_specs = [METHODS[m] for m in methods]
    shared_cosine = CosineSelector(k=3)
    if any((m.name in ('cosine', 'ours', 'full-library') for m in method_specs)):
        logger.info('warming up shared cosine selector...')
        shared_cosine.warmup(manifest, pool)
    selectors = _build_selectors(method_specs, shared_cosine)
    trial_indices = list(trial_indices)
    output_dir.mkdir(parents=True, exist_ok=True)
    tasks_3 = [(m, t, i) for m in methods for t in task_ids for i in trial_indices]
    total = len(tasks_3)
    logger.info('orchestrator: %d trials, concurrency=%d', total, concurrency)
    records: list[dict] = []
    completed = 0

    def _one(method: str, task: str, trial: int) -> dict | None:
        try:
            rec = run_one_trial(method_name=method, task_id=task, trial_idx=trial, manifest=manifest, pool=pool, selectors=selectors, output_dir=output_dir, seed=seed, max_steps=max_steps)
            return rec
        except Exception as e:
            logger.exception('trial (%s, %s, t%d) failed: %s', method, task, trial, e)
            if fail_fast:
                raise
            return None
    with ThreadPoolExecutor(max_workers=concurrency) as pool_ex:
        fut_to_key = {pool_ex.submit(_one, m, t, i): (m, t, i) for (m, t, i) in tasks_3}
        for fut in as_completed(fut_to_key):
            (m, t, i) = fut_to_key[fut]
            rec = fut.result()
            completed += 1
            if rec is not None:
                records.append(rec)
                logger.info('[%d/%d] (%s, %s, t%d) reward=%s stop=%s steps=%s', completed, total, m, t, i, rec.get('reward'), rec.get('stop_reason'), rec.get('n_steps'))
            else:
                logger.warning('[%d/%d] (%s, %s, t%d) FAILED', completed, total, m, t, i)
    return records
