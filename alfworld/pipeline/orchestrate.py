from __future__ import annotations
import hashlib
import json
import logging
import os
import queue
import threading
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Optional
from .paths import CACHE_DIR
from .skill_pool import load_skill_pool, SkillPool
from .selectors.no_skill import NoSkillSelector
from .selectors.random_sel import RandomSelector
from .selectors.bm25 import BM25Selector
from .selectors.cosine import CosineSelector
from .selectors.full_library import FullLibrarySelector
from .selectors.ours import OursSelector
from runner.envserver_client import EnvServerClient, EnvServerError
from runner.llm import LLMConfig, LLMClient
from runner.runner_alfworld import TrialSpec, run_trial_loop, write_trial_record, TrialRecord
from runner.prompt import extract_retrieval_query
from distill.prompts import REWRITER_SYSTEM, render_rewriter_user, REWRITER_PROMPT_VERSION
from distill.rewriter_io import parse_rewriter_response, RewriterParseError
log = logging.getLogger(__name__)
DETERMINISTIC_METHODS = ['no-skill', 'bm25', 'cosine', 'full-library', 'ours']
RANDOM_METHOD = 'random'
RANDOM_N_DRAWS = 5
DEFAULT_N_TRIALS = int(os.environ.get('SK_N_TRIALS', '5'))

def expected_trials(manifest: dict, methods: list[str]) -> list[dict]:
    out = []
    for t in manifest['tasks']:
        for m in methods:
            n = DEFAULT_N_TRIALS
            for i in range(n):
                out.append({'item_id': t['item_id'], 'task_id': t['task_id'], 'task_type': t['task_type'], 'method': m, 'trial_idx': i})
    return out

def make_selector(method: str, pool: SkillPool):
    if method == 'no-skill':
        return NoSkillSelector(pool)
    if method == 'random':
        return RandomSelector(pool, k=3)
    if method == 'bm25':
        return BM25Selector(pool, k=3)
    if method == 'cosine':
        return CosineSelector(pool, k=3)
    if method == 'full-library':
        return FullLibrarySelector(pool)
    if method == 'ours':
        return OursSelector(pool)
    raise ValueError(f'unknown method {method}')

def rewriter_cache_key(task_id, sorted_sha8s, retrieval_query, pool_fp) -> str:
    blob = json.dumps({'task_id': task_id, 'skill_content_hashes': list(sorted_sha8s), 'method': 'ours', 'rewriter_model': os.environ['SK_REWRITER_MODEL'], 'rewriter_prompt_version': REWRITER_PROMPT_VERSION, 'skill_pool_fp': pool_fp, 'retrieval_query_hash': hashlib.sha256(retrieval_query.encode()).hexdigest()[:16]}, sort_keys=True)
    return hashlib.sha256(blob.encode()).hexdigest()

def _to_dict(skill) -> dict:
    return {'skill_id': skill.skill_id, 'display_name': skill.display_name, 'sha8': skill.sha8, 'content': skill.body}

def rewrite_skills(selected_skills: list, retrieval_query: str, task_id: str, pool_fp: str, rewriter: Optional[LLMClient], cache_dir: Path, *, cache_only: bool=False) -> tuple[list[dict], dict]:
    if len(selected_skills) <= 1:
        return ([_to_dict(s) for s in selected_skills], {'rewritten': False, 'fallback': False})
    sorted_sha8s = sorted((s.sha8 for s in selected_skills))
    key = rewriter_cache_key(task_id, sorted_sha8s, retrieval_query, pool_fp)
    cache_file = cache_dir / f'{key}.json'
    if cache_file.exists():
        try:
            cached = json.loads(cache_file.read_text())
            return (cached['skills'], {'rewritten': True, 'fallback': False, 'cache_hit': True, 'key': key})
        except Exception:
            pass
    if cache_only:
        return ([_to_dict(s) for s in selected_skills], {'rewritten': False, 'fallback': True, 'fallback_reason': 'cache_miss_cache_only_mode', 'cache_hit': False, 'key': key})
    if rewriter is None:
        raise RuntimeError(f'rewrite_skills: rewriter=None and cache miss (key={key}). Either construct an LLMClient or pass cache_only=True.')
    in_skills_for_prompt = [{'sha8': s.sha8, 'display_name': s.display_name, 'content': s.body} for s in selected_skills]
    user_msg = render_rewriter_user(retrieval_query, in_skills_for_prompt)
    (text, _u) = rewriter.chat([{'role': 'system', 'content': REWRITER_SYSTEM}, {'role': 'user', 'content': user_msg}], temperature=0.7, top_p=0.9, max_tokens=2500)
    try:
        parsed = parse_rewriter_response(text, in_skills_for_prompt)
    except RewriterParseError as e:
        log.warning(f'  rewriter parse_failure (task={task_id}, identity fallback): {e}')
        return ([_to_dict(s) for s in selected_skills], {'rewritten': False, 'fallback': True, 'fallback_reason': 'rewriter_parse_error', 'error': str(e)[:200], 'key': key})
    rewritten = []
    for (s, out) in zip(selected_skills, parsed):
        rewritten.append({'skill_id': s.skill_id, 'display_name': out['display_name'], 'sha8': s.sha8, 'content': out['rewritten_content']})
    cache_file.parent.mkdir(parents=True, exist_ok=True)
    cache_file.write_text(json.dumps({'skills': rewritten, 'raw_response': text, 'key_inputs': {'task_id': task_id, 'sha8s': sorted_sha8s, 'pool_fp': pool_fp, 'rewriter_model': os.environ['SK_REWRITER_MODEL'], 'rewriter_prompt_version': REWRITER_PROMPT_VERSION}}, indent=2))
    return (rewritten, {'rewritten': True, 'fallback': False, 'cache_hit': False, 'key': key})

def _write_failure_stub(item: dict, stop_reason: str, error: str, output_dir: Path, admissible_policy: str, backend_label: str) -> Path:
    spec = TrialSpec(item_id=item['item_id'], task_id=item['task_id'], task_type=item['task_type'], method=item['method'], trial_idx=item['trial_idx'], skills=[], skill_ids=[], selection_meta={'failed_before_loop': True, 'error': error[:300]}, rewriter_meta={}, admissible_policy=admissible_policy, backend_label=backend_label)
    rec = TrialRecord(spec=asdict(spec), reward=0.0, stop_reason=stop_reason, n_steps=0, parse_errors=0, invalid_action_count=0, invalid_action_streak_max=0, rewriter_fallback=False, duration_s=0.0, started_at=datetime.utcnow().isoformat(), messages=[], commands=[], usage_total={})
    return write_trial_record(rec, output_dir)

def run_trials_for_worker(worker_id: int, q: queue.Queue, *, pool: SkillPool, selectors: dict, env_url: str, llm_cfg: LLMConfig, rewriter: Optional[LLMClient], output_dir: Path, admissible_policy: str, tokenizer, pool_fp: str, completed: list, failed: list, progress_lock: threading.Lock, total: int, rewriter_cache_only: bool=False) -> None:
    env: Optional[EnvServerClient] = None
    env_id: Optional[int] = None
    llm: Optional[LLMClient] = None
    try:
        env = EnvServerClient(base_url=env_url)
        env_id = env.create()
        log.info(f'  worker {worker_id}: env_id={env_id}')
        llm = LLMClient(llm_cfg)
        while True:
            try:
                item = q.get_nowait()
            except queue.Empty:
                break
            try:
                r = env.reset(env_id, game=item['item_id'], world_type='Text')
                initial_obs = r['observation']
                avail = r.get('available_actions', [])
            except EnvServerError as e:
                log.error(f"  worker {worker_id} reset fail item_id={item['item_id']}: {e}")
                _write_failure_stub(item, 'env_error', str(e), output_dir, admissible_policy, llm_cfg.backend_label)
                with progress_lock:
                    failed.append({**item, 'stop_reason': 'env_error', 'error': str(e)[:300]})
                q.task_done()
                continue
            try:
                query = extract_retrieval_query(initial_obs)
                method = item['method']
                trial_idx = item['trial_idx']
                seed = trial_idx * 10000 + item['item_id']
                selector = selectors[method]
                sel = selector.select({'task_id': item['task_id'], 'task_type': item['task_type'], 'item_id': item['item_id']}, query=query, trial_idx=trial_idx, seed=seed)
            except Exception as e:
                log.error(f'  worker {worker_id} selection fail: {e}')
                _write_failure_stub(item, 'selection_error', str(e), output_dir, admissible_policy, llm_cfg.backend_label)
                with progress_lock:
                    failed.append({**item, 'stop_reason': 'selection_error', 'error': str(e)[:300]})
                q.task_done()
                continue
            rewriter_meta: dict = {}
            skills_for_prompt = [_to_dict(s) for s in sel.selected_skills]
            do_rewrite = method == 'ours' and len(sel.selected_skills) > 1 and (rewriter is not None or rewriter_cache_only)
            if do_rewrite:
                try:
                    (skills_for_prompt, rewriter_meta) = rewrite_skills(sel.selected_skills, query, item['task_id'], pool_fp, rewriter, CACHE_DIR / 'rewriter', cache_only=rewriter_cache_only)
                except Exception as e:
                    log.error(f'  worker {worker_id} rewriter API fail: {e}')
                    _write_failure_stub(item, 'api_error', f'rewriter: {e}', output_dir, admissible_policy, llm_cfg.backend_label)
                    with progress_lock:
                        failed.append({**item, 'stop_reason': 'api_error', 'error': f'rewriter: {str(e)[:200]}'})
                    q.task_done()
                    continue
            spec = TrialSpec(item_id=item['item_id'], task_id=item['task_id'], task_type=item['task_type'], method=method, trial_idx=trial_idx, skills=skills_for_prompt, skill_ids=[s['skill_id'] for s in skills_for_prompt], selection_meta={'query': query[:300], 'k': len(skills_for_prompt), **sel.extra}, rewriter_meta=rewriter_meta, admissible_policy=admissible_policy, backend_label=llm_cfg.backend_label)
            try:
                record = run_trial_loop(spec, env_id=env_id, initial_observation=initial_obs, available_actions=avail, llm=llm, env=env, tokenizer=tokenizer)
                p = write_trial_record(record, output_dir)
                with progress_lock:
                    completed.append(str(p))
                    if (len(completed) + len(failed)) % 50 == 0:
                        log.info(f'progress: {len(completed) + len(failed)}/{total}, completed={len(completed)}, failed={len(failed)}')
            except Exception as e:
                log.error(f'  worker {worker_id} run_trial_loop unexpected: {e}')
                _write_failure_stub(item, 'api_error', str(e), output_dir, admissible_policy, llm_cfg.backend_label)
                with progress_lock:
                    failed.append({**item, 'stop_reason': 'api_error', 'error': str(e)[:300]})
            finally:
                q.task_done()
    finally:
        if llm:
            llm.close()
        if env:
            env.close()
        log.info(f'  worker {worker_id} finished')
