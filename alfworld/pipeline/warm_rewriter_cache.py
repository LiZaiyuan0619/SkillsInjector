from __future__ import annotations
import argparse
import json
import logging
import os
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from .paths import STATE_DIR, CACHE_DIR
from .skill_pool import load_skill_pool
from .selectors.ours import OursSelector
from .orchestrate import rewrite_skills, _to_dict
from runner.envserver_client import EnvServerClient, EnvServerError
from runner.llm import LLMConfig, LLMClient
from runner.prompt import extract_retrieval_query
log = logging.getLogger(__name__)

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--tasks-file', required=True)
    p.add_argument('--concurrency', type=int, default=4, help='Parallel 235B calls (default 4; shared endpoint, be polite)')
    p.add_argument('--output-dir', required=True, help='run dir to write warmup_summary.json (mirrors run_eval --output-dir)')
    args = p.parse_args()
    manifest = json.loads((STATE_DIR / 'manifest.json').read_text())
    manifest_by_iid = {t['item_id']: t for t in manifest['tasks']}
    selected_ids = [int(line.strip()) for line in Path(args.tasks_file).read_text().splitlines() if line.strip()]
    extra = set(selected_ids) - set(manifest_by_iid)
    if extra:
        raise RuntimeError(f'R9: tasks-file has {len(extra)} item_ids NOT in manifest (first 10: {sorted(extra)[:10]})')
    log.info(f'warming cache for {len(selected_ids)} tasks')
    pool = load_skill_pool()
    selector = OursSelector(pool)
    selector.warmup()
    rewriter_cfg = LLMConfig.from_env(prefix='SK_REWRITER')
    rewriter = LLMClient(rewriter_cfg)
    cache_dir = CACHE_DIR / 'rewriter'
    cache_dir.mkdir(parents=True, exist_ok=True)
    env_url = os.environ.get('AGENTGYM_ENVSERVER', 'http://127.0.0.1:36001')
    counts = {'cache_hits': 0, 'new_entries': 0, 'skipped_singleton': 0, 'api_errors': 0, 'parse_fallbacks': 0, 'env_errors': 0}
    counts_lock = threading.Lock()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    def warm_one(item_id: int):
        t = manifest_by_iid[item_id]
        try:
            with EnvServerClient(base_url=env_url) as env:
                eid = env.create()
                r = env.reset(eid, game=item_id, world_type='Text')
                initial_obs = r['observation']
        except EnvServerError as e:
            with counts_lock:
                counts['env_errors'] += 1
            log.error(f'  item_id={item_id} env_error: {e}')
            return
        query = extract_retrieval_query(initial_obs)
        sel = selector.select({'task_id': t['task_id'], 'task_type': t['task_type'], 'item_id': item_id}, query=query, trial_idx=0, seed=item_id)
        if len(sel.selected_skills) <= 1:
            with counts_lock:
                counts['skipped_singleton'] += 1
            return
        try:
            (_, meta) = rewrite_skills(sel.selected_skills, query, t['task_id'], pool.fingerprint, rewriter, cache_dir)
        except Exception as e:
            with counts_lock:
                counts['api_errors'] += 1
            log.error(f'  item_id={item_id} rewriter API error: {e}')
            return
        with counts_lock:
            if meta.get('cache_hit'):
                counts['cache_hits'] += 1
            elif meta.get('fallback'):
                counts['parse_fallbacks'] += 1
            else:
                counts['new_entries'] += 1
    with ThreadPoolExecutor(max_workers=args.concurrency) as ex:
        futures = [ex.submit(warm_one, iid) for iid in selected_ids]
        for (i, f) in enumerate(as_completed(futures), 1):
            f.result()
            if i % 25 == 0:
                log.info(f'progress: {i}/{len(selected_ids)} | counts={counts}')
    rewriter.close()
    summary = {'ended_at': datetime.utcnow().isoformat(), 'n_tasks': len(selected_ids), 'skill_pool_fp': pool.fingerprint, 'rewriter_model': os.environ.get('SK_REWRITER_MODEL'), 'rewriter_endpoint': os.environ.get('SK_REWRITER_BASE_URL'), **counts}
    (output_dir / 'warmup_summary.json').write_text(json.dumps(summary, indent=2))
    log.info(f'=== warmup done: {summary} ===')
    log.info(f"Phase 5 full eval will see {counts['new_entries'] + counts['cache_hits']} cache hits out of {len(selected_ids) - counts['skipped_singleton']} non-singleton trials")
if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(name)s: %(message)s')
    main()
