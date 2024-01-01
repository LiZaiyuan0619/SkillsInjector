from __future__ import annotations
import argparse
import json
import logging
import os
import queue
import threading
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path
from .paths import STATE_DIR
from .skill_pool import load_skill_pool
from .orchestrate import expected_trials, make_selector, run_trials_for_worker, DETERMINISTIC_METHODS, RANDOM_METHOD
from runner.llm import LLMConfig, LLMClient
log = logging.getLogger(__name__)

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--tasks-file', required=True)
    p.add_argument('--methods', default=','.join(DETERMINISTIC_METHODS + [RANDOM_METHOD]))
    p.add_argument('--concurrency', type=int, default=24)
    p.add_argument('--output-dir', required=True)
    p.add_argument('--admissible-policy', default='strict-stock', choices=['strict-stock', 'robust-patch'])
    p.add_argument('--rewriter-cache-only', action='store_true', help='GPT review #1: enforce eval-time 0× 235B calls. Skip LLMClient construction for the rewriter; on cache miss, write identity-fallback with fallback_reason=cache_miss_cache_only_mode. Use after warm-up.')
    args = p.parse_args()
    if os.environ.get('SK_REWRITER_CACHE_ONLY', '').lower() in ('1', 'true', 'yes'):
        args.rewriter_cache_only = True
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((STATE_DIR / 'manifest.json').read_text())
    manifest_ids = {t['item_id'] for t in manifest['tasks']}
    selected_ids = {int(line.strip()) for line in Path(args.tasks_file).read_text().splitlines() if line.strip()}
    extra = selected_ids - manifest_ids
    if extra:
        raise RuntimeError(f'R9: tasks-file contains {len(extra)} item_ids NOT in current manifest ({sorted(extra)[:10]}...). Run pipeline.write_task_lists or pipeline._preflight_envs to sync.')
    manifest['tasks'] = [t for t in manifest['tasks'] if t['item_id'] in selected_ids]
    log.info(f"running on {len(manifest['tasks'])} tasks")
    methods = args.methods.split(',')
    expected = expected_trials(manifest, methods)
    log.info(f'expected: {len(expected)} trials')
    tau_star_path = STATE_DIR / 'tau_star.json'
    tau_star_content = json.loads(tau_star_path.read_text()) if tau_star_path.exists() else None
    if 'ours' in methods and tau_star_content is None:
        raise FileNotFoundError(f'R10: {tau_star_path} required for Ours method (Phase 1.0)')
    pool = load_skill_pool()
    (output_dir / 'manifest_run.json').write_text(json.dumps({'started_at': datetime.utcnow().isoformat(), 'n_tasks': len(manifest['tasks']), 'methods': methods, 'admissible_policy': args.admissible_policy, 'concurrency': args.concurrency, 'rewriter_cache_only': args.rewriter_cache_only, 'tau_star': tau_star_content, 'skill_pool_fp': pool.fingerprint, 'skill_pool_meta': pool.meta, 'env_vars': {'SK_RUNNER_MODEL': os.environ.get('SK_RUNNER_MODEL'), 'SK_RUNNER_BASE_URL': os.environ.get('SK_RUNNER_BASE_URL'), 'SK_REWRITER_MODEL': os.environ.get('SK_REWRITER_MODEL') if 'ours' in methods else None, 'AGENTGYM_ENVSERVER': os.environ.get('AGENTGYM_ENVSERVER')}}, indent=2))
    (output_dir / 'expected_trials.json').write_text(json.dumps(expected, indent=2))
    selectors = {m: make_selector(m, pool) for m in methods}
    for s in selectors.values():
        s.warmup()
    from transformers import AutoTokenizer
    from .paths import QWEN3_8B_PATH
    tokenizer = AutoTokenizer.from_pretrained(str(QWEN3_8B_PATH))
    llm_cfg = LLMConfig.from_env(prefix='SK_RUNNER')
    rewriter: LLMClient | None = None
    if 'ours' in methods and (not args.rewriter_cache_only):
        rewriter_cfg = LLMConfig.from_env(prefix='SK_REWRITER')
        rewriter = LLMClient(rewriter_cfg)
    if args.rewriter_cache_only:
        log.info('rewriter cache-only mode: 235B endpoint will NOT be called. Run pipeline.warm_rewriter_cache first to populate cache/rewriter/.')
    env_url = os.environ.get('AGENTGYM_ENVSERVER', 'http://127.0.0.1:36001')
    q: queue.Queue = queue.Queue()
    for item in expected:
        q.put(item)
    failed_trials: list = []
    completed_files: list = []
    progress_lock = threading.Lock()
    with ThreadPoolExecutor(max_workers=args.concurrency) as ex:
        futures = [ex.submit(run_trials_for_worker, worker_id=wid, q=q, pool=pool, selectors=selectors, env_url=env_url, llm_cfg=llm_cfg, rewriter=rewriter, output_dir=output_dir, admissible_policy=args.admissible_policy, tokenizer=tokenizer, pool_fp=pool.fingerprint, completed=completed_files, failed=failed_trials, progress_lock=progress_lock, total=len(expected), rewriter_cache_only=args.rewriter_cache_only) for wid in range(args.concurrency)]
        for f in futures:
            f.result()
    if rewriter:
        rewriter.close()
    (output_dir / 'failed_trials.jsonl').write_text('\n'.join((json.dumps(f) for f in failed_trials)) + ('\n' if failed_trials else ''))
    summary = {'expected': len(expected), 'completed_files': len(completed_files), 'failed': len(failed_trials), 'missing': len(expected) - len(completed_files) - len(failed_trials), 'ended_at': datetime.utcnow().isoformat()}
    (output_dir / 'completion_summary.json').write_text(json.dumps(summary, indent=2))
    log.info(f'=== eval done: {summary} ===')
if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(name)s: %(message)s')
    main()
