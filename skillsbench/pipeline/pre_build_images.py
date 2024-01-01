from __future__ import annotations
import argparse
import logging
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from .manifest import load_manifest
from .paths import MANIFEST_PATH
_BUILD_ENV = {**os.environ, 'DOCKER_BUILDKIT': '1'}

def _image_exists(tag: str) -> bool:
    r = subprocess.run(['docker', 'image', 'inspect', tag], capture_output=True, text=True, timeout=10)
    return r.returncode == 0

def _build_one(task_id: str, env_dir: Path, rebuild: bool, timeout_s: int=1800) -> tuple[str, str, float]:
    tag = f'skbench-{task_id}:latest'
    if not rebuild and _image_exists(tag):
        return (task_id, 'cached', 0.0)
    last_err = ''
    t0 = time.time()
    for attempt in (1, 2):
        try:
            r = subprocess.run(['docker', 'build', '-t', tag, str(env_dir)], capture_output=True, text=True, timeout=timeout_s, env=_BUILD_ENV)
            if r.returncode == 0:
                return (task_id, 'built' if attempt == 1 else 'built-retry', time.time() - t0)
            last_err = f'rc={r.returncode}: {r.stderr[-300:]}'
        except subprocess.TimeoutExpired:
            last_err = f'timeout after {timeout_s}s'
    return (task_id, f'FAIL: {last_err}', time.time() - t0)

def _docker_disk_usage() -> str:
    try:
        r = subprocess.run(['docker', 'system', 'df'], capture_output=True, text=True, timeout=60)
    except subprocess.TimeoutExpired:
        return '(docker system df timed out >60s)'
    return r.stdout if r.returncode == 0 else '(docker system df failed)'

def main(argv: list[str] | None=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument('--tasks', help='comma-separated task_ids; default = all in manifest')
    p.add_argument('--tasks-file')
    p.add_argument('--concurrency', type=int, default=2, help='parallel docker builds; 2 is safe (apt-get is heavy)')
    p.add_argument('--limit', type=int, default=0, help='if > 0, stop after this many builds (for size probing)')
    p.add_argument('--rebuild', action='store_true', help='re-build even if image already exists')
    p.add_argument('--timeout', type=int, default=1800, help='per-build timeout in seconds (default 1800 = 30 min)')
    args = p.parse_args(argv if argv is not None else sys.argv[1:])
    logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(name)s: %(message)s')
    m = load_manifest(MANIFEST_PATH)
    if args.tasks:
        wanted = [t.strip() for t in args.tasks.split(',') if t.strip()]
    elif args.tasks_file:
        text = Path(args.tasks_file).read_text()
        wanted = [ln.strip() for ln in text.splitlines() if ln.strip() and (not ln.startswith('#'))]
    else:
        wanted = [t.task_id for t in m.tasks]
    by_id = {t.task_id: t for t in m.tasks}
    plan = []
    for tid in wanted:
        if tid not in by_id:
            print(f'WARN: task {tid} not in manifest, skipping')
            continue
        env_dir = Path(by_id[tid].task_dir) / 'environment'
        plan.append((tid, env_dir))
    if args.limit > 0:
        plan = plan[:args.limit]
    print(f'=== pre-building {len(plan)} images, concurrency={args.concurrency} ===')
    print('docker disk usage BEFORE:')
    print(_docker_disk_usage())
    t0 = time.time()
    results: list[tuple[str, str, float]] = []
    completed = 0
    with ThreadPoolExecutor(max_workers=args.concurrency) as ex:
        futs = {ex.submit(_build_one, tid, ed, args.rebuild, args.timeout): tid for (tid, ed) in plan}
        for fut in as_completed(futs):
            tid = futs[fut]
            try:
                res = fut.result()
                results.append(res)
                completed += 1
                tag = res[1] if res[1] in ('cached', 'built') else 'FAIL'
                print(f'[{completed}/{len(plan)}] {tid:<45} {tag:<8} {res[2]:>6.0f}s')
            except Exception as e:
                completed += 1
                results.append((tid, f'EXC: {e}', 0.0))
                print(f'[{completed}/{len(plan)}] {tid:<45} EXC      —')
    dur = time.time() - t0
    n_built = sum((1 for (_, st, _) in results if st == 'built'))
    n_cached = sum((1 for (_, st, _) in results if st == 'cached'))
    n_fail = sum((1 for (_, st, _) in results if st not in ('built', 'cached')))
    print(f'\n=== summary ({dur / 60:.1f} min) ===')
    print(f'  built:  {n_built}')
    print(f'  cached: {n_cached}')
    print(f'  fail:   {n_fail}')
    if n_fail:
        print('\nFAILURES:')
        for (tid, st, _) in results:
            if st not in ('built', 'cached'):
                print(f'  {tid}: {st[:200]}')
    print('\ndocker disk usage AFTER:')
    print(_docker_disk_usage())
    return 0 if n_fail == 0 else 1
if __name__ == '__main__':
    sys.exit(main())
