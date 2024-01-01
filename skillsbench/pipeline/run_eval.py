from __future__ import annotations
import argparse
import logging
import sys
import time
from collections import defaultdict
from pathlib import Path
from .methods import METHODS
from .orchestrate import run_grid

def _parse_tasks(args: argparse.Namespace) -> list[str]:
    if args.tasks_file:
        text = Path(args.tasks_file).read_text()
        return [ln.strip() for ln in text.splitlines() if ln.strip() and (not ln.startswith('#'))]
    if args.tasks:
        return [t.strip() for t in args.tasks.split(',') if t.strip()]
    raise SystemExit('Either --tasks or --tasks-file is required')

def _parse_args(argv: list[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser()
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument('--tasks', help='comma-separated task_ids')
    g.add_argument('--tasks-file', help='file with one task_id per line (# comments allowed)')
    p.add_argument('--methods', default='no-skill,random,bm25,cosine,full-library,ours', help='comma-separated method names')
    p.add_argument('--n-trials', type=int, default=3)
    p.add_argument('--output-dir', required=True, type=Path)
    p.add_argument('--max-steps', type=int, default=50)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--concurrency', type=int, default=4)
    p.add_argument('--fail-fast', action='store_true')
    return p.parse_args(argv)

def _print_summary(records: list[dict]) -> None:
    by_mt: dict[tuple[str, str], list[int]] = defaultdict(list)
    for r in records:
        method = r.get('method')
        task = r.get('task_id')
        rew = r.get('reward')
        if method is None or task is None or rew is None:
            continue
        by_mt[method, task].append(int(rew))
    methods = sorted({m for (m, _) in by_mt})
    tasks = sorted({t for (_, t) in by_mt})
    print(f'\n=== eval summary ({len(records)} trials) ===')
    print(f"{'task':<40} | " + '  '.join((f'{m:>12}' for m in methods)))
    print('-' * (40 + 1 + 14 * len(methods)))
    for t in tasks:
        cells = []
        for m in methods:
            xs = by_mt.get((m, t), [])
            if not xs:
                cells.append('    -')
            else:
                cells.append(f'{sum(xs)}/{len(xs)}')
        print(f'{t:<40} | ' + '  '.join((f'{c:>12}' for c in cells)))
    print()
    print(f"{'method':<14}  trials  rewards  pass_rate")
    for m in methods:
        all_r = [r for ((m2, _), rs) in by_mt.items() if m2 == m for r in rs]
        if all_r:
            print(f'{m:<14}  {len(all_r):>6}  {sum(all_r):>7}  {sum(all_r) / len(all_r):>8.3f}')

def main(argv: list[str] | None=None) -> int:
    args = _parse_args(argv if argv is not None else sys.argv[1:])
    logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(name)s: %(message)s')
    methods = [m.strip() for m in args.methods.split(',') if m.strip()]
    unknown = [m for m in methods if m not in METHODS]
    if unknown:
        print(f'Unknown methods: {unknown}. Known: {list(METHODS)}', file=sys.stderr)
        return 2
    tasks = _parse_tasks(args)
    n_total = len(methods) * len(tasks) * args.n_trials
    print(f'=== eval plan ===')
    print(f'  methods:      {methods}')
    print(f"  tasks:        {len(tasks)} ({(tasks if len(tasks) <= 6 else tasks[:3] + ['...'] + tasks[-2:])})")
    print(f'  n_trials:     {args.n_trials}')
    print(f'  concurrency:  {args.concurrency}')
    print(f'  max_steps:    {args.max_steps}')
    print(f'  total trials: {n_total}')
    print(f'  output:       {args.output_dir}')
    t0 = time.time()
    records = run_grid(methods=methods, task_ids=tasks, trial_indices=list(range(args.n_trials)), output_dir=args.output_dir, seed=args.seed, max_steps=args.max_steps, fail_fast=args.fail_fast, concurrency=args.concurrency)
    dur = time.time() - t0
    print(f'\n=== completed in {dur / 60:.1f} min ({dur:.0f}s) — {len(records)}/{n_total} trials ===')
    _print_summary(records)
    return 0
if __name__ == '__main__':
    sys.exit(main())
