from __future__ import annotations
import argparse
import logging
import sys
import time
from pathlib import Path
from .methods import METHODS
from .orchestrate import run_grid

def _parse_args(argv: list[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument('--task', required=True, help='task_id (single)')
    p.add_argument('--methods', default='no-skill,random,bm25,cosine,full-library,ours', help='comma-separated method names')
    p.add_argument('--trial', type=int, default=0)
    p.add_argument('--output-dir', required=True, type=Path)
    p.add_argument('--max-steps', type=int, default=50)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--fail-fast', action='store_true', help='raise on first trial error instead of skipping')
    return p.parse_args(argv)

def main(argv: list[str] | None=None) -> int:
    args = _parse_args(argv if argv is not None else sys.argv[1:])
    logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(name)s: %(message)s')
    methods = [m.strip() for m in args.methods.split(',') if m.strip()]
    unknown = [m for m in methods if m not in METHODS]
    if unknown:
        print(f'Unknown methods: {unknown}. Known: {list(METHODS)}', file=sys.stderr)
        return 2
    print(f'B-step: {len(methods)} methods × 1 task ({args.task}) × 1 trial → {args.output_dir}')
    print(f'methods: {methods}')
    t0 = time.time()
    records = run_grid(methods=methods, task_ids=[args.task], trial_indices=[args.trial], output_dir=args.output_dir, seed=args.seed, max_steps=args.max_steps, fail_fast=args.fail_fast)
    dur = time.time() - t0
    print(f'\n=== B-step summary ({dur:.0f}s) ===')
    print(f"{'method':<14}  {'reward':>6}  {'stop_reason':<14}  {'steps':>5}  {'tool_calls':>10}  budget")
    for r in records:
        sm = r.get('selection_meta', {})
        budget = len(sm.get('skill_ids', []))
        if sm.get('extra', {}).get('budget'):
            budget = sm['extra']['budget']
        print(f"{r.get('method', ''):<14}  {str(r.get('reward', '-')):>6}  {str(r.get('stop_reason', '-')):<14}  {str(r.get('n_steps', '-')):>5}  {str(r.get('n_tool_calls', '-')):>10}  {budget}")
    return 0
if __name__ == '__main__':
    sys.exit(main())
