from __future__ import annotations
import argparse
import logging
import sys
from pathlib import Path
from .runner import TrialSpec, run_trial

def _parse_args(argv: list[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(description='SkillsBench minimal-runner trial driver.')
    p.add_argument('--task-dir', required=True, type=Path, help='Path to a single task dir (containing environment/, tests/, instruction.md)')
    p.add_argument('--method', required=True, choices=['no-skill', 'random', 'bm25', 'cosine', 'full-library', 'ours'])
    p.add_argument('--trial', required=True, type=int, dest='trial_idx')
    p.add_argument('--output-dir', required=True, type=Path)
    p.add_argument('--max-steps', type=int, default=50)
    return p.parse_args(argv)

def main(argv: list[str] | None=None) -> int:
    args = _parse_args(argv if argv is not None else sys.argv[1:])
    logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(name)s: %(message)s')
    if not args.task_dir.is_dir():
        print(f'--task-dir does not exist: {args.task_dir}', file=sys.stderr)
        return 2
    task_id = args.task_dir.name
    spec = TrialSpec(task_id=task_id, task_dir=args.task_dir, method=args.method, trial_idx=args.trial_idx, max_steps=args.max_steps, skill_bodies=None, skill_ids=[])
    rec = run_trial(spec, args.output_dir)
    print(f"reward={rec.get('reward')} stop_reason={rec.get('stop_reason')} n_steps={rec.get('n_steps')}")
    return 0
if __name__ == '__main__':
    sys.exit(main())
