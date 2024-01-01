import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
from temp.selector.freeze_selections import _build_selection_row, _compute_all_logits, _write_jsonl
from temp.selector.datasets import load_pair_records

def compute_utility_ranking(data_dir: Path) -> list[tuple[str, float]]:
    records = load_pair_records(data_dir)
    scores: dict[str, list[float]] = defaultdict(list)
    for r in records:
        scores[r.skill_name].append(r.p_skill)
    ranking = [(name, sum(vals) / len(vals)) for (name, vals) in scores.items()]
    ranking.sort(key=lambda x: (-x[1], x[0]))
    return ranking

def generate_all_inject_rows(domain: str, task_ids: list[str], pool: list[str], logits_by_key: dict[tuple[str, str], float], pool_size: int) -> list[dict[str, Any]]:
    rows = []
    for tid in task_ids:
        scored = sorted([(s, logits_by_key.get((tid, s), -999.0)) for s in pool], key=lambda x: -x[1])
        names = [s for (s, _) in scored]
        score_vals = [sc for (_, sc) in scored]
        rows.append(_build_selection_row(domain=domain, task_id=tid, method='trained_selector', selected_skill_name=names[0] if names else None, selected_skill_names=names, selection_seed=None, selection_score=score_vals[0] if score_vals else None, selection_scores=score_vals, selection_rank=1, selection_source='all_inject', checkpoint_path=None, budget=len(names), candidate_pool_size=pool_size, training_scope='single_domain'))
    return rows

def generate_ours_rows(domain: str, task_ids: list[str], pool: list[str], logits_by_key: dict[tuple[str, str], float], checkpoint_path: str, threshold: float, max_budget: int, pool_size: int, normalize_mode: str='absolute') -> list[dict[str, Any]]:
    rows = []
    for tid in task_ids:
        scored = sorted([(s, logits_by_key.get((tid, s), -999.0)) for s in pool], key=lambda x: -x[1])
        if normalize_mode == 'per_task_minmax':
            if not scored:
                selected = []
            else:
                values = [sc for (_, sc) in scored]
                (t_min, t_max) = (min(values), max(values))
                t_range = t_max - t_min
                if t_range < 1e-09:
                    normalized = [(s, sc, 1.0) for (s, sc) in scored]
                else:
                    normalized = [(s, sc, (sc - t_min) / t_range) for (s, sc) in scored]
                selected = [(s, sc) for (s, sc, ns) in normalized if ns >= threshold]
        else:
            selected = [(s, sc) for (s, sc) in scored if sc > threshold]
        if len(selected) > max_budget:
            selected = selected[:max_budget]
        names = [s for (s, _) in selected]
        score_vals = [sc for (_, sc) in selected]
        rows.append(_build_selection_row(domain=domain, task_id=tid, method='trained_selector', selected_skill_name=names[0] if names else None, selected_skill_names=names, selection_seed=None, selection_score=score_vals[0] if score_vals else None, selection_scores=score_vals, selection_rank=1, selection_source='scaling_selector', checkpoint_path=checkpoint_path, budget=len(names), candidate_pool_size=pool_size, training_scope='single_domain'))
    return rows

def main() -> None:
    parser = argparse.ArgumentParser(description='Generate selection files for Figure 2 scaling curve')
    parser.add_argument('--domain', required=True)
    parser.add_argument('--data-dir', required=True, type=Path)
    parser.add_argument('--checkpoint', required=True, type=Path)
    parser.add_argument('--threshold', required=True, type=float)
    parser.add_argument('--pool-sizes', nargs='+', type=int, default=[1, 2, 3, 4, 8, 16, 32, 64])
    parser.add_argument('--output-dir', required=True, type=Path)
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--max-budget', type=int, default=10)
    parser.add_argument('--normalize-mode', choices=['absolute', 'per_task_minmax'], default='absolute', help="Score handling before threshold check. 'absolute' (default, legacy): compare raw scores against τ. 'per_task_minmax': min-max normalize each task's pool scores to [0,1], then compare against τ using >= (τ=1.0 still picks top-1). The latter is calibration-invariant and τ becomes 'relative gap from task's best skill in the pool'.")
    parser.add_argument('--skip-ours', action='store_true', help='If set, skip generating the Ours line (useful when τ is pending; only all_inject files will be produced).')
    args = parser.parse_args()
    domain_leaf = args.data_dir.name
    print('Computing utility ranking ...')
    ranking = compute_utility_ranking(args.data_dir)
    utility_names = [name for (name, _) in ranking]
    print(f'  {len(ranking)} skills total')
    for (i, (name, score)) in enumerate(ranking[:5]):
        print(f'  #{i + 1} {name}: {score:.4f}')
    print()
    print(f'Loading selector from {args.checkpoint} ...')
    (logits_by_key, tasks_by_id, all_skill_names) = _compute_all_logits(domain=args.domain, data_dir=str(args.data_dir), checkpoint_path=str(args.checkpoint), device=args.device)
    task_ids = sorted(tasks_by_id.keys())
    available = set(all_skill_names)
    utility_names = [s for s in utility_names if s in available]
    print(f'  {len(task_ids)} tasks × {len(utility_names)} skills (after intersection)\n')
    print(f"{'pool':>5}  {'ai_rows':>8}  {'ours_rows':>9}  {'avg_B':>6}  {'zero_B':>6}")
    print('-' * 45)
    for N in args.pool_sizes:
        actual_N = min(N, len(utility_names))
        if actual_N < N:
            print(f'WARN: pool {N} capped to {actual_N} (only {len(utility_names)} skills available)')
        pool = utility_names[:actual_N]
        ai_rows = generate_all_inject_rows(args.domain, task_ids, pool, logits_by_key, actual_N)
        ai_dir = args.output_dir / 'all_inject' / domain_leaf
        ai_dir.mkdir(parents=True, exist_ok=True)
        _write_jsonl(ai_dir / f'pool_{N:03d}.jsonl', ai_rows)
        if args.skip_ours:
            print(f"{N:5d}  {len(ai_rows):8d}  {'skipped':>9}  {'-':>6}  {'-':>6}")
            continue
        ours_rows = generate_ours_rows(args.domain, task_ids, pool, logits_by_key, str(args.checkpoint), args.threshold, args.max_budget, actual_N, args.normalize_mode)
        ours_dir = args.output_dir / 'ours' / domain_leaf
        ours_dir.mkdir(parents=True, exist_ok=True)
        _write_jsonl(ours_dir / f'pool_{N:03d}.jsonl', ours_rows)
        budgets = [r['budget'] for r in ours_rows]
        avg_b = sum(budgets) / len(budgets) if budgets else 0
        zero_b = sum((1 for b in budgets if b == 0))
        print(f'{N:5d}  {len(ai_rows):8d}  {len(ours_rows):9d}  {avg_b:6.1f}  {zero_b:6d}')
    print(f'\nDone. Output directory: {args.output_dir}/')
    print(f'  all_inject/{domain_leaf}/pool_*.jsonl')
    print(f'  ours/{domain_leaf}/pool_*.jsonl')
    print(f'\nUtility ranking (top-10):')
    for (i, (name, score)) in enumerate(ranking[:10]):
        print(f'  #{i + 1:2d}  {score:.4f}  {name}')
if __name__ == '__main__':
    main()
