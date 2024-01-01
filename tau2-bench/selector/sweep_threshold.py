import argparse
import json
import statistics
import sys
from collections import Counter
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
from temp.selector.freeze_selections import _build_selection_row, _write_jsonl, dump_score_matrix

def analyze_score_matrix(matrix_path: Path) -> None:
    payload = json.loads(matrix_path.read_text(encoding='utf-8'))
    matrix = payload['matrix']
    num_tasks = payload['num_tasks']
    num_skills = payload['num_skills']
    all_scores = []
    top1_scores = []
    for (task_id, skill_scores) in matrix.items():
        scores = sorted(skill_scores.values(), reverse=True)
        all_scores.extend(scores)
        top1_scores.append(scores[0])
    print(f"\n{'=' * 60}")
    print(f'Score Matrix Analysis: {num_tasks} tasks × {num_skills} skills')
    print(f"{'=' * 60}")
    print(f'All scores:  min={min(all_scores):.4f}  max={max(all_scores):.4f}  mean={statistics.mean(all_scores):.4f}  std={statistics.stdev(all_scores):.4f}')
    print(f'Top-1 only:  min={min(top1_scores):.4f}  max={max(top1_scores):.4f}  mean={statistics.mean(top1_scores):.4f}  std={statistics.stdev(top1_scores):.4f}')
    n_positive = sum((1 for s in all_scores if s > 0))
    print(f'Positive scores: {n_positive}/{len(all_scores)} ({100 * n_positive / len(all_scores):.1f}%)')
    print(f'Tasks with negative top-1: {sum((1 for s in top1_scores if s <= 0))}/{num_tasks}')

def analyze_threshold(rows: list[dict], threshold: float) -> dict:
    budgets = [len(r['selected_skill_names']) for r in rows]
    budget_counter = Counter(budgets)
    return {'threshold': threshold, 'mean_budget': statistics.mean(budgets) if budgets else 0, 'median_budget': statistics.median(budgets) if budgets else 0, 'min_budget': min(budgets) if budgets else 0, 'max_budget': max(budgets) if budgets else 0, 'num_b0': budget_counter.get(0, 0), 'num_b1': budget_counter.get(1, 0), 'num_b2': budget_counter.get(2, 0), 'num_b3': budget_counter.get(3, 0), 'num_b4_plus': sum((v for (k, v) in budget_counter.items() if k >= 4)), 'budget_distribution': dict(sorted(budget_counter.items()))}

def main() -> None:
    parser = argparse.ArgumentParser(description='Sweep thresholds for adaptive budget selection')
    parser.add_argument('--domain', required=True)
    parser.add_argument('--data-dir', required=False, default=None, help='Required only if --matrix-path is not set.')
    parser.add_argument('--checkpoint', required=False, default=None, help='Required only if --matrix-path is not set.')
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--max-budget', type=int, default=10, help='Maximum number of skills per task (cap)')
    parser.add_argument('--thresholds', type=float, nargs='+', default=[-0.1, 0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8], help='Threshold values to sweep')
    parser.add_argument('--normalize-mode', choices=['absolute', 'per_task_minmax'], default='absolute', help="Score handling before threshold check. 'absolute' (default, legacy): compare raw scores against τ. 'per_task_minmax': min-max normalize each task's scores to [0,1] (top=1, bottom=0), then compare against τ. The latter is calibration-invariant and τ becomes 'relative gap from this task's best skill'.")
    parser.add_argument('--matrix-path', default=None, help='Path to a pre-computed score_matrix.json. If set, skip the GPU encoding step and reuse this matrix. --data-dir and --checkpoint are then optional.')
    args = parser.parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if args.matrix_path:
        matrix_path = Path(args.matrix_path)
        if not matrix_path.exists():
            raise SystemExit(f'--matrix-path file not found: {matrix_path}')
        print(f'Reusing existing score matrix at {matrix_path}')
    else:
        if not args.data_dir or not args.checkpoint:
            raise SystemExit('--data-dir and --checkpoint are required when --matrix-path is not set')
        matrix_path = output_dir / 'score_matrix.json'
        print(f'Dumping score matrix to {matrix_path} ...')
        dump_score_matrix(domain=args.domain, data_dir=args.data_dir, checkpoint_path=args.checkpoint, output_path=matrix_path, device=args.device)
    analyze_score_matrix(matrix_path)
    payload = json.loads(matrix_path.read_text(encoding='utf-8'))
    matrix = payload['matrix']
    checkpoint_path_str = payload['checkpoint_path']
    sweep_summary = []
    print(f"\n{'=' * 60}")
    print(f'Sweeping {len(args.thresholds)} thresholds: {args.thresholds}  [mode={args.normalize_mode}]')
    print(f"{'=' * 60}")
    for threshold in sorted(args.thresholds):
        rows = []
        for task_id in sorted(matrix.keys(), key=lambda t: (int(t) if t.isdigit() else 0, t)):
            skill_scores = matrix[task_id]
            if args.normalize_mode == 'per_task_minmax':
                values = list(skill_scores.values())
                (t_min, t_max) = (min(values), max(values))
                t_range = t_max - t_min
                if t_range < 1e-09:
                    effective = {sn: 1.0 for sn in skill_scores}
                else:
                    effective = {sn: (sc - t_min) / t_range for (sn, sc) in skill_scores.items()}
                ranked_eff = sorted(effective.items(), key=lambda kv: -kv[1])
                selected_pairs = [(sn, sc) for (sn, sc) in ranked_eff if sc >= threshold][:args.max_budget]
            else:
                ranked_raw = sorted(skill_scores.items(), key=lambda kv: -kv[1])
                selected_pairs = [(sn, sc) for (sn, sc) in ranked_raw if sc > threshold][:args.max_budget]
            selected_names = [sn for (sn, _) in selected_pairs]
            selected_scores_list = [skill_scores[sn] for sn in selected_names]
            rows.append(_build_selection_row(domain=args.domain, task_id=task_id, method='trained_selector', selected_skill_name=selected_names[0] if selected_names else None, selected_skill_names=selected_names, selection_seed=None, selection_score=selected_scores_list[0] if selected_scores_list else None, selection_scores=selected_scores_list, selection_rank=1, selection_source='threshold_selector', checkpoint_path=checkpoint_path_str, budget=len(selected_names), candidate_pool_size=None, training_scope='single_domain'))
        tag = f't{threshold:.2f}'.replace('-', 'neg').replace('.', '')
        sel_path = output_dir / f'threshold_{tag}.jsonl'
        _write_jsonl(sel_path, rows)
        analysis = analyze_threshold(rows, threshold)
        sweep_summary.append(analysis)
        dist = analysis['budget_distribution']
        print(f"  τ={threshold:+.2f}: mean_B={analysis['mean_budget']:.1f}, B=0:{analysis['num_b0']}, B=1:{analysis['num_b1']}, B=2:{analysis['num_b2']}, B=3:{analysis['num_b3']}, B≥4:{analysis['num_b4_plus']}  → {sel_path.name}")
    summary_path = output_dir / 'sweep_summary.json'
    summary_path.write_text(json.dumps(sweep_summary, indent=2), encoding='utf-8')
    print(f"\n{'=' * 60}")
    print('Online eval commands (copy & run):')
    print(f"{'=' * 60}")
    print(f'cd {PROJECT_ROOT}')
    for threshold in sorted(args.thresholds):
        tag = f't{threshold:.2f}'.replace('-', 'neg').replace('.', '')
        sel_file = output_dir / f'threshold_{tag}.jsonl'
        res_dir = output_dir.parent / f'online_results/threshold_{tag}' / 'trained_selector'
        print(f'\npython -m temp.selector.run_online_eval \\')
        print(f'    --domain {args.domain} --method trained_selector \\')
        print(f'    --selection-file {sel_file} \\')
        print(f'    --output-dir {res_dir} \\')
        print(f'    --num-trials 5 --max-concurrency 24')
if __name__ == '__main__':
    main()
