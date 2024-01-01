import argparse
import json
import shutil
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

def get_unique_skills(ranked_skills: list[dict]) -> list[dict]:
    seen = set()
    unique = []
    for s in ranked_skills:
        name = s['skill_name']
        if name not in seen:
            seen.add(name)
            unique.append(s)
    return unique

def install_single_skill(skill: dict, parsed_dir: Path, skills_dir: Path) -> str:
    skills_dir.mkdir(parents=True, exist_ok=True)
    init_file = skills_dir / '__init__.py'
    if not init_file.exists():
        init_file.write_text('')
    src = parsed_dir / skill['file']
    dst = skills_dir / f"{skill['skill_name']}.py"
    shutil.copy2(src, dst)
    domain = skills_dir.name
    return f"{domain}/{skill['skill_name']}"

def generate_single_skill_config(domain: str, module_path: str, skill_name: str, config_dir: Path, template: dict) -> Path:
    config_dir.mkdir(parents=True, exist_ok=True)
    config_path = config_dir / f'{domain}_single_{skill_name}.yaml'
    lines = [f'# Single-skill test: {skill_name}', f'domain: {domain}', f'skills:', f'  - {module_path}', f"model: {template.get('model', 'openai/Qwen/Qwen3-235B-A22B-Instruct-2507')}", f'llm_args:', f"  api_base: {template.get('llm_args', {}).get('api_base', 'http://localhost:32800/v1')}", f"  api_key: {template.get('llm_args', {}).get('api_key', 'not-needed')}", f"  temperature: {template.get('llm_args', {}).get('temperature', 0.0)}", f"task_split: {template.get('task_split', 'base')}", f'task_ids: null', f"max_steps: {template.get('max_steps', 200)}", f"seed: {template.get('seed', 300)}"]
    config_path.write_text('\n'.join(lines) + '\n')
    return config_path

def check_existing_result(output_dir: Path, domain: str, skill_name: str, num_trials: int) -> Path | None:
    if num_trials > 1:
        pattern = f'{domain}_single_{skill_name}_agg{num_trials}_*.json'
        existing = sorted(output_dir.glob(pattern))
        if existing:
            return existing[-1]
    else:
        pattern = f'{domain}_single_{skill_name}_*.json'
        existing = [path for path in sorted(output_dir.glob(pattern)) if '_trial' not in path.name and '_agg' not in path.name]
        if existing:
            return existing[-1]
    return None

def check_baseline_result(output_dir: Path, domain: str, num_trials: int) -> Path | None:
    if num_trials == 1:
        pattern = f'{domain}_baseline_*.json'
        existing = [path for path in sorted(output_dir.glob(pattern)) if '_trial' not in path.name and '_agg' not in path.name]
        if existing:
            return existing[-1]
    pattern = f'{domain}_baseline_agg{num_trials}_*.json'
    existing = sorted(output_dir.glob(pattern))
    if existing:
        return existing[-1]
    pattern = f'{domain}_baseline_agg*_*.json'
    existing = sorted(output_dir.glob(pattern))
    if existing:
        return existing[-1]
    pattern = f'baseline_agg*_*.json'
    existing = sorted(output_dir.glob(pattern))
    if existing:
        return existing[-1]
    return None

def run_single_experiment(config_path: Path, concurrency: int, num_trials: int, output_dir: Path, start_trial: int=0, save_trajectories: bool=False) -> int:
    cmd = [sys.executable, 'temp/run_experiment.py', '--config', str(config_path), '-j', str(concurrency), '-k', str(num_trials), '--output-dir', str(output_dir), '--start-trial', str(start_trial)]
    if save_trajectories:
        cmd.append('--save-trajectories')
    proc = subprocess.run(cmd, capture_output=False)
    return proc.returncode

def find_result_files(output_dir: Path, config_name: str, num_trials: int) -> dict:
    result = {'trials': [], 'aggregate': None}
    if num_trials > 1:
        agg_pattern = f'{config_name}_agg{num_trials}_*.json'
        agg_files = sorted(output_dir.glob(agg_pattern))
        if agg_files:
            result['aggregate'] = json.loads(agg_files[-1].read_text())
        for i in range(num_trials):
            trial_pattern = f'{config_name}_trial{i}_*.json'
            trial_files = sorted(output_dir.glob(trial_pattern))
            if trial_files:
                result['trials'].append(json.loads(trial_files[-1].read_text()))
    else:
        pattern = f'{config_name}_*.json'
        files = sorted(output_dir.glob(pattern))
        if files:
            result['trials'].append(json.loads(files[-1].read_text()))
    return result

def build_summary_report(all_skill_results: list[dict], baseline_result: dict | None, domain: str, num_trials: int) -> dict:
    baseline_pass_rate = 0.0
    baseline_majority_pass_rate = 0.0
    baseline_per_task = {}
    if baseline_result:
        bs = baseline_result.get('summary', {})
        baseline_pass_rate = bs.get('pass_rate_mean', bs.get('pass_rate', 0))
        baseline_majority_pass_rate = bs.get('majority_pass_rate', bs.get('pass_rate', 0))
        for r in baseline_result.get('results', []):
            baseline_per_task[r['task_id']] = r.get('reward', 0)
    skill_rows = []
    for sr in all_skill_results:
        skill_name = sr['skill_name']
        frequency = sr['frequency']
        cluster_size = sr['cluster_size']
        agg = sr.get('aggregate_data')
        if agg is None:
            skill_rows.append({'skill_name': skill_name, 'frequency': frequency, 'cluster_size': cluster_size, 'status': 'not_run'})
            continue
        summary = agg.get('summary', {})
        pass_rate_mean = summary.get('pass_rate_mean', summary.get('pass_rate', 0))
        pass_rate_std = summary.get('pass_rate_std', 0)
        majority_pass_rate = summary.get('majority_pass_rate', summary.get('pass_rate', 0))
        pass_counts = summary.get('pass_counts', [summary.get('pass_count', 0)])
        delta_mean = pass_rate_mean - baseline_pass_rate
        delta_majority = majority_pass_rate - baseline_majority_pass_rate
        improved_tasks = []
        degraded_tasks = []
        for r in agg.get('results', []):
            tid = r['task_id']
            skill_reward = r.get('reward', 0)
            base_reward = baseline_per_task.get(tid, 0)
            if skill_reward > base_reward + 0.01:
                improved_tasks.append(tid)
            elif skill_reward < base_reward - 0.01:
                degraded_tasks.append(tid)
        skill_rows.append({'skill_name': skill_name, 'frequency': frequency, 'cluster_size': cluster_size, 'status': 'completed', 'pass_rate_mean': round(pass_rate_mean, 4), 'pass_rate_std': round(pass_rate_std, 4), 'pass_counts': pass_counts, 'majority_pass_rate': round(majority_pass_rate, 4), 'delta_pass_rate_mean': round(delta_mean, 4), 'delta_majority_pass_rate': round(delta_majority, 4), 'improved_tasks': improved_tasks, 'degraded_tasks': degraded_tasks, 'net_task_change': len(improved_tasks) - len(degraded_tasks)})
    completed = [r for r in skill_rows if r['status'] == 'completed']
    not_run = [r for r in skill_rows if r['status'] == 'not_run']
    completed.sort(key=lambda r: r['delta_pass_rate_mean'], reverse=True)
    return {'experiment_info': {'domain': domain, 'timestamp': datetime.now().isoformat(), 'num_trials': num_trials, 'total_skills': len(all_skill_results), 'completed_skills': len(completed), 'pending_skills': len(not_run), 'baseline_pass_rate_mean': round(baseline_pass_rate, 4), 'baseline_majority_pass_rate': round(baseline_majority_pass_rate, 4)}, 'utility_ranking': completed, 'not_run': not_run}

def print_utility_table(report: dict):
    info = report['experiment_info']
    print(f"\n{'=' * 90}")
    print(f"  SINGLE-SKILL UTILITY RANKING — {info['domain'].upper()}")
    num_k = info.get('num_trials', 5)
    print(f"  Baseline pass@{num_k} mean: {info['baseline_pass_rate_mean']:.1%}")
    print(f"  Completed: {info['completed_skills']}/{info['total_skills']}")
    print(f"{'=' * 90}")
    print(f"  {'Rank':>4} {'Skill Name':<45} {'Freq':>5} {'Pass@' + str(num_k):>8} {'Δ Mean':>8} {'Δ Maj':>8} {'Net':>4}")
    print(f"  {'-' * 4} {'-' * 45} {'-' * 5} {'-' * 8} {'-' * 8} {'-' * 8} {'-' * 4}")
    for (i, row) in enumerate(report['utility_ranking']):
        delta_str = f"{row['delta_pass_rate_mean']:+.1%}"
        delta_maj_str = f"{row['delta_majority_pass_rate']:+.1%}"
        net_str = f"{row['net_task_change']:+d}"
        print(f"  {i + 1:>4} {row['skill_name']:<45} {row['frequency']:>5} {row['pass_rate_mean']:>7.1%} {delta_str:>8} {delta_maj_str:>8} {net_str:>4}")
    if report['not_run']:
        print(f"\n  Pending: {len(report['not_run'])} skills not yet tested")
    print(f"{'=' * 90}")

def main():
    parser = argparse.ArgumentParser(description='Step 5: Single-skill utility testing')
    parser.add_argument('--ranked-json', required=True, help='Path to skills_ranked.json from deduplication')
    parser.add_argument('--domain', default='airline', help='Domain name (default: airline)')
    parser.add_argument('-j', '--concurrency', type=int, default=4, help='Concurrent task count (default: 4)')
    parser.add_argument('-k', '--num-trials', type=int, default=3, help='Number of trials per skill (default: 3)')
    parser.add_argument('--start-trial', type=int, default=0, help='Starting trial index (default: 0). Use to extend existing trials. E.g., --start-trial 3 -k 5 runs only trials 3 and 4.')
    parser.add_argument('--skill-range', nargs=2, type=int, default=None, metavar=('START', 'END'), help='Only test skills in index range [START, END)')
    parser.add_argument('--output-dir', default='<MODEL_ROOT>/agent/skills/single-skill-data', help='Output directory for results')
    parser.add_argument('--dry-run', action='store_true', help='Show what would be run without executing')
    parser.add_argument('--skip-baseline', action='store_true', help='Skip baseline run (use existing)')
    parser.add_argument('--report-only', action='store_true', help='Only generate report from existing results')
    parser.add_argument('--save-trajectories', action='store_true', help='Save full conversation trajectories (messages) for each result')
    args = parser.parse_args()
    ranked_path = Path(args.ranked_json)
    ranked = json.loads(ranked_path.read_text())
    unique_skills = get_unique_skills(ranked)
    print(f'Loaded {len(ranked)} ranked skills, {len(unique_skills)} unique names')
    if args.skill_range:
        (start, end) = args.skill_range
        unique_skills = unique_skills[start:end]
        print(f'Filtered to skill index [{start}, {end}): {len(unique_skills)} skills')
    base_dir = Path('temp')
    parsed_dir = ranked_path.parent.parent / 'parsed'
    skills_dir = base_dir / 'skills' / args.domain
    config_dir = base_dir / 'configs' / 'single_skill'
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    baseline_config_path = base_dir / 'configs' / 'baseline.yaml'
    template = {}
    if baseline_config_path.exists():
        import yaml
        template = yaml.safe_load(baseline_config_path.read_text()) or {}
    if 'task_split' not in template:
        template['task_split'] = 'base'
    print(f"\n{'=' * 70}")
    print(f'  STEP 0: Baseline (no skills)')
    print(f"{'=' * 70}")
    baseline_config_name = f'{args.domain}_baseline'
    baseline_result_path = check_baseline_result(output_dir, args.domain, args.num_trials)
    if baseline_result_path:
        print(f'  Found existing baseline: {baseline_result_path.name}')
        baseline_data = json.loads(baseline_result_path.read_text())
    elif args.skip_baseline or args.report_only:
        print(f'  Skipping baseline (--skip-baseline or --report-only)')
        baseline_data = None
    elif args.dry_run:
        print(f'  [DRY RUN] Would run baseline with -k {args.num_trials}')
        baseline_data = None
    else:
        bl_config_path = config_dir / f'{baseline_config_name}.yaml'
        config_dir.mkdir(parents=True, exist_ok=True)
        lines = [f'# Baseline: no skills', f'domain: {args.domain}', f'skills: []', f"model: {template.get('model', 'openai/Qwen/Qwen3-235B-A22B-Instruct-2507')}", f'llm_args:', f"  api_base: {template.get('llm_args', {}).get('api_base', 'http://localhost:32800/v1')}", f"  api_key: {template.get('llm_args', {}).get('api_key', 'not-needed')}", f"  temperature: {template.get('llm_args', {}).get('temperature', 0.0)}", f"task_split: {template.get('task_split', 'base')}", f'task_ids: null', f"max_steps: {template.get('max_steps', 200)}", f"seed: {template.get('seed', 300)}"]
        bl_config_path.write_text('\n'.join(lines) + '\n')
        print(f'  Running baseline...')
        start = time.time()
        rc = run_single_experiment(bl_config_path, args.concurrency, args.num_trials, output_dir, args.start_trial, args.save_trajectories)
        elapsed = time.time() - start
        print(f'  Baseline completed in {elapsed:.0f}s (rc={rc})')
        baseline_result_path = check_baseline_result(output_dir, args.domain, args.num_trials)
        if baseline_result_path:
            baseline_data = json.loads(baseline_result_path.read_text())
        else:
            bl_files = sorted(output_dir.glob(f'{baseline_config_name}_*.json'))
            baseline_data = json.loads(bl_files[-1].read_text()) if bl_files else None
    if baseline_data:
        bs = baseline_data.get('summary', {})
        print(f"  Baseline pass@{args.num_trials} mean: {bs.get('pass_rate_mean', bs.get('pass_rate', 0)):.1%}")
    print(f"\n{'=' * 70}")
    print(f'  STEP 1: Single-Skill Tests ({len(unique_skills)} skills)')
    print(f"{'=' * 70}")
    to_run = []
    already_done = []
    all_skill_results = []
    for (idx, skill) in enumerate(unique_skills):
        skill_name = skill['skill_name']
        existing = check_existing_result(output_dir, args.domain, skill_name, args.num_trials)
        skill_info = {'skill_name': skill_name, 'frequency': skill['frequency'], 'cluster_size': skill['cluster_size'], 'index': idx}
        if existing:
            already_done.append(skill_name)
            skill_info['aggregate_data'] = json.loads(existing.read_text())
        else:
            to_run.append(skill)
            skill_info['aggregate_data'] = None
        all_skill_results.append(skill_info)
    print(f'  Already completed: {len(already_done)}')
    print(f'  To run: {len(to_run)}')
    if args.dry_run:
        print(f'\n  [DRY RUN] Would run {len(to_run)} single-skill experiments:')
        for (i, s) in enumerate(to_run):
            print(f"    {i + 1:3d}. {s['skill_name']:<45s} (freq={s['frequency']})")
        est_minutes = len(to_run) * 40
        print(f'\n  Estimated total time: ~{est_minutes} min ({est_minutes / 60:.1f} hours)')
    elif not args.report_only:
        total_to_run = len(to_run)
        start_all = time.time()
        for (run_idx, skill) in enumerate(to_run):
            skill_name = skill['skill_name']
            config_name = f'{args.domain}_single_{skill_name}'
            print(f"\n{'─' * 70}")
            print(f"  [{run_idx + 1}/{total_to_run}] Testing: {skill_name} (freq={skill['frequency']}, cluster={skill['cluster_size']})")
            print(f"{'─' * 70}")
            module_path = install_single_skill(skill, parsed_dir, skills_dir)
            config_path = generate_single_skill_config(args.domain, module_path, skill_name, config_dir, template)
            start = time.time()
            rc = run_single_experiment(config_path, args.concurrency, args.num_trials, output_dir, args.start_trial, args.save_trajectories)
            elapsed = time.time() - start
            elapsed_total = time.time() - start_all
            avg_per_skill = elapsed_total / (run_idx + 1)
            remaining = avg_per_skill * (total_to_run - run_idx - 1)
            print(f'  Completed in {elapsed:.0f}s (rc={rc})')
            print(f'  Progress: {run_idx + 1}/{total_to_run} | Elapsed: {elapsed_total / 60:.0f}m | ETA: {remaining / 60:.0f}m')
            existing = check_existing_result(output_dir, args.domain, skill_name, args.num_trials)
            if existing:
                agg_data = json.loads(existing.read_text())
            else:
                result_files = sorted(output_dir.glob(f'{config_name}_*.json'))
                agg_data = json.loads(result_files[-1].read_text()) if result_files else None
            for sr in all_skill_results:
                if sr['skill_name'] == skill_name:
                    sr['aggregate_data'] = agg_data
                    break
    print(f"\n{'=' * 70}")
    print(f'  STEP 2: Generating Summary Report')
    print(f"{'=' * 70}")
    report = build_summary_report(all_skill_results, baseline_data, args.domain, args.num_trials)
    print_utility_table(report)
    report_path = output_dir / f"single_skill_report_{args.domain}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False))
    print(f'\n  Report saved to: {report_path}')
    csv_path = output_dir / f"single_skill_ranking_{args.domain}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
    lines = [f'# Single-Skill Utility Ranking — {args.domain.upper()}', f"# Baseline pass@{args.num_trials} mean: {report['experiment_info']['baseline_pass_rate_mean']:.4f}", f'# Generated: {datetime.now().isoformat()}', f'#', f"# {'Rank':>4} {'Skill':<45} {'Freq':>5} {'Pass@' + str(args.num_trials):>8} {'ΔMean':>8} {'ΔMaj':>8} {'Net':>4}"]
    for (i, row) in enumerate(report['utility_ranking']):
        lines.append(f"  {i + 1:>4} {row['skill_name']:<45} {row['frequency']:>5} {row['pass_rate_mean']:>7.4f} {row['delta_pass_rate_mean']:>+7.4f} {row['delta_majority_pass_rate']:>+7.4f} {row['net_task_change']:>+4d}")
    csv_path.write_text('\n'.join(lines) + '\n')
    print(f'  Ranking saved to: {csv_path}')
if __name__ == '__main__':
    main()
