import argparse
import glob
import json
import os
import shutil
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

def select_top_n_unique(ranked_skills: list[dict], n: int) -> list[dict]:
    seen_names = set()
    selected = []
    for skill in ranked_skills:
        name = skill['skill_name']
        if name in seen_names:
            continue
        seen_names.add(name)
        selected.append(skill)
        if len(selected) >= n:
            break
    return selected

def install_skills(selected_skills: list[dict], parsed_dir: Path, skills_dir: Path) -> list[str]:
    skills_dir.mkdir(parents=True, exist_ok=True)
    init_file = skills_dir / '__init__.py'
    if not init_file.exists():
        init_file.write_text('')
    module_paths = []
    for skill in selected_skills:
        src = parsed_dir / skill['file']
        dst = skills_dir / f"{skill['skill_name']}.py"
        shutil.copy2(src, dst)
        domain = skills_dir.name
        module_paths.append(f"{domain}/{skill['skill_name']}")
    return module_paths

def generate_config(domain: str, module_paths: list[str], n: int, config_dir: Path, template: dict) -> Path:
    config_dir.mkdir(parents=True, exist_ok=True)
    config_path = config_dir / f'{domain}_top{n}_skills.yaml'
    lines = [f'# Scaling experiment: top-{n} skills injected', f'domain: {domain}', f'skills:']
    for mp in module_paths:
        lines.append(f'  - {mp}')
    lines.extend([f"model: {template.get('model', 'openai/Qwen/Qwen3-235B-A22B-Instruct-2507')}", f'llm_args:', f"  api_base: {template.get('llm_args', {}).get('api_base', 'http://localhost:32800/v1')}", f"  api_key: {template.get('llm_args', {}).get('api_key', 'not-needed')}", f"  temperature: {template.get('llm_args', {}).get('temperature', 0.0)}", f"task_split: {template.get('task_split', 'base')}", f'task_ids: null', f"max_steps: {template.get('max_steps', 200)}", f"seed: {template.get('seed', 300)}"])
    config_path.write_text('\n'.join(lines) + '\n')
    return config_path

def find_new_files(output_dir: Path, before_files: set[str], config_name: str) -> list[Path]:
    all_files = set((str(p) for p in output_dir.glob('*.json')))
    new_files = all_files - before_files
    matched = [Path(f) for f in new_files if config_name in Path(f).name]
    return sorted(matched, key=lambda p: p.stat().st_mtime)

def extract_metrics_from_result(result_path: Path) -> dict:
    data = json.loads(result_path.read_text())
    summary = data.get('summary', {})
    per_task = []
    for r in data.get('results', []):
        per_task.append({'task_id': r['task_id'], 'reward': r['reward'], 'num_messages': r['num_messages'], 'num_tool_calls': r['num_tool_calls'], 'tool_sequence': r.get('tool_sequence', [])})
    return {'file': result_path.name, 'pass_rate': summary.get('pass_rate', 0), 'pass_count': summary.get('pass_count', 0), 'num_tasks': summary.get('num_tasks', 0), 'avg_reward': summary.get('avg_reward', 0), 'avg_messages': summary.get('avg_messages', 0), 'avg_tool_calls': summary.get('avg_tool_calls', 0), 'seed': summary.get('seed', 0), 'per_task': per_task}

def extract_metrics_from_aggregate(agg_path: Path) -> dict:
    data = json.loads(agg_path.read_text())
    summary = data.get('summary', {})
    per_task = []
    for r in data.get('results', []):
        per_task.append({'task_id': r.get('task_id', ''), 'reward': r.get('majority_reward', r.get('reward', 0)), 'num_messages': r.get('avg_messages', r.get('num_messages', 0)), 'num_tool_calls': r.get('avg_tool_calls', r.get('num_tool_calls', 0))})
    return {'file': agg_path.name, 'pass_rate_mean': summary.get('pass_rate_mean', summary.get('pass_rate', 0)), 'pass_rate_std': summary.get('pass_rate_std', 0), 'pass_counts': summary.get('pass_counts', [summary.get('pass_count', 0)]), 'majority_pass_rate': summary.get('majority_pass_rate', summary.get('pass_rate', 0)), 'majority_pass_count': summary.get('majority_pass_count', summary.get('pass_count', 0)), 'num_tasks': summary.get('num_tasks', 0), 'avg_messages': summary.get('avg_messages', 0), 'avg_tool_calls': summary.get('avg_tool_calls', 0), 'per_task': per_task, 'trial_summaries': data.get('trial_summaries', [])}

def collect_experiment_results(output_dir: Path, before_files: set[str], config_name: str, num_trials: int) -> dict:
    new_files = find_new_files(output_dir, before_files, config_name)
    if not new_files:
        return {'error': 'No result files found', 'files': []}
    result = {'files': [str(f.name) for f in new_files]}
    if num_trials > 1:
        agg_files = [f for f in new_files if '_agg' in f.name]
        trial_files = [f for f in new_files if '_trial' in f.name]
        if agg_files:
            result['aggregate'] = extract_metrics_from_aggregate(agg_files[-1])
        result['trials'] = []
        for tf in trial_files:
            result['trials'].append(extract_metrics_from_result(tf))
    else:
        non_log = [f for f in new_files if 'scaling_log' not in f.name]
        if non_log:
            metrics = extract_metrics_from_result(non_log[-1])
            result['single_trial'] = metrics
    return result

def run_experiment(config_path: Path, concurrency: int, num_trials: int, output_dir: Path, save_trajectories: bool, start_trial: int=0) -> dict:
    config_name = config_path.stem
    output_dir.mkdir(parents=True, exist_ok=True)
    before_files = set((str(p) for p in output_dir.glob('*.json')))
    cmd = [sys.executable, 'temp/run_experiment.py', '--config', str(config_path), '-j', str(concurrency), '-k', str(num_trials), '--output-dir', str(output_dir), '--start-trial', str(start_trial)]
    if save_trajectories:
        cmd.append('--save-trajectories')
    print(f"\n{'=' * 70}")
    print(f'  Running: {config_path.name}')
    print(f"  Command: {' '.join(cmd)}")
    print(f"{'=' * 70}")
    start = time.time()
    proc = subprocess.run(cmd, capture_output=False)
    elapsed = time.time() - start
    experiment_results = collect_experiment_results(output_dir, before_files, config_name, num_trials)
    return {'config': config_path.name, 'returncode': proc.returncode, 'elapsed_seconds': round(elapsed, 1), 'metrics': experiment_results}

def build_scaling_summary(all_results: list[dict], num_trials: int) -> dict:
    rows = []
    for r in all_results:
        n = r['n_skills']
        metrics = r.get('metrics', {})
        if num_trials > 1 and 'aggregate' in metrics:
            agg = metrics['aggregate']
            row = {'n_skills': n, 'pass_rate_mean': agg['pass_rate_mean'], 'pass_rate_std': agg['pass_rate_std'], 'pass_counts': agg['pass_counts'], 'majority_pass_rate': agg['majority_pass_rate'], 'majority_pass_count': agg['majority_pass_count'], 'num_tasks': agg['num_tasks'], 'avg_messages': agg['avg_messages'], 'avg_tool_calls': agg['avg_tool_calls']}
        elif 'single_trial' in metrics:
            st = metrics['single_trial']
            row = {'n_skills': n, 'pass_rate_mean': st['pass_rate'], 'pass_rate_std': 0, 'pass_counts': [st['pass_count']], 'majority_pass_rate': st['pass_rate'], 'majority_pass_count': st['pass_count'], 'num_tasks': st['num_tasks'], 'avg_messages': st['avg_messages'], 'avg_tool_calls': st['avg_tool_calls']}
        else:
            row = {'n_skills': n, 'pass_rate_mean': -1, 'pass_rate_std': 0, 'pass_counts': [], 'num_tasks': 0, 'error': 'No metrics collected'}
        rows.append(row)
    return {'scaling_table': rows}

def build_per_task_comparison(all_results: list[dict], num_trials: int) -> list[dict]:
    task_data = {}
    for r in all_results:
        n = r['n_skills']
        metrics = r.get('metrics', {})
        per_task = []
        if num_trials > 1 and 'aggregate' in metrics:
            per_task = metrics['aggregate'].get('per_task', [])
        elif 'single_trial' in metrics:
            per_task = metrics['single_trial'].get('per_task', [])
        for t in per_task:
            tid = t['task_id']
            if tid not in task_data:
                task_data[tid] = {}
            task_data[tid][n] = {'reward': t['reward'], 'num_tool_calls': t.get('num_tool_calls', 0)}
    comparison = []
    for tid in sorted(task_data.keys(), key=lambda x: int(x) if x.isdigit() else x):
        entry = {'task_id': tid}
        for (n, data) in sorted(task_data[tid].items()):
            entry[f'top{n}_reward'] = data['reward']
            entry[f'top{n}_tools'] = data['num_tool_calls']
        comparison.append(entry)
    return comparison

def print_scaling_table(summary: dict, domain: str):
    rows = summary['scaling_table']
    print(f"\n{'=' * 70}")
    print(f'  SCALING EXPERIMENT RESULTS — {domain.upper()}')
    print(f"{'=' * 70}")
    print(f"  {'N Skills':>10} | {'Pass Rate':>12} | {'Pass Count':>12} | {'Avg Msgs':>10} | {'Avg Tools':>10}")
    print(f"  {'-' * 10}-+-{'-' * 12}-+-{'-' * 12}-+-{'-' * 10}-+-{'-' * 10}")
    for row in rows:
        n = row['n_skills']
        if row.get('error'):
            print(f"  {n:>10} | {'ERROR':>12} |")
            continue
        if row['pass_rate_std'] > 0:
            pr_str = f"{row['pass_rate_mean']:.1%}±{row['pass_rate_std']:.1%}"
            pc_str = '/'.join((str(c) for c in row['pass_counts']))
        else:
            pr_str = f"{row['pass_rate_mean']:.1%}"
            pc_str = str(row['pass_counts'][0]) if row['pass_counts'] else '?'
        print(f"  {n:>10} | {pr_str:>12} | {pc_str:>12} | {row.get('avg_messages', 0):>10.1f} | {row.get('avg_tool_calls', 0):>10.1f}")
    print(f"{'=' * 70}")

def main():
    parser = argparse.ArgumentParser(description='Step 4: Run scaling experiment with top-N skills')
    parser.add_argument('--ranked-json', required=True, help='Path to skills_ranked.json from deduplication')
    parser.add_argument('--domain', default='airline', help='Domain name (default: airline)')
    parser.add_argument('--scales', nargs='+', type=int, default=[4, 8, 16, 32, 64], help='Scale points to test (default: 4 8 16 32 64)')
    parser.add_argument('-j', '--concurrency', type=int, default=4, help='Concurrent task count (default: 4)')
    parser.add_argument('-k', '--num-trials', type=int, default=3, help='Number of trials per config (default: 3)')
    parser.add_argument('--start-trial', type=int, default=0, help='Starting trial index (default: 0). Use to extend existing trials. E.g., --start-trial 3 -k 5 runs only trials 3 and 4.')
    parser.add_argument('--prepare-only', action='store_true', help="Only generate configs, don't run experiments")
    parser.add_argument('--save-trajectories', action='store_true', help='Save full conversation trajectories')
    parser.add_argument('--output-dir', default='<MODEL_ROOT>/agent/skills/scaling-data', help='Output directory for results')
    args = parser.parse_args()
    ranked = json.loads(Path(args.ranked_json).read_text())
    print(f'Loaded {len(ranked)} ranked skills from {args.ranked_json}')
    unique_names = []
    seen = set()
    for s in ranked:
        if s['skill_name'] not in seen:
            seen.add(s['skill_name'])
            unique_names.append(s['skill_name'])
    print(f'Unique skill names: {len(unique_names)}')
    max_scale = max(args.scales)
    if max_scale > len(unique_names):
        print(f'WARNING: max scale {max_scale} > unique skills {len(unique_names)}. Capping at {len(unique_names)}.')
        args.scales = [s for s in args.scales if s <= len(unique_names)]
    base_dir = Path('temp')
    parsed_dir = Path(args.ranked_json).parent.parent / 'parsed'
    skills_dir = base_dir / 'skills' / args.domain
    config_dir = base_dir / 'configs' / 'scaling'
    output_dir = Path(args.output_dir)
    baseline_config_path = base_dir / 'configs' / 'baseline.yaml'
    template = {}
    if baseline_config_path.exists():
        import yaml
        template = yaml.safe_load(baseline_config_path.read_text()) or {}
    configs = []
    print(f'\n--- Preparing {len(args.scales)} scale points ---')
    for n in args.scales:
        selected = select_top_n_unique(ranked, n)
        module_paths = install_skills(selected, parsed_dir, skills_dir)
        config_path = generate_config(args.domain, module_paths, n, config_dir, template)
        configs.append({'n': n, 'config_path': config_path, 'skills': [{'name': s['skill_name'], 'frequency': s['frequency'], 'cluster_size': s['cluster_size']} for s in selected]})
        print(f'  top-{n:2d}: {len(selected)} unique skills, config → {config_path}')
    print(f'\n--- Skill Selection Overview ---')
    max_n = max(args.scales)
    all_selected = select_top_n_unique(ranked, max_n)
    for (i, s) in enumerate(all_selected):
        markers = ' '.join((f'[top-{n}]' for n in args.scales if i < n))
        freq_info = f"freq={s['frequency']}, cluster_size={s['cluster_size']}"
        print(f"  #{i:2d} {s['skill_name']:<45s} ({freq_info}) {markers}")
    if args.prepare_only:
        print(f'\n--prepare-only: configs generated, not running experiments.')
        print(f'To run manually:')
        for c in configs:
            print(f"  python temp/run_experiment.py --config {c['config_path']} -j {args.concurrency} -k {args.num_trials}")
        return
    print(f'\n--- Running {len(configs)} experiments ---')
    print(f'  Concurrency: {args.concurrency}')
    print(f'  Trials: {args.num_trials}')
    print(f'  Estimated time: ~{len(configs) * 45} min')
    output_dir.mkdir(parents=True, exist_ok=True)
    all_results = []
    for c in configs:
        result = run_experiment(c['config_path'], args.concurrency, args.num_trials, output_dir, args.save_trajectories, args.start_trial)
        result['n_skills'] = c['n']
        result['skills'] = c['skills']
        all_results.append(result)
    scaling_summary = build_scaling_summary(all_results, args.num_trials)
    per_task_comparison = build_per_task_comparison(all_results, args.num_trials)
    print_scaling_table(scaling_summary, args.domain)
    report = {'experiment_info': {'timestamp': datetime.now().isoformat(), 'domain': args.domain, 'model': template.get('model', 'openai/Qwen/Qwen3-235B-A22B-Instruct-2507'), 'scales': args.scales, 'concurrency': args.concurrency, 'num_trials': args.num_trials, 'ranked_json': args.ranked_json, 'total_elapsed_seconds': sum((r['elapsed_seconds'] for r in all_results))}, 'scaling_summary': scaling_summary['scaling_table'], 'per_scale_details': [{'n_skills': r['n_skills'], 'skills': r['skills'], 'config': r['config'], 'returncode': r['returncode'], 'elapsed_seconds': r['elapsed_seconds'], 'metrics': r['metrics']} for r in all_results], 'per_task_comparison': per_task_comparison}
    log_path = output_dir / f"scaling_report_{args.domain}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    log_path.write_text(json.dumps(report, indent=2, ensure_ascii=False))
    print(f'\nScaling report: {log_path}')
    print(f"Total time: {report['experiment_info']['total_elapsed_seconds']:.0f}s")
if __name__ == '__main__':
    main()
