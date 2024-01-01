import argparse
import json
import math
import re
from collections import defaultdict
from datetime import datetime
from pathlib import Path
SKILLS_ROOT = '<MODEL_ROOT>/agent/skills'
SCALING_DIRS = {'naive': f'{SKILLS_ROOT}/scaling-data', 'mmr': f'{SKILLS_ROOT}/scaling-data-mmr', 'utility': f'{SKILLS_ROOT}/scaling-data-utility'}
SINGLE_SKILL_DIR = f'{SKILLS_ROOT}/single-skill-data'
PAT_SCALING_TRIAL = re.compile('^(?P<domain>[^_]+)_top(?P<n>\\d+)_skills_trial(?P<trial>\\d+)_seed(?P<seed>\\d+)_(?P<ts>\\d{8}_\\d{6})\\.json$')
PAT_SINGLE_TRIAL = re.compile('^(?P<domain>[^_]+)_single_(?P<skill>.+)_trial(?P<trial>\\d+)_seed(?P<seed>\\d+)_(?P<ts>\\d{8}_\\d{6})\\.json$')
PAT_BASELINE_TRIAL = re.compile('^(?P<domain>[^_]+)_baseline_trial(?P<trial>\\d+)_seed(?P<seed>\\d+)_(?P<ts>\\d{8}_\\d{6})\\.json$')

def mean(xs):
    return sum(xs) / len(xs) if xs else 0

def std(xs):
    if len(xs) <= 1:
        return 0
    m = mean(xs)
    return math.sqrt(sum(((x - m) ** 2 for x in xs)) / len(xs))

def aggregate_trial_files(trial_paths: list[Path]) -> dict:
    trials = []
    for p in sorted(trial_paths, key=lambda x: x.name):
        trials.append(json.loads(p.read_text()))
    num_trials = len(trials)
    if num_trials == 1:
        return trials[0]
    pass_rates = [t['summary']['pass_rate'] for t in trials]
    pass_counts = [t['summary']['pass_count'] for t in trials]
    avg_messages_list = [t['summary']['avg_messages'] for t in trials]
    avg_tools_list = [t['summary']['avg_tool_calls'] for t in trials]
    task_pass_fractions = defaultdict(list)
    for trial in trials:
        for r in trial['results']:
            task_pass_fractions[r['task_id']].append(r['reward'])
    aggregated_results = []
    for (task_id, rewards) in sorted(task_pass_fractions.items(), key=lambda x: int(x[0]) if x[0].isdigit() else x[0]):
        aggregated_results.append({'task_id': task_id, 'reward': mean(rewards), 'pass_count': sum((1 for r in rewards if r == 1.0)), 'num_trials': len(rewards)})
    majority_pass = sum((1 for r in aggregated_results if r['reward'] > 0.5))
    aggregated_summary = {**trials[0]['summary'], 'num_trials': num_trials, 'seeds': [t['summary']['seed'] for t in trials], 'pass_rates': pass_rates, 'pass_counts': pass_counts, 'pass_rate_mean': mean(pass_rates), 'pass_rate_std': std(pass_rates), 'pass_count_mean': mean(pass_counts), 'pass_count_std': std(pass_counts), 'majority_pass_count': majority_pass, 'majority_pass_rate': majority_pass / trials[0]['summary']['num_tasks'], 'avg_messages_mean': mean(avg_messages_list), 'avg_tool_calls_mean': mean(avg_tools_list)}
    return {'summary': aggregated_summary, 'results': aggregated_results, 'trial_summaries': [t['summary'] for t in trials]}

def reaggregate_scaling(data_dir: Path, domain: str, target_k: int, dry_run: bool=False) -> dict:
    groups = defaultdict(dict)
    for f in sorted(data_dir.glob('*.json')):
        m = PAT_SCALING_TRIAL.match(f.name)
        if m and m.group('domain') == domain:
            n = int(m.group('n'))
            trial = int(m.group('trial'))
            groups[n][trial] = f
    if not groups:
        print(f'  No trial files found for domain={domain} in {data_dir}')
        return {}
    print(f'\n  Re-aggregating scaling data: {data_dir}')
    print(f'  Domain: {domain}, Target k: {target_k}')
    results = {}
    scaling_summary = []
    per_task_comparison = []
    task_data = {}
    for n in sorted(groups.keys()):
        trial_dict = groups[n]
        trial_indices = sorted(trial_dict.keys())
        trial_paths = [trial_dict[i] for i in trial_indices]
        actual_k = len(trial_paths)
        status = 'OK' if actual_k >= target_k else f'INCOMPLETE ({actual_k}/{target_k})'
        print(f'    top{n:>3d}: {actual_k} trials {trial_indices} [{status}]', end='')
        if dry_run:
            print()
            continue
        agg = aggregate_trial_files(trial_paths)
        s = agg['summary']
        ts = datetime.now().strftime('%Y%m%d_%H%M%S')
        agg_filename = f'{domain}_top{n}_skills_agg{actual_k}_{ts}.json'
        agg_path = data_dir / agg_filename
        agg_save = {'summary': agg['summary'], 'results': agg['results'], 'trial_summaries': agg.get('trial_summaries', [])}
        agg_path.write_text(json.dumps(agg_save, indent=2, ensure_ascii=False))
        print(f" -> {agg_filename}  pass_rate={s['pass_rate_mean']:.4f}±{s['pass_rate_std']:.4f}")
        results[n] = agg
        scaling_summary.append({'n_skills': n, 'pass_rate_mean': s['pass_rate_mean'], 'pass_rate_std': s['pass_rate_std'], 'pass_counts': s['pass_counts'], 'majority_pass_rate': s['majority_pass_rate'], 'majority_pass_count': s['majority_pass_count'], 'num_tasks': s['num_tasks'], 'avg_messages': s.get('avg_messages_mean', s.get('avg_messages', 0)), 'avg_tool_calls': s.get('avg_tool_calls_mean', s.get('avg_tool_calls', 0))})
        for r in agg['results']:
            tid = r['task_id']
            if tid not in task_data:
                task_data[tid] = {}
            task_data[tid][n] = r['reward']
    if dry_run:
        return {}
    for tid in sorted(task_data.keys(), key=lambda x: int(x) if x.isdigit() else x):
        entry = {'task_id': tid}
        for (n, reward) in sorted(task_data[tid].items()):
            entry[f'top{n}_reward'] = reward
            entry[f'top{n}_tools'] = 0
        per_task_comparison.append(entry)
    report = {'experiment_info': {'timestamp': datetime.now().isoformat(), 'domain': domain, 'reaggregated': True, 'target_k': target_k, 'scales': sorted(groups.keys())}, 'scaling_summary': sorted(scaling_summary, key=lambda x: x['n_skills']), 'per_task_comparison': per_task_comparison}
    report_path = data_dir / f"scaling_report_{domain}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False))
    print(f'  Scaling report: {report_path.name}')
    return results

def reaggregate_single_skill(data_dir: Path, domain: str, target_k: int, dry_run: bool=False) -> dict:
    skill_groups = defaultdict(dict)
    baseline_trials = {}
    for f in sorted(data_dir.glob('*.json')):
        m = PAT_BASELINE_TRIAL.match(f.name)
        if m and m.group('domain') == domain:
            trial = int(m.group('trial'))
            baseline_trials[trial] = f
            continue
        m = PAT_SINGLE_TRIAL.match(f.name)
        if m and m.group('domain') == domain:
            skill = m.group('skill')
            trial = int(m.group('trial'))
            skill_groups[skill][trial] = f
    print(f'\n  Re-aggregating single-skill data: {data_dir}')
    print(f'  Domain: {domain}, Target k: {target_k}')
    print(f'  Found {len(skill_groups)} skills, baseline has {len(baseline_trials)} trials')
    if not skill_groups:
        print(f'  No trial files found!')
        return {}
    if baseline_trials and len(baseline_trials) >= target_k:
        bl_paths = [baseline_trials[i] for i in sorted(baseline_trials.keys())[:target_k]]
        if not dry_run:
            bl_agg = aggregate_trial_files(bl_paths)
            ts = datetime.now().strftime('%Y%m%d_%H%M%S')
            bl_filename = f'{domain}_baseline_agg{len(bl_paths)}_{ts}.json'
            bl_path = data_dir / bl_filename
            bl_save = {'summary': bl_agg['summary'], 'results': bl_agg['results'], 'trial_summaries': bl_agg.get('trial_summaries', [])}
            bl_path.write_text(json.dumps(bl_save, indent=2, ensure_ascii=False))
            bl_s = bl_agg['summary']
            print(f"    baseline: {len(bl_paths)} trials -> {bl_filename}  pass_rate={bl_s['pass_rate_mean']:.4f}±{bl_s['pass_rate_std']:.4f}")
    complete = 0
    incomplete = 0
    for skill_name in sorted(skill_groups.keys()):
        trial_dict = skill_groups[skill_name]
        trial_indices = sorted(trial_dict.keys())
        trial_paths = [trial_dict[i] for i in trial_indices]
        actual_k = len(trial_paths)
        if actual_k < target_k:
            incomplete += 1
            if actual_k < 3:
                print(f'    SKIP {skill_name}: only {actual_k} trials')
                continue
        if dry_run:
            status = 'OK' if actual_k >= target_k else f'INCOMPLETE ({actual_k}/{target_k})'
            print(f'    {skill_name}: {actual_k} trials [{status}]')
            continue
        agg = aggregate_trial_files(trial_paths)
        s = agg['summary']
        ts = datetime.now().strftime('%Y%m%d_%H%M%S')
        agg_filename = f'{domain}_single_{skill_name}_agg{actual_k}_{ts}.json'
        agg_path = data_dir / agg_filename
        agg_save = {'summary': agg['summary'], 'results': agg['results'], 'trial_summaries': agg.get('trial_summaries', [])}
        agg_path.write_text(json.dumps(agg_save, indent=2, ensure_ascii=False))
        complete += 1
    if not dry_run:
        print(f'  Done: {complete} skills re-aggregated' + (f', {incomplete} incomplete' if incomplete else ''))
    return {'complete': complete, 'incomplete': incomplete}

def main():
    parser = argparse.ArgumentParser(description='Re-aggregate trial files into combined agg files')
    parser.add_argument('--data-dir', type=str, default=None, help='Data directory to re-aggregate')
    parser.add_argument('--domain', type=str, default='airline', help='Domain name (default: airline)')
    parser.add_argument('--type', choices=['scaling', 'single-skill'], default='scaling', help='Type of data (default: scaling)')
    parser.add_argument('--target-k', type=int, default=5, help='Target number of trials (default: 5)')
    parser.add_argument('--dry-run', action='store_true', help='Show what would be done without writing files')
    parser.add_argument('--all-scaling', action='store_true', help='Re-aggregate all 3 scaling data directories')
    parser.add_argument('--all', action='store_true', help='Re-aggregate everything (3 scaling + single-skill)')
    args = parser.parse_args()
    if args.all or args.all_scaling:
        for (label, dir_path) in SCALING_DIRS.items():
            p = Path(dir_path)
            if p.is_dir():
                print(f"\n{'=' * 60}")
                print(f'  {label.upper()}: {dir_path}')
                print(f"{'=' * 60}")
                reaggregate_scaling(p, args.domain, args.target_k, args.dry_run)
            else:
                print(f'  SKIP {label}: {dir_path} not found')
        if args.all:
            p = Path(SINGLE_SKILL_DIR)
            if p.is_dir():
                print(f"\n{'=' * 60}")
                print(f'  SINGLE-SKILL: {SINGLE_SKILL_DIR}')
                print(f"{'=' * 60}")
                reaggregate_single_skill(p, args.domain, args.target_k, args.dry_run)
    else:
        if not args.data_dir:
            parser.error('--data-dir is required when not using --all/--all-scaling')
        p = Path(args.data_dir)
        if not p.is_dir():
            print(f'ERROR: {args.data_dir} not found')
            return
        if args.type == 'scaling':
            reaggregate_scaling(p, args.domain, args.target_k, args.dry_run)
        else:
            reaggregate_single_skill(p, args.domain, args.target_k, args.dry_run)
if __name__ == '__main__':
    main()
