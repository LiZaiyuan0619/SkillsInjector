import argparse
import json
from pathlib import Path

def classify_task(rewards: list[float]) -> str:
    if all((r == 0.0 for r in rewards)):
        return 'ALL_FAIL'
    elif all((r == 1.0 for r in rewards)):
        return 'ALL_PASS'
    else:
        return 'MIXED'

def main():
    parser = argparse.ArgumentParser(description='Step 1: Classify tasks by trial outcomes')
    parser.add_argument('--trial-files', nargs='+', required=True, help='Trial result JSON files (3 files, one per trial)')
    parser.add_argument('--output', type=str, required=True, help='Output classification JSON file')
    args = parser.parse_args()
    trials = []
    seeds = []
    for f in args.trial_files:
        data = json.loads(Path(f).read_text())
        trials.append(data)
        seeds.append(data['summary']['seed'])
    domain = trials[0]['summary']['domain']
    print(f'Domain: {domain}, Seeds: {seeds}')
    task_rewards = {}
    for trial in trials:
        for r in trial['results']:
            tid = str(r['task_id'])
            if tid not in task_rewards:
                task_rewards[tid] = []
            task_rewards[tid].append(r['reward'])
    try:
        sorted_ids = sorted(task_rewards.keys(), key=lambda x: int(x))
    except ValueError:
        sorted_ids = sorted(task_rewards.keys())
    classifications = {}
    counts = {'ALL_FAIL': 0, 'ALL_PASS': 0, 'MIXED': 0}
    for tid in sorted_ids:
        rewards = task_rewards[tid]
        category = classify_task(rewards)
        classifications[tid] = {'category': category, 'rewards': rewards, 'pass_count': sum((1 for r in rewards if r == 1.0))}
        counts[category] += 1
    total = len(classifications)
    print(f'\nClassification Summary ({total} tasks):')
    print(f"  ALL_FAIL:  {counts['ALL_FAIL']:3d} ({counts['ALL_FAIL'] / total:.1%})")
    print(f"  ALL_PASS:  {counts['ALL_PASS']:3d} ({counts['ALL_PASS'] / total:.1%})")
    print(f"  MIXED:     {counts['MIXED']:3d} ({counts['MIXED'] / total:.1%})")
    for cat in ['ALL_FAIL', 'ALL_PASS', 'MIXED']:
        task_list = [tid for (tid, c) in classifications.items() if c['category'] == cat]
        print(f'\n  {cat}: {task_list}')
    output = {'domain': domain, 'seeds': seeds, 'total_tasks': total, 'counts': counts, 'classifications': classifications}
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(output, indent=2, ensure_ascii=False))
    print(f'\nSaved to {out_path}')
if __name__ == '__main__':
    main()
