import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from utils import summarize_trace, save_jsonl
TRIAL_RE = re.compile('^(?P<domain>[a-z]+)_single_(?P<skill>.+)_trial(?P<trial>\\d+)_seed(?P<seed>\\d+)_\\d{8}_\\d{6}\\.json$')

def parse_filename(fname: str) -> dict | None:
    m = TRIAL_RE.match(fname)
    if not m:
        return None
    return {'domain': m.group('domain'), 'skill_name': m.group('skill'), 'trial_idx': int(m.group('trial')), 'seed': int(m.group('seed'))}

def _select_success_traces(traces: list[dict], max_count: int=2) -> list[dict]:
    if not traces:
        return []
    sorted_traces = sorted(traces, key=lambda t: t['num_messages'])
    return sorted_traces[:max_count]

def _select_failure_traces(traces: list[dict], max_count: int=3) -> list[dict]:
    if not traces:
        return []
    sorted_traces = sorted(traces, key=lambda t: t['num_messages'])
    if len(sorted_traces) <= max_count:
        return sorted_traces
    min_msg_threshold = 4
    informative = [t for t in sorted_traces if t['num_messages'] >= min_msg_threshold]
    if len(informative) < max_count:
        informative = sorted_traces
    n = len(informative)
    if n <= max_count:
        return informative
    upper_bound = max(max_count, int(n * 0.8))
    candidates = informative[:upper_bound]
    if len(candidates) <= max_count:
        return candidates
    mid = len(candidates) // 2
    half = max_count // 2
    start = max(0, mid - half)
    end = start + max_count
    if end > len(candidates):
        end = len(candidates)
        start = max(0, end - max_count)
    return candidates[start:end]

def collect_traces_for_domain(traces_dir: Path, domain: str) -> list[dict]:
    skill_trials: dict[str, list[dict]] = defaultdict(list)
    file_count = 0
    for fpath in sorted(traces_dir.glob(f'{domain}_single_*_trial*_seed*.json')):
        info = parse_filename(fpath.name)
        if info is None:
            continue
        if info['domain'] != domain:
            continue
        file_count += 1
        skill_name = info['skill_name']
        trial_idx = info['trial_idx']
        try:
            with open(fpath, encoding='utf-8') as f:
                data = json.load(f)
        except (json.JSONDecodeError, OSError) as e:
            print(f'  [WARN] Failed to load {fpath.name}: {e}')
            continue
        results = data.get('results', [])
        for r in results:
            task_id = r.get('task_id', '?')
            reward = r.get('reward', None)
            trajectory = r.get('trajectory', {})
            if not isinstance(trajectory, dict):
                trajectory = {}
            messages = trajectory.get('messages', [])
            num_messages = len(messages) if messages else r.get('num_messages', 0)
            skill_trials[skill_name].append({'task_id': task_id, 'trial_idx': trial_idx, 'reward': reward, 'num_messages': num_messages, 'trajectory': trajectory})
    print(f'Scanned {file_count} trial files, found {len(skill_trials)} skills')
    records = []
    total_success = 0
    total_failure = 0
    skills_no_success = 0
    skills_no_failure = 0
    for skill_name in sorted(skill_trials.keys()):
        all_traces = skill_trials[skill_name]
        successes = [t for t in all_traces if t['reward'] == 1.0]
        failures = [t for t in all_traces if t['reward'] == 0.0]
        selected_success = _select_success_traces(successes)
        selected_failure = _select_failure_traces(failures)
        for trace in selected_success + selected_failure:
            trace['summary'] = summarize_trace(trace['trajectory'])
        total_success += len(successes)
        total_failure += len(failures)
        if not successes:
            skills_no_success += 1
        if not failures:
            skills_no_failure += 1
        records.append({'skill_name': skill_name, 'num_success': len(successes), 'num_failure': len(failures), 'success_traces': selected_success, 'failure_traces': selected_failure})
    n_skills = len(records)
    print(f"\n{'=' * 60}")
    print(f'Domain: {domain}')
    print(f'Total skills:            {n_skills}')
    print(f'Total success traces:    {total_success} (avg {total_success / max(n_skills, 1):.1f}/skill)')
    print(f'Total failure traces:    {total_failure} (avg {total_failure / max(n_skills, 1):.1f}/skill)')
    print(f'Skills with no success:  {skills_no_success}')
    print(f'Skills with no failure:  {skills_no_failure}')
    print(f"Selected success traces: {sum((len(r['success_traces']) for r in records))}")
    print(f"Selected failure traces: {sum((len(r['failure_traces']) for r in records))}")
    print(f"{'=' * 60}\n")
    return records

def main():
    parser = argparse.ArgumentParser(description='Collect success/failure execution traces for rewriter training.')
    parser.add_argument('--traces-dir', type=str, default='<MODEL_ROOT>/agent/skills/single-skill-data-with-traces', help='Directory containing single-skill trial result files.')
    parser.add_argument('--domain', type=str, default='airline', help='Domain to collect traces for (default: airline).')
    parser.add_argument('--output', type=str, default=None, help='Output JSONL path. Default: data/{domain}/collected_traces.jsonl')
    args = parser.parse_args()
    traces_dir = Path(args.traces_dir)
    if not traces_dir.is_dir():
        print(f'ERROR: Traces directory not found: {traces_dir}')
        sys.exit(1)
    output_path = args.output
    if output_path is None:
        output_path = str(Path(__file__).resolve().parent / 'data' / args.domain / 'collected_traces.jsonl')
    print(f'Traces dir: {traces_dir}')
    print(f'Domain:     {args.domain}')
    print(f'Output:     {output_path}')
    print()
    records = collect_traces_for_domain(traces_dir, args.domain)
    if not records:
        print('No records collected. Check traces directory and domain.')
        sys.exit(1)
    save_jsonl(output_path, records)
    print('Done.')
if __name__ == '__main__':
    main()
