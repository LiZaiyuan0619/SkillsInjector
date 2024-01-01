import argparse
import json
import sys
import threading
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from copy import deepcopy
from datetime import datetime
from pathlib import Path
import yaml
TEMP_DIR = Path(__file__).parent
if str(TEMP_DIR) not in sys.path:
    sys.path.insert(0, str(TEMP_DIR))

def _latest_matching_file(output_dir: Path, pattern: str, exclude_tokens: tuple[str, ...]=()) -> Path | None:
    matches = [path for path in output_dir.glob(pattern) if all((token not in path.name for token in exclude_tokens))]
    if not matches:
        return None
    return sorted(matches)[-1]

def load_existing_trial_results(output_dir: Path, config_name: str, num_trials: int) -> list[dict]:
    existing_results = []
    for trial_idx in range(num_trials):
        result_path = _latest_matching_file(output_dir, f'{config_name}_trial{trial_idx}_*.json')
        if result_path is None and trial_idx == 0:
            result_path = _latest_matching_file(output_dir, f'{config_name}_*.json', exclude_tokens=('_trial', '_agg'))
        if result_path is None:
            continue
        result = json.loads(result_path.read_text())
        result['config_name'] = config_name
        result['trial_idx'] = trial_idx
        existing_results.append(result)
    existing_results.sort(key=lambda r: r['trial_idx'])
    return existing_results

def load_config(config_path: str) -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)

def _serialize_message(msg):
    d = msg.model_dump()
    d.pop('raw_data', None)
    return d

def _safe_dump(obj):
    if hasattr(obj, 'model_dump'):
        return obj.model_dump()
    elif isinstance(obj, dict):
        return obj
    elif obj is None:
        return None
    else:
        return str(obj)

def run_with_config(config: dict, task_id_override: list=None, max_concurrency: int=1, seed_override: int=None, save_trajectories: bool=False, skip_injection: bool=False) -> dict:
    from skill_injector import SkillInjector
    from tau2.run import run_task, get_tasks
    from tau2.evaluator.evaluator import EvaluationType
    domain = config['domain']
    model = config['model']
    llm_args = config.get('llm_args', {})
    task_ids = task_id_override or config.get('task_ids')
    task_split = config.get('task_split', None)
    max_steps = config.get('max_steps', 200)
    seed = seed_override if seed_override is not None else config.get('seed', 300)
    skill_paths = config.get('skills', [])
    injector = None
    if skill_paths and (not skip_injection):
        injector = SkillInjector()
        for sp in skill_paths:
            injector.load_skill_module(sp)
        print(injector.get_summary())
        injector.apply()
    tasks = get_tasks(task_set_name=domain, task_split_name=task_split, task_ids=[str(t) for t in task_ids] if task_ids else None)
    print(f'\nRunning {len(tasks)} tasks | domain={domain} | skills={len(skill_paths)} | seed={seed} | concurrency={max_concurrency}')
    lock = threading.Lock()
    completed_count = [0]

    def _run_single(task):
        trajectory = None
        try:
            simulation = run_task(domain=domain, task=task, agent='llm_agent', user='user_simulator', llm_agent=model, llm_args_agent=deepcopy(llm_args), llm_user=model, llm_args_user=deepcopy(llm_args), max_steps=max_steps, max_errors=5, evaluation_type=EvaluationType.ALL, seed=seed)
            reward = simulation.reward_info.reward if simulation.reward_info else 0.0
            traces = SkillInjector.extract_tool_traces(simulation)
            num_messages = len(simulation.messages)
            if save_trajectories:
                trajectory = {'messages': [_serialize_message(m) for m in simulation.messages], 'reward_info': _safe_dump(simulation.reward_info), 'termination_reason': str(simulation.termination_reason) if simulation.termination_reason else None, 'task_description': _safe_dump(task.description), 'user_scenario': _safe_dump(task.user_scenario), 'evaluation_criteria': _safe_dump(task.evaluation_criteria)}
        except Exception as e:
            with lock:
                print(f'  Task {task.id} ERROR: {e}')
            reward = 0.0
            traces = []
            num_messages = 0
        with lock:
            completed_count[0] += 1
            print(f'  [{completed_count[0]}/{len(tasks)}] Task {task.id}: reward={reward}, msgs={num_messages}, tools={len(traces)}')
        result = {'task_id': task.id, 'reward': reward, 'num_messages': num_messages, 'num_tool_calls': len(traces), 'tool_sequence': [t['tool'] for t in traces], 'tool_traces': traces}
        if trajectory is not None:
            result['trajectory'] = trajectory
        return result
    results = []
    if max_concurrency <= 1:
        for task in tasks:
            results.append(_run_single(task))
    else:
        with ThreadPoolExecutor(max_workers=max_concurrency) as executor:
            futures = {executor.submit(_run_single, task): task for task in tasks}
            for future in as_completed(futures):
                results.append(future.result())
        results.sort(key=lambda r: int(r['task_id']) if r['task_id'].isdigit() else r['task_id'])
    pass_count = sum((1 for r in results if r['reward'] == 1.0))
    if injector and (not skip_injection):
        injector.restore()
    total = len(results)
    avg_reward = sum((r['reward'] for r in results)) / total if total > 0 else 0
    avg_messages = sum((r['num_messages'] for r in results)) / total if total > 0 else 0
    avg_tools = sum((r['num_tool_calls'] for r in results)) / total if total > 0 else 0
    summary = {'domain': domain, 'model': model, 'num_skills': len(skill_paths), 'skill_names': skill_paths, 'num_tasks': total, 'pass_count': pass_count, 'pass_rate': pass_count / total if total > 0 else 0, 'avg_reward': avg_reward, 'avg_messages': avg_messages, 'avg_tool_calls': avg_tools, 'max_concurrency': max_concurrency, 'seed': seed, 'timestamp': datetime.now().isoformat()}
    print(f"\n{'=' * 60}")
    print(f"  Pass: {pass_count}/{total} ({summary['pass_rate']:.1%})")
    print(f'  Avg Messages: {avg_messages:.1f}')
    print(f'  Avg Tool Calls: {avg_tools:.1f}')
    print(f"{'=' * 60}")
    return {'summary': summary, 'results': results}

def aggregate_trials(trial_results_list: list, config_name: str) -> dict:
    import math
    num_trials = len(trial_results_list)
    if num_trials == 1:
        return trial_results_list[0]
    pass_rates = [r['summary']['pass_rate'] for r in trial_results_list]
    pass_counts = [r['summary']['pass_count'] for r in trial_results_list]
    avg_messages_list = [r['summary']['avg_messages'] for r in trial_results_list]
    avg_tools_list = [r['summary']['avg_tool_calls'] for r in trial_results_list]

    def mean(xs):
        return sum(xs) / len(xs)

    def std(xs):
        m = mean(xs)
        return math.sqrt(sum(((x - m) ** 2 for x in xs)) / len(xs))
    task_pass_fractions = defaultdict(list)
    for trial in trial_results_list:
        for r in trial['results']:
            task_pass_fractions[r['task_id']].append(r['reward'])
    aggregated_results = []
    for (task_id, rewards) in sorted(task_pass_fractions.items(), key=lambda x: int(x[0]) if x[0].isdigit() else x[0]):
        aggregated_results.append({'task_id': task_id, 'reward': mean(rewards), 'pass_count': sum((1 for r in rewards if r == 1.0)), 'num_trials': len(rewards)})
    majority_pass = sum((1 for r in aggregated_results if r['reward'] > 0.5))
    aggregated_summary = {**trial_results_list[0]['summary'], 'num_trials': num_trials, 'seeds': [r['summary']['seed'] for r in trial_results_list], 'pass_rates': pass_rates, 'pass_counts': pass_counts, 'pass_rate_mean': mean(pass_rates), 'pass_rate_std': std(pass_rates), 'pass_count_mean': mean(pass_counts), 'pass_count_std': std(pass_counts), 'majority_pass_count': majority_pass, 'majority_pass_rate': majority_pass / trial_results_list[0]['summary']['num_tasks'], 'avg_messages_mean': mean(avg_messages_list), 'avg_tool_calls_mean': mean(avg_tools_list)}
    return {'summary': aggregated_summary, 'results': aggregated_results, 'trial_results': trial_results_list}

def compare_results(results_list: list, config_names: list):
    print(f"\n{'=' * 70}")
    print('COMPARISON')
    print(f"{'=' * 70}")
    header = f"  {'Metric':<25s}" + ''.join((f'{n:>15s}' for n in config_names))
    print(header)
    print(f"  {'-' * 25}" + ''.join((f"{'-' * 15}" for _ in config_names)))
    any_multi = any(('num_trials' in r['summary'] and r['summary']['num_trials'] > 1 for r in results_list))
    if any_multi:
        metrics = [('Skills', lambda s: str(s['num_skills'])), ('Tasks', lambda s: str(s['num_tasks'])), ('Trials', lambda s: str(s.get('num_trials', 1))), ('Per-trial Pass', lambda s: ' / '.join((str(c) for c in s['pass_counts'])) if 'pass_counts' in s else f"{s['pass_count']}"), ('Per-trial Rate', lambda s: ' / '.join((f'{r:.1%}' for r in s['pass_rates'])) if 'pass_rates' in s else f"{s['pass_rate']:.1%}"), ('Mean Pass Rate', lambda s: f"{s.get('pass_rate_mean', s['pass_rate']):.1%}"), ('Std Pass Rate', lambda s: f"{s.get('pass_rate_std', 0):.1%}"), ('Majority Pass', lambda s: f"{s['majority_pass_count']}/{s['num_tasks']}" if 'majority_pass_count' in s else f"{s['pass_count']}/{s['num_tasks']}"), ('Majority Rate', lambda s: f"{s.get('majority_pass_rate', s['pass_rate']):.1%}"), ('Avg Messages', lambda s: f"{s.get('avg_messages_mean', s['avg_messages']):.1f}"), ('Avg Tool Calls', lambda s: f"{s.get('avg_tool_calls_mean', s['avg_tool_calls']):.1f}")]
    else:
        metrics = [('Skills', lambda s: str(s['num_skills'])), ('Tasks', lambda s: str(s['num_tasks'])), ('Pass', lambda s: f"{s['pass_count']}/{s['num_tasks']}"), ('Pass Rate', lambda s: f"{s['pass_rate']:.1%}"), ('Avg Messages', lambda s: f"{s['avg_messages']:.1f}"), ('Avg Tool Calls', lambda s: f"{s['avg_tool_calls']:.1f}")]
    for (name, fn) in metrics:
        row = f'  {name:<25s}' + ''.join((f"{fn(r['summary']):>15s}" for r in results_list))
        print(row)
    if len(results_list) == 2:
        (a, b) = results_list
        a_by_id = {r['task_id']: r for r in a['results']}
        b_by_id = {r['task_id']: r for r in b['results']}
        improved = []
        degraded = []
        for task_id in a_by_id:
            ra = a_by_id[task_id]
            rb = b_by_id.get(task_id)
            if rb is None:
                continue
            a_pass = ra['reward'] > 0.5 if any_multi else ra['reward'] == 1.0
            b_pass = rb['reward'] > 0.5 if any_multi else rb['reward'] == 1.0
            if not a_pass and b_pass:
                improved.append(task_id)
            elif a_pass and (not b_pass):
                degraded.append(task_id)
        print(f"\n  Improved (0→1): {(improved if improved else 'none')}")
        print(f"  Degraded (1→0): {(degraded if degraded else 'none')}")
        print(f'  Net change: {len(improved) - len(degraded):+d} tasks')
        if any_multi:
            print(f'\n  (Improved/Degraded based on majority vote across trials)')

def main():
    parser = argparse.ArgumentParser(description='Run tau2-bench skill injection experiment')
    parser.add_argument('--config', nargs='+', required=True, help='YAML config file(s)')
    parser.add_argument('--task-ids', nargs='+', type=str, default=None, help='Override task IDs')
    parser.add_argument('--output-dir', type=str, default='temp/results', help='Output directory')
    parser.add_argument('-j', '--concurrency', type=int, default=4, help='Number of concurrent tasks (default: 4)')
    parser.add_argument('-k', '--num-trials', type=int, default=1, help='Number of trials per config (default: 1). Each trial uses a different seed for stable averaging.')
    parser.add_argument('--start-trial', type=int, default=0, help='Starting trial index (default: 0). Use to resume/extend trials. E.g., --start-trial 3 -k 5 runs only trials 3 and 4.')
    parser.add_argument('--save-trajectories', action='store_true', default=False, help='Save full message trajectories per task (for Phase 0 pattern extraction). Increases output file size significantly.')
    args = parser.parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    all_aggregated = []
    config_names = []
    for config_path in args.config:
        config = load_config(config_path)
        config_name = Path(config_path).stem
        config_names.append(config_name)
        base_seed = config.get('seed', 300)
        trial_results = []
        existing_trials_by_idx = {}
        if args.num_trials > 1:
            existing_trials = load_existing_trial_results(output_dir, config_name, args.num_trials)
            if existing_trials:
                loaded = ', '.join((str(r['trial_idx']) for r in existing_trials))
                print(f'Found existing trials for {config_name}: {loaded}')
                for result in existing_trials:
                    existing_trials_by_idx[result['trial_idx']] = result
                trial_results.extend(existing_trials)
        for trial_idx in range(args.start_trial, args.num_trials):
            if trial_idx in existing_trials_by_idx:
                print(f'Reusing existing trial {trial_idx} for {config_name}')
                continue
            trial_seed = base_seed + trial_idx
            print(f"\n{'=' * 70}")
            if args.num_trials > 1:
                print(f'EXPERIMENT: {config_name} (trial {trial_idx + 1}/{args.num_trials}, seed={trial_seed})')
            else:
                print(f'EXPERIMENT: {config_name}')
            print(f"{'=' * 70}")
            result = run_with_config(config, task_id_override=args.task_ids, max_concurrency=args.concurrency, seed_override=trial_seed, save_trajectories=args.save_trajectories)
            result['config_name'] = config_name
            result['config_path'] = config_path
            result['trial_idx'] = trial_idx
            trial_results.append(result)
            if args.num_trials > 1:
                out_file = output_dir / f"{config_name}_trial{trial_idx}_seed{trial_seed}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            else:
                out_file = output_dir / f"{config_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            out_file.write_text(json.dumps(result, ensure_ascii=False, indent=2))
            print(f'  Saved to {out_file}')
        trial_results.sort(key=lambda r: r['trial_idx'])
        aggregated = aggregate_trials(trial_results, config_name)
        aggregated['config_name'] = config_name
        all_aggregated.append(aggregated)
        actual_trials = len(trial_results)
        if actual_trials > 1:
            s = aggregated['summary']
            trial_indices = sorted({r['trial_idx'] for r in trial_results})
            complete_indices = list(range(args.num_trials))
            aggregate_label = args.num_trials if trial_indices == complete_indices else actual_trials
            print(f"\n{'=' * 60}")
            print(f'  AGGREGATE ({config_name}, {actual_trials} trials)')
            print(f"  Per-trial pass: {' / '.join((str(c) for c in s['pass_counts']))}")
            print(f"  Mean Pass Rate: {s['pass_rate_mean']:.1%} ± {s['pass_rate_std']:.1%}")
            print(f"  Majority Pass:  {s['majority_pass_count']}/{s['num_tasks']} ({s['majority_pass_rate']:.1%})")
            print(f"{'=' * 60}")
            agg_file = output_dir / f"{config_name}_agg{aggregate_label}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            agg_save = {'summary': aggregated['summary'], 'results': aggregated['results'], 'trial_summaries': [r['summary'] for r in trial_results]}
            agg_file.write_text(json.dumps(agg_save, ensure_ascii=False, indent=2))
            print(f'  Aggregated saved to {agg_file}')
    if len(all_aggregated) > 1:
        compare_results(all_aggregated, config_names)
if __name__ == '__main__':
    main()
