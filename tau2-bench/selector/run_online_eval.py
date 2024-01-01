import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import inspect
import json
import math
import threading
import sys
from pathlib import Path
from typing import Any
import yaml
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
from temp.selector.domain_registry import get_baseline_config_path
from temp.selector.common import resolve_domain_spec

def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding='utf-8').splitlines():
        rows.append(json.loads(line))
    return rows

def _write_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding='utf-8')

def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text('\n'.join((json.dumps(row, sort_keys=True) for row in rows)) + ('\n' if rows else ''), encoding='utf-8')

def _append_jsonl_row(path: Path, row: dict[str, Any]) -> None:
    with path.open('a', encoding='utf-8') as fp:
        fp.write(json.dumps(row, sort_keys=True))
        fp.write('\n')

def _sort_task_key(task_id: str) -> tuple[int, str]:
    return (0, f'{int(task_id):012d}') if task_id.isdigit() else (1, task_id)

def _sort_task_row_key(row: dict[str, Any]) -> tuple[int, str, tuple[int, str]]:
    return (int(row['trial_index']), str(row['trial_seed']), _sort_task_key(str(row['task_id'])))

def _resolve_selection_file(selection_file: str | Path, method: str) -> Path:
    path = Path(selection_file)
    if path.exists():
        return path
    if method == 'random' and path.name == 'random.jsonl':
        matches = sorted(path.parent.glob('random_seed*.jsonl'))
        if len(matches) == 1:
            return matches[0]
    raise FileNotFoundError(path)

def _load_existing_jsonl_rows(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return _read_jsonl(path)

def load_selection_rows(domain: str, method: str, selection_file: str | Path) -> list[dict[str, Any]]:
    spec = resolve_domain_spec(domain)
    rows = _read_jsonl(_resolve_selection_file(selection_file, method))
    if not rows:
        raise ValueError('selection file must not be empty')
    seen_task_ids: set[str] = set()
    normalized_rows: list[dict[str, Any]] = []
    for row in rows:
        task_id = str(row['task_id'])
        if task_id in seen_task_ids:
            raise ValueError(f'duplicate task_id in selection file: {task_id}')
        seen_task_ids.add(task_id)
        if str(row['domain']) != spec.domain:
            raise ValueError('selection row domain does not match requested domain')
        if str(row['method']) != method:
            raise ValueError('selection row method does not match requested method')
        selected_skill_names = row.get('selected_skill_names')
        if selected_skill_names is None:
            selected_skill_name = row.get('selected_skill_name')
            selected_skill_names = [] if selected_skill_name is None else [str(selected_skill_name)]
        normalized_skill_names = [str(skill_name) for skill_name in selected_skill_names]
        selected_skill_name = str(row['selected_skill_name']) if row.get('selected_skill_name') is not None else normalized_skill_names[0] if normalized_skill_names else None
        normalized_rows.append({**row, 'task_id': task_id, 'domain': spec.domain, 'method': method, 'budget': int(row.get('budget', max(len(normalized_skill_names), 1))), 'candidate_pool_size': row.get('candidate_pool_size'), 'training_scope': str(row.get('training_scope', 'single_domain')), 'selected_skill_name': selected_skill_name, 'selected_skill_names': normalized_skill_names})
    return sorted(normalized_rows, key=lambda row: _sort_task_key(row['task_id']))

def build_online_task_config(domain: str, selected_skill_names: list[str]) -> dict[str, Any]:
    spec = resolve_domain_spec(domain)
    config_path = get_baseline_config_path(spec.domain)
    with config_path.open('r', encoding='utf-8') as fp:
        config = yaml.safe_load(fp)
    if not isinstance(config, dict):
        raise ValueError(f'Expected YAML object in baseline config: {config_path}')
    config['domain'] = spec.domain
    if spec.task_split is not None:
        config['task_split'] = spec.task_split
    else:
        config.pop('task_split', None)
    config['skills'] = [f'{spec.domain}/{skill_name}' for skill_name in selected_skill_names]
    return config

def execute_online_task(config: dict[str, Any], task_id: str, trial_seed: int, max_concurrency: int=1, skip_injection: bool=False) -> dict[str, Any]:
    from temp.run_experiment import run_with_config
    result = run_with_config(config, task_id_override=[task_id], max_concurrency=max_concurrency, seed_override=trial_seed, save_trajectories=False, skip_injection=skip_injection)
    task_results = result.get('results', [])
    if len(task_results) != 1:
        raise ValueError(f'Expected exactly one task result for task_id={task_id}')
    return task_results[0]

def _execute_online_task_compat(config: dict[str, Any], task_id: str, trial_seed: int, max_concurrency: int, skip_injection: bool=False) -> dict[str, Any]:
    return execute_online_task(config=config, task_id=task_id, trial_seed=trial_seed, max_concurrency=max_concurrency, skip_injection=skip_injection)

def _mean(values: list[float]) -> float:
    return sum(values) / float(len(values)) if values else 0.0

def _std(values: list[float]) -> float:
    if not values:
        return 0.0
    mean_value = _mean(values)
    return math.sqrt(sum(((value - mean_value) ** 2 for value in values)) / float(len(values)))

def _load_rewrite_overrides(rewrite_file: str | Path) -> dict[str, dict]:
    overrides: dict[str, dict] = {}
    for line in Path(rewrite_file).read_text(encoding='utf-8').splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        skill_name = record.get('skill_name', '')
        rewritten = record.get('rewritten', {})
        if not isinstance(rewritten, dict):
            continue
        override = {}
        for k in ('prompt_guidance', 'docstring', 'body_content'):
            v = rewritten.get(k)
            if isinstance(v, str) and v.strip():
                override[k] = v
        if override:
            overrides[skill_name] = override
    return overrides

def _load_dynamic_rewrite_overrides(rewrite_file: str | Path) -> dict[tuple[str, ...], dict[str, dict]]:
    result: dict[tuple[str, ...], dict[str, dict]] = {}
    for line in Path(rewrite_file).read_text(encoding='utf-8').splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        skill_set = record.get('skill_set', [])
        skill_name = record.get('skill_name', '')
        rewritten = record.get('rewritten', {})
        if not isinstance(rewritten, dict) or not skill_name:
            continue
        override = {}
        for k in ('prompt_guidance', 'docstring', 'body_content'):
            v = rewritten.get(k)
            if isinstance(v, str) and v.strip():
                override[k] = v
        if override:
            set_key = tuple(sorted(skill_set))
            if set_key not in result:
                result[set_key] = {}
            result[set_key][skill_name] = override
    return result

def run_online_eval(domain: str, method: str, selection_file: str | Path, output_dir: str | Path, num_trials: int=5, base_seed: int | None=None, budget: int | None=None, candidate_pool_size: int | None=None, max_concurrency: int=1, rewrite_file: str | Path | None=None, dynamic_rewrite_file: str | Path | None=None) -> dict[str, Any]:
    spec = resolve_domain_spec(domain)
    resolved_selection_file = _resolve_selection_file(selection_file, method)
    rows = load_selection_rows(domain=spec.domain, method=method, selection_file=resolved_selection_file)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    task_results_path = output_dir / 'task_results.jsonl'
    trial_summaries_path = output_dir / 'trial_summaries.jsonl'
    all_budgets = [int(row['budget']) for row in rows]
    adaptive_budget = len(set(all_budgets)) > 1
    resolved_budget = int(budget) if budget is not None else all_budgets[0] if not adaptive_budget else -1
    resolved_candidate_pool_size = rows[0].get('candidate_pool_size') if candidate_pool_size is None else candidate_pool_size
    if any((row.get('candidate_pool_size') != resolved_candidate_pool_size for row in rows)):
        raise ValueError('selection rows contain inconsistent candidate_pool_size values')
    resolved_training_scope = str(rows[0].get('training_scope', 'single_domain'))
    if any((str(row.get('training_scope', 'single_domain')) != resolved_training_scope for row in rows)):
        raise ValueError('selection rows contain inconsistent training_scope values')
    first_config = build_online_task_config(spec.domain, rows[0]['selected_skill_names'])
    resolved_base_seed = int(first_config.get('seed', 300)) if base_seed is None else int(base_seed)
    num_tasks = len(rows)
    row_by_task_id = {str(row['task_id']): row for row in rows}
    existing_task_rows = _load_existing_jsonl_rows(task_results_path)
    task_row_by_key: dict[tuple[int, str], dict[str, Any]] = {}
    for row in existing_task_rows:
        task_row_by_key[int(row['trial_index']), str(row['task_id'])] = row
    existing_trial_summaries = _load_existing_jsonl_rows(trial_summaries_path)
    trial_summary_by_index = {int(row['trial_index']): row for row in existing_trial_summaries}
    lock = threading.Lock()

    def _build_task_row(selection_row: dict[str, Any], trial_index: int) -> dict[str, Any]:
        trial_seed = resolved_base_seed + trial_index
        selected_skill_name = selection_row['selected_skill_name']
        selected_skill_names = list(selection_row['selected_skill_names'])
        task_id = str(selection_row['task_id'])
        config = build_online_task_config(spec.domain, selected_skill_names)
        result = _execute_online_task_compat(config=config, task_id=task_id, trial_seed=trial_seed, max_concurrency=1, skip_injection=True)
        reward = float(result.get('reward', 0.0))
        success = reward == 1.0
        return {'domain': spec.domain, 'method': method, 'trial_index': trial_index, 'trial_seed': trial_seed, 'task_id': task_id, 'selected_skill_name': selected_skill_name, 'selected_skill_names': selected_skill_names, 'used_baseline': not selected_skill_names, 'budget': int(selection_row['budget']) if adaptive_budget else resolved_budget, 'candidate_pool_size': resolved_candidate_pool_size, 'training_scope': resolved_training_scope, 'max_concurrency': max_concurrency, 'success': success, 'reward': reward, 'num_messages': int(result.get('num_messages', 0)), 'num_tool_calls': int(result.get('num_tool_calls', 0)), 'selection_file': str(resolved_selection_file), 'selection_seed': selection_row.get('selection_seed'), 'checkpoint_path': selection_row.get('checkpoint_path')}

    def _build_trial_summary(trial_index: int) -> dict[str, Any]:
        trial_seed = resolved_base_seed + trial_index
        current_rows = [row for ((current_trial_index, _task_id), row) in task_row_by_key.items() if current_trial_index == trial_index]
        trial_num_passed = sum((1 for row in current_rows if bool(row['success'])))
        return {'domain': spec.domain, 'method': method, 'trial_index': trial_index, 'trial_seed': trial_seed, 'num_tasks': num_tasks, 'num_passed': trial_num_passed, 'pass_rate': trial_num_passed / float(num_tasks) if num_tasks else 0.0}

    def _record_task_row(task_row: dict[str, Any]) -> None:
        task_key = (int(task_row['trial_index']), str(task_row['task_id']))
        with lock:
            if task_key in task_row_by_key:
                return
            task_row_by_key[task_key] = task_row
            _append_jsonl_row(task_results_path, task_row)
            trial_index = int(task_row['trial_index'])
            current_trial_count = sum((1 for (current_trial_index, _) in task_row_by_key if current_trial_index == trial_index))
            if current_trial_count == num_tasks and trial_index not in trial_summary_by_index:
                summary_row = _build_trial_summary(trial_index)
                trial_summary_by_index[trial_index] = summary_row
                _append_jsonl_row(trial_summaries_path, summary_row)
    pending_jobs = [(trial_index, row_by_task_id[str(selection_row['task_id'])]) for trial_index in range(num_trials) for selection_row in rows if (trial_index, str(selection_row['task_id'])) not in task_row_by_key]
    if pending_jobs:
        from collections import defaultdict
        skill_groups: dict[tuple[str, ...], list[tuple[int, dict]]] = defaultdict(list)
        for (trial_index, selection_row) in pending_jobs:
            skill_key = tuple(sorted(selection_row['selected_skill_names']))
            skill_groups[skill_key].append((trial_index, selection_row))
        no_skill_key = ()
        if no_skill_key in skill_groups:
            group_jobs = skill_groups.pop(no_skill_key)
            if max_concurrency <= 1:
                for (trial_index, selection_row) in group_jobs:
                    _record_task_row(_build_task_row(selection_row, trial_index))
            else:
                with ThreadPoolExecutor(max_workers=max_concurrency) as executor:
                    for future in as_completed({executor.submit(_build_task_row, sr, ti): None for (ti, sr) in group_jobs}):
                        _record_task_row(future.result())
        rewrite_overrides: dict[str, dict] = {}
        if rewrite_file is not None:
            rewrite_overrides = _load_rewrite_overrides(rewrite_file)
        dynamic_rewrites: dict[tuple[str, ...], dict[str, dict]] = {}
        if dynamic_rewrite_file is not None:
            dynamic_rewrites = _load_dynamic_rewrite_overrides(dynamic_rewrite_file)
        for (skill_key, group_jobs) in skill_groups.items():
            from temp.skill_injector import SkillInjector
            _injector = SkillInjector()
            sample_config = build_online_task_config(spec.domain, list(skill_key))
            set_overrides = dynamic_rewrites.get(skill_key, {})
            for skill_path in sample_config.get('skills', []):
                skill_name = skill_path.split('/')[-1]
                ovr = set_overrides.get(skill_name) or rewrite_overrides.get(skill_name)
                _injector.load_skill_module(skill_path, overrides=ovr)
            _injector.apply()
            try:
                if max_concurrency <= 1:
                    for (trial_index, selection_row) in group_jobs:
                        _record_task_row(_build_task_row(selection_row, trial_index))
                else:
                    with ThreadPoolExecutor(max_workers=max_concurrency) as executor:
                        for future in as_completed({executor.submit(_build_task_row, sr, ti): None for (ti, sr) in group_jobs}):
                            _record_task_row(future.result())
            finally:
                _injector.restore()
    for trial_index in range(num_trials):
        if trial_index in trial_summary_by_index:
            continue
        current_trial_count = sum((1 for (current_trial_index, _) in task_row_by_key if current_trial_index == trial_index))
        if current_trial_count != num_tasks:
            raise ValueError(f'incomplete trial results for trial_index={trial_index}')
        trial_summary_by_index[trial_index] = _build_trial_summary(trial_index)
    task_rows = sorted(task_row_by_key.values(), key=_sort_task_row_key)
    trial_summaries = [trial_summary_by_index[trial_index] for trial_index in sorted(trial_summary_by_index)]
    trial_pass_rates = [float(row['pass_rate']) for row in trial_summaries]
    aggregate = {'domain': spec.domain, 'method': method, 'budget': 'adaptive' if adaptive_budget else resolved_budget, 'candidate_pool_size': resolved_candidate_pool_size, 'training_scope': resolved_training_scope, 'num_trials': num_trials, 'max_concurrency': max_concurrency, 'mean_pass_rate': _mean(trial_pass_rates), 'std_pass_rate': _std(trial_pass_rates), 'trial_pass_rates': trial_pass_rates, 'selection_file': str(resolved_selection_file), 'selection_seed': rows[0].get('selection_seed'), 'checkpoint_path': rows[0].get('checkpoint_path'), 'rewrite_file': str(rewrite_file) if rewrite_file else None, 'dynamic_rewrite_file': str(dynamic_rewrite_file) if dynamic_rewrite_file else None}
    _write_jsonl(task_results_path, task_rows)
    _write_jsonl(trial_summaries_path, trial_summaries)
    _write_json(output_dir / 'aggregate.json', aggregate)
    return aggregate

def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description='Run fixed-mapping online selector evaluation')
    parser.add_argument('--domain', required=True)
    parser.add_argument('--method', required=True)
    parser.add_argument('--selection-file', required=True)
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--num-trials', type=int, default=5)
    parser.add_argument('--base-seed', type=int, default=None)
    parser.add_argument('--budget', type=int, default=None)
    parser.add_argument('--candidate-pool-size', type=int, default=None)
    parser.add_argument('--max-concurrency', type=int, default=1)
    parser.add_argument('--rewrite-file', type=str, default=None, help='Path to rewritten skills JSONL (static, same override for all tasks)')
    parser.add_argument('--dynamic-rewrite-file', type=str, default=None, help='Path to dynamic rewrite JSONL (per-skill-set overrides from v3a)')
    return parser

def main() -> None:
    parser = _build_arg_parser()
    args = parser.parse_args()
    payload = run_online_eval(domain=args.domain, method=args.method, selection_file=args.selection_file, output_dir=args.output_dir, num_trials=args.num_trials, base_seed=args.base_seed, budget=args.budget, candidate_pool_size=args.candidate_pool_size, max_concurrency=args.max_concurrency, rewrite_file=args.rewrite_file, dynamic_rewrite_file=args.dynamic_rewrite_file)
    print(json.dumps(payload))
if __name__ == '__main__':
    main()
