import json
import sys
import threading
import time
from pathlib import Path
import pytest
PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
from temp.selector.run_online_eval import run_online_eval

def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.write_text('\n'.join((json.dumps(row, sort_keys=True) for row in rows)) + '\n', encoding='utf-8')

def test_online_runner_reuses_same_mapping_across_trials(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    selection_file = tmp_path / 'trained_selector.jsonl'
    output_dir = tmp_path / 'results'
    _write_jsonl(selection_file, [{'domain': 'airline', 'task_id': '1', 'method': 'trained_selector', 'selected_skill_name': 'alpha', 'selection_seed': None, 'selection_score': 0.9, 'selection_rank': 1, 'selection_source': 'trained_selector', 'checkpoint_path': '/tmp/model.pt'}, {'domain': 'airline', 'task_id': '2', 'method': 'trained_selector', 'selected_skill_name': 'beta', 'selection_seed': None, 'selection_score': 0.8, 'selection_rank': 1, 'selection_source': 'trained_selector', 'checkpoint_path': '/tmp/model.pt'}])
    calls: list[dict[str, object]] = []

    def _fake_execute(config: dict[str, object], task_id: str, trial_seed: int) -> dict[str, object]:
        calls.append({'task_id': task_id, 'trial_seed': trial_seed, 'skills': list(config.get('skills', []))})
        reward = 1.0 if task_id == '1' else 0.0
        return {'task_id': task_id, 'reward': reward, 'num_messages': 3, 'num_tool_calls': 1 if config.get('skills') else 0, 'tool_sequence': ['alpha'] if config.get('skills') else [], 'tool_traces': []}
    monkeypatch.setattr('temp.selector.run_online_eval.execute_online_task', _fake_execute)
    result = run_online_eval(domain='airline', method='trained_selector', selection_file=selection_file, output_dir=output_dir, num_trials=5)
    assert result['num_trials'] == 5
    assert result['mean_pass_rate'] == pytest.approx(0.5)
    assert result['trial_pass_rates'] == [0.5] * 5
    assert len(calls) == 10
    assert {(call['task_id'], tuple(call['skills'])) for call in calls if call['task_id'] == '1'} == {('1', ('airline/alpha',))}
    assert {(call['task_id'], tuple(call['skills'])) for call in calls if call['task_id'] == '2'} == {('2', ('airline/beta',))}
    assert (output_dir / 'task_results.jsonl').exists()
    assert (output_dir / 'trial_summaries.jsonl').exists()
    assert (output_dir / 'aggregate.json').exists()

def test_online_runner_supports_baseline_without_injected_skill(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    selection_file = tmp_path / 'baseline.jsonl'
    output_dir = tmp_path / 'baseline-results'
    _write_jsonl(selection_file, [{'domain': 'telecom', 'task_id': '10', 'method': 'baseline', 'selected_skill_name': None, 'selection_seed': None, 'selection_score': None, 'selection_rank': None, 'selection_source': 'baseline', 'checkpoint_path': None}])
    calls: list[dict[str, object]] = []

    def _fake_execute(config: dict[str, object], task_id: str, trial_seed: int) -> dict[str, object]:
        calls.append({'task_id': task_id, 'trial_seed': trial_seed, 'skills': list(config.get('skills', [])), 'task_split': config.get('task_split')})
        return {'task_id': task_id, 'reward': 1.0, 'num_messages': 2, 'num_tool_calls': 0, 'tool_sequence': [], 'tool_traces': []}
    monkeypatch.setattr('temp.selector.run_online_eval.execute_online_task', _fake_execute)
    result = run_online_eval(domain='telecom', method='baseline', selection_file=selection_file, output_dir=output_dir, num_trials=2)
    assert result['domain'] == 'telecom'
    assert result['method'] == 'baseline'
    assert result['num_trials'] == 2
    assert result['mean_pass_rate'] == pytest.approx(1.0)
    assert calls == [{'task_id': '10', 'trial_seed': 300, 'skills': [], 'task_split': 'base'}, {'task_id': '10', 'trial_seed': 301, 'skills': [], 'task_split': 'base'}]
    task_rows = [json.loads(line) for line in (output_dir / 'task_results.jsonl').read_text(encoding='utf-8').splitlines()]
    assert all((row['used_baseline'] is True for row in task_rows))
    assert all((row['selected_skill_name'] is None for row in task_rows))

def test_online_runner_supports_multiple_injected_skills_and_runtime_metadata(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    selection_file = tmp_path / 'trained_selector_multi.jsonl'
    output_dir = tmp_path / 'multi-results'
    _write_jsonl(selection_file, [{'domain': 'airline', 'task_id': '1', 'method': 'trained_selector', 'budget': 2, 'candidate_pool_size': 4, 'training_scope': 'cross_domain', 'selected_skill_name': 'alpha', 'selected_skill_names': ['alpha', 'beta'], 'selection_seed': None, 'selection_score': 0.9, 'selection_scores': [0.9, 0.7], 'selection_rank': 1, 'selection_source': 'trained_selector', 'checkpoint_path': '/tmp/model.pt'}])
    calls: list[dict[str, object]] = []

    def _fake_execute(config: dict[str, object], task_id: str, trial_seed: int) -> dict[str, object]:
        calls.append({'task_id': task_id, 'trial_seed': trial_seed, 'skills': list(config.get('skills', []))})
        return {'task_id': task_id, 'reward': 1.0, 'num_messages': 3, 'num_tool_calls': 2, 'tool_sequence': ['alpha', 'beta'], 'tool_traces': []}
    monkeypatch.setattr('temp.selector.run_online_eval.execute_online_task', _fake_execute)
    result = run_online_eval(domain='airline', method='trained_selector', selection_file=selection_file, output_dir=output_dir, num_trials=2, max_concurrency=3)
    assert result['budget'] == 2
    assert result['candidate_pool_size'] == 4
    assert result['training_scope'] == 'cross_domain'
    assert result['max_concurrency'] == 3
    assert calls == [{'task_id': '1', 'trial_seed': 300, 'skills': ['airline/alpha', 'airline/beta']}, {'task_id': '1', 'trial_seed': 301, 'skills': ['airline/alpha', 'airline/beta']}]
    task_rows = [json.loads(line) for line in (output_dir / 'task_results.jsonl').read_text(encoding='utf-8').splitlines()]
    assert all((row['selected_skill_names'] == ['alpha', 'beta'] for row in task_rows))
    assert all((row['budget'] == 2 for row in task_rows))
    assert all((row['candidate_pool_size'] == 4 for row in task_rows))

def test_online_runner_executes_benchmark_tasks_concurrently(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    selection_file = tmp_path / 'trained_selector.jsonl'
    output_dir = tmp_path / 'concurrent-results'
    _write_jsonl(selection_file, [{'domain': 'airline', 'task_id': task_id, 'method': 'trained_selector', 'selected_skill_name': f'skill_{task_id}', 'selected_skill_names': [f'skill_{task_id}'], 'selection_seed': None, 'selection_score': 0.9, 'selection_scores': [0.9], 'selection_rank': 1, 'selection_source': 'trained_selector', 'checkpoint_path': '/tmp/model.pt'} for task_id in ('1', '2', '3', '4')])
    state = {'active': 0, 'peak': 0}
    lock = threading.Lock()

    def _fake_execute(config: dict[str, object], task_id: str, trial_seed: int, max_concurrency: int=1) -> dict[str, object]:
        del config, trial_seed, max_concurrency
        with lock:
            state['active'] += 1
            state['peak'] = max(state['peak'], state['active'])
        time.sleep(0.05)
        with lock:
            state['active'] -= 1
        return {'task_id': task_id, 'reward': 1.0, 'num_messages': 1, 'num_tool_calls': 1, 'tool_sequence': [], 'tool_traces': []}
    monkeypatch.setattr('temp.selector.run_online_eval.execute_online_task', _fake_execute)
    result = run_online_eval(domain='airline', method='trained_selector', selection_file=selection_file, output_dir=output_dir, num_trials=1, max_concurrency=3)
    assert result['mean_pass_rate'] == pytest.approx(1.0)
    assert state['peak'] >= 2

def test_online_runner_persists_partial_progress_and_resumes(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    selection_file = tmp_path / 'trained_selector.jsonl'
    output_dir = tmp_path / 'resume-results'
    _write_jsonl(selection_file, [{'domain': 'airline', 'task_id': '1', 'method': 'trained_selector', 'selected_skill_name': 'alpha', 'selected_skill_names': ['alpha'], 'selection_seed': None, 'selection_score': 0.9, 'selection_scores': [0.9], 'selection_rank': 1, 'selection_source': 'trained_selector', 'checkpoint_path': '/tmp/model.pt'}, {'domain': 'airline', 'task_id': '2', 'method': 'trained_selector', 'selected_skill_name': 'beta', 'selected_skill_names': ['beta'], 'selection_seed': None, 'selection_score': 0.8, 'selection_scores': [0.8], 'selection_rank': 1, 'selection_source': 'trained_selector', 'checkpoint_path': '/tmp/model.pt'}])
    calls: list[tuple[str, int]] = []
    failure_state = {'remaining': 1}

    def _flaky_execute(config: dict[str, object], task_id: str, trial_seed: int, max_concurrency: int=1) -> dict[str, object]:
        del config, max_concurrency
        calls.append((task_id, trial_seed))
        if failure_state['remaining'] == 0:
            raise RuntimeError('boom')
        failure_state['remaining'] -= 1
        return {'task_id': task_id, 'reward': 1.0, 'num_messages': 1, 'num_tool_calls': 1, 'tool_sequence': [], 'tool_traces': []}
    monkeypatch.setattr('temp.selector.run_online_eval.execute_online_task', _flaky_execute)
    with pytest.raises(RuntimeError, match='boom'):
        run_online_eval(domain='airline', method='trained_selector', selection_file=selection_file, output_dir=output_dir, num_trials=2)
    task_rows_after_failure = [json.loads(line) for line in (output_dir / 'task_results.jsonl').read_text(encoding='utf-8').splitlines()]
    assert len(task_rows_after_failure) == 1
    assert task_rows_after_failure[0]['task_id'] == '1'

    def _resume_execute(config: dict[str, object], task_id: str, trial_seed: int, max_concurrency: int=1) -> dict[str, object]:
        del config, max_concurrency
        calls.append((task_id, trial_seed))
        return {'task_id': task_id, 'reward': 1.0 if task_id == '1' else 0.0, 'num_messages': 2, 'num_tool_calls': 1, 'tool_sequence': [], 'tool_traces': []}
    monkeypatch.setattr('temp.selector.run_online_eval.execute_online_task', _resume_execute)
    result = run_online_eval(domain='airline', method='trained_selector', selection_file=selection_file, output_dir=output_dir, num_trials=2)
    assert calls == [('1', 300), ('2', 300), ('2', 300), ('1', 301), ('2', 301)]
    assert result['trial_pass_rates'] == [0.5, 0.5]
    task_rows = [json.loads(line) for line in (output_dir / 'task_results.jsonl').read_text(encoding='utf-8').splitlines()]
    assert len(task_rows) == 4
    trial_rows = [json.loads(line) for line in (output_dir / 'trial_summaries.jsonl').read_text(encoding='utf-8').splitlines()]
    assert trial_rows == [{'domain': 'airline', 'method': 'trained_selector', 'trial_index': 0, 'trial_seed': 300, 'num_tasks': 2, 'num_passed': 1, 'pass_rate': 0.5}, {'domain': 'airline', 'method': 'trained_selector', 'trial_index': 1, 'trial_seed': 301, 'num_tasks': 2, 'num_passed': 1, 'pass_rate': 0.5}]

def test_online_runner_accepts_random_seed_suffix_selection_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    requested_selection_file = tmp_path / 'random.jsonl'
    actual_selection_file = tmp_path / 'random_seed13.jsonl'
    output_dir = tmp_path / 'random-results'
    _write_jsonl(actual_selection_file, [{'domain': 'retail', 'task_id': '5', 'method': 'random', 'selected_skill_name': 'alpha', 'selected_skill_names': ['alpha'], 'selection_seed': 13, 'selection_score': None, 'selection_scores': None, 'selection_rank': 1, 'selection_source': 'random', 'checkpoint_path': None}])

    def _fake_execute(config: dict[str, object], task_id: str, trial_seed: int, max_concurrency: int=1) -> dict[str, object]:
        del config, task_id, trial_seed, max_concurrency
        return {'task_id': '5', 'reward': 1.0, 'num_messages': 2, 'num_tool_calls': 1, 'tool_sequence': [], 'tool_traces': []}
    monkeypatch.setattr('temp.selector.run_online_eval.execute_online_task', _fake_execute)
    result = run_online_eval(domain='retail', method='random', selection_file=requested_selection_file, output_dir=output_dir, num_trials=1)
    assert result['selection_seed'] == 13
    assert result['selection_file'] == str(actual_selection_file)
