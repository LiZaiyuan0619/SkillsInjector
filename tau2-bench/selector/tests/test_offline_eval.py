import json
import subprocess
import sys
from pathlib import Path
import pytest
PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
from temp.selector.offline_eval import run_offline_baselines, validate_labels_table, validate_offline_output_dir
from temp.selector.domain_registry import get_domain_spec

def _write_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload), encoding='utf-8')

def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.write_text('\n'.join((json.dumps(row, sort_keys=True) for row in rows)) + '\n', encoding='utf-8')

def _build_tiny_dataset(data_dir: Path, domain: str='airline') -> None:
    spec = get_domain_spec(domain)
    data_dir.mkdir(parents=True)
    _write_json(data_dir / 'manifest.json', {'schema_version': 'selector_contract.v1', 'domain': domain, 'budget': 1, 'split_policy': 'feasibility_full_matrix', 'task_split_name': spec.task_split or 'base', 'domain_output_name': spec.output_name, 'output_filenames': {'manifest': 'manifest.json', 'tasks': 'tasks.jsonl', 'skills': 'skills.jsonl', 'labels': 'labels.jsonl'}, 'label_definition': {'label_name': 'utility_label', 'label_type': 'float', 'value_range': [0.0, 1.0], 'semantic': 'Estimated single-skill pass probability p_skill(t, s).', 'source': 'offline single-skill outcomes'}})
    _write_jsonl(data_dir / 'tasks.jsonl', [{'task_id': 't1', 'task_text': 'task 1', 'k_base': 1, 'n_base': 2}, {'task_id': 't2', 'task_text': 'task 2', 'k_base': 0, 'n_base': 2}])
    _write_jsonl(data_dir / 'skills.jsonl', [{'skill_name': 'alpha', 'skill_text': 'alpha text'}, {'skill_name': 'beta', 'skill_text': 'beta text'}, {'skill_name': 'gamma', 'skill_text': 'gamma text'}])
    _write_jsonl(data_dir / 'labels.jsonl', [{'task_id': 't1', 'skill_name': 'alpha', 'k_skill': 2, 'n_skill': 2}, {'task_id': 't1', 'skill_name': 'beta', 'k_skill': 1, 'n_skill': 2}, {'task_id': 't1', 'skill_name': 'gamma', 'k_skill': 0, 'n_skill': 2}, {'task_id': 't2', 'skill_name': 'alpha', 'k_skill': 0, 'n_skill': 2}, {'task_id': 't2', 'skill_name': 'beta', 'k_skill': 2, 'n_skill': 2}, {'task_id': 't2', 'skill_name': 'gamma', 'k_skill': 1, 'n_skill': 2}])

def test_validate_labels_table_rejects_empty() -> None:
    with pytest.raises(ValueError, match='labels table must not be empty'):
        validate_labels_table([])

@pytest.mark.parametrize(('domain', 'output_name'), [('airline', 'airline'), ('retail', 'retail'), ('telecom', 'telecom_base')])
def test_validate_offline_output_dir_creates_domain_parent_for_offline_layout(tmp_path: Path, domain: str, output_name: str) -> None:
    output_dir = tmp_path / 'results' / output_name / 'offline'
    assert validate_offline_output_dir(output_dir, domain=domain) == output_dir
    assert output_dir.exists()
    assert output_dir.parent.exists()

def test_run_offline_baselines_writes_expected_metric_contract(tmp_path: Path) -> None:
    data_dir = tmp_path / 'data'
    output_dir = tmp_path / 'results'
    _build_tiny_dataset(data_dir)
    results = run_offline_baselines(domain='airline', data_dir=data_dir, baseline_suite=['global_utility', 'random'], budget=1, output_dir=output_dir, seed=13)
    assert [result['baseline_name'] for result in results] == ['global_utility', 'random']
    global_utility = results[0]
    assert global_utility == {'baseline_name': 'global_utility', 'budget': 1, 'mrr': pytest.approx(0.75), 'num_tasks': 2, 'positive_hit_rate': pytest.approx(0.5), 'recall_at_1': pytest.approx(1 / 3), 'top1_delta_mean': pytest.approx(0.5)}
    random_summary_path = output_dir / 'random.summary.json'
    global_summary_path = output_dir / 'global_utility.summary.json'
    comparison_path = output_dir / 'comparison.json'
    random_predictions_path = output_dir / 'random.predictions.jsonl'
    assert random_summary_path.exists()
    assert global_summary_path.exists()
    assert comparison_path.exists()
    assert random_predictions_path.exists()
    comparison = json.loads(comparison_path.read_text(encoding='utf-8'))
    assert [row['baseline_name'] for row in comparison['summaries']] == ['global_utility', 'random']
    assert comparison['domain'] == 'airline'
    assert comparison['budget'] == 1

def test_random_baseline_is_deterministic_for_same_seed(tmp_path: Path) -> None:
    data_dir = tmp_path / 'data'
    output_dir_a = tmp_path / 'results-a'
    output_dir_b = tmp_path / 'results-b'
    _build_tiny_dataset(data_dir)
    results_a = run_offline_baselines(domain='airline', data_dir=data_dir, baseline_suite=['random'], budget=1, output_dir=output_dir_a, seed=7)
    results_b = run_offline_baselines(domain='airline', data_dir=data_dir, baseline_suite=['random'], budget=1, output_dir=output_dir_b, seed=7)
    assert results_a == results_b
    assert (output_dir_a / 'random.predictions.jsonl').read_text(encoding='utf-8') == (output_dir_b / 'random.predictions.jsonl').read_text(encoding='utf-8')

@pytest.mark.parametrize('domain', ['retail', 'telecom'])
def test_run_offline_baselines_accepts_supported_non_airline_manifests(tmp_path: Path, domain: str) -> None:
    data_dir = tmp_path / f'{domain}-data'
    output_dir = tmp_path / 'results'
    _build_tiny_dataset(data_dir, domain=domain)
    results = run_offline_baselines(domain=domain, data_dir=data_dir, baseline_suite=['global_utility'], budget=1, output_dir=output_dir, seed=13)
    assert results == [{'baseline_name': 'global_utility', 'budget': 1, 'mrr': pytest.approx(0.75), 'num_tasks': 2, 'positive_hit_rate': pytest.approx(0.5), 'recall_at_1': pytest.approx(1 / 3), 'top1_delta_mean': pytest.approx(0.5)}]
    comparison = json.loads((output_dir / 'comparison.json').read_text(encoding='utf-8'))
    assert comparison['domain'] == domain
    assert comparison['budget'] == 1

def test_offline_eval_cli_smoke(tmp_path: Path) -> None:
    data_dir = tmp_path / 'data'
    output_dir = tmp_path / 'results'
    _build_tiny_dataset(data_dir)
    result = subprocess.run([sys.executable, str(PROJECT_ROOT / 'temp/selector/offline_eval.py'), '--domain', 'airline', '--data-dir', str(data_dir), '--baseline-suite', 'random,global_utility', '--budget', '1', '--output-dir', str(output_dir), '--seed', '13'], check=False, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout)
    assert payload['domain'] == 'airline'
    assert payload['budget'] == 1
    assert payload['baseline_names'] == ['random', 'global_utility']
