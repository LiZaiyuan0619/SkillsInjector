import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import cast
import pytest
PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
from temp.selector.export_dataset import export_airline_dataset, export_dataset

def _write_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload), encoding='utf-8')

def _read_jsonl(path: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for line in path.read_text(encoding='utf-8').splitlines():
        rows.append(json.loads(line))
    return rows

def _make_task(task_id: str, purpose: str) -> object:
    return SimpleNamespace(id=task_id, description=SimpleNamespace(purpose=purpose, relevant_policies='policy', notes='note'), user_scenario=SimpleNamespace(persona='persona', instructions={'task_instructions': f'instructions for {task_id}'}))

def test_export_airline_dataset_writes_canonical_files_and_rows(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    baseline_root = tmp_path / 'single-skill-data'
    baseline_root.mkdir()
    parsed_dir = tmp_path / 'parsed'
    parsed_dir.mkdir()
    ranked_path = tmp_path / 'skills_ranked.json'
    output_dir = tmp_path / 'out'
    _write_json(baseline_root / 'airline_baseline_agg3_20260315_152801.json', {'summary': {'domain': 'airline', 'skill_names': [], 'num_trials': 3}, 'results': [{'task_id': '0', 'reward': 1 / 3, 'pass_count': 1, 'num_trials': 3}, {'task_id': '1', 'reward': 2 / 3, 'pass_count': 2, 'num_trials': 3}]})
    _write_json(baseline_root / 'airline_single_alpha_skill_agg5_20260317_101704.json', {'summary': {'domain': 'airline', 'skill_names': ['airline/alpha_skill'], 'num_trials': 5}, 'results': [{'task_id': '0', 'reward': 0.8, 'pass_count': 4, 'num_trials': 5}, {'task_id': '1', 'reward': 0.2, 'pass_count': 1, 'num_trials': 5}]})
    _write_json(baseline_root / 'airline_single_beta_skill_agg4_20260317_101704.json', {'summary': {'domain': 'airline', 'skill_names': ['beta_skill'], 'num_trials': 4}, 'results': [{'task_id': '0', 'reward': 0.25, 'pass_count': 1, 'num_trials': 4}, {'task_id': '1', 'reward': 0.5, 'pass_count': 2, 'num_trials': 4}]})
    _write_json(ranked_path, [{'rank': 0, 'skill_name': 'alpha_skill', 'file': 'skill_alpha.py', 'frequency': 11, 'cluster_size': 3, 'cluster_id': 7, 'code': ''}, {'rank': 1, 'skill_name': 'beta_skill', 'file': 'skill_beta.py', 'frequency': 5, 'cluster_size': 2, 'cluster_id': 2, 'code': ''}])
    (parsed_dir / 'skill_alpha.py').write_text('\n'.join(['SKILL_NAME = "alpha_skill"', 'SKILL_DOMAIN = "airline"', 'SKILL_DESCRIPTION = "alpha desc"', 'PROMPT_GUIDANCE = "alpha guide"', 'TARGET_PATTERN = "alpha pattern"']), encoding='utf-8')
    (parsed_dir / 'skill_beta.py').write_text('\n'.join(['SKILL_NAME = "beta_skill"', 'SKILL_DOMAIN = "airline"', 'SKILL_DESCRIPTION = "beta desc"', 'PROMPT_GUIDANCE = "beta guide"', 'TARGET_PATTERN = "beta pattern"']), encoding='utf-8')
    monkeypatch.setattr('temp.selector.export_dataset.load_airline_tasks', lambda task_split_name='base': [_make_task('0', 'purpose 0'), _make_task('1', 'purpose 1')])
    manifest = export_airline_dataset(output_dir=output_dir, baseline_and_skill_root=baseline_root, ranked_skills_path=ranked_path, parsed_skill_dir=parsed_dir)
    assert sorted((path.name for path in output_dir.iterdir())) == ['labels.jsonl', 'manifest.json', 'skills.jsonl', 'tasks.jsonl']
    source_runs = cast(dict[str, object], manifest['source_runs'])
    assert str(source_runs['baseline_aggregate_file']).endswith('airline_baseline_agg3_20260315_152801.json')
    tasks_rows = _read_jsonl(output_dir / 'tasks.jsonl')
    skills_rows = _read_jsonl(output_dir / 'skills.jsonl')
    labels_rows = _read_jsonl(output_dir / 'labels.jsonl')
    assert len(tasks_rows) == 2
    assert tasks_rows[0]['task_id'] == '0'
    assert tasks_rows[0]['k_base'] == 1
    assert tasks_rows[0]['n_base'] == 3
    assert 'purpose: purpose 0' in str(tasks_rows[0]['task_text'])
    assert len(skills_rows) == 2
    assert {row['skill_name'] for row in skills_rows} == {'alpha_skill', 'beta_skill'}
    alpha_row = next((row for row in skills_rows if row['skill_name'] == 'alpha_skill'))
    assert alpha_row['frequency'] == 11
    assert alpha_row['cluster_size'] == 3
    assert alpha_row['source_file'] == 'skill_alpha.py'
    assert 'skill_description: alpha desc' in str(alpha_row['skill_text'])
    assert len(labels_rows) == 4
    alpha_task0 = next((row for row in labels_rows if row['task_id'] == '0' and row['skill_name'] == 'alpha_skill'))
    assert alpha_task0['k_skill'] == 4
    assert alpha_task0['n_skill'] == 5
    assert alpha_task0['n_skill'] != tasks_rows[0]['n_base']

def test_export_dataset_supports_retail_and_ignores_other_domain_aggregates(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    source_root = tmp_path / 'single-skill-data'
    source_root.mkdir()
    parsed_dir = tmp_path / 'parsed'
    parsed_dir.mkdir()
    ranked_path = tmp_path / 'skills_ranked.json'
    output_dir = tmp_path / 'retail'
    _write_json(source_root / 'retail_baseline_agg5_20260320_144520.json', {'summary': {'domain': 'retail', 'skill_names': [], 'num_trials': 5}, 'results': [{'task_id': 'r-0', 'reward': 0.6, 'pass_count': 3, 'num_trials': 5}, {'task_id': 'r-1', 'reward': 0.2, 'pass_count': 1, 'num_trials': 5}]})
    _write_json(source_root / 'airline_baseline_agg9_20260320_144520.json', {'summary': {'domain': 'airline', 'skill_names': [], 'num_trials': 9}, 'results': [{'task_id': 'a-0', 'reward': 1.0, 'pass_count': 9, 'num_trials': 9}]})
    _write_json(source_root / 'retail_single_alpha_skill_agg5_20260320_144520.json', {'summary': {'domain': 'retail', 'skill_names': ['retail/alpha_skill'], 'num_trials': 5}, 'results': [{'task_id': 'r-0', 'reward': 0.8, 'pass_count': 4, 'num_trials': 5}, {'task_id': 'r-1', 'reward': 0.4, 'pass_count': 2, 'num_trials': 5}]})
    _write_json(source_root / 'retail_single_beta_skill_agg5_20260320_144521.json', {'summary': {'domain': 'retail', 'skill_names': ['beta_skill'], 'num_trials': 5}, 'results': [{'task_id': 'r-0', 'reward': 0.6, 'pass_count': 3, 'num_trials': 5}, {'task_id': 'r-1', 'reward': 0.2, 'pass_count': 1, 'num_trials': 5}]})
    _write_json(source_root / 'airline_single_intruder_skill_agg5_20260320_144522.json', {'summary': {'domain': 'airline', 'skill_names': ['airline/intruder_skill'], 'num_trials': 5}, 'results': [{'task_id': 'a-0', 'reward': 1.0, 'pass_count': 5, 'num_trials': 5}]})
    _write_json(ranked_path, [{'rank': 0, 'skill_name': 'retail/alpha_skill', 'file': 'skill_alpha.py', 'frequency': 8, 'cluster_size': 2, 'cluster_id': 11, 'code': ''}, {'rank': 1, 'skill_name': 'beta_skill', 'file': 'skill_beta.py', 'frequency': 4, 'cluster_size': 1, 'cluster_id': 5, 'code': ''}])
    (parsed_dir / 'skill_alpha.py').write_text('\n'.join(['SKILL_NAME = "alpha_skill"', 'SKILL_DOMAIN = "retail"', 'SKILL_DESCRIPTION = "alpha desc"', 'PROMPT_GUIDANCE = "alpha guide"', 'TARGET_PATTERN = "alpha pattern"']), encoding='utf-8')
    (parsed_dir / 'skill_beta.py').write_text('\n'.join(['SKILL_NAME = "retail/beta_skill"', 'SKILL_DOMAIN = "retail"', 'SKILL_DESCRIPTION = "beta desc"', 'PROMPT_GUIDANCE = "beta guide"', 'TARGET_PATTERN = "beta pattern"']), encoding='utf-8')
    monkeypatch.setattr('temp.selector.export_dataset.load_domain_tasks', lambda domain, task_split_name=None: [_make_task('r-0', 'retail purpose 0'), _make_task('r-1', 'retail purpose 1')])
    manifest = export_dataset(domain='retail', output_dir=output_dir, baseline_and_skill_root=source_root, ranked_skills_path=ranked_path, parsed_skill_dir=parsed_dir, task_split_name='base')
    source_runs = cast(dict[str, object], manifest['source_runs'])
    assert str(source_runs['baseline_aggregate_file']).endswith('retail_baseline_agg5_20260320_144520.json')
    assert set(cast(dict[str, object], source_runs['single_skill_aggregate_files'])) == {'alpha_skill', 'beta_skill'}
    task_rows = _read_jsonl(output_dir / 'tasks.jsonl')
    skill_rows = _read_jsonl(output_dir / 'skills.jsonl')
    label_rows = _read_jsonl(output_dir / 'labels.jsonl')
    assert [row['task_id'] for row in task_rows] == ['r-0', 'r-1']
    assert {row['skill_name'] for row in skill_rows} == {'alpha_skill', 'beta_skill'}
    assert all((row['skill_name'] != 'intruder_skill' for row in skill_rows))
    assert len(label_rows) == 4

def test_export_dataset_supports_telecom_base_split_metadata(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    source_root = tmp_path / 'single-skill-data'
    source_root.mkdir()
    parsed_dir = tmp_path / 'parsed'
    parsed_dir.mkdir()
    ranked_path = tmp_path / 'skills_ranked.json'
    output_dir = tmp_path / 'telecom_base'
    _write_json(source_root / 'telecom_baseline_agg5_20260320_144520.json', {'summary': {'domain': 'telecom', 'skill_names': [], 'num_trials': 5}, 'results': [{'task_id': 'ticket-0', 'reward': 0.4, 'pass_count': 2, 'num_trials': 5}]})
    _write_json(source_root / 'telecom_single_network_mode_preference_guide_agg5_20260320_144522.json', {'summary': {'domain': 'telecom', 'skill_names': ['telecom/network_mode_preference_guide'], 'num_trials': 5}, 'results': [{'task_id': 'ticket-0', 'reward': 0.8, 'pass_count': 4, 'num_trials': 5}]})
    _write_json(ranked_path, [{'rank': 0, 'skill_name': 'telecom/network_mode_preference_guide', 'file': 'skill_network_mode.py', 'frequency': 12, 'cluster_size': 3, 'cluster_id': 4, 'code': ''}])
    (parsed_dir / 'skill_network_mode.py').write_text('\n'.join(['SKILL_NAME = "network_mode_preference_guide"', 'SKILL_DOMAIN = "telecom"', 'SKILL_DESCRIPTION = "network mode desc"', 'PROMPT_GUIDANCE = "network mode guide"', 'TARGET_PATTERN = "network mode pattern"']), encoding='utf-8')

    def _raise_missing(*args, **kwargs):
        raise ModuleNotFoundError("No module named 'tau2.run'")
    monkeypatch.setattr('temp.selector.export_dataset.load_domain_tasks', _raise_missing)
    tasks_json_path = tmp_path / 'tasks.json'
    split_tasks_path = tmp_path / 'split_tasks.json'
    _write_json(tasks_json_path, [{'id': 'ticket-0', 'description': {'purpose': 'telecom base purpose', 'relevant_policies': None, 'notes': None}, 'user_scenario': {'persona': None, 'instructions': {'task_instructions': 'telecom base instructions'}}}])
    _write_json(split_tasks_path, {'base': ['ticket-0'], 'full': ['ticket-0', 'ticket-9']})
    manifest = export_dataset(domain='telecom', output_dir=output_dir, baseline_and_skill_root=source_root, ranked_skills_path=ranked_path, parsed_skill_dir=parsed_dir, local_tasks_json_path=tasks_json_path, local_split_tasks_json_path=split_tasks_path)
    assert sorted((path.name for path in output_dir.iterdir())) == ['labels.jsonl', 'manifest.json', 'skills.jsonl', 'tasks.jsonl']
    assert manifest['domain'] == 'telecom'
    assert manifest['task_split_name'] == 'base'
    assert manifest['domain_output_name'] == 'telecom_base'
    source_runs = cast(dict[str, object], manifest['source_runs'])
    task_source = cast(dict[str, object], source_runs['task_source'])
    assert task_source['type'] == 'local_tasks_json_fallback'
    assert task_source['task_split_name'] == 'base'
    task_rows = _read_jsonl(output_dir / 'tasks.jsonl')
    assert task_rows[0]['task_id'] == 'ticket-0'
    assert task_rows[0]['n_base'] == 5

def test_export_dataset_rejects_non_base_telecom_split_override(tmp_path: Path):
    source_root = tmp_path / 'single-skill-data'
    source_root.mkdir()
    parsed_dir = tmp_path / 'parsed'
    parsed_dir.mkdir()
    ranked_path = tmp_path / 'skills_ranked.json'
    _write_json(source_root / 'telecom_baseline_agg5_20260320_144520.json', {'summary': {'domain': 'telecom', 'skill_names': [], 'num_trials': 5}, 'results': [{'task_id': 'ticket-0', 'reward': 0.4, 'pass_count': 2, 'num_trials': 5}]})
    _write_json(source_root / 'telecom_single_network_mode_preference_guide_agg5_20260320_144522.json', {'summary': {'domain': 'telecom', 'skill_names': ['telecom/network_mode_preference_guide'], 'num_trials': 5}, 'results': [{'task_id': 'ticket-0', 'reward': 0.8, 'pass_count': 4, 'num_trials': 5}]})
    _write_json(ranked_path, [{'rank': 0, 'skill_name': 'telecom/network_mode_preference_guide', 'file': 'skill_network_mode.py', 'frequency': 12, 'cluster_size': 3, 'cluster_id': 4, 'code': ''}])
    (parsed_dir / 'skill_network_mode.py').write_text('\n'.join(['SKILL_NAME = "network_mode_preference_guide"', 'SKILL_DOMAIN = "telecom"', 'SKILL_DESCRIPTION = "network mode desc"', 'PROMPT_GUIDANCE = "network mode guide"', 'TARGET_PATTERN = "network mode pattern"']), encoding='utf-8')
    with pytest.raises(ValueError, match='telecom.*base'):
        export_dataset(domain='telecom', output_dir=tmp_path / 'telecom_full', baseline_and_skill_root=source_root, ranked_skills_path=ranked_path, parsed_skill_dir=parsed_dir, task_split_name='full')

def test_export_airline_dataset_falls_back_to_local_tasks_and_records_source(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    source_root = tmp_path / 'single-skill-data'
    source_root.mkdir()
    parsed_dir = tmp_path / 'parsed'
    parsed_dir.mkdir()
    ranked_path = tmp_path / 'skills_ranked.json'
    tasks_json_path = tmp_path / 'tasks.json'
    split_tasks_path = tmp_path / 'split_tasks.json'
    _write_json(source_root / 'airline_baseline_agg1_20260315_000000.json', {'summary': {'domain': 'airline', 'skill_names': [], 'num_trials': 2}, 'results': [{'task_id': '10', 'reward': 0.5, 'pass_count': 1, 'num_trials': 2}]})
    _write_json(source_root / 'airline_single_gamma_skill_agg1_20260315_000000.json', {'summary': {'domain': 'airline', 'skill_names': ['gamma_skill'], 'num_trials': 2}, 'results': [{'task_id': '10', 'reward': 1.0, 'pass_count': 2, 'num_trials': 2}]})
    _write_json(ranked_path, [{'rank': 0, 'skill_name': 'gamma_skill', 'file': 'skill_gamma.py', 'frequency': 9, 'cluster_size': 1, 'cluster_id': 0, 'code': ''}])
    (parsed_dir / 'skill_gamma.py').write_text('\n'.join(['SKILL_NAME = "gamma_skill"', 'SKILL_DOMAIN = "airline"', 'SKILL_DESCRIPTION = "gamma desc"', 'PROMPT_GUIDANCE = "gamma guide"', 'TARGET_PATTERN = "gamma pattern"']), encoding='utf-8')
    _write_json(tasks_json_path, [{'id': '10', 'description': {'purpose': 'fallback purpose', 'relevant_policies': None, 'notes': None}, 'user_scenario': {'persona': None, 'instructions': {'task_instructions': 'fallback instructions'}}}])
    _write_json(split_tasks_path, {'base': ['10']})

    def _raise_missing(*args, **kwargs):
        raise ModuleNotFoundError("No module named 'tau2.run'")
    monkeypatch.setattr('temp.selector.export_dataset.load_airline_tasks', _raise_missing)
    manifest = export_airline_dataset(output_dir=tmp_path / 'out', baseline_and_skill_root=source_root, ranked_skills_path=ranked_path, parsed_skill_dir=parsed_dir, local_tasks_json_path=tasks_json_path, local_split_tasks_json_path=split_tasks_path)
    source_runs = cast(dict[str, object], manifest['source_runs'])
    task_source = cast(dict[str, object], source_runs['task_source'])
    assert task_source['type'] == 'local_tasks_json_fallback'
    assert str(task_source['tasks_json']).endswith('tasks.json')
    task_rows = _read_jsonl(tmp_path / 'out' / 'tasks.jsonl')
    assert len(task_rows) == 1
    assert 'purpose: fallback purpose' in str(task_rows[0]['task_text'])

def test_export_airline_dataset_fails_when_loaded_tasks_miss_baseline_task_ids(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    source_root = tmp_path / 'single-skill-data'
    source_root.mkdir()
    parsed_dir = tmp_path / 'parsed'
    parsed_dir.mkdir()
    ranked_path = tmp_path / 'skills_ranked.json'
    _write_json(source_root / 'airline_baseline_agg1_20260315_000000.json', {'summary': {'domain': 'airline', 'skill_names': [], 'num_trials': 2}, 'results': [{'task_id': '0', 'reward': 0.0, 'pass_count': 0, 'num_trials': 2}, {'task_id': '1', 'reward': 0.5, 'pass_count': 1, 'num_trials': 2}]})
    _write_json(source_root / 'airline_single_alpha_skill_agg1_20260315_000000.json', {'summary': {'domain': 'airline', 'skill_names': ['alpha_skill'], 'num_trials': 2}, 'results': [{'task_id': '0', 'reward': 0.5, 'pass_count': 1, 'num_trials': 2}, {'task_id': '1', 'reward': 0.5, 'pass_count': 1, 'num_trials': 2}]})
    _write_json(ranked_path, [{'rank': 0, 'skill_name': 'alpha_skill', 'file': 'skill_alpha.py', 'frequency': 1, 'cluster_size': 1, 'cluster_id': 0, 'code': ''}])
    (parsed_dir / 'skill_alpha.py').write_text('\n'.join(['SKILL_NAME = "alpha_skill"', 'SKILL_DOMAIN = "airline"', 'SKILL_DESCRIPTION = "alpha desc"', 'PROMPT_GUIDANCE = "alpha guide"', 'TARGET_PATTERN = "alpha pattern"']), encoding='utf-8')
    monkeypatch.setattr('temp.selector.export_dataset.load_airline_tasks', lambda task_split_name='base': [_make_task('0', 'only one loaded task')])
    with pytest.raises(ValueError, match='Missing task_ids in loaded tasks'):
        export_airline_dataset(output_dir=tmp_path / 'out', baseline_and_skill_root=source_root, ranked_skills_path=ranked_path, parsed_skill_dir=parsed_dir)

def test_export_airline_dataset_fails_when_loaded_tasks_include_extra_task_ids(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    source_root = tmp_path / 'single-skill-data'
    source_root.mkdir()
    parsed_dir = tmp_path / 'parsed'
    parsed_dir.mkdir()
    ranked_path = tmp_path / 'skills_ranked.json'
    _write_json(source_root / 'airline_baseline_agg1_20260315_000000.json', {'summary': {'domain': 'airline', 'skill_names': [], 'num_trials': 2}, 'results': [{'task_id': '0', 'reward': 0.0, 'pass_count': 0, 'num_trials': 2}]})
    _write_json(source_root / 'airline_single_alpha_skill_agg1_20260315_000000.json', {'summary': {'domain': 'airline', 'skill_names': ['alpha_skill'], 'num_trials': 2}, 'results': [{'task_id': '0', 'reward': 0.5, 'pass_count': 1, 'num_trials': 2}]})
    _write_json(ranked_path, [{'rank': 0, 'skill_name': 'alpha_skill', 'file': 'skill_alpha.py', 'frequency': 1, 'cluster_size': 1, 'cluster_id': 0, 'code': ''}])
    (parsed_dir / 'skill_alpha.py').write_text('\n'.join(['SKILL_NAME = "alpha_skill"', 'SKILL_DOMAIN = "airline"', 'SKILL_DESCRIPTION = "alpha desc"', 'PROMPT_GUIDANCE = "alpha guide"', 'TARGET_PATTERN = "alpha pattern"']), encoding='utf-8')
    monkeypatch.setattr('temp.selector.export_dataset.load_airline_tasks', lambda task_split_name='base': [_make_task('0', 'baseline task'), _make_task('999', 'extra loaded task')])
    with pytest.raises(ValueError, match='Extra task_ids in loaded tasks'):
        export_airline_dataset(output_dir=tmp_path / 'out', baseline_and_skill_root=source_root, ranked_skills_path=ranked_path, parsed_skill_dir=parsed_dir)

def test_export_airline_dataset_handles_duplicate_ranked_skill_entries(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    source_root = tmp_path / 'single-skill-data'
    source_root.mkdir()
    parsed_dir = tmp_path / 'parsed'
    parsed_dir.mkdir()
    ranked_path = tmp_path / 'skills_ranked.json'
    _write_json(source_root / 'airline_baseline_agg1_20260315_000000.json', {'summary': {'domain': 'airline', 'skill_names': [], 'num_trials': 2}, 'results': [{'task_id': '0', 'reward': 0.0, 'pass_count': 0, 'num_trials': 2}]})
    _write_json(source_root / 'airline_single_dupe_skill_agg1_20260315_000000.json', {'summary': {'domain': 'airline', 'skill_names': ['dupe_skill'], 'num_trials': 2}, 'results': [{'task_id': '0', 'reward': 0.5, 'pass_count': 1, 'num_trials': 2}]})
    _write_json(ranked_path, [{'rank': 0, 'skill_name': 'dupe_skill', 'file': 'skill_dupe_primary.py', 'frequency': 10, 'cluster_size': 5, 'cluster_id': 3, 'code': ''}, {'rank': 1, 'skill_name': 'dupe_skill', 'file': 'skill_dupe_secondary.py', 'frequency': 7, 'cluster_size': 4, 'cluster_id': 6, 'code': ''}])
    (parsed_dir / 'skill_dupe_primary.py').write_text('\n'.join(['SKILL_NAME = "dupe_skill"', 'SKILL_DOMAIN = "airline"', 'SKILL_DESCRIPTION = "primary desc"', 'PROMPT_GUIDANCE = "primary guide"', 'TARGET_PATTERN = "primary pattern"']), encoding='utf-8')
    monkeypatch.setattr('temp.selector.export_dataset.load_airline_tasks', lambda task_split_name='base': [_make_task('0', 'duplicate ranked entries')])
    export_airline_dataset(output_dir=tmp_path / 'out', baseline_and_skill_root=source_root, ranked_skills_path=ranked_path, parsed_skill_dir=parsed_dir)
    skills_rows = _read_jsonl(tmp_path / 'out' / 'skills.jsonl')
    assert len(skills_rows) == 1
    assert skills_rows[0]['skill_name'] == 'dupe_skill'
    assert skills_rows[0]['source_file'] == 'skill_dupe_primary.py'

def test_exporter_cli_surfaces_unrelated_loader_dependency_errors(tmp_path: Path):
    source_root = tmp_path / 'single-skill-data'
    source_root.mkdir()
    parsed_dir = tmp_path / 'parsed'
    parsed_dir.mkdir()
    ranked_path = tmp_path / 'skills_ranked.json'
    tasks_json_path = tmp_path / 'tasks.json'
    split_tasks_path = tmp_path / 'split_tasks.json'
    output_dir = tmp_path / 'out'
    _write_json(source_root / 'airline_baseline_agg1_20260315_000000.json', {'summary': {'domain': 'airline', 'skill_names': [], 'num_trials': 2}, 'results': [{'task_id': '0', 'reward': 0.0, 'pass_count': 0, 'num_trials': 2}]})
    _write_json(source_root / 'airline_single_cli_skill_agg1_20260315_000000.json', {'summary': {'domain': 'airline', 'skill_names': ['cli_skill'], 'num_trials': 2}, 'results': [{'task_id': '0', 'reward': 0.5, 'pass_count': 1, 'num_trials': 2}]})
    _write_json(ranked_path, [{'rank': 0, 'skill_name': 'cli_skill', 'file': 'skill_cli.py', 'frequency': 3, 'cluster_size': 1, 'cluster_id': 0, 'code': ''}])
    (parsed_dir / 'skill_cli.py').write_text('\n'.join(['SKILL_NAME = "cli_skill"', 'SKILL_DOMAIN = "airline"', 'SKILL_DESCRIPTION = "cli desc"', 'PROMPT_GUIDANCE = "cli guidance"', 'TARGET_PATTERN = "cli pattern"']), encoding='utf-8')
    _write_json(tasks_json_path, [{'id': '0', 'description': {'purpose': 'cli fallback purpose', 'relevant_policies': None, 'notes': None}, 'user_scenario': {'persona': None, 'instructions': {'task_instructions': 'cli fallback instructions'}}}])
    _write_json(split_tasks_path, {'base': ['0']})
    command = [sys.executable, str(PROJECT_ROOT / 'temp/selector/export_dataset.py'), '--domain', 'airline', '--output-dir', str(output_dir), '--baseline-and-skill-root', str(source_root), '--ranked-skills-path', str(ranked_path), '--parsed-skill-dir', str(parsed_dir), '--local-tasks-json-path', str(tasks_json_path), '--local-split-tasks-json-path', str(split_tasks_path)]
    result = subprocess.run(command, cwd=PROJECT_ROOT, text=True, capture_output=True, check=False)
    assert result.returncode != 0
    assert 'loguru' in result.stderr
    assert not (output_dir / 'manifest.json').exists()

def test_export_airline_dataset_skips_malformed_single_skill_files(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    source_root = tmp_path / 'single-skill-data'
    source_root.mkdir()
    parsed_dir = tmp_path / 'parsed'
    parsed_dir.mkdir()
    ranked_path = tmp_path / 'skills_ranked.json'
    _write_json(source_root / 'airline_baseline_agg1_20260315_000000.json', {'summary': {'domain': 'airline', 'skill_names': [], 'num_trials': 2}, 'results': [{'task_id': '0', 'reward': 0.0, 'pass_count': 0, 'num_trials': 2}]})
    _write_json(source_root / 'airline_single_malformed_skill_agg1_20260315_000000.json', {'summary': {'domain': 'airline', 'skill_names': ['malformed_skill'], 'num_trials': 2}, 'results': [{'task_id': '0', 'reward': 0.5, 'num_trials': 2}]})
    _write_json(source_root / 'airline_single_valid_skill_agg1_20260315_000000.json', {'summary': {'domain': 'airline', 'skill_names': ['valid_skill'], 'num_trials': 2}, 'results': [{'task_id': '0', 'reward': 0.5, 'pass_count': 1, 'num_trials': 2}]})
    _write_json(ranked_path, [{'rank': 0, 'skill_name': 'valid_skill', 'file': 'skill_valid.py', 'frequency': 1, 'cluster_size': 1, 'cluster_id': 0, 'code': ''}])
    (parsed_dir / 'skill_valid.py').write_text('\n'.join(['SKILL_NAME = "valid_skill"', 'SKILL_DOMAIN = "airline"', 'SKILL_DESCRIPTION = "valid desc"', 'PROMPT_GUIDANCE = "valid guidance"', 'TARGET_PATTERN = "valid pattern"']), encoding='utf-8')
    monkeypatch.setattr('temp.selector.export_dataset.load_airline_tasks', lambda task_split_name='base': [_make_task('0', 'skip malformed')])
    manifest = export_airline_dataset(output_dir=tmp_path / 'out', baseline_and_skill_root=source_root, ranked_skills_path=ranked_path, parsed_skill_dir=parsed_dir)
    skills_rows = _read_jsonl(tmp_path / 'out' / 'skills.jsonl')
    assert len(skills_rows) == 1
    assert skills_rows[0]['skill_name'] == 'valid_skill'
    source_runs = cast(dict[str, object], manifest['source_runs'])
    malformed_count = cast(int, source_runs['malformed_single_skill_file_count'])
    malformed_files = cast(list[dict[str, object]], source_runs['malformed_single_skill_files'])
    assert malformed_count == 1
    assert len(malformed_files) == 1
    assert str(malformed_files[0]['path']).endswith('airline_single_malformed_skill_agg1_20260315_000000.json')

def test_export_airline_dataset_does_not_fallback_on_unexpected_task_loader_errors(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    source_root = tmp_path / 'single-skill-data'
    source_root.mkdir()
    parsed_dir = tmp_path / 'parsed'
    parsed_dir.mkdir()
    ranked_path = tmp_path / 'skills_ranked.json'
    _write_json(source_root / 'airline_baseline_agg1_20260315_000000.json', {'summary': {'domain': 'airline', 'skill_names': [], 'num_trials': 2}, 'results': [{'task_id': '0', 'reward': 0.0, 'pass_count': 0, 'num_trials': 2}]})
    _write_json(source_root / 'airline_single_alpha_skill_agg1_20260315_000000.json', {'summary': {'domain': 'airline', 'skill_names': ['alpha_skill'], 'num_trials': 2}, 'results': [{'task_id': '0', 'reward': 0.5, 'pass_count': 1, 'num_trials': 2}]})
    _write_json(ranked_path, [{'rank': 0, 'skill_name': 'alpha_skill', 'file': 'skill_alpha.py', 'frequency': 1, 'cluster_size': 1, 'cluster_id': 0, 'code': ''}])
    (parsed_dir / 'skill_alpha.py').write_text('\n'.join(['SKILL_NAME = "alpha_skill"', 'SKILL_DOMAIN = "airline"', 'SKILL_DESCRIPTION = "alpha desc"', 'PROMPT_GUIDANCE = "alpha guidance"', 'TARGET_PATTERN = "alpha pattern"']), encoding='utf-8')

    def _raise_runtime_error(*args, **kwargs):
        raise RuntimeError('unexpected loader failure')
    monkeypatch.setattr('temp.selector.export_dataset.load_airline_tasks', _raise_runtime_error)
    with pytest.raises(RuntimeError, match='unexpected loader failure'):
        export_airline_dataset(output_dir=tmp_path / 'out', baseline_and_skill_root=source_root, ranked_skills_path=ranked_path, parsed_skill_dir=parsed_dir)

@pytest.mark.parametrize(('loader_exc', 'match'), [(ModuleNotFoundError("No module named 'pydantic'"), 'pydantic'), (ImportError("cannot import name 'get_tasks' from 'tau2.run'"), 'cannot import name')])
def test_export_airline_dataset_does_not_fallback_on_unrelated_import_errors(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, loader_exc: BaseException, match: str):
    source_root = tmp_path / 'single-skill-data'
    source_root.mkdir()
    parsed_dir = tmp_path / 'parsed'
    parsed_dir.mkdir()
    ranked_path = tmp_path / 'skills_ranked.json'
    _write_json(source_root / 'airline_baseline_agg1_20260315_000000.json', {'summary': {'domain': 'airline', 'skill_names': [], 'num_trials': 2}, 'results': [{'task_id': '0', 'reward': 0.0, 'pass_count': 0, 'num_trials': 2}]})
    _write_json(source_root / 'airline_single_alpha_skill_agg1_20260315_000000.json', {'summary': {'domain': 'airline', 'skill_names': ['alpha_skill'], 'num_trials': 2}, 'results': [{'task_id': '0', 'reward': 0.5, 'pass_count': 1, 'num_trials': 2}]})
    _write_json(ranked_path, [{'rank': 0, 'skill_name': 'alpha_skill', 'file': 'skill_alpha.py', 'frequency': 1, 'cluster_size': 1, 'cluster_id': 0, 'code': ''}])
    (parsed_dir / 'skill_alpha.py').write_text('\n'.join(['SKILL_NAME = "alpha_skill"', 'SKILL_DOMAIN = "airline"', 'SKILL_DESCRIPTION = "alpha desc"', 'PROMPT_GUIDANCE = "alpha guidance"', 'TARGET_PATTERN = "alpha pattern"']), encoding='utf-8')

    def _raise_loader_error(*args, **kwargs):
        raise loader_exc
    monkeypatch.setattr('temp.selector.export_dataset.load_airline_tasks', _raise_loader_error)
    with pytest.raises(type(loader_exc), match=match):
        export_airline_dataset(output_dir=tmp_path / 'out', baseline_and_skill_root=source_root, ranked_skills_path=ranked_path, parsed_skill_dir=parsed_dir)

def test_export_airline_dataset_fails_on_extra_single_skill_task_ids(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    source_root = tmp_path / 'single-skill-data'
    source_root.mkdir()
    parsed_dir = tmp_path / 'parsed'
    parsed_dir.mkdir()
    ranked_path = tmp_path / 'skills_ranked.json'
    _write_json(source_root / 'airline_baseline_agg1_20260315_000000.json', {'summary': {'domain': 'airline', 'skill_names': [], 'num_trials': 2}, 'results': [{'task_id': '0', 'reward': 0.0, 'pass_count': 0, 'num_trials': 2}]})
    _write_json(source_root / 'airline_single_alpha_skill_agg1_20260315_000000.json', {'summary': {'domain': 'airline', 'skill_names': ['alpha_skill'], 'num_trials': 2}, 'results': [{'task_id': '0', 'reward': 0.5, 'pass_count': 1, 'num_trials': 2}, {'task_id': '999', 'reward': 0.5, 'pass_count': 1, 'num_trials': 2}]})
    _write_json(ranked_path, [{'rank': 0, 'skill_name': 'alpha_skill', 'file': 'skill_alpha.py', 'frequency': 1, 'cluster_size': 1, 'cluster_id': 0, 'code': ''}])
    (parsed_dir / 'skill_alpha.py').write_text('\n'.join(['SKILL_NAME = "alpha_skill"', 'SKILL_DOMAIN = "airline"', 'SKILL_DESCRIPTION = "alpha desc"', 'PROMPT_GUIDANCE = "alpha guidance"', 'TARGET_PATTERN = "alpha pattern"']), encoding='utf-8')
    monkeypatch.setattr('temp.selector.export_dataset.load_airline_tasks', lambda task_split_name='base': [_make_task('0', 'baseline only task')])
    with pytest.raises(ValueError, match='Extra task_ids'):
        export_airline_dataset(output_dir=tmp_path / 'out', baseline_and_skill_root=source_root, ranked_skills_path=ranked_path, parsed_skill_dir=parsed_dir)
