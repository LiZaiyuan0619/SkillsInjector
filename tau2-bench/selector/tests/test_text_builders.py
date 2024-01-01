import json
import sys
from pathlib import Path
from types import SimpleNamespace
import pytest
PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
from temp.selector.common import extract_skill_metadata, load_aggregate_task_results, normalize_skill_name, select_airline_baseline_aggregate, select_airline_single_skill_aggregate
from temp.selector.domain_registry import get_domain_spec
import temp.selector.text_builders as text_builders_module
from temp.selector.text_builders import build_skill_text, build_task_text, load_airline_tasks, load_domain_tasks, load_retail_tasks, load_telecom_tasks

def _write_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload), encoding='utf-8')

def test_normalize_skill_name_removes_domain_prefix():
    assert normalize_skill_name('airline/user_pressure_response_guide') == 'user_pressure_response_guide'
    assert normalize_skill_name('user_pressure_response_guide') == 'user_pressure_response_guide'

def test_select_baseline_and_single_skill_aggregate_inputs(tmp_path: Path):
    baseline_path = tmp_path / 'airline_baseline_agg3_20260315_152801.json'
    single_skill_path = tmp_path / 'airline_single_user_pressure_response_guide_agg5_20260317_101704.json'
    _write_json(baseline_path, {'summary': {'domain': 'airline', 'skill_names': [], 'num_trials': 3}, 'results': [{'task_id': '0', 'reward': 0.5, 'pass_count': 1, 'num_trials': 2}]})
    _write_json(single_skill_path, {'summary': {'domain': 'airline', 'skill_names': ['airline/user_pressure_response_guide'], 'num_trials': 5}, 'results': [{'task_id': '0', 'reward': 0.6, 'pass_count': 3, 'num_trials': 5}]})
    chosen_baseline = select_airline_baseline_aggregate(tmp_path)
    chosen_single = select_airline_single_skill_aggregate(tmp_path, 'user_pressure_response_guide')
    assert chosen_baseline == baseline_path
    assert chosen_single == single_skill_path

def test_baseline_aggregate_selection_prefers_highest_agg_then_latest_timestamp(tmp_path: Path):
    agg9 = tmp_path / 'airline_baseline_agg9_20260315_152801.json'
    agg10_old = tmp_path / 'airline_baseline_agg10_20260315_152801.json'
    agg10_new = tmp_path / 'airline_baseline_agg10_20260316_010101.json'
    payload = {'summary': {'domain': 'airline', 'skill_names': [], 'num_trials': 3}, 'results': []}
    _write_json(agg9, payload)
    _write_json(agg10_old, payload)
    _write_json(agg10_new, payload)
    chosen = select_airline_baseline_aggregate(tmp_path)
    assert chosen == agg10_new

def test_baseline_aggregate_selection_uses_lexical_tiebreak_last(tmp_path: Path):
    first = tmp_path / 'airline_baseline_agg10_20260316_010101_a.json'
    second = tmp_path / 'airline_baseline_agg10_20260316_010101_b.json'
    payload = {'summary': {'domain': 'airline', 'skill_names': [], 'num_trials': 3}, 'results': []}
    _write_json(first, payload)
    _write_json(second, payload)
    chosen = select_airline_baseline_aggregate(tmp_path)
    assert chosen == second

def test_missing_baseline_file_raises(tmp_path: Path):
    with pytest.raises(FileNotFoundError, match='baseline'):
        select_airline_baseline_aggregate(tmp_path)

def test_single_skill_aggregate_selection_prefers_highest_agg_then_latest_timestamp(tmp_path: Path):
    agg2 = tmp_path / 'airline_single_user_pressure_response_guide_agg2_20260315_101704.json'
    agg3 = tmp_path / 'airline_single_user_pressure_response_guide_agg3_20260316_101704.json'
    agg5_old = tmp_path / 'airline_single_user_pressure_response_guide_agg5_20260317_101704.json'
    agg5_new = tmp_path / 'airline_single_user_pressure_response_guide_agg5_20260318_101704.json'
    payload = {'summary': {'domain': 'airline', 'skill_names': ['airline/user_pressure_response_guide'], 'num_trials': 5}, 'results': []}
    _write_json(agg2, payload)
    _write_json(agg3, payload)
    _write_json(agg5_old, payload)
    _write_json(agg5_new, payload)
    chosen = select_airline_single_skill_aggregate(tmp_path, 'user_pressure_response_guide')
    assert chosen == agg5_new

def test_single_skill_aggregate_selection_uses_lexical_tiebreak_last(tmp_path: Path):
    first = tmp_path / 'airline_single_user_pressure_response_guide_agg5_20260318_101704_a.json'
    second = tmp_path / 'airline_single_user_pressure_response_guide_agg5_20260318_101704_b.json'
    payload = {'summary': {'domain': 'airline', 'skill_names': ['airline/user_pressure_response_guide'], 'num_trials': 5}, 'results': []}
    _write_json(first, payload)
    _write_json(second, payload)
    chosen = select_airline_single_skill_aggregate(tmp_path, 'user_pressure_response_guide')
    assert chosen == second

def test_load_aggregate_task_results_keeps_trial_counts_separate(tmp_path: Path):
    baseline_path = tmp_path / 'baseline.json'
    skill_path = tmp_path / 'skill.json'
    _write_json(baseline_path, {'summary': {'domain': 'airline', 'skill_names': [], 'num_trials': 3}, 'results': [{'task_id': '0', 'reward': 2 / 3, 'pass_count': 2, 'num_trials': 3}]})
    _write_json(skill_path, {'summary': {'domain': 'airline', 'skill_names': ['airline/user_pressure_response_guide'], 'num_trials': 5}, 'results': [{'task_id': '0', 'reward': 0.8, 'pass_count': 4, 'num_trials': 5}]})
    baseline_data = load_aggregate_task_results(baseline_path)
    skill_data = load_aggregate_task_results(skill_path)
    assert baseline_data['num_trials'] == 3
    assert skill_data['num_trials'] == 5
    assert baseline_data['results_by_task_id']['0']['num_trials'] == 3
    assert skill_data['results_by_task_id']['0']['num_trials'] == 5

def test_extract_skill_metadata_prefers_parsed_file_when_available(tmp_path: Path):
    parsed_dir = tmp_path / 'parsed'
    parsed_dir.mkdir()
    ranked_json = tmp_path / 'skills_ranked.json'
    parsed_file = parsed_dir / 'skill_0294_user_pressure_response_guide.py'
    parsed_file.write_text('\n'.join(['SKILL_NAME = "user_pressure_response_guide"', 'SKILL_DOMAIN = "airline"', 'SKILL_DESCRIPTION = "parsed description"', 'PROMPT_GUIDANCE = "parsed guidance"', 'TARGET_PATTERN = "parsed pattern"']), encoding='utf-8')
    _write_json(ranked_json, {'items': [{'skill_name': 'user_pressure_response_guide', 'file': 'skill_0294_user_pressure_response_guide.py', 'code': '\n'.join(['SKILL_NAME = "user_pressure_response_guide"', 'SKILL_DOMAIN = "airline"', 'SKILL_DESCRIPTION = "ranked description"', 'PROMPT_GUIDANCE = "ranked guidance"', 'TARGET_PATTERN = "ranked pattern"'])}]})
    metadata = extract_skill_metadata('airline/user_pressure_response_guide', parsed_dir, ranked_json)
    assert metadata['skill_name'] == 'user_pressure_response_guide'
    assert metadata['skill_description'] == 'parsed description'
    assert metadata['prompt_guidance'] == 'parsed guidance'
    assert metadata['target_pattern'] == 'parsed pattern'

def test_extract_skill_metadata_falls_back_to_ranked_code_and_rejects_duplicates(tmp_path: Path):
    parsed_dir = tmp_path / 'parsed'
    parsed_dir.mkdir()
    ranked_json = tmp_path / 'skills_ranked.json'
    _write_json(ranked_json, [{'skill_name': 'cabin_upgrade_policy_reference', 'file': 'skill_a.py', 'code': '\n'.join(['SKILL_NAME = "cabin_upgrade_policy_reference"', 'SKILL_DOMAIN = "airline"', 'SKILL_DESCRIPTION = "first"', 'PROMPT_GUIDANCE = "first guidance"', 'TARGET_PATTERN = "first pattern"'])}, {'skill_name': 'cabin_upgrade_policy_reference', 'file': 'skill_b.py', 'code': '\n'.join(['SKILL_NAME = "cabin_upgrade_policy_reference"', 'SKILL_DOMAIN = "airline"', 'SKILL_DESCRIPTION = "second"', 'PROMPT_GUIDANCE = "second guidance"', 'TARGET_PATTERN = "second pattern"'])}])
    with pytest.raises(ValueError, match='Duplicate skill_name entries'):
        extract_skill_metadata('cabin_upgrade_policy_reference', parsed_dir, ranked_json)

def test_load_airline_tasks_prefers_tau2_run_helper(monkeypatch: pytest.MonkeyPatch):
    captured: dict[str, object] = {}

    def fake_get_tasks(**kwargs):
        captured['kwargs'] = kwargs
        return ['ok']

    def fake_import_module(name: str):
        assert name == 'tau2.run'
        return SimpleNamespace(get_tasks=fake_get_tasks)
    monkeypatch.setattr(text_builders_module, 'import_module', fake_import_module)
    result = load_airline_tasks(task_split_name='base')
    assert result == ['ok']
    assert captured['kwargs'] == {'task_set_name': 'airline', 'task_split_name': 'base'}

def test_load_airline_tasks_falls_back_only_when_tau2_run_module_missing(monkeypatch: pytest.MonkeyPatch):
    calls: list[str] = []

    def fake_import_module(name: str):
        calls.append(name)
        if name == 'tau2.run':
            raise ModuleNotFoundError("No module named 'tau2.run'")
        if name == 'tau2.domains.airline.environment':
            return SimpleNamespace(get_tasks=lambda **kwargs: ['fallback', kwargs])
        raise AssertionError(f'Unexpected import request: {name}')
    monkeypatch.setattr(text_builders_module, 'import_module', fake_import_module)
    result = load_airline_tasks(task_split_name='base')
    assert result == ['fallback', {'task_split_name': 'base'}]
    assert calls == ['tau2.run', 'tau2.domains.airline.environment']

def test_load_airline_tasks_does_not_swallow_unrelated_module_not_found(monkeypatch: pytest.MonkeyPatch):

    def fake_import_module(name: str):
        if name == 'tau2.run':
            raise ModuleNotFoundError("No module named 'pydantic'")
        raise AssertionError(f'Fallback import should not happen: {name}')
    monkeypatch.setattr(text_builders_module, 'import_module', fake_import_module)
    with pytest.raises(ModuleNotFoundError, match='pydantic'):
        load_airline_tasks(task_split_name='base')

def test_load_domain_tasks_uses_domain_spec_defaults(monkeypatch: pytest.MonkeyPatch):
    captured: list[dict[str, object]] = []

    def fake_get_tasks(**kwargs):
        captured.append(kwargs)
        return [kwargs['task_set_name'], kwargs['task_split_name']]

    def fake_import_module(name: str):
        assert name == 'tau2.run'
        return SimpleNamespace(get_tasks=fake_get_tasks)
    monkeypatch.setattr(text_builders_module, 'import_module', fake_import_module)
    airline_tasks = load_domain_tasks('airline')
    retail_tasks = load_domain_tasks('retail')
    telecom_tasks = load_domain_tasks(get_domain_spec('telecom'))
    assert airline_tasks == ['airline', None]
    assert retail_tasks == ['retail', None]
    assert telecom_tasks == ['telecom', 'base']
    assert captured == [{'task_set_name': 'airline', 'task_split_name': None}, {'task_set_name': 'retail', 'task_split_name': None}, {'task_set_name': 'telecom', 'task_split_name': 'base'}]

def test_domain_specific_task_helpers_delegate_with_expected_defaults(monkeypatch: pytest.MonkeyPatch):
    calls: list[tuple[object, object]] = []

    def fake_load_domain_tasks(domain, task_split_name=None):
        calls.append((domain, task_split_name))
        return [domain, task_split_name]
    monkeypatch.setattr(text_builders_module, 'load_domain_tasks', fake_load_domain_tasks)
    airline_tasks = load_airline_tasks()
    retail_tasks = load_retail_tasks()
    telecom_tasks = load_telecom_tasks()
    assert airline_tasks == ['airline', None]
    assert retail_tasks == ['retail', None]
    assert telecom_tasks == ['telecom', 'base']
    assert calls == [('airline', None), ('retail', None), ('telecom', 'base')]

def test_extract_skill_metadata_prefers_exact_domain_qualified_match(tmp_path: Path):
    parsed_dir = tmp_path / 'parsed'
    parsed_dir.mkdir()
    ranked_json = tmp_path / 'skills_ranked.json'
    _write_json(ranked_json, [{'skill_name': 'airline/shared_skill', 'file': 'airline_shared_skill.py', 'code': '\n'.join(['SKILL_NAME = "shared_skill"', 'SKILL_DOMAIN = "airline"', 'SKILL_DESCRIPTION = "airline description"', 'PROMPT_GUIDANCE = "airline guidance"', 'TARGET_PATTERN = "airline pattern"'])}, {'skill_name': 'retail/shared_skill', 'file': 'retail_shared_skill.py', 'code': '\n'.join(['SKILL_NAME = "shared_skill"', 'SKILL_DOMAIN = "retail"', 'SKILL_DESCRIPTION = "retail description"', 'PROMPT_GUIDANCE = "retail guidance"', 'TARGET_PATTERN = "retail pattern"'])}])
    metadata = extract_skill_metadata('retail/shared_skill', parsed_dir, ranked_json)
    assert metadata['skill_name'] == 'shared_skill'
    assert metadata['skill_domain'] == 'retail'
    assert metadata['skill_description'] == 'retail description'

def test_build_task_and_skill_text_are_stable_and_handle_empty_sections():
    task = SimpleNamespace(description=SimpleNamespace(purpose='test purpose', relevant_policies='policy text', notes='note text'), user_scenario=SimpleNamespace(persona='customer persona', instructions={'reason_for_call': 'need rebooking'}))
    task_text = build_task_text(task)
    assert 'purpose:' in task_text
    assert 'relevant_policies:' in task_text
    assert 'notes:' in task_text
    assert 'persona:' in task_text
    assert 'instructions:' in task_text
    assert '\n\n' not in task_text
    assert 'instructions: {"reason_for_call": "need rebooking"}' in task_text
    empty_like_skill_text = build_skill_text({'skill_description': '', 'prompt_guidance': '', 'target_pattern': ''})
    assert empty_like_skill_text == ''
    partial_skill_text = build_skill_text({'skill_description': 'short desc', 'prompt_guidance': '', 'target_pattern': 'some pattern'})
    assert partial_skill_text == 'skill_description: short desc\ntarget_pattern: some pattern'

def test_build_task_text_accepts_domain_spec_and_canonicalizes_mapping_instructions():
    task = SimpleNamespace(description=SimpleNamespace(purpose=None, relevant_policies='policy text', notes=None), user_scenario=SimpleNamespace(persona=None, instructions={'task_instructions': 'Resolve the billing issue first.', 'domain': 'telecom', 'reason_for_call': 'Billing problem'}))
    task_text = build_task_text(task, domain=get_domain_spec('telecom'))
    assert task_text == 'purpose: \nrelevant_policies: policy text\nnotes: \npersona: \ninstructions: {"domain": "telecom", "reason_for_call": "Billing problem", "task_instructions": "Resolve the billing issue first."}'
