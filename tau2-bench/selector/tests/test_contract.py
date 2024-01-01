import sys
from pathlib import Path
from typing import cast
import pytest
PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
from temp.selector.common import BUDGET, LABEL_DEFINITION, OUTPUT_FILENAMES, REQUIRED_MANIFEST_KEYS, SCHEMA_VERSION, SPLIT_POLICY, build_manifest, select_baseline_aggregate, select_single_skill_aggregate, validate_required_manifest_keys

def test_scope_markers_are_frozen_for_task1_slice():
    assert BUDGET == 1
    assert SPLIT_POLICY == 'feasibility_full_matrix'

def test_schema_and_output_filenames_are_frozen():
    assert SCHEMA_VERSION == 'selector_contract.v1'
    assert OUTPUT_FILENAMES == {'manifest': 'manifest.json', 'tasks': 'tasks.jsonl', 'skills': 'skills.jsonl', 'labels': 'labels.jsonl'}

def test_label_definition_metadata_is_frozen():
    assert LABEL_DEFINITION == {'label_name': 'utility_label', 'label_type': 'float', 'value_range': [0.0, 1.0], 'semantic': 'Estimated single-skill pass probability p_skill(t, s).', 'source': 'offline single-skill outcomes'}

def test_manifest_contract_and_missing_key_guard():
    manifest = build_manifest()
    assert tuple(sorted(manifest.keys())) == tuple(sorted(REQUIRED_MANIFEST_KEYS))
    validate_required_manifest_keys(manifest)
    missing_domain = dict(manifest)
    missing_domain.pop('domain')
    with pytest.raises(ValueError, match='Missing required manifest keys: domain'):
        validate_required_manifest_keys(missing_domain)

def test_build_manifest_defaults_to_airline_for_legacy_callers():
    manifest = build_manifest()
    assert manifest['domain'] == 'airline'

def test_build_manifest_supports_parameterized_metadata():
    manifest = build_manifest(domain='retail', budget=4, training_scope='cross_domain', source_domains=['airline', 'retail', 'telecom'])
    assert manifest['domain'] == 'retail'
    assert manifest['budget'] == 4
    assert manifest['training_scope'] == 'cross_domain'
    assert manifest['source_domains'] == ['airline', 'retail', 'telecom']

@pytest.mark.parametrize('domain', ['airline', 'retail', 'telecom'])
def test_build_manifest_supports_all_registered_domains(domain: str):
    manifest = build_manifest(domain=domain)
    assert manifest['schema_version'] == SCHEMA_VERSION
    assert manifest['domain'] == domain
    assert manifest['budget'] == BUDGET
    assert manifest['split_policy'] == SPLIT_POLICY
    assert tuple(sorted(manifest.keys())) == tuple(sorted(REQUIRED_MANIFEST_KEYS))

def test_build_manifest_rejects_unsupported_domain():
    with pytest.raises(ValueError, match='Unsupported domain: lodging'):
        build_manifest(domain='lodging')

def test_build_manifest_returns_independent_nested_mutables():
    manifest = build_manifest()
    output_filenames = cast(dict[str, str], manifest['output_filenames'])
    label_definition = cast(dict[str, object], manifest['label_definition'])
    output_filenames['manifest'] = 'mutated.json'
    label_definition['label_name'] = 'mutated_label'
    assert OUTPUT_FILENAMES['manifest'] == 'manifest.json'
    assert LABEL_DEFINITION['label_name'] == 'utility_label'

def test_select_baseline_aggregate_uses_domain_aware_prefix(tmp_path: Path):
    airline = tmp_path / 'airline_baseline_agg1_20260315_000000.json'
    retail = tmp_path / 'retail_baseline_agg4_20260315_000000.json'
    airline.write_text('{}', encoding='utf-8')
    retail.write_text('{}', encoding='utf-8')
    chosen = select_baseline_aggregate(tmp_path, domain='retail')
    assert chosen == retail

def test_select_single_skill_aggregate_uses_domain_aware_prefix(tmp_path: Path):
    airline = tmp_path / 'airline_single_alpha_skill_agg1_20260315_000000.json'
    retail = tmp_path / 'retail_single_alpha_skill_agg4_20260315_000000.json'
    airline.write_text('{"summary":{"skill_names":["alpha_skill"]}}', encoding='utf-8')
    retail.write_text('{"summary":{"skill_names":["alpha_skill"]}}', encoding='utf-8')
    chosen = select_single_skill_aggregate(tmp_path, 'alpha_skill', domain='retail')
    assert chosen == retail
