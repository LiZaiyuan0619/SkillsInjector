import ast
import json
import re
from copy import deepcopy
from collections.abc import Mapping
from pathlib import Path
from typing import Any
from temp.selector.domain_registry import DomainSpec, get_domain_spec
SCHEMA_VERSION = 'selector_contract.v1'
CONTRACT_DOMAIN = 'airline'
BUDGET = 1
SPLIT_POLICY = 'feasibility_full_matrix'
OUTPUT_FILENAMES = {'manifest': 'manifest.json', 'tasks': 'tasks.jsonl', 'skills': 'skills.jsonl', 'labels': 'labels.jsonl'}
LABEL_DEFINITION = {'label_name': 'utility_label', 'label_type': 'float', 'value_range': [0.0, 1.0], 'semantic': 'Estimated single-skill pass probability p_skill(t, s).', 'source': 'offline single-skill outcomes'}
REQUIRED_MANIFEST_KEYS = ('schema_version', 'domain', 'budget', 'split_policy', 'output_filenames', 'label_definition')

def resolve_domain_spec(domain: str | DomainSpec=CONTRACT_DOMAIN) -> DomainSpec:
    if isinstance(domain, DomainSpec):
        return domain
    return get_domain_spec(domain)

def resolve_task_split_name(domain: str | DomainSpec=CONTRACT_DOMAIN, task_split_name: str | None=None) -> str | None:
    if task_split_name is not None:
        return task_split_name
    return resolve_domain_spec(domain).task_split

def build_manifest(domain: str | DomainSpec=CONTRACT_DOMAIN, budget: int=BUDGET, training_scope: str | None=None, source_domains: list[str] | None=None) -> dict[str, object]:
    spec = resolve_domain_spec(domain)
    manifest: dict[str, object] = {'schema_version': SCHEMA_VERSION, 'domain': spec.domain, 'budget': int(budget), 'split_policy': SPLIT_POLICY, 'output_filenames': dict(OUTPUT_FILENAMES), 'label_definition': deepcopy(LABEL_DEFINITION)}
    if training_scope is not None:
        manifest['training_scope'] = training_scope
    if source_domains is not None:
        manifest['source_domains'] = list(source_domains)
    return manifest

def validate_required_manifest_keys(manifest: Mapping[str, object]) -> None:
    missing = [key for key in REQUIRED_MANIFEST_KEYS if key not in manifest]
    if missing:
        raise ValueError(f"Missing required manifest keys: {', '.join(missing)}")

def normalize_skill_name(skill_name: str) -> str:
    value = skill_name.strip()
    if '/' in value:
        return value.split('/')[-1]
    return value

def _load_json_file(path: Path) -> dict[str, Any]:
    with path.open('r', encoding='utf-8') as fp:
        payload = json.load(fp)
    if not isinstance(payload, dict):
        raise ValueError(f'Expected JSON object at {path}')
    return payload

def get_baseline_aggregate_glob(domain: str | DomainSpec=CONTRACT_DOMAIN) -> str:
    spec = resolve_domain_spec(domain)
    return f'{spec.domain}_baseline_agg*.json'

def get_single_skill_aggregate_glob(domain: str | DomainSpec=CONTRACT_DOMAIN) -> str:
    spec = resolve_domain_spec(domain)
    return f'{spec.domain}_single_*_agg*.json'

def select_baseline_aggregate(path_or_dir: str | Path, domain: str | DomainSpec=CONTRACT_DOMAIN) -> Path:
    path = Path(path_or_dir)
    if path.is_file():
        return path
    if not path.exists():
        raise FileNotFoundError(f'Missing baseline aggregate directory: {path}')
    candidates = sorted(path.glob(get_baseline_aggregate_glob(domain)))
    if not candidates:
        raise FileNotFoundError(f'Missing baseline aggregate file under: {path}')
    return max(candidates, key=_aggregate_file_sort_key)

def select_single_skill_aggregate(path_or_dir: str | Path, skill_name: str, domain: str | DomainSpec=CONTRACT_DOMAIN) -> Path:
    path = Path(path_or_dir)
    normalized_target = normalize_skill_name(skill_name)
    if path.is_file():
        payload = _load_json_file(path)
        summary = payload.get('summary', {})
        skill_names = [normalize_skill_name(name) for name in summary.get('skill_names', [])]
        if normalized_target in skill_names:
            return path
        raise FileNotFoundError(f'Single-skill aggregate for {normalized_target} not found in file: {path}')
    if not path.exists():
        raise FileNotFoundError(f'Missing single-skill aggregate directory: {path}')
    matches: list[Path] = []
    for candidate in sorted(path.glob(get_single_skill_aggregate_glob(domain))):
        payload = _load_json_file(candidate)
        summary = payload.get('summary', {})
        skill_names = [normalize_skill_name(name) for name in summary.get('skill_names', [])]
        if normalized_target in skill_names:
            matches.append(candidate)
    if not matches:
        raise FileNotFoundError(f'No single-skill aggregate files found for {normalized_target} under: {path}')
    return max(matches, key=_aggregate_file_sort_key)

def select_airline_baseline_aggregate(path_or_dir: str | Path) -> Path:
    return select_baseline_aggregate(path_or_dir, domain='airline')

def select_airline_single_skill_aggregate(path_or_dir: str | Path, skill_name: str) -> Path:
    return select_single_skill_aggregate(path_or_dir, skill_name, domain='airline')

def _aggregate_file_sort_key(path: Path) -> tuple[int, str, str]:
    stem = path.stem
    agg_match = re.search('_agg(?P<agg>\\d+)', stem)
    agg_num = int(agg_match.group('agg')) if agg_match else -1
    timestamp_match = re.search('_agg\\d+_(?P<ts>\\d{8}_\\d{6})', stem)
    timestamp = timestamp_match.group('ts') if timestamp_match else ''
    return (agg_num, timestamp, path.name)

def load_aggregate_task_results(path: str | Path) -> dict[str, Any]:
    payload = _load_json_file(Path(path))
    summary = payload.get('summary', {})
    raw_results = payload.get('results', [])
    if not isinstance(raw_results, list):
        raise ValueError(f'Aggregate results must be a list: {path}')
    results_by_task_id: dict[str, dict[str, Any]] = {}
    for row in raw_results:
        task_id = str(row['task_id'])
        results_by_task_id[task_id] = {'task_id': task_id, 'reward': float(row['reward']), 'pass_count': int(row['pass_count']), 'num_trials': int(row['num_trials'])}
    skill_names = [normalize_skill_name(skill_name) for skill_name in summary.get('skill_names', []) if isinstance(skill_name, str)]
    return {'domain': summary.get('domain'), 'num_trials': int(summary.get('num_trials', 0)), 'skill_names': skill_names, 'results_by_task_id': results_by_task_id}

def _extract_skill_fields_from_python(source: str) -> dict[str, str]:
    module = ast.parse(source)
    fields: dict[str, str] = {}
    wanted = {'SKILL_NAME', 'SKILL_DOMAIN', 'SKILL_DESCRIPTION', 'PROMPT_GUIDANCE', 'TARGET_PATTERN'}
    for node in module.body:
        if not isinstance(node, ast.Assign) or len(node.targets) != 1:
            continue
        target = node.targets[0]
        if not isinstance(target, ast.Name) or target.id not in wanted:
            continue
        value = node.value
        if isinstance(value, ast.Constant) and isinstance(value.value, str):
            fields[target.id] = value.value
    return fields

def extract_skill_function_parts(source: str) -> dict[str, str]:
    result = {'docstring': '', 'body': ''}
    try:
        module = ast.parse(source)
    except SyntaxError:
        return result
    for node in module.body:
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        docstring = ast.get_docstring(node)
        if docstring:
            result['docstring'] = docstring
        lines = source.splitlines()
        func_start = node.lineno - 1
        func_end = node.end_lineno
        if func_end is not None:
            func_lines = lines[func_start:func_end]
            result['body'] = '\n'.join(func_lines)
        break
    return result

def _load_ranked_skill_entries(path: Path) -> list[dict[str, Any]]:
    with path.open('r', encoding='utf-8') as fp:
        payload = json.load(fp)
    if isinstance(payload, list):
        return payload
    if isinstance(payload, dict) and isinstance(payload.get('items'), list):
        return payload['items']
    raise ValueError(f'Unsupported ranked skills JSON shape: {path}')

def _select_ranked_skill_entry(ranked_entries: list[dict[str, Any]], skill_name: str) -> tuple[dict[str, Any], str]:
    requested_skill_name = skill_name.strip()
    exact_matches = [entry for entry in ranked_entries if str(entry.get('skill_name', '')).strip() == requested_skill_name]
    if len(exact_matches) > 1:
        raise ValueError(f'Duplicate skill_name entries for {requested_skill_name}')
    if exact_matches:
        return (exact_matches[0], requested_skill_name)
    normalized_target = normalize_skill_name(requested_skill_name)
    normalized_matches = [entry for entry in ranked_entries if normalize_skill_name(str(entry.get('skill_name', ''))) == normalized_target]
    if len(normalized_matches) > 1:
        raise ValueError(f'Duplicate skill_name entries for {normalized_target}')
    if not normalized_matches:
        raise FileNotFoundError(f'Skill metadata for {normalized_target} not found')
    return (normalized_matches[0], normalized_target)

def extract_skill_metadata(skill_name: str, parsed_skill_dir: str | Path, ranked_skills_path: str | Path) -> dict[str, str]:
    normalized_target = normalize_skill_name(skill_name)
    ranked_entries = _load_ranked_skill_entries(Path(ranked_skills_path))
    (ranked_entry, _matched_skill_name) = _select_ranked_skill_entry(ranked_entries, skill_name)
    parsed_dir = Path(parsed_skill_dir)
    parsed_fields: dict[str, str] = {}
    source_code: str = ''
    ranked_file = ranked_entry.get('file')
    if isinstance(ranked_file, str):
        parsed_path = parsed_dir / ranked_file
        if parsed_path.exists():
            source_code = parsed_path.read_text(encoding='utf-8')
            parsed_fields = _extract_skill_fields_from_python(source_code)
    if not parsed_fields:
        pattern_matches = sorted(parsed_dir.glob(f'*{normalized_target}.py'))
        if len(pattern_matches) == 1:
            source_code = pattern_matches[0].read_text(encoding='utf-8')
            parsed_fields = _extract_skill_fields_from_python(source_code)
    if not parsed_fields:
        ranked_code = ranked_entry.get('code')
        if not isinstance(ranked_code, str):
            raise ValueError(f'Ranked skill entry for {normalized_target} has no code string')
        source_code = ranked_code
        parsed_fields = _extract_skill_fields_from_python(ranked_code)
    func_parts = extract_skill_function_parts(source_code) if source_code else {'docstring': '', 'body': ''}
    return {'skill_name': normalize_skill_name(parsed_fields.get('SKILL_NAME', normalized_target)), 'skill_domain': parsed_fields.get('SKILL_DOMAIN', ''), 'skill_description': parsed_fields.get('SKILL_DESCRIPTION', ''), 'prompt_guidance': parsed_fields.get('PROMPT_GUIDANCE', ''), 'target_pattern': parsed_fields.get('TARGET_PATTERN', ''), 'docstring': func_parts['docstring'], 'body': func_parts['body']}
