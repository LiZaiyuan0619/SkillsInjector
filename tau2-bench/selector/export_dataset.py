import argparse
import ast
import json
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
from temp.selector.common import build_manifest, extract_skill_function_parts, extract_skill_metadata, get_single_skill_aggregate_glob, load_aggregate_task_results, normalize_skill_name, select_baseline_aggregate, select_single_skill_aggregate, validate_required_manifest_keys
from temp.selector.domain_registry import get_default_local_split_tasks_json_path, get_default_local_tasks_json_path, get_default_parsed_skill_dir, get_default_ranked_skills_path, get_domain_spec, resolve_export_identity
from temp.selector.text_builders import DEFAULT_SKILL_FIELDS_MODE, SKILL_FIELDS_BY_MODE, build_skill_text, build_task_text, load_airline_tasks, load_domain_tasks
DEFAULT_BASELINE_AND_SKILL_ROOT = Path('<MODEL_ROOT>/agent/skills/single-skill-data')
DEFAULT_RANKED_SKILLS_PATH = get_default_ranked_skills_path('airline')
DEFAULT_PARSED_SKILL_DIR = get_default_parsed_skill_dir('airline')
DEFAULT_LOCAL_TASKS_JSON_PATH = get_default_local_tasks_json_path('airline')
DEFAULT_LOCAL_SPLIT_TASKS_JSON_PATH = get_default_local_split_tasks_json_path('airline')

def _sort_task_id(task_id: str) -> tuple[int, str]:
    if task_id.isdigit():
        return (0, f'{int(task_id):08d}')
    return (1, task_id)

def _parse_skill_fields_from_source(source: str) -> dict[str, str]:
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

def _load_ranked_entries(ranked_skills_path: Path) -> list[dict[str, Any]]:
    payload = json.loads(ranked_skills_path.read_text(encoding='utf-8'))
    if isinstance(payload, list):
        return [entry for entry in payload if isinstance(entry, dict)]
    if isinstance(payload, dict) and isinstance(payload.get('items'), list):
        return [entry for entry in payload['items'] if isinstance(entry, dict)]
    raise ValueError(f'Unsupported ranked skills JSON shape: {ranked_skills_path}')

def _ranked_entry_index(ranked_entries: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    index: dict[str, dict[str, Any]] = {}
    for entry in ranked_entries:
        skill_name = normalize_skill_name(str(entry.get('skill_name', '')))
        if not skill_name:
            continue
        if skill_name not in index:
            index[skill_name] = entry
    return index

def _discover_single_skill_aggregate_files(root: Path, domain: str) -> tuple[dict[str, Path], list[dict[str, str]]]:
    discovered: dict[str, Path] = {}
    malformed_candidates: list[dict[str, str]] = []
    for candidate in sorted(root.glob(get_single_skill_aggregate_glob(domain))):
        try:
            aggregate_data = load_aggregate_task_results(candidate)
        except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
            malformed_candidates.append({'path': str(candidate), 'error_type': type(exc).__name__, 'error': str(exc)})
            continue
        for skill_name in aggregate_data.get('skill_names', []):
            normalized = normalize_skill_name(skill_name)
            discovered.setdefault(normalized, candidate)
    selected: dict[str, Path] = {}
    for skill_name in sorted(discovered):
        selected[skill_name] = select_single_skill_aggregate(root, skill_name, domain=domain)
    return (selected, malformed_candidates)

def _is_task_loader_unavailable_error(exc: BaseException, domain: str) -> bool:
    if not isinstance(exc, ModuleNotFoundError):
        return False
    expected_names = {'tau2', 'tau2.run', f'tau2.domains.{domain}', f'tau2.domains.{domain}.environment'}
    missing_name = getattr(exc, 'name', None)
    if missing_name in expected_names:
        return True
    message = str(exc)
    return any((f"No module named '{name}'" in message for name in expected_names))

def _read_task_id(task: Any) -> str:
    if hasattr(task, 'id'):
        return str(task.id)
    if hasattr(task, 'task_id'):
        return str(task.task_id)
    raise ValueError('Task object is missing id/task_id')

def _load_tasks_from_local_files(task_split_name: str, local_tasks_json_path: Path, local_split_tasks_json_path: Path) -> list[Any]:
    all_tasks = json.loads(local_tasks_json_path.read_text(encoding='utf-8'))
    split_payload = json.loads(local_split_tasks_json_path.read_text(encoding='utf-8'))
    if not isinstance(all_tasks, list):
        raise ValueError(f'Expected task list in {local_tasks_json_path}')
    if not isinstance(split_payload, dict):
        raise ValueError(f'Expected split mapping in {local_split_tasks_json_path}')
    split_ids_raw = split_payload.get(task_split_name, [])
    if not isinstance(split_ids_raw, list):
        raise ValueError(f"Expected list for split '{task_split_name}'")
    split_ids = {str(task_id) for task_id in split_ids_raw}
    tasks_by_id: dict[str, Any] = {}
    for row in all_tasks:
        if not isinstance(row, dict):
            continue
        task_id = str(row.get('id', ''))
        if task_id and task_id in split_ids:
            tasks_by_id[task_id] = row
    selected_rows = [tasks_by_id[task_id] for task_id in sorted(split_ids, key=_sort_task_id)]
    tasks: list[Any] = []
    for row in selected_rows:
        description = row.get('description') or {}
        user_scenario = row.get('user_scenario') or {}
        task_obj = SimpleNamespace(id=str(row.get('id')), description=SimpleNamespace(purpose=description.get('purpose'), relevant_policies=description.get('relevant_policies'), notes=description.get('notes')), user_scenario=SimpleNamespace(persona=user_scenario.get('persona'), instructions=user_scenario.get('instructions')))
        tasks.append(task_obj)
    return tasks

def _load_tasks_for_export(domain: str, task_split_name: str, local_tasks_json_path: Path, local_split_tasks_json_path: Path) -> tuple[list[Any], dict[str, str]]:
    try:
        if domain == 'airline':
            tasks = load_airline_tasks(task_split_name=task_split_name)
        else:
            tasks = load_domain_tasks(domain, task_split_name=task_split_name)
        return (tasks, {'type': 'tau2_loader', 'domain': domain, 'task_split_name': task_split_name})
    except Exception as exc:
        if not _is_task_loader_unavailable_error(exc, domain):
            raise
        tasks = _load_tasks_from_local_files(task_split_name=task_split_name, local_tasks_json_path=local_tasks_json_path, local_split_tasks_json_path=local_split_tasks_json_path)
        return (tasks, {'type': 'local_tasks_json_fallback', 'domain': domain, 'task_split_name': task_split_name, 'tasks_json': str(local_tasks_json_path), 'split_tasks_json': str(local_split_tasks_json_path)})

def _validate_task_parity(tasks: list[Any], baseline_task_ids: set[str]) -> None:
    loaded_task_ids = {_read_task_id(task) for task in tasks}
    missing_task_ids = sorted(baseline_task_ids - loaded_task_ids, key=_sort_task_id)
    extra_task_ids = sorted(loaded_task_ids - baseline_task_ids, key=_sort_task_id)
    if not missing_task_ids and (not extra_task_ids):
        return
    message_parts: list[str] = []
    if missing_task_ids:
        message_parts.append(f'Missing task_ids in loaded tasks: {missing_task_ids}')
    if extra_task_ids:
        message_parts.append(f'Extra task_ids in loaded tasks: {extra_task_ids}')
    raise ValueError('; '.join(message_parts))

def _resolve_skill_metadata(skill_name: str, ranked_entry_index: dict[str, dict[str, Any]], parsed_skill_dir: Path, ranked_skills_path: Path) -> dict[str, Any]:
    metadata: dict[str, str]
    try:
        metadata = extract_skill_metadata(skill_name=skill_name, parsed_skill_dir=parsed_skill_dir, ranked_skills_path=ranked_skills_path)
    except ValueError:
        entry = ranked_entry_index.get(skill_name, {})
        fields: dict[str, str] = {}
        source_code: str = ''
        ranked_file = entry.get('file')
        if isinstance(ranked_file, str):
            candidate = parsed_skill_dir / ranked_file
            if candidate.exists():
                source_code = candidate.read_text(encoding='utf-8')
                fields = _parse_skill_fields_from_source(source_code)
        if not fields and isinstance(entry.get('code'), str):
            source_code = str(entry['code'])
            fields = _parse_skill_fields_from_source(source_code)
        func_parts = extract_skill_function_parts(source_code) if source_code else {'docstring': '', 'body': ''}
        metadata = {'skill_name': normalize_skill_name(fields.get('SKILL_NAME', skill_name)), 'skill_domain': fields.get('SKILL_DOMAIN', ''), 'skill_description': fields.get('SKILL_DESCRIPTION', ''), 'prompt_guidance': fields.get('PROMPT_GUIDANCE', ''), 'target_pattern': fields.get('TARGET_PATTERN', ''), 'docstring': func_parts['docstring'], 'body': func_parts['body']}
    ranked_entry = ranked_entry_index.get(skill_name, {})
    return {**metadata, 'frequency': int(ranked_entry.get('frequency', 0) or 0), 'cluster_size': int(ranked_entry.get('cluster_size', 0) or 0), 'cluster_id': int(ranked_entry.get('cluster_id', -1) or -1), 'source_file': str(ranked_entry.get('file', ''))}

def _write_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding='utf-8')

def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    lines = [json.dumps(row, sort_keys=True) for row in rows]
    path.write_text('\n'.join(lines) + ('\n' if lines else ''), encoding='utf-8')

def export_dataset(domain: str, output_dir: str | Path, baseline_and_skill_root: str | Path=DEFAULT_BASELINE_AND_SKILL_ROOT, ranked_skills_path: str | Path | None=None, parsed_skill_dir: str | Path | None=None, local_tasks_json_path: str | Path | None=None, local_split_tasks_json_path: str | Path | None=None, task_split_name: str | None=None, fields_mode: str=DEFAULT_SKILL_FIELDS_MODE) -> dict[str, object]:
    if fields_mode not in SKILL_FIELDS_BY_MODE:
        supported = ', '.join(sorted(SKILL_FIELDS_BY_MODE))
        raise ValueError(f'Unsupported fields_mode={fields_mode!r}; expected one of: {supported}')
    spec = get_domain_spec(domain)
    export_identity = resolve_export_identity(spec.domain, task_split_name=task_split_name)
    output_dir = Path(output_dir)
    source_root = Path(baseline_and_skill_root)
    ranked_skills_path = Path(ranked_skills_path or get_default_ranked_skills_path(spec.domain))
    parsed_skill_dir = Path(parsed_skill_dir or get_default_parsed_skill_dir(spec.domain))
    local_tasks_json_path = Path(local_tasks_json_path or get_default_local_tasks_json_path(spec.domain))
    local_split_tasks_json_path = Path(local_split_tasks_json_path or get_default_local_split_tasks_json_path(spec.domain))
    resolved_task_split_name = export_identity['task_split_name']
    output_dir.mkdir(parents=True, exist_ok=True)
    baseline_path = select_baseline_aggregate(source_root, domain=spec)
    baseline_data = load_aggregate_task_results(baseline_path)
    ranked_entries = _load_ranked_entries(ranked_skills_path)
    ranked_entry_index = _ranked_entry_index(ranked_entries)
    (single_skill_files, malformed_single_skill_files) = _discover_single_skill_aggregate_files(source_root, domain=spec.domain)
    (tasks, task_source) = _load_tasks_for_export(domain=spec.domain, task_split_name=resolved_task_split_name, local_tasks_json_path=local_tasks_json_path, local_split_tasks_json_path=local_split_tasks_json_path)
    baseline_task_ids = set(baseline_data['results_by_task_id'].keys())
    _validate_task_parity(tasks, baseline_task_ids)
    task_text_by_id = {_read_task_id(task): build_task_text(task, domain=spec) for task in tasks}
    task_rows: list[dict[str, object]] = []
    baseline_results = baseline_data['results_by_task_id']
    for task_id in sorted(baseline_results, key=_sort_task_id):
        baseline_row = baseline_results[task_id]
        task_rows.append({'task_id': task_id, 'task_text': task_text_by_id[task_id], 'k_base': int(baseline_row['pass_count']), 'n_base': int(baseline_row['num_trials'])})
    skill_rows: list[dict[str, object]] = []
    labels_rows: list[dict[str, object]] = []
    single_skill_file_map: dict[str, str] = {}
    baseline_task_ids = {str(task_row['task_id']) for task_row in task_rows}
    for skill_name in sorted(single_skill_files):
        selected_skill_path = select_single_skill_aggregate(source_root, skill_name, domain=spec)
        single_skill_file_map[skill_name] = str(selected_skill_path)
        skill_data = load_aggregate_task_results(selected_skill_path)
        metadata = _resolve_skill_metadata(skill_name=skill_name, ranked_entry_index=ranked_entry_index, parsed_skill_dir=parsed_skill_dir, ranked_skills_path=ranked_skills_path)
        skill_rows.append({'skill_name': skill_name, 'skill_text': build_skill_text(metadata, fields_mode=fields_mode), 'frequency': int(metadata['frequency']), 'cluster_size': int(metadata['cluster_size']), 'cluster_id': int(metadata['cluster_id']), 'source_file': str(metadata['source_file'])})
        skill_results = skill_data['results_by_task_id']
        skill_task_ids = set(skill_results.keys())
        missing_task_ids = sorted(baseline_task_ids - skill_task_ids, key=_sort_task_id)
        extra_task_ids = sorted(skill_task_ids - baseline_task_ids, key=_sort_task_id)
        if missing_task_ids or extra_task_ids:
            message_parts: list[str] = []
            if missing_task_ids:
                message_parts.append(f'Missing task_ids in single-skill aggregate for {skill_name}: {missing_task_ids}')
            if extra_task_ids:
                message_parts.append(f'Extra task_ids in single-skill aggregate for {skill_name}: {extra_task_ids}')
            raise ValueError('; '.join(message_parts))
        for task_row in task_rows:
            task_id = str(task_row['task_id'])
            skill_task = skill_results[task_id]
            labels_rows.append({'task_id': task_id, 'skill_name': skill_name, 'k_skill': int(skill_task['pass_count']), 'n_skill': int(skill_task['num_trials'])})
    manifest = build_manifest(spec)
    manifest['task_split_name'] = export_identity['task_split_name']
    manifest['domain_output_name'] = export_identity['output_name']
    manifest['fields_mode'] = fields_mode
    manifest['skill_text_fields'] = list(SKILL_FIELDS_BY_MODE[fields_mode])
    manifest['source_runs'] = {'baseline_aggregate_file': str(baseline_path), 'single_skill_aggregate_root': str(source_root), 'single_skill_aggregate_files': single_skill_file_map, 'malformed_single_skill_file_count': len(malformed_single_skill_files), 'malformed_single_skill_files': malformed_single_skill_files, 'ranked_skills_path': str(ranked_skills_path), 'parsed_skill_dir': str(parsed_skill_dir), 'task_source': task_source}
    validate_required_manifest_keys(manifest)
    _write_json(output_dir / 'manifest.json', manifest)
    _write_jsonl(output_dir / 'tasks.jsonl', task_rows)
    _write_jsonl(output_dir / 'skills.jsonl', skill_rows)
    _write_jsonl(output_dir / 'labels.jsonl', labels_rows)
    return manifest

def export_airline_dataset(output_dir: str | Path, baseline_and_skill_root: str | Path=DEFAULT_BASELINE_AND_SKILL_ROOT, ranked_skills_path: str | Path=DEFAULT_RANKED_SKILLS_PATH, parsed_skill_dir: str | Path=DEFAULT_PARSED_SKILL_DIR, local_tasks_json_path: str | Path=DEFAULT_LOCAL_TASKS_JSON_PATH, local_split_tasks_json_path: str | Path=DEFAULT_LOCAL_SPLIT_TASKS_JSON_PATH, task_split_name: str='base', fields_mode: str=DEFAULT_SKILL_FIELDS_MODE) -> dict[str, object]:
    return export_dataset(domain='airline', output_dir=output_dir, baseline_and_skill_root=baseline_and_skill_root, ranked_skills_path=ranked_skills_path, parsed_skill_dir=parsed_skill_dir, local_tasks_json_path=local_tasks_json_path, local_split_tasks_json_path=local_split_tasks_json_path, task_split_name=task_split_name, fields_mode=fields_mode)

def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description='Export canonical selector dataset')
    parser.add_argument('--domain', required=True)
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--baseline-and-skill-root', default=str(DEFAULT_BASELINE_AND_SKILL_ROOT))
    parser.add_argument('--ranked-skills-path')
    parser.add_argument('--parsed-skill-dir')
    parser.add_argument('--task-split-name')
    parser.add_argument('--local-tasks-json-path')
    parser.add_argument('--local-split-tasks-json-path', default=None)
    parser.add_argument('--fields-mode', default=DEFAULT_SKILL_FIELDS_MODE, choices=sorted(SKILL_FIELDS_BY_MODE), help="Which fields to pack into skill_text: '3f' = description+guidance+target_pattern, '6f' = 3f + skill_name+docstring+body")
    return parser

def main() -> None:
    parser = _build_arg_parser()
    args = parser.parse_args()
    manifest = export_dataset(domain=args.domain, output_dir=args.output_dir, baseline_and_skill_root=args.baseline_and_skill_root, ranked_skills_path=args.ranked_skills_path, parsed_skill_dir=args.parsed_skill_dir, task_split_name=args.task_split_name, local_tasks_json_path=args.local_tasks_json_path, local_split_tasks_json_path=args.local_split_tasks_json_path, fields_mode=args.fields_mode)
    print(json.dumps({'output_dir': args.output_dir, 'schema_version': manifest['schema_version'], 'fields_mode': manifest.get('fields_mode')}))
if __name__ == '__main__':
    main()
