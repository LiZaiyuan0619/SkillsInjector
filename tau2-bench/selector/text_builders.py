import json
import sys
from collections.abc import Mapping, Sequence
from importlib import import_module
from pathlib import Path
from typing import Any
from temp.selector.common import CONTRACT_DOMAIN, resolve_domain_spec, resolve_task_split_name
from temp.selector.domain_registry import DomainSpec
PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / 'src'
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

def load_domain_tasks(domain: str | DomainSpec=CONTRACT_DOMAIN, task_split_name: str | None=None) -> list[Any]:
    spec = resolve_domain_spec(domain)
    resolved_task_split_name = resolve_task_split_name(spec, task_split_name)
    try:
        get_tasks = import_module('tau2.run').get_tasks
        return get_tasks(task_set_name=spec.domain, task_split_name=resolved_task_split_name)
    except ModuleNotFoundError as exc:
        missing_name = getattr(exc, 'name', None)
        if missing_name not in {'tau2', 'tau2.run'} and 'tau2.run' not in str(exc):
            raise
        domain_get_tasks = import_module(f'tau2.domains.{spec.domain}.environment').get_tasks
        return domain_get_tasks(task_split_name=resolved_task_split_name)

def load_airline_tasks(task_split_name: str | None=None) -> list[Any]:
    return load_domain_tasks('airline', task_split_name=resolve_task_split_name('airline', task_split_name))

def load_retail_tasks(task_split_name: str | None=None) -> list[Any]:
    return load_domain_tasks('retail', task_split_name=resolve_task_split_name('retail', task_split_name))

def load_telecom_tasks(task_split_name: str | None=None) -> list[Any]:
    return load_domain_tasks('telecom', task_split_name=resolve_task_split_name('telecom', task_split_name))

def _normalize_for_json(value: Any) -> Any:
    if value is None:
        return None
    if hasattr(value, 'model_dump') and callable(value.model_dump):
        return _normalize_for_json(value.model_dump(mode='json'))
    if isinstance(value, Mapping):
        return {str(key): _normalize_for_json(value[key]) for key in sorted(value)}
    if isinstance(value, Sequence) and (not isinstance(value, (str, bytes, bytearray))):
        return [_normalize_for_json(item) for item in value]
    return value

def _stringify_field(value: Any) -> str:
    if value is None:
        return ''
    if hasattr(value, 'model_dump') and callable(value.model_dump):
        value = value.model_dump(mode='json')
    if isinstance(value, Mapping):
        return json.dumps(_normalize_for_json(value), sort_keys=True).strip()
    if isinstance(value, Sequence) and (not isinstance(value, (str, bytes, bytearray))):
        return json.dumps(_normalize_for_json(value), sort_keys=True).strip()
    return str(value).strip()

def build_task_text(task: Any, domain: str | DomainSpec=CONTRACT_DOMAIN) -> str:
    resolve_domain_spec(domain)
    description = task.description
    fields = [('purpose', _stringify_field(description.purpose) if description else ''), ('relevant_policies', _stringify_field(description.relevant_policies) if description else ''), ('notes', _stringify_field(description.notes) if description else ''), ('persona', _stringify_field(task.user_scenario.persona)), ('instructions', _stringify_field(task.user_scenario.instructions))]
    return '\n'.join((f'{key}: {value}' for (key, value) in fields))
SKILL_FIELDS_BY_MODE: dict[str, tuple[str, ...]] = {'3f': ('skill_description', 'prompt_guidance', 'target_pattern'), '6f': ('skill_name', 'skill_description', 'prompt_guidance', 'target_pattern', 'docstring', 'body')}
DEFAULT_SKILL_FIELDS_MODE = '6f'

def build_skill_text(skill_metadata: dict[str, str], fields_mode: str=DEFAULT_SKILL_FIELDS_MODE) -> str:
    try:
        field_names = SKILL_FIELDS_BY_MODE[fields_mode]
    except KeyError as exc:
        supported = ', '.join(sorted(SKILL_FIELDS_BY_MODE))
        raise ValueError(f'Unsupported fields_mode={fields_mode!r}; expected one of: {supported}') from exc
    fields = [(name, _stringify_field(skill_metadata.get(name))) for name in field_names]
    return '\n'.join((f'{key}: {value}' for (key, value) in fields if value))
