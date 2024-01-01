from dataclasses import dataclass
from pathlib import Path

@dataclass(frozen=True)
class DomainSpec:
    domain: str
    task_split: str | None
    output_name: str
    baseline_config: str
PROJECT_ROOT = Path(__file__).resolve().parents[2]
_CONFIGS_DIR = Path(__file__).resolve().parents[1] / 'configs'
_GENERATED_SKILLS_DIR = PROJECT_ROOT / 'temp/generated_skills'
_LOCAL_TASKS_DIR = PROJECT_ROOT / 'data/tau2/domains'
_DOMAIN_SPECS = {'airline': DomainSpec(domain='airline', task_split=None, output_name='airline', baseline_config='baseline.yaml'), 'retail': DomainSpec(domain='retail', task_split=None, output_name='retail', baseline_config='retail_baseline.yaml'), 'telecom': DomainSpec(domain='telecom', task_split='base', output_name='telecom_base', baseline_config='telecom_baseline.yaml')}

def get_supported_domains() -> tuple[str, ...]:
    return tuple(_DOMAIN_SPECS)

def get_domain_spec(domain: str) -> DomainSpec:
    try:
        return _DOMAIN_SPECS[domain]
    except KeyError as exc:
        supported = ', '.join(get_supported_domains())
        raise ValueError(f'Unsupported domain: {domain}. Supported domains: {supported}') from exc

def get_baseline_config_path(domain: str) -> Path:
    return _CONFIGS_DIR / get_domain_spec(domain).baseline_config

def get_export_task_split_name(domain: str) -> str:
    spec = get_domain_spec(domain)
    return spec.task_split or 'base'

def get_default_ranked_skills_path(domain: str) -> Path:
    spec = get_domain_spec(domain)
    return _GENERATED_SKILLS_DIR / spec.domain / 'deduplicated_090' / 'skills_ranked.json'

def get_default_parsed_skill_dir(domain: str) -> Path:
    spec = get_domain_spec(domain)
    return _GENERATED_SKILLS_DIR / spec.domain / 'parsed'

def get_default_local_tasks_json_path(domain: str) -> Path:
    spec = get_domain_spec(domain)
    return _LOCAL_TASKS_DIR / spec.domain / 'tasks.json'

def get_default_local_split_tasks_json_path(domain: str) -> Path:
    spec = get_domain_spec(domain)
    return _LOCAL_TASKS_DIR / spec.domain / 'split_tasks.json'

def resolve_export_identity(domain: str, task_split_name: str | None=None) -> dict[str, str]:
    spec = get_domain_spec(domain)
    resolved_task_split_name = task_split_name or get_export_task_split_name(domain)
    if spec.task_split is not None and resolved_task_split_name != spec.task_split:
        raise ValueError(f"Exporter v1 only supports domain '{spec.domain}' split '{spec.task_split}', got '{resolved_task_split_name}'")
    return {'domain': spec.domain, 'task_split_name': resolved_task_split_name, 'output_name': spec.output_name}
