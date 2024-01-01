import sys
from pathlib import Path
import pytest
PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
from temp.selector.domain_registry import get_baseline_config_path, get_domain_spec, get_supported_domains

def test_supported_domains_are_frozen_for_task1():
    assert get_supported_domains() == ('airline', 'retail', 'telecom')

@pytest.mark.parametrize(('domain', 'task_split', 'output_name', 'baseline_config_name'), [('airline', None, 'airline', 'baseline.yaml'), ('retail', None, 'retail', 'retail_baseline.yaml'), ('telecom', 'base', 'telecom_base', 'telecom_baseline.yaml')])
def test_domain_specs_expose_split_output_and_config(domain: str, task_split: str | None, output_name: str, baseline_config_name: str):
    spec = get_domain_spec(domain)
    assert spec.domain == domain
    assert spec.task_split == task_split
    assert spec.output_name == output_name
    assert spec.baseline_config == baseline_config_name
    assert get_baseline_config_path(domain).name == baseline_config_name
    assert get_baseline_config_path(domain).exists()

def test_get_domain_spec_rejects_unknown_domain():
    with pytest.raises(ValueError, match='Unsupported domain: lodging'):
        get_domain_spec('lodging')
