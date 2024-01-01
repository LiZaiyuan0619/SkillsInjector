import os
import subprocess
import sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
SCRIPTS_DIR = PROJECT_ROOT / 'scripts' / 'selector'

def _run_script(script_name: str, extra_env: dict[str, str]) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    env.update(extra_env)
    return subprocess.run(['bash', str(SCRIPTS_DIR / script_name)], cwd=PROJECT_ROOT, env=env, text=True, capture_output=True, check=False)

def test_run_selector_matrix_freezes_all_domains_before_eval() -> None:
    result = _run_script('run_selector_matrix.sh', {'DRY_RUN': '1', 'EXPERIMENT_TAG': 'cross_domain_b1', 'TRAINING_SCOPE': 'cross_domain', 'BUDGET': '1', 'CHECKPOINT_SCOPE': 'shared', 'CHECKPOINT_PATH': 'temp/selector/checkpoints/cross_domain/all/final/model.pt', 'POOL_SIZES': '1 2', 'DOMAINS': 'airline telecom', 'EVAL_METHODS': 'baseline trained_selector', 'RUN_SUMMARY': '0'})
    assert result.returncode == 0, result.stderr
    lines = [line for line in result.stdout.splitlines() if line.startswith('DRY_RUN:')]
    assert len(lines) == 12
    assert 'freeze_selections.py' in lines[0]
    assert 'freeze_selections.py' in lines[1]
    assert 'freeze_selections.py' in lines[2]
    assert 'freeze_selections.py' in lines[3]
    assert all(('run_online_eval.py' in line for line in lines[4:]))

def test_run_selector_matrix_uses_random_seed_selection_filename() -> None:
    result = _run_script('run_selector_matrix.sh', {'DRY_RUN': '1', 'EXPERIMENT_TAG': 'cross_domain_b1', 'TRAINING_SCOPE': 'cross_domain', 'BUDGET': '1', 'CHECKPOINT_SCOPE': 'shared', 'CHECKPOINT_PATH': 'temp/selector/checkpoints/cross_domain/all/final/model.pt', 'POOL_SIZES': '8', 'DOMAINS': 'retail', 'EVAL_METHODS': 'random', 'RUN_SUMMARY': '0'})
    assert result.returncode == 0, result.stderr
    assert 'random_seed13.jsonl' in result.stdout
    assert '/random.jsonl' not in result.stdout

def test_machine_c_wrapper_defaults_to_trained_selector_only() -> None:
    result = _run_script('machine_c_cross_domain_b4.sh', {'DRY_RUN': '1', 'POOL_SIZES': '4', 'DOMAINS': 'airline', 'RUN_SUMMARY': '0'})
    assert result.returncode == 0, result.stderr
    assert '--method trained_selector' in result.stdout
    assert '--method baseline' not in result.stdout
    assert '--method random' not in result.stdout
    assert '--method global_utility' not in result.stdout
