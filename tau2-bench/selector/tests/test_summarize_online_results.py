import json
import sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
from temp.selector.summarize_online_results import summarize_online_results

def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding='utf-8')

def test_summarize_online_results_writes_markdown_and_json(tmp_path: Path) -> None:
    results_root = tmp_path / 'results'
    output_dir = tmp_path / 'benchmark_summary'
    _write_json(results_root / 'airline' / 'online' / 'single_domain' / 'b1' / 'm8' / 'baseline' / 'aggregate.json', {'domain': 'airline', 'method': 'baseline', 'training_scope': 'single_domain', 'budget': 1, 'candidate_pool_size': 8, 'num_trials': 5, 'max_concurrency': 1, 'mean_pass_rate': 0.2, 'std_pass_rate': 0.02, 'trial_pass_rates': [0.2] * 5, 'selection_file': '/tmp/baseline.jsonl', 'selection_seed': None, 'checkpoint_path': None})
    _write_json(results_root / 'airline' / 'online' / 'cross_domain' / 'b4' / 'm8' / 'trained_selector' / 'aggregate.json', {'domain': 'airline', 'method': 'trained_selector', 'training_scope': 'cross_domain', 'budget': 4, 'candidate_pool_size': 8, 'num_trials': 5, 'max_concurrency': 3, 'mean_pass_rate': 0.44, 'std_pass_rate': 0.03, 'trial_pass_rates': [0.4, 0.4, 0.4, 0.6, 0.4], 'selection_file': '/tmp/trained_selector.jsonl', 'selection_seed': None, 'checkpoint_path': '/tmp/model.pt'})
    _write_json(results_root / 'retail' / 'online' / 'cross_domain' / 'b1' / 'm4' / 'global_utility' / 'aggregate.json', {'domain': 'retail', 'method': 'global_utility', 'training_scope': 'cross_domain', 'budget': 1, 'candidate_pool_size': 4, 'num_trials': 5, 'max_concurrency': 2, 'mean_pass_rate': 0.31, 'std_pass_rate': 0.01, 'trial_pass_rates': [0.3, 0.3, 0.3, 0.3, 0.35], 'selection_file': '/tmp/global.jsonl', 'selection_seed': None, 'checkpoint_path': None})
    payload = summarize_online_results(results_root=results_root, output_dir=output_dir)
    assert payload['num_results'] == 3
    assert (output_dir / 'benchmark_summary.json').exists()
    assert (output_dir / 'summary.md').exists()
    benchmark_json = json.loads((output_dir / 'benchmark_summary.json').read_text(encoding='utf-8'))
    assert benchmark_json['domains'] == ['airline', 'retail']
    assert benchmark_json['methods'] == ['baseline', 'global_utility', 'trained_selector']
    assert benchmark_json['training_scopes'] == ['cross_domain', 'single_domain']
    assert benchmark_json['budgets'] == [1, 4]
    assert benchmark_json['candidate_pool_sizes'] == [4, 8]
    markdown = (output_dir / 'summary.md').read_text(encoding='utf-8')
    assert '| Domain | Training Scope | Budget | Candidate Pool Size | Method | Trials | Mean Pass Rate | Std Pass Rate | Max Concurrency |' in markdown
    assert '| airline | single_domain | 1 | 8 | baseline | 5 | 0.2000 | 0.0200 | 1 |' in markdown
    assert '| airline | cross_domain | 4 | 8 | trained_selector | 5 | 0.4400 | 0.0300 | 3 |' in markdown
    assert '| retail | cross_domain | 1 | 4 | global_utility | 5 | 0.3100 | 0.0100 | 2 |' in markdown
