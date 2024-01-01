import json
import sys
from pathlib import Path
import pytest
import torch
PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
from temp.selector.freeze_selections import freeze_all_selections, freeze_baseline, freeze_global_utility, freeze_random, freeze_trained_selector
from temp.selector.train_biencoder import train_selector
from temp.selector.domain_registry import get_domain_spec

class DummyEncoder:

    def __init__(self, dimension: int=4):
        self.dimension = dimension

    def encode(self, texts: list[str], convert_to_tensor: bool=True, normalize_embeddings: bool=True, **_: object) -> torch.Tensor:
        rows = []
        for text in texts:
            base = float(sum((ord(char) for char in text)) % 17 + 1)
            rows.append([base, base / 2.0, base / 3.0, base / 4.0])
        tensor = torch.tensor(rows, dtype=torch.float32)
        if normalize_embeddings:
            tensor = torch.nn.functional.normalize(tensor, dim=-1)
        if convert_to_tensor:
            return tensor
        return tensor.numpy()

def _write_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload), encoding='utf-8')

def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.write_text('\n'.join((json.dumps(row, sort_keys=True) for row in rows)) + '\n', encoding='utf-8')

def _build_tiny_dataset(data_dir: Path, domain: str='airline') -> None:
    spec = get_domain_spec(domain)
    data_dir.mkdir(parents=True)
    _write_json(data_dir / 'manifest.json', {'schema_version': 'selector_contract.v1', 'domain': domain, 'budget': 1, 'split_policy': 'feasibility_full_matrix', 'task_split_name': spec.task_split or 'base', 'domain_output_name': spec.output_name, 'output_filenames': {'manifest': 'manifest.json', 'tasks': 'tasks.jsonl', 'skills': 'skills.jsonl', 'labels': 'labels.jsonl'}, 'label_definition': {'label_name': 'utility_label', 'label_type': 'float', 'value_range': [0.0, 1.0], 'semantic': 'Estimated single-skill pass probability p_skill(t, s).', 'source': 'offline single-skill outcomes'}})
    _write_jsonl(data_dir / 'tasks.jsonl', [{'task_id': 't1', 'task_text': 'task 1', 'k_base': 1, 'n_base': 2}, {'task_id': 't2', 'task_text': 'task 2', 'k_base': 0, 'n_base': 2}])
    _write_jsonl(data_dir / 'skills.jsonl', [{'skill_name': 'alpha', 'skill_text': 'alpha text'}, {'skill_name': 'beta', 'skill_text': 'beta text'}, {'skill_name': 'gamma', 'skill_text': 'gamma text'}])
    _write_jsonl(data_dir / 'labels.jsonl', [{'task_id': 't1', 'skill_name': 'alpha', 'k_skill': 2, 'n_skill': 2}, {'task_id': 't1', 'skill_name': 'beta', 'k_skill': 1, 'n_skill': 2}, {'task_id': 't1', 'skill_name': 'gamma', 'k_skill': 0, 'n_skill': 2}, {'task_id': 't2', 'skill_name': 'alpha', 'k_skill': 0, 'n_skill': 2}, {'task_id': 't2', 'skill_name': 'beta', 'k_skill': 2, 'n_skill': 2}, {'task_id': 't2', 'skill_name': 'gamma', 'k_skill': 1, 'n_skill': 2}])

def _train_tiny_checkpoint(data_dir: Path, checkpoint_dir: Path, monkeypatch: pytest.MonkeyPatch, domain: str='airline') -> Path:
    checkpoint_dir.parent.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr('temp.selector.train_biencoder.load_text_encoder', lambda encoder_name_or_path, device: DummyEncoder())
    monkeypatch.setattr('temp.selector.freeze_selections.load_text_encoder', lambda encoder_name_or_path, device: DummyEncoder())
    train_selector(domain=domain, data_dir=data_dir, output_dir=checkpoint_dir, encoder_name_or_path='dummy', max_steps=2, device='cpu', hidden_dim=8, pairwise_weight=0.5, learning_rate=0.01)
    return checkpoint_dir / 'model.pt'

def test_freeze_baseline_writes_null_skill_rows(tmp_path: Path) -> None:
    data_dir = tmp_path / 'data'
    _build_tiny_dataset(data_dir)
    rows = freeze_baseline(domain='airline', data_dir=data_dir)
    assert {row['task_id'] for row in rows} == {'t1', 't2'}
    assert {row['method'] for row in rows} == {'baseline'}
    assert all((row['selected_skill_name'] is None for row in rows))
    assert all((row['selection_source'] == 'baseline' for row in rows))
    assert all((row['selection_rank'] is None for row in rows))
    assert all((row['checkpoint_path'] is None for row in rows))

def test_freeze_random_is_deterministic_for_same_seed(tmp_path: Path) -> None:
    data_dir = tmp_path / 'data'
    _build_tiny_dataset(data_dir)
    rows_a = freeze_random(domain='airline', data_dir=data_dir, selection_seed=13)
    rows_b = freeze_random(domain='airline', data_dir=data_dir, selection_seed=13)
    assert rows_a == rows_b
    assert all((row['selection_seed'] == 13 for row in rows_a))
    assert all((row['selected_skill_name'] in {'alpha', 'beta', 'gamma'} for row in rows_a))

def test_freeze_random_supports_budget_and_candidate_pool_size(tmp_path: Path) -> None:
    data_dir = tmp_path / 'data'
    _build_tiny_dataset(data_dir)
    rows = freeze_random(domain='airline', data_dir=data_dir, selection_seed=13, budget=2, candidate_pool_size=2, training_scope='cross_domain')
    assert all((row['budget'] == 2 for row in rows))
    assert all((row['candidate_pool_size'] == 2 for row in rows))
    assert all((row['training_scope'] == 'cross_domain' for row in rows))
    assert all((len(row['selected_skill_names']) == 2 for row in rows))
    assert all((row['selected_skill_name'] == row['selected_skill_names'][0] for row in rows))

def test_freeze_global_utility_uses_same_top_skill_for_every_task(tmp_path: Path) -> None:
    data_dir = tmp_path / 'data'
    _build_tiny_dataset(data_dir)
    rows = freeze_global_utility(domain='airline', data_dir=data_dir)
    assert {row['task_id'] for row in rows} == {'t1', 't2'}
    assert {row['selected_skill_name'] for row in rows} == {'beta'}
    assert all((row['method'] == 'global_utility' for row in rows))
    assert all((row['selection_rank'] == 1 for row in rows))

def test_freeze_trained_selector_writes_one_row_per_task(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    data_dir = tmp_path / 'data'
    checkpoint_dir = tmp_path / 'checkpoints' / 'airline' / 'final'
    _build_tiny_dataset(data_dir)
    checkpoint_path = _train_tiny_checkpoint(data_dir, checkpoint_dir, monkeypatch)
    rows = freeze_trained_selector(domain='airline', data_dir=data_dir, checkpoint_path=checkpoint_path, device='cpu')
    assert {row['task_id'] for row in rows} == {'t1', 't2'}
    assert {row['method'] for row in rows} == {'trained_selector'}
    assert all((row['selection_rank'] == 1 for row in rows))
    assert all((row['selection_source'] == 'trained_selector' for row in rows))
    assert all((Path(str(row['checkpoint_path'])) == checkpoint_path for row in rows))

def test_freeze_all_selections_writes_method_files(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    data_dir = tmp_path / 'data'
    checkpoint_dir = tmp_path / 'checkpoints' / 'retail' / 'final'
    output_dir = tmp_path / 'selections' / 'retail'
    _build_tiny_dataset(data_dir, domain='retail')
    checkpoint_path = _train_tiny_checkpoint(data_dir, checkpoint_dir, monkeypatch, domain='retail')
    result = freeze_all_selections(domain='retail', data_dir=data_dir, checkpoint_path=checkpoint_path, output_dir=output_dir, selection_seed=13, device='cpu')
    assert result['domain'] == 'retail'
    assert sorted(result['methods']) == ['baseline', 'global_utility', 'random', 'trained_selector']
    assert (output_dir / 'baseline.jsonl').exists()
    assert (output_dir / 'random_seed13.jsonl').exists()
    assert (output_dir / 'global_utility.jsonl').exists()
    assert (output_dir / 'trained_selector.jsonl').exists()
    random_rows = [json.loads(line) for line in (output_dir / 'random_seed13.jsonl').read_text(encoding='utf-8').splitlines()]
    assert {row['domain'] for row in random_rows} == {'retail'}
    assert {row['selection_seed'] for row in random_rows} == {13}
