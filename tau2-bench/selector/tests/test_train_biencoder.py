import json
import sys
from pathlib import Path
import pytest
import torch
PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
from temp.selector.train_biencoder import train_selector, validate_output_dir
from temp.selector.domain_registry import get_domain_spec

class DummyEncoder:

    def __init__(self, dimension: int=4):
        self.dimension = dimension

    def get_sentence_embedding_dimension(self) -> int:
        return self.dimension

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

def test_train_selector_writes_smoke_checkpoint_and_metrics(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    data_dir = tmp_path / 'data'
    output_dir = tmp_path / 'checkpoints'
    _build_tiny_dataset(data_dir)
    monkeypatch.setattr('temp.selector.train_biencoder.load_text_encoder', lambda encoder_name_or_path, device: DummyEncoder())
    metrics = train_selector(domain='airline', data_dir=data_dir, output_dir=output_dir, encoder_name_or_path='dummy', max_steps=2, device='cpu', hidden_dim=8, pairwise_weight=0.5, learning_rate=0.01)
    assert metrics['max_steps'] == 2
    assert metrics['num_pair_records'] == 6
    assert metrics['num_pairwise_records'] > 0
    assert (output_dir / 'model.pt').exists()
    assert (output_dir / 'config.json').exists()
    assert (output_dir / 'train_metrics.json').exists()

@pytest.mark.parametrize(('domain', 'output_name'), [('retail', 'retail'), ('telecom', 'telecom_base')])
def test_train_selector_supports_multidomain_final_checkpoint_layout(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, domain: str, output_name: str) -> None:
    data_dir = tmp_path / f'{domain}-data'
    output_dir = tmp_path / 'checkpoints' / output_name / 'final'
    output_dir.parent.mkdir(parents=True)
    _build_tiny_dataset(data_dir, domain=domain)
    monkeypatch.setattr('temp.selector.train_biencoder.load_text_encoder', lambda encoder_name_or_path, device: DummyEncoder())
    metrics = train_selector(domain=domain, data_dir=data_dir, output_dir=output_dir, encoder_name_or_path='dummy', max_steps=2, device='cpu', hidden_dim=8, pairwise_weight=0.5, learning_rate=0.01)
    assert metrics['domain'] == domain
    assert metrics['budget'] == 1
    config = json.loads((output_dir / 'config.json').read_text(encoding='utf-8'))
    assert config['domain'] == domain
    assert (output_dir / 'model.pt').exists()
    assert (output_dir / 'train_metrics.json').exists()

def test_validate_output_dir_creates_missing_parent(tmp_path: Path) -> None:
    output_dir = tmp_path / 'new-parent' / 'child'
    assert validate_output_dir(output_dir) == output_dir
    assert output_dir.exists()
    assert output_dir.parent.exists()

def test_train_selector_supports_cross_domain_checkpoint_layout_and_metadata(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    airline_data_dir = tmp_path / 'airline-data'
    retail_data_dir = tmp_path / 'retail-data'
    output_dir = tmp_path / 'checkpoints' / 'cross_domain' / 'all' / 'final'
    output_dir.parent.mkdir(parents=True)
    _build_tiny_dataset(airline_data_dir, domain='airline')
    _build_tiny_dataset(retail_data_dir, domain='retail')
    monkeypatch.setattr('temp.selector.train_biencoder.load_text_encoder', lambda encoder_name_or_path, device: DummyEncoder())
    metrics = train_selector(domain='cross_domain', data_dir=airline_data_dir, output_dir=output_dir, encoder_name_or_path='dummy', max_steps=2, device='cpu', hidden_dim=8, pairwise_weight=0.5, learning_rate=0.01, training_scope='cross_domain', source_data_dirs=[airline_data_dir, retail_data_dir])
    assert metrics['domain'] == 'cross_domain'
    assert metrics['training_scope'] == 'cross_domain'
    assert metrics['source_domains'] == ['airline', 'retail']
    assert metrics['num_pair_records'] == 12
    config = json.loads((output_dir / 'config.json').read_text(encoding='utf-8'))
    assert config['domain'] == 'cross_domain'
    assert config['training_scope'] == 'cross_domain'
    assert config['source_domains'] == ['airline', 'retail']

@pytest.mark.parametrize(('domain', 'output_name'), [('airline', 'airline'), ('retail', 'retail'), ('telecom', 'telecom_base')])
def test_validate_output_dir_enforces_domain_final_layout(tmp_path: Path, domain: str, output_name: str) -> None:
    wrong_output_dir = tmp_path / 'checkpoints' / 'wrong-domain' / 'final'
    wrong_output_dir.parent.mkdir(parents=True)
    with pytest.raises(ValueError, match=output_name):
        validate_output_dir(wrong_output_dir, domain=domain)
    right_output_dir = tmp_path / 'checkpoints' / output_name / 'final'
    right_output_dir.parent.mkdir(parents=True)
    assert validate_output_dir(right_output_dir, domain=domain) == right_output_dir

def test_validate_output_dir_supports_cross_domain_final_layout(tmp_path: Path) -> None:
    wrong_output_dir = tmp_path / 'checkpoints' / 'cross_domain' / 'wrong' / 'final'
    wrong_output_dir.parent.mkdir(parents=True)
    with pytest.raises(ValueError, match='cross_domain/all/final'):
        validate_output_dir(wrong_output_dir, domain='cross_domain', training_scope='cross_domain')
    right_output_dir = tmp_path / 'checkpoints' / 'cross_domain' / 'all' / 'final'
    right_output_dir.parent.mkdir(parents=True)
    assert validate_output_dir(right_output_dir, domain='cross_domain', training_scope='cross_domain') == right_output_dir
