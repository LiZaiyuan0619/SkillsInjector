import json
import sys
from pathlib import Path
import pytest
import torch
PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
from temp.selector.offline_eval import run_trained_evaluation, validate_manifest_schema
from temp.selector.train_biencoder import train_selector
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

def test_run_trained_evaluation_writes_comparison_outputs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    data_dir = tmp_path / 'data'
    checkpoint_dir = tmp_path / 'checkpoint'
    output_dir = tmp_path / 'trained-results'
    _build_tiny_dataset(data_dir)
    monkeypatch.setattr('temp.selector.train_biencoder.load_text_encoder', lambda encoder_name_or_path, device: DummyEncoder())
    monkeypatch.setattr('temp.selector.offline_eval.load_text_encoder', lambda encoder_name_or_path, device: DummyEncoder())
    train_selector(domain='airline', data_dir=data_dir, output_dir=checkpoint_dir, encoder_name_or_path='dummy', max_steps=2, device='cpu', hidden_dim=8, pairwise_weight=0.5, learning_rate=0.01)
    payload = run_trained_evaluation(domain='airline', data_dir=data_dir, checkpoint_path=checkpoint_dir / 'model.pt', baseline_suite=['random', 'global_utility'], budget=1, output_dir=output_dir, seed=13)
    assert payload['trained_summary']['baseline_name'] == 'trained_selector'
    assert len(payload['baseline_summaries']) == 2
    assert (output_dir / 'trained.summary.json').exists()
    assert (output_dir / 'trained.predictions.jsonl').exists()
    assert (output_dir / 'comparison.json').exists()

@pytest.mark.parametrize(('domain', 'output_name'), [('retail', 'retail'), ('telecom', 'telecom_base')])
def test_run_trained_evaluation_supports_supported_non_airline_domains(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, domain: str, output_name: str) -> None:
    data_dir = tmp_path / f'{domain}-data'
    checkpoint_dir = tmp_path / 'checkpoints' / output_name / 'final'
    output_dir = tmp_path / 'results' / output_name / 'offline'
    checkpoint_dir.parent.mkdir(parents=True)
    _build_tiny_dataset(data_dir, domain=domain)
    monkeypatch.setattr('temp.selector.train_biencoder.load_text_encoder', lambda encoder_name_or_path, device: DummyEncoder())
    monkeypatch.setattr('temp.selector.offline_eval.load_text_encoder', lambda encoder_name_or_path, device: DummyEncoder())
    train_selector(domain=domain, data_dir=data_dir, output_dir=checkpoint_dir, encoder_name_or_path='dummy', max_steps=2, device='cpu', hidden_dim=8, pairwise_weight=0.5, learning_rate=0.01)
    payload = run_trained_evaluation(domain=domain, data_dir=data_dir, checkpoint_path=checkpoint_dir / 'model.pt', baseline_suite=['global_utility'], budget=1, output_dir=output_dir, seed=13)
    assert payload['domain'] == domain
    assert payload['trained_summary']['baseline_name'] == 'trained_selector'
    assert len(payload['baseline_summaries']) == 1
    assert (output_dir / 'trained.summary.json').exists()
    assert (output_dir / 'comparison.json').exists()

def test_validate_manifest_schema_rejects_bad_version() -> None:
    with pytest.raises(ValueError, match='unsupported manifest schema_version'):
        validate_manifest_schema({'schema_version': 'bad-version'}, expected_domain='airline', budget=1)
