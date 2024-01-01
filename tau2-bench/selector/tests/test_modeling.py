import json
import sys
from pathlib import Path
import pytest
import torch
PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
from temp.selector.datasets import PairRecord, build_pair_batch, build_pairwise_records, load_pair_records
from temp.selector.modeling import SelectorScorer, validate_probability_counts

def _write_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload), encoding='utf-8')

def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.write_text('\n'.join((json.dumps(row, sort_keys=True) for row in rows)) + '\n', encoding='utf-8')

def _build_tiny_dataset(data_dir: Path) -> None:
    data_dir.mkdir(parents=True)
    _write_json(data_dir / 'manifest.json', {'schema_version': 'selector_contract.v1', 'domain': 'airline', 'budget': 1, 'split_policy': 'feasibility_full_matrix', 'output_filenames': {'manifest': 'manifest.json', 'tasks': 'tasks.jsonl', 'skills': 'skills.jsonl', 'labels': 'labels.jsonl'}, 'label_definition': {'label_name': 'utility_label', 'label_type': 'float', 'value_range': [0.0, 1.0], 'semantic': 'Estimated single-skill pass probability p_skill(t, s).', 'source': 'offline single-skill outcomes'}})
    _write_jsonl(data_dir / 'tasks.jsonl', [{'task_id': 't1', 'task_text': 'task 1', 'k_base': 1, 'n_base': 2}, {'task_id': 't2', 'task_text': 'task 2', 'k_base': 0, 'n_base': 2}])
    _write_jsonl(data_dir / 'skills.jsonl', [{'skill_name': 'alpha', 'skill_text': 'alpha text'}, {'skill_name': 'beta', 'skill_text': 'beta text'}, {'skill_name': 'gamma', 'skill_text': 'gamma text'}])
    _write_jsonl(data_dir / 'labels.jsonl', [{'task_id': 't1', 'skill_name': 'alpha', 'k_skill': 2, 'n_skill': 2}, {'task_id': 't1', 'skill_name': 'beta', 'k_skill': 1, 'n_skill': 2}, {'task_id': 't1', 'skill_name': 'gamma', 'k_skill': 0, 'n_skill': 2}, {'task_id': 't2', 'skill_name': 'alpha', 'k_skill': 0, 'n_skill': 2}, {'task_id': 't2', 'skill_name': 'beta', 'k_skill': 2, 'n_skill': 2}, {'task_id': 't2', 'skill_name': 'gamma', 'k_skill': 1, 'n_skill': 2}])

def test_load_pair_records_preserves_ids_counts_and_probabilities(tmp_path: Path) -> None:
    data_dir = tmp_path / 'data'
    _build_tiny_dataset(data_dir)
    records = load_pair_records(data_dir)
    assert len(records) == 6
    assert records[0].task_id == 't1'
    assert records[0].skill_name == 'alpha'
    assert records[0].k_skill == 2
    assert records[0].n_skill == 2
    assert records[0].p_base == pytest.approx(0.5)
    assert records[0].p_skill == pytest.approx(1.0)
    assert records[0].delta == pytest.approx(0.5)

def test_build_pair_batch_returns_tensorized_fields_with_stable_ids(tmp_path: Path) -> None:
    data_dir = tmp_path / 'data'
    _build_tiny_dataset(data_dir)
    records = load_pair_records(data_dir)[:2]
    task_embeddings = {'t1': torch.tensor([1.0, 0.0, 0.5]), 't2': torch.tensor([0.5, 0.5, 0.5])}
    skill_embeddings = {'alpha': torch.tensor([0.0, 1.0, 0.0]), 'beta': torch.tensor([0.0, 0.0, 1.0]), 'gamma': torch.tensor([1.0, 0.0, 0.0])}
    batch = build_pair_batch(records, task_embeddings=task_embeddings, skill_embeddings=skill_embeddings)
    assert batch['task_ids'] == ['t1', 't1']
    assert batch['skill_names'] == ['alpha', 'beta']
    assert tuple(batch['task_embeddings'].shape) == (2, 3)
    assert tuple(batch['skill_embeddings'].shape) == (2, 3)
    assert torch.allclose(batch['p_skill'], torch.tensor([1.0, 0.5]))

def test_build_pairwise_records_rejects_same_skill_duplicates_and_zero_contrast() -> None:
    records = [PairRecord(task_id='t1', skill_name='alpha', task_text='task', skill_text='alpha', k_base=1, n_base=2, k_skill=2, n_skill=2, p_base=0.5, p_skill=1.0, delta=0.5), PairRecord(task_id='t1', skill_name='beta', task_text='task', skill_text='beta', k_base=1, n_base=2, k_skill=1, n_skill=2, p_base=0.5, p_skill=0.5, delta=0.0), PairRecord(task_id='t2', skill_name='gamma', task_text='task2', skill_text='gamma', k_base=0, n_base=2, k_skill=1, n_skill=2, p_base=0.0, p_skill=0.5, delta=0.5), PairRecord(task_id='t2', skill_name='delta', task_text='task2', skill_text='delta', k_base=0, n_base=2, k_skill=1, n_skill=2, p_base=0.0, p_skill=0.5, delta=0.5)]
    pairwise = build_pairwise_records(records)
    assert len(pairwise) == 1
    assert pairwise[0].task_id == 't1'
    assert pairwise[0].positive_skill_name == 'alpha'
    assert pairwise[0].negative_skill_name == 'beta'

def test_validate_probability_counts_rejects_zero_denominator() -> None:
    with pytest.raises(ValueError, match='n_skill must be positive'):
        validate_probability_counts(k_skill=0, n_skill=0)

def test_selector_scorer_returns_bounded_probabilities() -> None:
    model = SelectorScorer(embedding_dim=3, hidden_dim=5)
    task_embeddings = torch.tensor([[1.0, 0.0, 0.5], [0.5, 0.5, 0.5]])
    skill_embeddings = torch.tensor([[0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
    output = model(task_embeddings=task_embeddings, skill_embeddings=skill_embeddings)
    assert tuple(output['logits'].shape) == (2,)
    assert tuple(output['probabilities'].shape) == (2,)
    assert torch.all(output['probabilities'] >= 0.0)
    assert torch.all(output['probabilities'] <= 1.0)
