import argparse
import json
import sys
from pathlib import Path
from typing import Any
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
from temp.selector.common import resolve_domain_spec
from temp.selector.datasets import load_pair_records
from temp.selector.offline_eval import _encode_lookup, _load_exported_dataset, load_text_encoder
DEFAULT_ENCODER = '<MODEL_ROOT>/model/Qwen3-Embedding-0.6B'

def cosine_select(domain: str, data_dir: Path, encoder_name_or_path: str=DEFAULT_ENCODER, device: str='cuda') -> list[dict[str, Any]]:
    import torch
    spec = resolve_domain_spec(domain)
    (_manifest, tasks_by_id, skill_names, _labels) = _load_exported_dataset(domain=spec.domain, data_dir=data_dir, budget=1, validate_budget=False)
    pair_records = load_pair_records(data_dir)
    task_texts = {r.task_id: r.task_text for r in pair_records}
    skill_texts = {r.skill_name: r.skill_text for r in pair_records}
    encoder = load_text_encoder(encoder_name_or_path, device=device)
    task_embeddings = _encode_lookup(encoder, task_texts, device=device)
    skill_embeddings = _encode_lookup(encoder, skill_texts, device=device)
    skill_matrix = torch.stack([skill_embeddings[name] for name in skill_names])
    rows: list[dict[str, Any]] = []
    for task_id in sorted(tasks_by_id):
        task_vec = task_embeddings[task_id]
        sims = (skill_matrix @ task_vec).tolist()
        ranked_indices = sorted(range(len(skill_names)), key=lambda i: (-sims[i], skill_names[i]))
        top_idx = ranked_indices[0]
        selected_name = skill_names[top_idx]
        rows.append({'domain': spec.domain, 'task_id': task_id, 'method': 'trained_selector', 'budget': 1, 'candidate_pool_size': None, 'training_scope': 'single_domain', 'selected_skill_name': selected_name, 'selected_skill_names': [selected_name], 'selection_seed': None, 'selection_score': float(sims[top_idx]), 'selection_scores': [float(sims[top_idx])], 'selection_rank': 1, 'selection_source': 'cosine_frozen_encoder', 'checkpoint_path': None})
    return rows

def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text('\n'.join((json.dumps(row, sort_keys=True) for row in rows)) + ('\n' if rows else ''), encoding='utf-8')

def main() -> None:
    parser = argparse.ArgumentParser(description='Cosine frozen-encoder baseline (top-1)')
    parser.add_argument('--domain', required=True)
    parser.add_argument('--data-dir', required=True)
    parser.add_argument('--output', required=True, help='Output selection JSONL path')
    parser.add_argument('--encoder', default=DEFAULT_ENCODER)
    parser.add_argument('--device', default='cuda')
    args = parser.parse_args()
    rows = cosine_select(args.domain, Path(args.data_dir), encoder_name_or_path=args.encoder, device=args.device)
    output_path = Path(args.output)
    _write_jsonl(output_path, rows)
    print(json.dumps({'domain': args.domain, 'num_rows': len(rows), 'output': str(output_path), 'encoder': args.encoder, 'selection_source': 'cosine_frozen_encoder', 'row_method_field': 'trained_selector'}))
if __name__ == '__main__':
    main()
