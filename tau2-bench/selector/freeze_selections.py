import argparse
import json
import sys
from pathlib import Path
from typing import Any
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
from temp.selector.common import BUDGET, resolve_domain_spec
from temp.selector.datasets import build_pair_batch, load_pair_records
from temp.selector.modeling import load_selector_checkpoint
from temp.selector.offline_eval import _encode_lookup, _load_exported_dataset, _rank_global_utility, _rank_random, load_text_encoder

def _build_selection_row(domain: str, task_id: str, method: str, selected_skill_name: str | None, selected_skill_names: list[str], selection_seed: int | None, selection_score: float | None, selection_scores: list[float] | None, selection_rank: int | None, selection_source: str, checkpoint_path: str | None, budget: int, candidate_pool_size: int | None, training_scope: str) -> dict[str, Any]:
    return {'domain': domain, 'task_id': task_id, 'method': method, 'budget': budget, 'candidate_pool_size': candidate_pool_size, 'training_scope': training_scope, 'selected_skill_name': selected_skill_name, 'selected_skill_names': selected_skill_names, 'selection_seed': selection_seed, 'selection_score': selection_score, 'selection_scores': selection_scores, 'selection_rank': selection_rank, 'selection_source': selection_source, 'checkpoint_path': checkpoint_path}

def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text('\n'.join((json.dumps(row, sort_keys=True) for row in rows)) + ('\n' if rows else ''), encoding='utf-8')

def _validate_selection_output_dir(output_dir: Path, domain: str) -> Path:
    spec = resolve_domain_spec(domain)
    if output_dir.name != spec.output_name:
        raise ValueError(f"Selection output for domain '{spec.domain}' must end with '{spec.output_name}'")
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir

def freeze_baseline(domain: str, data_dir: str | Path, budget: int=BUDGET, candidate_pool_size: int | None=None, training_scope: str='single_domain') -> list[dict[str, Any]]:
    spec = resolve_domain_spec(domain)
    (_manifest, tasks_by_id, _skill_names, _labels_by_task) = _load_exported_dataset(domain=spec.domain, data_dir=Path(data_dir), budget=budget, validate_budget=False)
    return [_build_selection_row(domain=spec.domain, task_id=task_id, method='baseline', selected_skill_name=None, selected_skill_names=[], selection_seed=None, selection_score=None, selection_scores=None, selection_rank=None, selection_source='baseline', checkpoint_path=None, budget=budget, candidate_pool_size=candidate_pool_size, training_scope=training_scope) for task_id in sorted(tasks_by_id)]

def freeze_random(domain: str, data_dir: str | Path, selection_seed: int=13, budget: int=BUDGET, candidate_pool_size: int | None=None, training_scope: str='single_domain') -> list[dict[str, Any]]:
    spec = resolve_domain_spec(domain)
    (_manifest, tasks_by_id, skill_names, _labels_by_task) = _load_exported_dataset(domain=spec.domain, data_dir=Path(data_dir), budget=budget, validate_budget=False)
    rows: list[dict[str, Any]] = []
    for task_id in sorted(tasks_by_id):
        ranked = _rank_random(task_id, skill_names, selection_seed)
        candidate_names = ranked[:candidate_pool_size] if candidate_pool_size is not None else ranked
        selected_names = candidate_names[:budget]
        rows.append(_build_selection_row(domain=spec.domain, task_id=task_id, method='random', selected_skill_name=selected_names[0] if selected_names else None, selected_skill_names=selected_names, selection_seed=selection_seed, selection_score=None, selection_scores=None, selection_rank=1, selection_source='random', checkpoint_path=None, budget=budget, candidate_pool_size=candidate_pool_size, training_scope=training_scope))
    return rows

def freeze_global_utility(domain: str, data_dir: str | Path, budget: int=BUDGET, candidate_pool_size: int | None=None, training_scope: str='single_domain') -> list[dict[str, Any]]:
    spec = resolve_domain_spec(domain)
    (_manifest, tasks_by_id, skill_names, labels_by_task) = _load_exported_dataset(domain=spec.domain, data_dir=Path(data_dir), budget=budget, validate_budget=False)
    utility_scores = {skill_name: sum((labels_by_task[current_task_id][skill_name]['p_skill'] for current_task_id in labels_by_task)) / float(len(labels_by_task)) for skill_name in skill_names}
    rows: list[dict[str, Any]] = []
    for task_id in sorted(tasks_by_id):
        ranked = _rank_global_utility(task_id, skill_names, labels_by_task)
        candidate_names = ranked[:candidate_pool_size] if candidate_pool_size is not None else ranked
        selected_names = candidate_names[:budget]
        selected_skill = selected_names[0]
        rows.append(_build_selection_row(domain=spec.domain, task_id=task_id, method='global_utility', selected_skill_name=selected_skill, selected_skill_names=selected_names, selection_seed=None, selection_score=float(utility_scores[selected_skill]), selection_scores=[float(utility_scores[name]) for name in selected_names], selection_rank=1, selection_source='global_utility', checkpoint_path=None, budget=budget, candidate_pool_size=candidate_pool_size, training_scope=training_scope))
    return rows

def freeze_trained_selector(domain: str, data_dir: str | Path, checkpoint_path: str | Path, device: str='cpu', budget: int=BUDGET, candidate_pool_size: int | None=None, training_scope: str='single_domain') -> list[dict[str, Any]]:
    import torch
    spec = resolve_domain_spec(domain)
    data_dir = Path(data_dir)
    checkpoint_path = Path(checkpoint_path)
    pair_records = load_pair_records(data_dir)
    (model, checkpoint_payload) = load_selector_checkpoint(str(checkpoint_path))
    encoder_name_or_path = str(checkpoint_payload['encoder_name_or_path'])
    hidden_dim = int(checkpoint_payload.get('extra_config', {}).get('hidden_dim', 128))
    checkpoint_domain = checkpoint_payload.get('extra_config', {}).get('domain')
    if checkpoint_domain is not None and str(checkpoint_domain) not in {spec.domain, 'cross_domain'}:
        raise ValueError('checkpoint domain does not match requested domain')
    if hasattr(model, 'mlp') and getattr(model.mlp[0], 'out_features', hidden_dim) != hidden_dim:
        (model, checkpoint_payload) = load_selector_checkpoint(str(checkpoint_path), hidden_dim=hidden_dim)
        encoder_name_or_path = str(checkpoint_payload['encoder_name_or_path'])
    encoder = load_text_encoder(encoder_name_or_path, device=device)
    task_texts = {record.task_id: record.task_text for record in pair_records}
    skill_texts = {record.skill_name: record.skill_text for record in pair_records}
    task_embeddings = _encode_lookup(encoder, task_texts, device=device)
    skill_embeddings = _encode_lookup(encoder, skill_texts, device=device)
    batch = build_pair_batch(pair_records, task_embeddings=task_embeddings, skill_embeddings=skill_embeddings)
    model.eval()
    with torch.no_grad():
        output = model(task_embeddings=batch['task_embeddings'], skill_embeddings=batch['skill_embeddings'])
    logits_by_key = {(task_id, skill_name): float(output['logits'][idx].cpu()) for (idx, (task_id, skill_name)) in enumerate(zip(batch['task_ids'], batch['skill_names'], strict=True))}
    (_manifest, tasks_by_id, skill_names, _labels_by_task) = _load_exported_dataset(domain=spec.domain, data_dir=data_dir, budget=budget, validate_budget=False)
    rows: list[dict[str, Any]] = []
    for task_id in sorted(tasks_by_id):
        ranked = sorted(skill_names, key=lambda skill_name: (-logits_by_key[task_id, skill_name], skill_name))
        candidate_names = ranked[:candidate_pool_size] if candidate_pool_size is not None else ranked
        selected_names = candidate_names[:budget]
        selected_skill = selected_names[0]
        rows.append(_build_selection_row(domain=spec.domain, task_id=task_id, method='trained_selector', selected_skill_name=selected_skill, selected_skill_names=selected_names, selection_seed=None, selection_score=float(logits_by_key[task_id, selected_skill]), selection_scores=[float(logits_by_key[task_id, name]) for name in selected_names], selection_rank=1, selection_source='trained_selector', checkpoint_path=str(checkpoint_path), budget=budget, candidate_pool_size=candidate_pool_size, training_scope=training_scope))
    return rows

def _compute_all_logits(domain: str, data_dir: str | Path, checkpoint_path: str | Path, device: str='cpu') -> tuple[dict[tuple[str, str], float], dict[str, Any], list[str]]:
    import torch
    spec = resolve_domain_spec(domain)
    data_dir = Path(data_dir)
    checkpoint_path = Path(checkpoint_path)
    pair_records = load_pair_records(data_dir)
    (model, checkpoint_payload) = load_selector_checkpoint(str(checkpoint_path))
    encoder_name_or_path = str(checkpoint_payload['encoder_name_or_path'])
    hidden_dim = int(checkpoint_payload.get('extra_config', {}).get('hidden_dim', 128))
    checkpoint_domain = checkpoint_payload.get('extra_config', {}).get('domain')
    if checkpoint_domain is not None and str(checkpoint_domain) not in {spec.domain, 'cross_domain'}:
        raise ValueError('checkpoint domain does not match requested domain')
    if hasattr(model, 'mlp') and getattr(model.mlp[0], 'out_features', hidden_dim) != hidden_dim:
        (model, checkpoint_payload) = load_selector_checkpoint(str(checkpoint_path), hidden_dim=hidden_dim)
        encoder_name_or_path = str(checkpoint_payload['encoder_name_or_path'])
    encoder = load_text_encoder(encoder_name_or_path, device=device)
    task_texts = {record.task_id: record.task_text for record in pair_records}
    skill_texts = {record.skill_name: record.skill_text for record in pair_records}
    task_embeddings = _encode_lookup(encoder, task_texts, device=device)
    skill_embeddings = _encode_lookup(encoder, skill_texts, device=device)
    batch = build_pair_batch(pair_records, task_embeddings=task_embeddings, skill_embeddings=skill_embeddings)
    model.eval()
    with torch.no_grad():
        output = model(task_embeddings=batch['task_embeddings'], skill_embeddings=batch['skill_embeddings'])
    logits_by_key = {(task_id, skill_name): float(output['logits'][idx].cpu()) for (idx, (task_id, skill_name)) in enumerate(zip(batch['task_ids'], batch['skill_names'], strict=True))}
    (_manifest, tasks_by_id, skill_names, _labels_by_task) = _load_exported_dataset(domain=spec.domain, data_dir=data_dir, budget=1, validate_budget=False)
    return (logits_by_key, tasks_by_id, skill_names)

def freeze_threshold_selector(domain: str, data_dir: str | Path, checkpoint_path: str | Path, threshold: float, device: str='cpu', max_budget: int=10, training_scope: str='single_domain') -> list[dict[str, Any]]:
    spec = resolve_domain_spec(domain)
    (logits_by_key, tasks_by_id, skill_names) = _compute_all_logits(domain=spec.domain, data_dir=data_dir, checkpoint_path=checkpoint_path, device=device)
    rows: list[dict[str, Any]] = []
    for task_id in sorted(tasks_by_id):
        ranked = sorted(skill_names, key=lambda sn: (-logits_by_key[task_id, sn], sn))
        selected_names = [sn for sn in ranked if logits_by_key[task_id, sn] > threshold][:max_budget]
        actual_budget = len(selected_names)
        rows.append(_build_selection_row(domain=spec.domain, task_id=task_id, method='trained_selector', selected_skill_name=selected_names[0] if selected_names else None, selected_skill_names=selected_names, selection_seed=None, selection_score=float(logits_by_key[task_id, selected_names[0]]) if selected_names else None, selection_scores=[float(logits_by_key[task_id, sn]) for sn in selected_names], selection_rank=1, selection_source='threshold_selector', checkpoint_path=str(checkpoint_path), budget=actual_budget, candidate_pool_size=None, training_scope=training_scope))
    return rows

def dump_score_matrix(domain: str, data_dir: str | Path, checkpoint_path: str | Path, output_path: str | Path, device: str='cpu') -> dict[str, Any]:
    spec = resolve_domain_spec(domain)
    (logits_by_key, tasks_by_id, skill_names) = _compute_all_logits(domain=spec.domain, data_dir=data_dir, checkpoint_path=checkpoint_path, device=device)
    matrix: dict[str, dict[str, float]] = {}
    for task_id in sorted(tasks_by_id):
        matrix[task_id] = {sn: float(logits_by_key[task_id, sn]) for sn in sorted(skill_names, key=lambda sn: -logits_by_key[task_id, sn])}
    payload = {'domain': spec.domain, 'num_tasks': len(tasks_by_id), 'num_skills': len(skill_names), 'checkpoint_path': str(checkpoint_path), 'matrix': matrix}
    Path(output_path).write_text(json.dumps(payload, indent=2), encoding='utf-8')
    return payload
FREEZE_MODES = ('all', 'baselines_only', 'trained_only')

def freeze_selections_by_mode(mode: str, domain: str, data_dir: str | Path, output_dir: str | Path, checkpoint_path: str | Path | None=None, selection_seed: int=13, device: str='cpu', budget: int=BUDGET, candidate_pool_size: int | None=None, training_scope: str='single_domain') -> dict[str, Any]:
    if mode not in FREEZE_MODES:
        raise ValueError(f"Unsupported mode={mode!r}; expected one of: {', '.join(FREEZE_MODES)}")
    if mode in ('all', 'trained_only') and (not checkpoint_path):
        raise ValueError(f'checkpoint_path is required for mode={mode!r}')
    spec = resolve_domain_spec(domain)
    output_dir = _validate_selection_output_dir(Path(output_dir), spec.domain)
    written: list[str] = []
    methods: list[str] = []
    if mode in ('all', 'baselines_only'):
        baseline_rows = freeze_baseline(domain=spec.domain, data_dir=data_dir, budget=budget, candidate_pool_size=candidate_pool_size, training_scope=training_scope)
        _write_jsonl(output_dir / 'baseline.jsonl', baseline_rows)
        written.append('baseline.jsonl')
        methods.append('baseline')
        random_rows = freeze_random(domain=spec.domain, data_dir=data_dir, selection_seed=selection_seed, budget=budget, candidate_pool_size=candidate_pool_size, training_scope=training_scope)
        _write_jsonl(output_dir / f'random_seed{selection_seed}.jsonl', random_rows)
        written.append(f'random_seed{selection_seed}.jsonl')
        methods.append('random')
        global_rows = freeze_global_utility(domain=spec.domain, data_dir=data_dir, budget=budget, candidate_pool_size=candidate_pool_size, training_scope=training_scope)
        _write_jsonl(output_dir / 'global_utility.jsonl', global_rows)
        written.append('global_utility.jsonl')
        methods.append('global_utility')
    if mode in ('all', 'trained_only'):
        trained_rows = freeze_trained_selector(domain=spec.domain, data_dir=data_dir, checkpoint_path=checkpoint_path, device=device, budget=budget, candidate_pool_size=candidate_pool_size, training_scope=training_scope)
        _write_jsonl(output_dir / 'trained_selector.jsonl', trained_rows)
        written.append('trained_selector.jsonl')
        methods.append('trained_selector')
    return {'domain': spec.domain, 'mode': mode, 'budget': budget, 'candidate_pool_size': candidate_pool_size, 'training_scope': training_scope, 'selection_seed': selection_seed, 'methods': methods, 'output_dir': str(output_dir), 'written_files': written, 'checkpoint_path': str(checkpoint_path) if checkpoint_path else None}

def freeze_all_selections(domain: str, data_dir: str | Path, checkpoint_path: str | Path, output_dir: str | Path, selection_seed: int=13, device: str='cpu', budget: int=BUDGET, candidate_pool_size: int | None=None, training_scope: str='single_domain') -> dict[str, Any]:
    return freeze_selections_by_mode(mode='all', domain=domain, data_dir=data_dir, output_dir=output_dir, checkpoint_path=checkpoint_path, selection_seed=selection_seed, device=device, budget=budget, candidate_pool_size=candidate_pool_size, training_scope=training_scope)

def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description='Freeze selector choices for online eval')
    parser.add_argument('--domain', required=True)
    parser.add_argument('--data-dir', required=True)
    parser.add_argument('--checkpoint', default=None, help='Required unless --mode baselines_only')
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--selection-seed', type=int, default=13)
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--budget', type=int, default=BUDGET)
    parser.add_argument('--candidate-pool-size', type=int, default=None)
    parser.add_argument('--training-scope', default='single_domain')
    parser.add_argument('--mode', default='all', choices=list(FREEZE_MODES), help="Which selection files to produce: 'all' writes baseline/random/global_utility/trained_selector (needs --checkpoint); 'baselines_only' writes the first 3 (no --checkpoint needed); 'trained_only' writes only trained_selector (needs --checkpoint).")
    return parser

def main() -> None:
    parser = _build_arg_parser()
    args = parser.parse_args()
    if args.mode in ('all', 'trained_only') and (not args.checkpoint):
        parser.error(f'--checkpoint is required for --mode {args.mode}')
    payload = freeze_selections_by_mode(mode=args.mode, domain=args.domain, data_dir=args.data_dir, checkpoint_path=args.checkpoint, output_dir=args.output_dir, selection_seed=args.selection_seed, device=args.device, budget=args.budget, candidate_pool_size=args.candidate_pool_size, training_scope=args.training_scope)
    print(json.dumps(payload))
if __name__ == '__main__':
    main()
