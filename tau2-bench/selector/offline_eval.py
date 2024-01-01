import argparse
import json
import random
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
from temp.selector.common import BUDGET, SCHEMA_VERSION, resolve_domain_spec
from temp.selector.datasets import build_pair_batch, load_export_artifacts, load_pair_records
from temp.selector.modeling import load_selector_checkpoint

def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding='utf-8'))
    if not isinstance(payload, dict):
        raise ValueError(f'Expected JSON object at {path}')
    return payload

def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding='utf-8').splitlines():
        rows.append(json.loads(line))
    return rows

def validate_manifest_schema(manifest: dict[str, Any], expected_domain: str, budget: int) -> None:
    spec = resolve_domain_spec(expected_domain)
    if manifest.get('schema_version') != SCHEMA_VERSION:
        raise ValueError('unsupported manifest schema_version')
    if manifest.get('domain') != spec.domain:
        raise ValueError('manifest domain does not match requested domain')
    if int(manifest.get('budget', -1)) != budget:
        raise ValueError('manifest budget does not match requested budget')
    expected_task_split_name = spec.task_split or 'base'
    if manifest.get('task_split_name', expected_task_split_name) != expected_task_split_name:
        raise ValueError('manifest task_split_name does not match requested domain')
    if manifest.get('domain_output_name', spec.output_name) != spec.output_name:
        raise ValueError('manifest domain_output_name does not match requested domain')

def validate_labels_table(rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError('labels table must not be empty')

def _load_exported_dataset(domain: str, data_dir: Path, budget: int, validate_budget: bool=True) -> tuple[dict[str, Any], dict[str, dict[str, Any]], list[str], dict[str, dict[str, dict[str, float]]]]:
    (manifest, task_rows, skill_rows, label_rows) = load_export_artifacts(data_dir)
    if validate_budget:
        validate_manifest_schema(manifest, expected_domain=domain, budget=budget)
    else:
        validate_manifest_schema({**manifest, 'budget': budget}, expected_domain=domain, budget=budget)
    validate_labels_table(label_rows)
    tasks_by_id: dict[str, dict[str, Any]] = {}
    for row in task_rows:
        task_id = str(row['task_id'])
        n_base = int(row['n_base'])
        if n_base <= 0:
            raise ValueError(f'task {task_id} has non-positive n_base')
        tasks_by_id[task_id] = {**row, 'task_id': task_id, 'k_base': int(row['k_base']), 'n_base': n_base, 'p_base': float(row['k_base']) / float(n_base)}
    skill_names = sorted((str(row['skill_name']) for row in skill_rows))
    labels_by_task: dict[str, dict[str, dict[str, float]]] = defaultdict(dict)
    for row in label_rows:
        task_id = str(row['task_id'])
        skill_name = str(row['skill_name'])
        n_skill = int(row['n_skill'])
        if n_skill <= 0:
            raise ValueError(f'task {task_id} / skill {skill_name} has non-positive n_skill')
        if task_id not in tasks_by_id:
            raise ValueError(f'labels row references unknown task_id: {task_id}')
        p_skill = float(row['k_skill']) / float(n_skill)
        p_base = float(tasks_by_id[task_id]['p_base'])
        labels_by_task[task_id][skill_name] = {'k_skill': int(row['k_skill']), 'n_skill': n_skill, 'p_skill': p_skill, 'delta': p_skill - p_base}
    if set(tasks_by_id) != set(labels_by_task):
        raise ValueError('labels table task universe does not match tasks table')
    for (task_id, labels) in labels_by_task.items():
        if set(labels) != set(skill_names):
            raise ValueError(f'skill universe mismatch for task {task_id}')
    return (manifest, tasks_by_id, skill_names, dict(labels_by_task))

def _rank_global_utility(task_id: str, skill_names: list[str], labels_by_task: dict[str, dict[str, dict[str, float]]]) -> list[str]:
    del task_id
    totals: dict[str, float] = {}
    for skill_name in skill_names:
        totals[skill_name] = sum((labels_by_task[current_task_id][skill_name]['p_skill'] for current_task_id in labels_by_task)) / float(len(labels_by_task))
    return sorted(skill_names, key=lambda name: (-totals[name], name))

def _rank_random(task_id: str, skill_names: list[str], seed: int) -> list[str]:
    rng = random.Random(f'{seed}:{task_id}')
    ranked = list(skill_names)
    rng.shuffle(ranked)
    return ranked

def _summarize_ranking(baseline_name: str, budget: int, tasks_by_id: dict[str, dict[str, Any]], labels_by_task: dict[str, dict[str, dict[str, float]]], rankings_by_task: dict[str, list[str]]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    task_ids = sorted(tasks_by_id)
    top1_deltas: list[float] = []
    selected_positive_count = 0
    total_positive_count = 0
    reciprocal_ranks: list[float] = []
    prediction_rows: list[dict[str, Any]] = []
    for task_id in task_ids:
        ranked_skills = rankings_by_task[task_id]
        labels = labels_by_task[task_id]
        selected_skill = ranked_skills[0]
        selected = labels[selected_skill]
        top1_deltas.append(float(selected['delta']))
        positive_skills = {skill_name for (skill_name, row) in labels.items() if float(row['delta']) > 0.0}
        if float(selected['delta']) > 0.0:
            selected_positive_count += 1
        total_positive_count += len(positive_skills)
        reciprocal_rank = 0.0
        for (rank_index, skill_name) in enumerate(ranked_skills, start=1):
            if skill_name in positive_skills:
                reciprocal_rank = 1.0 / float(rank_index)
                break
        reciprocal_ranks.append(reciprocal_rank)
        prediction_rows.append({'baseline_name': baseline_name, 'task_id': task_id, 'selected_skill_name': selected_skill, 'selected_delta': float(selected['delta']), 'selected_p_skill': float(selected['p_skill']), 'ranked_skill_names': ranked_skills, 'positive_skill_names': sorted(positive_skills)})
    num_tasks = len(task_ids)
    summary = {'baseline_name': baseline_name, 'budget': budget, 'num_tasks': num_tasks, 'top1_delta_mean': sum(top1_deltas) / float(num_tasks), 'positive_hit_rate': selected_positive_count / float(num_tasks), 'recall_at_1': selected_positive_count / float(total_positive_count) if total_positive_count else 0.0, 'mrr': sum(reciprocal_ranks) / float(num_tasks)}
    return (summary, prediction_rows)

def validate_offline_output_dir(output_dir: Path, domain: str) -> Path:
    spec = resolve_domain_spec(domain)
    if output_dir.name == 'offline' and output_dir.parent.name != spec.output_name:
        raise ValueError(f"Offline results for domain '{spec.domain}' must be written under '{spec.output_name}/offline'")
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir

def _write_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding='utf-8')

def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text('\n'.join((json.dumps(row, sort_keys=True) for row in rows)) + ('\n' if rows else ''), encoding='utf-8')

def run_offline_baselines(domain: str, data_dir: str | Path, baseline_suite: list[str], budget: int, output_dir: str | Path, seed: int=13) -> list[dict[str, Any]]:
    spec = resolve_domain_spec(domain)
    if budget != BUDGET:
        raise ValueError(f'Unsupported budget for offline evaluation: {budget}')
    data_dir = Path(data_dir)
    output_dir = validate_offline_output_dir(Path(output_dir), spec.domain)
    (_manifest, tasks_by_id, skill_names, labels_by_task) = _load_exported_dataset(domain=spec.domain, data_dir=data_dir, budget=budget)
    summaries: list[dict[str, Any]] = []
    for baseline_name in baseline_suite:
        if baseline_name == 'global_utility':
            rankings_by_task = {task_id: _rank_global_utility(task_id, skill_names, labels_by_task) for task_id in tasks_by_id}
        elif baseline_name == 'random':
            rankings_by_task = {task_id: _rank_random(task_id, skill_names, seed) for task_id in tasks_by_id}
        else:
            raise ValueError(f'Unsupported baseline: {baseline_name}')
        (summary, prediction_rows) = _summarize_ranking(baseline_name=baseline_name, budget=budget, tasks_by_id=tasks_by_id, labels_by_task=labels_by_task, rankings_by_task=rankings_by_task)
        summaries.append(summary)
        _write_json(output_dir / f'{baseline_name}.summary.json', summary)
        _write_jsonl(output_dir / f'{baseline_name}.predictions.jsonl', prediction_rows)
    _write_json(output_dir / 'comparison.json', {'domain': spec.domain, 'budget': budget, 'seed': seed, 'summaries': summaries})
    return summaries

def load_text_encoder(encoder_name_or_path: str, device: str):
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer(encoder_name_or_path, trust_remote_code=True, device=device)

def _encode_lookup(encoder: Any, texts_by_key: dict[str, str], device: str) -> dict[str, Any]:
    ordered_keys = sorted(texts_by_key)
    embeddings = encoder.encode([texts_by_key[key] for key in ordered_keys], batch_size=16, convert_to_tensor=True, normalize_embeddings=True, device=device)
    try:
        import torch
        if not isinstance(embeddings, torch.Tensor):
            embeddings = torch.tensor(embeddings, dtype=torch.float32)
        embeddings = embeddings.detach().cpu().float()
        return {key: embeddings[idx] for (idx, key) in enumerate(ordered_keys)}
    except Exception as exc:
        raise RuntimeError('failed to convert embeddings to torch tensors') from exc

def run_trained_evaluation(domain: str, data_dir: str | Path, checkpoint_path: str | Path, baseline_suite: list[str], budget: int, output_dir: str | Path, seed: int=13, device: str='cpu') -> dict[str, Any]:
    import torch
    spec = resolve_domain_spec(domain)
    data_dir = Path(data_dir)
    output_dir = validate_offline_output_dir(Path(output_dir), spec.domain)
    checkpoint_path = Path(checkpoint_path)
    baseline_summaries = run_offline_baselines(domain=spec.domain, data_dir=data_dir, baseline_suite=baseline_suite, budget=budget, output_dir=output_dir, seed=seed)
    pair_records = load_pair_records(data_dir)
    (model, checkpoint_payload) = load_selector_checkpoint(str(checkpoint_path))
    encoder_name_or_path = str(checkpoint_payload['encoder_name_or_path'])
    checkpoint_domain = checkpoint_payload.get('extra_config', {}).get('domain')
    if checkpoint_domain is not None and str(checkpoint_domain) != spec.domain:
        raise ValueError('checkpoint domain does not match requested domain')
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
    (_manifest, tasks_by_id, skill_names, labels_by_task) = _load_exported_dataset(domain=spec.domain, data_dir=data_dir, budget=budget)
    rankings_by_task = {task_id: sorted(skill_names, key=lambda skill_name: (-logits_by_key[task_id, skill_name], skill_name)) for task_id in tasks_by_id}
    (trained_summary, trained_predictions) = _summarize_ranking(baseline_name='trained_selector', budget=budget, tasks_by_id=tasks_by_id, labels_by_task=labels_by_task, rankings_by_task=rankings_by_task)
    _write_json(output_dir / 'trained.summary.json', trained_summary)
    _write_jsonl(output_dir / 'trained.predictions.jsonl', trained_predictions)
    delta_vs_baselines = []
    for summary in baseline_summaries:
        delta_vs_baselines.append({'baseline_name': summary['baseline_name'], 'top1_delta_mean_delta': trained_summary['top1_delta_mean'] - summary['top1_delta_mean'], 'positive_hit_rate_delta': trained_summary['positive_hit_rate'] - summary['positive_hit_rate'], 'recall_at_1_delta': trained_summary['recall_at_1'] - summary['recall_at_1'], 'mrr_delta': trained_summary['mrr'] - summary['mrr']})
    payload = {'domain': spec.domain, 'budget': budget, 'seed': seed, 'checkpoint_path': str(checkpoint_path), 'trained_summary': trained_summary, 'baseline_summaries': baseline_summaries, 'delta_vs_baselines': delta_vs_baselines}
    _write_json(output_dir / 'comparison.json', payload)
    return payload

def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description='Run offline selector baselines')
    parser.add_argument('--domain', required=True)
    parser.add_argument('--data-dir', required=True)
    parser.add_argument('--baseline-suite', required=True)
    parser.add_argument('--budget', required=True, type=int)
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--seed', type=int, default=13)
    parser.add_argument('--checkpoint')
    parser.add_argument('--device', default='cpu')
    return parser

def main() -> None:
    parser = _build_arg_parser()
    args = parser.parse_args()
    baseline_names = [name.strip() for name in args.baseline_suite.split(',') if name.strip()]
    if args.checkpoint:
        payload = run_trained_evaluation(domain=args.domain, data_dir=args.data_dir, checkpoint_path=args.checkpoint, baseline_suite=baseline_names, budget=args.budget, output_dir=args.output_dir, seed=args.seed, device=args.device)
        print(json.dumps(payload))
    else:
        run_offline_baselines(domain=args.domain, data_dir=args.data_dir, baseline_suite=baseline_names, budget=args.budget, output_dir=args.output_dir, seed=args.seed)
        print(json.dumps({'domain': args.domain, 'budget': args.budget, 'baseline_names': baseline_names, 'output_dir': args.output_dir}))
if __name__ == '__main__':
    main()
