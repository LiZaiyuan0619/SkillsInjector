import argparse
import json
import sys
from pathlib import Path
from typing import Any
import torch
from torch import optim
from torch.nn import functional as F
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
from temp.selector.common import BUDGET, CONTRACT_DOMAIN, resolve_domain_spec
from temp.selector.datasets import build_pair_batch, build_pairwise_records, load_export_artifacts, load_pair_records, load_pair_records_multi
from temp.selector.modeling import SelectorScorer, listwise_cross_entropy_loss, pairwise_logistic_loss, ports_dual_loss, save_selector_checkpoint
from temp.selector.offline_eval import validate_manifest_schema
DEFAULT_ENCODER_NAME_OR_PATH = '<MODEL_ROOT>/model/Qwen3-Embedding-0.6B'

def load_text_encoder(encoder_name_or_path: str, device: str):
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer(encoder_name_or_path, trust_remote_code=True, device=device)

def validate_output_dir(output_dir: Path, domain: str=CONTRACT_DOMAIN, training_scope: str='single_domain') -> Path:
    parent = output_dir.parent
    parent.mkdir(parents=True, exist_ok=True)
    if training_scope == 'cross_domain':
        if output_dir.name == 'final' and (parent.name != 'all' or parent.parent.name != 'cross_domain'):
            raise ValueError("Selector checkpoints for cross-domain training must be written under 'cross_domain/all/final'")
    else:
        spec = resolve_domain_spec(domain)
        if output_dir.name == 'final' and parent.name != spec.output_name:
            raise ValueError(f"Selector checkpoints for domain '{spec.domain}' must be written under '{spec.output_name}/final'")
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir

def _encode_lookup(encoder: Any, texts_by_key: dict[str, str], device: str) -> dict[str, torch.Tensor]:
    ordered_keys = sorted(texts_by_key)
    embeddings = encoder.encode([texts_by_key[key] for key in ordered_keys], batch_size=16, convert_to_tensor=True, normalize_embeddings=True, device=device)
    if not isinstance(embeddings, torch.Tensor):
        embeddings = torch.tensor(embeddings, dtype=torch.float32)
    embeddings = embeddings.detach().cpu().float()
    return {key: embeddings[idx] for (idx, key) in enumerate(ordered_keys)}

def _resample_hard_negatives(pairwise_records: list, task_embeddings: dict[str, torch.Tensor], skill_embeddings: dict[str, torch.Tensor], hard_neg_ratio: float=0.7) -> list:
    from collections import defaultdict
    import random
    random.seed(42)
    pairs_by_task: dict[str, list] = defaultdict(list)
    for pr in pairwise_records:
        pairs_by_task[pr.task_id].append(pr)
    skill_names_sorted = sorted(skill_embeddings.keys())
    skill_emb_matrix = torch.stack([skill_embeddings[sn] for sn in skill_names_sorted])
    skill_name_to_idx = {sn: i for (i, sn) in enumerate(skill_names_sorted)}
    resampled: list = []
    for (task_id, task_pairs) in pairs_by_task.items():
        task_emb = task_embeddings[task_id]
        sim = torch.nn.functional.cosine_similarity(task_emb.unsqueeze(0), skill_emb_matrix, dim=1)
        neg_sim = {sn: float(sim[skill_name_to_idx[sn]]) for sn in skill_names_sorted}
        pair_sims = [neg_sim[pr.negative_skill_name] for pr in task_pairs]
        median_sim = sorted(pair_sims)[len(pair_sims) // 2]
        hard_pairs = [pr for pr in task_pairs if neg_sim[pr.negative_skill_name] >= median_sim]
        easy_pairs = [pr for pr in task_pairs if neg_sim[pr.negative_skill_name] < median_sim]
        n_total = len(task_pairs)
        n_hard = int(n_total * hard_neg_ratio)
        n_easy = n_total - n_hard
        sampled_hard = random.choices(hard_pairs, k=n_hard) if hard_pairs else []
        sampled_easy = random.choices(easy_pairs, k=n_easy) if easy_pairs else []
        resampled.extend(sampled_hard + sampled_easy)
    random.shuffle(resampled)
    return resampled

def train_selector(domain: str, data_dir: str | Path, output_dir: str | Path, encoder_name_or_path: str=DEFAULT_ENCODER_NAME_OR_PATH, max_steps: int=2, device: str='cpu', hidden_dim: int=128, pairwise_weight: float=0.5, learning_rate: float=0.001, training_scope: str='single_domain', source_data_dirs: list[str | Path] | None=None, loss_type: str='default', label_type: str='absolute', num_layers: int=3, dropout: float=0.1, hard_negative_mining: bool=False, hard_neg_ratio: float=0.7) -> dict[str, Any]:
    data_dir = Path(data_dir)
    output_dir = validate_output_dir(Path(output_dir), domain=domain, training_scope=training_scope)
    if training_scope == 'cross_domain':
        if not source_data_dirs:
            raise ValueError('source_data_dirs are required for cross_domain training')
        (pair_records, source_domains) = load_pair_records_multi(source_data_dirs)
        effective_domain = 'cross_domain'
    else:
        spec = resolve_domain_spec(domain)
        manifest = json.loads((data_dir / 'manifest.json').read_text(encoding='utf-8'))
        validate_manifest_schema(manifest, expected_domain=spec.domain, budget=BUDGET)
        pair_records = load_pair_records(data_dir)
        source_domains = [spec.domain]
        effective_domain = spec.domain
    pairwise_records = build_pairwise_records(pair_records)
    encoder = load_text_encoder(encoder_name_or_path, device=device)
    task_texts = {record.task_id: record.task_text for record in pair_records}
    skill_texts = {record.skill_name: record.skill_text for record in pair_records}
    task_embeddings = _encode_lookup(encoder, task_texts, device=device)
    skill_embeddings = _encode_lookup(encoder, skill_texts, device=device)
    embedding_dim = int(next(iter(task_embeddings.values())).shape[0])
    if hard_negative_mining:
        pairwise_records = _resample_hard_negatives(pairwise_records=pairwise_records, task_embeddings=task_embeddings, skill_embeddings=skill_embeddings, hard_neg_ratio=hard_neg_ratio)
    model = SelectorScorer(embedding_dim=embedding_dim, hidden_dim=hidden_dim, num_layers=num_layers, dropout=dropout)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    last_pointwise = 0.0
    last_pairwise = 0.0
    last_total = 0.0
    for _step in range(max_steps):
        model.train()
        batch = build_pair_batch(pair_records, task_embeddings=task_embeddings, skill_embeddings=skill_embeddings)
        output = model(task_embeddings=batch['task_embeddings'], skill_embeddings=batch['skill_embeddings'])
        if label_type == 'delta':
            training_labels = (batch['delta'] + 1.0) / 2.0
        else:
            training_labels = batch['p_skill']
        if loss_type == 'default':
            pointwise_loss = F.binary_cross_entropy(output['probabilities'], training_labels)
            if pairwise_records:
                logits_by_key = {(task_id, skill_name): output['logits'][idx] for (idx, (task_id, skill_name)) in enumerate(zip(batch['task_ids'], batch['skill_names'], strict=True))}
                positive_logits = torch.stack([logits_by_key[r.task_id, r.positive_skill_name] for r in pairwise_records])
                negative_logits = torch.stack([logits_by_key[r.task_id, r.negative_skill_name] for r in pairwise_records])
                ranking_loss = pairwise_logistic_loss(positive_logits, negative_logits)
            else:
                ranking_loss = torch.tensor(0.0)
            total_loss = pointwise_loss + pairwise_weight * ranking_loss
        elif loss_type == 'ports':
            total_loss = ports_dual_loss(logits=output['logits'], p_skill=training_labels, task_ids=batch['task_ids'])
        elif loss_type == 'listwise':
            total_loss = listwise_cross_entropy_loss(logits=output['logits'], p_skill=training_labels, task_ids=batch['task_ids'])
        elif loss_type == 'pairwise_only':
            if pairwise_records:
                logits_by_key = {(task_id, skill_name): output['logits'][idx] for (idx, (task_id, skill_name)) in enumerate(zip(batch['task_ids'], batch['skill_names'], strict=True))}
                positive_logits = torch.stack([logits_by_key[r.task_id, r.positive_skill_name] for r in pairwise_records])
                negative_logits = torch.stack([logits_by_key[r.task_id, r.negative_skill_name] for r in pairwise_records])
                total_loss = pairwise_logistic_loss(positive_logits, negative_logits)
            else:
                total_loss = torch.tensor(0.0)
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()
        last_pointwise = float(pointwise_loss.detach().cpu()) if loss_type == 'default' else 0.0
        last_pairwise = float(ranking_loss.detach().cpu()) if loss_type == 'default' else 0.0
        last_total = float(total_loss.detach().cpu())
    save_selector_checkpoint(path=str(output_dir / 'model.pt'), model=model, encoder_name_or_path=encoder_name_or_path, embedding_dim=embedding_dim, extra_config={'hidden_dim': hidden_dim, 'num_layers': num_layers, 'dropout': dropout, 'loss_type': loss_type, 'label_type': label_type, 'pairwise_weight': pairwise_weight, 'learning_rate': learning_rate, 'max_steps': max_steps, 'domain': effective_domain, 'training_scope': training_scope, 'source_domains': source_domains, 'hard_negative_mining': hard_negative_mining, 'hard_neg_ratio': hard_neg_ratio})
    config = {'domain': effective_domain, 'budget': BUDGET, 'training_scope': training_scope, 'source_domains': source_domains, 'encoder_name_or_path': encoder_name_or_path, 'embedding_dim': embedding_dim, 'hidden_dim': hidden_dim, 'num_layers': num_layers, 'dropout': dropout, 'loss_type': loss_type, 'label_type': label_type, 'pairwise_weight': pairwise_weight, 'learning_rate': learning_rate, 'max_steps': max_steps, 'device': device, 'hard_negative_mining': hard_negative_mining, 'hard_neg_ratio': hard_neg_ratio}
    metrics = {'domain': effective_domain, 'budget': BUDGET, 'training_scope': training_scope, 'source_domains': source_domains, 'encoder_name_or_path': encoder_name_or_path, 'hidden_dim': hidden_dim, 'num_layers': num_layers, 'dropout': dropout, 'loss_type': loss_type, 'label_type': label_type, 'max_steps': max_steps, 'device': device, 'hard_negative_mining': hard_negative_mining, 'hard_neg_ratio': hard_neg_ratio, 'num_pair_records': len(pair_records), 'num_pairwise_records': len(pairwise_records), 'last_pointwise_loss': last_pointwise, 'last_pairwise_loss': last_pairwise, 'last_total_loss': last_total}
    (output_dir / 'config.json').write_text(json.dumps(config, indent=2, sort_keys=True), encoding='utf-8')
    (output_dir / 'train_metrics.json').write_text(json.dumps(metrics, indent=2, sort_keys=True), encoding='utf-8')
    return metrics

def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description='Train selector biencoder')
    parser.add_argument('--domain', required=True)
    parser.add_argument('--data-dir', required=True)
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--encoder-name-or-path', default=DEFAULT_ENCODER_NAME_OR_PATH)
    parser.add_argument('--max-steps', type=int, default=2)
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--hidden-dim', type=int, default=128)
    parser.add_argument('--pairwise-weight', type=float, default=0.5)
    parser.add_argument('--learning-rate', type=float, default=0.001)
    parser.add_argument('--training-scope', default='single_domain')
    parser.add_argument('--source-data-dirs', default=None)
    parser.add_argument('--loss-type', default='default', choices=['default', 'ports', 'listwise', 'pairwise_only'], help='Loss function type')
    parser.add_argument('--label-type', default='absolute', choices=['absolute', 'delta'], help='Label type: absolute=p_skill, delta=p_skill-p_base')
    parser.add_argument('--num-layers', type=int, default=3, help='MLP scorer layers (default: 3)')
    parser.add_argument('--dropout', type=float, default=0.1, help='MLP dropout rate (default: 0.1)')
    parser.add_argument('--hard-negative-mining', action='store_true', help='Enable hard negative mining')
    parser.add_argument('--hard-neg-ratio', type=float, default=0.7, help='Hard pair ratio in resampled pairwise records')
    return parser

def main() -> None:
    parser = _build_arg_parser()
    args = parser.parse_args()
    metrics = train_selector(domain=args.domain, data_dir=args.data_dir, output_dir=args.output_dir, encoder_name_or_path=args.encoder_name_or_path, max_steps=args.max_steps, device=args.device, hidden_dim=args.hidden_dim, pairwise_weight=args.pairwise_weight, learning_rate=args.learning_rate, training_scope=args.training_scope, loss_type=args.loss_type, label_type=args.label_type, num_layers=args.num_layers, dropout=args.dropout, hard_negative_mining=args.hard_negative_mining, hard_neg_ratio=args.hard_neg_ratio, source_data_dirs=[part for part in str(args.source_data_dirs).split(',') if part] if args.source_data_dirs is not None else None)
    print(json.dumps(metrics))
if __name__ == '__main__':
    main()
