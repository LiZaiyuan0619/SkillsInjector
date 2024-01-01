import argparse
import json
import random
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any
import torch
from torch import nn, optim
from torch.nn import functional as F
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
from temp.selector.common import BUDGET, resolve_domain_spec
from temp.selector.datasets import load_pair_records, PairRecord
from temp.selector.modeling import CosineScorer, SelectorScorer, listwise_cross_entropy_loss, ports_dual_loss, save_selector_checkpoint
from temp.selector.offline_eval import validate_manifest_schema

def encode_with_grad(encoder: nn.Module, texts: list[str], device: str, batch_size: int=32) -> torch.Tensor:
    all_embeddings: list[torch.Tensor] = []
    for start in range(0, len(texts), batch_size):
        batch_texts = texts[start:start + batch_size]
        features = encoder.tokenize(batch_texts)
        features = {k: v.to(device) for (k, v) in features.items()}
        out = encoder.forward(features)
        emb = out['sentence_embedding'].float()
        emb = F.normalize(emb, p=2, dim=1)
        all_embeddings.append(emb)
    return torch.cat(all_embeddings, dim=0)

def train_finetune(domain: str, data_dir: str | Path, output_dir: str | Path, encoder_name_or_path: str, scorer_type: str='mlp', loss_type: str='ports', label_type: str='absolute', max_steps: int=200, device: str='cuda', hidden_dim: int=256, num_layers: int=2, dropout: float=0.1, encoder_lr: float=2e-05, head_lr: float=0.001, max_grad_norm: float=1.0, warmup_steps: int=20, batch_size: int=4, encode_batch_size: int=32) -> dict[str, Any]:
    from sentence_transformers import SentenceTransformer
    data_dir = Path(data_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    spec = resolve_domain_spec(domain)
    manifest = json.loads((data_dir / 'manifest.json').read_text(encoding='utf-8'))
    validate_manifest_schema(manifest, expected_domain=spec.domain, budget=BUDGET)
    pair_records = load_pair_records(data_dir)
    records_by_task: dict[str, list[PairRecord]] = defaultdict(list)
    for r in pair_records:
        records_by_task[r.task_id].append(r)
    all_task_ids = sorted(records_by_task.keys())
    print(f'Loaded {len(pair_records)} records across {len(all_task_ids)} tasks')
    encoder = SentenceTransformer(encoder_name_or_path, trust_remote_code=True, device=device)
    with torch.no_grad():
        probe = encoder.encode(['probe'], convert_to_tensor=True, device=device)
        embedding_dim = int(probe.shape[1])
    print(f'Encoder: {encoder_name_or_path}  embedding_dim={embedding_dim}')
    if scorer_type == 'cosine':
        scorer: nn.Module = CosineScorer(embedding_dim=embedding_dim)
    else:
        scorer = SelectorScorer(embedding_dim=embedding_dim, hidden_dim=hidden_dim, num_layers=num_layers, dropout=dropout)
    scorer = scorer.to(device)
    print(f'Scorer: {scorer_type}  params={sum((p.numel() for p in scorer.parameters())):,}')
    optimizer = optim.AdamW([{'params': encoder.parameters(), 'lr': encoder_lr, 'weight_decay': 0.01}, {'params': scorer.parameters(), 'lr': head_lr, 'weight_decay': 0.0}])

    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return float(step + 1) / float(max(1, warmup_steps))
        return 1.0
    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    encoder.train()
    scorer.train()
    random.seed(42)
    last_loss = 0.0
    for step in range(max_steps):
        optimizer.zero_grad()
        sampled = random.choices(all_task_ids, k=batch_size)
        all_logits: list[torch.Tensor] = []
        all_labels: list[torch.Tensor] = []
        all_tids: list[str] = []
        for task_id in sampled:
            task_records = records_by_task[task_id]
            task_text = task_records[0].task_text
            skill_texts = [r.skill_text for r in task_records]
            task_emb = encode_with_grad(encoder, [task_text], device, batch_size=1)
            skill_embs = encode_with_grad(encoder, skill_texts, device, batch_size=encode_batch_size)
            task_emb_expanded = task_emb.expand(len(skill_texts), -1)
            output = scorer(task_emb_expanded, skill_embs)
            if label_type == 'delta':
                labels = torch.tensor([(r.delta + 1.0) / 2.0 for r in task_records], dtype=torch.float32, device=device)
            else:
                labels = torch.tensor([r.p_skill for r in task_records], dtype=torch.float32, device=device)
            all_logits.append(output['logits'])
            all_labels.append(labels)
            all_tids.extend([task_id] * len(task_records))
        combined_logits = torch.cat(all_logits, dim=0)
        combined_labels = torch.cat(all_labels, dim=0)
        if loss_type == 'ports':
            loss = ports_dual_loss(combined_logits, combined_labels, all_tids)
        elif loss_type == 'listwise':
            loss = listwise_cross_entropy_loss(combined_logits, combined_labels, all_tids)
        else:
            loss = F.binary_cross_entropy(torch.sigmoid(combined_logits), combined_labels)
        loss.backward()
        all_params = list(encoder.parameters()) + list(scorer.parameters())
        torch.nn.utils.clip_grad_norm_(all_params, max_norm=max_grad_norm)
        optimizer.step()
        scheduler.step()
        last_loss = float(loss.detach().cpu())
        if (step + 1) % 10 == 0 or step == 0:
            enc_lr = optimizer.param_groups[0]['lr']
            head_lr_now = optimizer.param_groups[1]['lr']
            print(f'Step {step + 1}/{max_steps}  loss={last_loss:.4f}  enc_lr={enc_lr:.2e}  head_lr={head_lr_now:.2e}')
    encoder_save_dir = str(output_dir / 'encoder')
    encoder.save(encoder_save_dir)
    print(f'Saved fine-tuned encoder to {encoder_save_dir}')
    save_selector_checkpoint(path=str(output_dir / 'model.pt'), model=scorer, encoder_name_or_path=encoder_save_dir, embedding_dim=embedding_dim, extra_config={'hidden_dim': hidden_dim, 'domain': spec.domain, 'training_mode': 'finetune', 'scorer_type': scorer_type, 'loss_type': loss_type, 'label_type': label_type, 'encoder_lr': encoder_lr, 'head_lr': head_lr, 'max_steps': max_steps, 'batch_size': batch_size, 'warmup_steps': warmup_steps})
    config = {'domain': spec.domain, 'budget': BUDGET, 'training_mode': 'finetune', 'scorer_type': scorer_type, 'loss_type': loss_type, 'label_type': label_type, 'encoder_name_or_path': encoder_name_or_path, 'embedding_dim': embedding_dim, 'hidden_dim': hidden_dim, 'num_layers': num_layers, 'dropout': dropout, 'encoder_lr': encoder_lr, 'head_lr': head_lr, 'max_steps': max_steps, 'batch_size': batch_size, 'warmup_steps': warmup_steps, 'max_grad_norm': max_grad_norm}
    metrics = {**config, 'num_pair_records': len(pair_records), 'num_tasks': len(all_task_ids), 'last_loss': last_loss}
    (output_dir / 'config.json').write_text(json.dumps(config, indent=2, sort_keys=True), encoding='utf-8')
    (output_dir / 'train_metrics.json').write_text(json.dumps(metrics, indent=2, sort_keys=True), encoding='utf-8')
    print(f'Done. Last loss={last_loss:.4f}')
    return metrics
DEFAULT_ENCODER = '<MODEL_ROOT>/model/Qwen3-Embedding-0.6B'

def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description='Fine-tune encoder + scorer for selector')
    p.add_argument('--domain', required=True)
    p.add_argument('--data-dir', required=True)
    p.add_argument('--output-dir', required=True)
    p.add_argument('--encoder-name-or-path', default=DEFAULT_ENCODER)
    p.add_argument('--scorer-type', default='mlp', choices=['mlp', 'cosine'], help='Scoring head type: mlp (4x concat + MLP) or cosine (dot product)')
    p.add_argument('--loss-type', default='ports', choices=['ports', 'listwise', 'default'])
    p.add_argument('--label-type', default='absolute', choices=['absolute', 'delta'])
    p.add_argument('--max-steps', type=int, default=200)
    p.add_argument('--device', default='cuda')
    p.add_argument('--hidden-dim', type=int, default=256)
    p.add_argument('--num-layers', type=int, default=2)
    p.add_argument('--dropout', type=float, default=0.1)
    p.add_argument('--encoder-lr', type=float, default=2e-05)
    p.add_argument('--head-lr', type=float, default=0.001)
    p.add_argument('--max-grad-norm', type=float, default=1.0)
    p.add_argument('--warmup-steps', type=int, default=20)
    p.add_argument('--batch-size', type=int, default=4, help='Tasks per gradient step')
    p.add_argument('--encode-batch-size', type=int, default=32, help='Texts per encode micro-batch')
    return p

def main() -> None:
    args = _build_arg_parser().parse_args()
    metrics = train_finetune(domain=args.domain, data_dir=args.data_dir, output_dir=args.output_dir, encoder_name_or_path=args.encoder_name_or_path, scorer_type=args.scorer_type, loss_type=args.loss_type, label_type=args.label_type, max_steps=args.max_steps, device=args.device, hidden_dim=args.hidden_dim, num_layers=args.num_layers, dropout=args.dropout, encoder_lr=args.encoder_lr, head_lr=args.head_lr, max_grad_norm=args.max_grad_norm, warmup_steps=args.warmup_steps, batch_size=args.batch_size, encode_batch_size=args.encode_batch_size)
    print(json.dumps(metrics, indent=2))
if __name__ == '__main__':
    main()
