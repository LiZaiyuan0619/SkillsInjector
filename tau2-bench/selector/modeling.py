from typing import Any
import torch
from torch import nn
from torch.nn import functional as F

def validate_probability_counts(k_skill: int, n_skill: int) -> None:
    if n_skill <= 0:
        raise ValueError('n_skill must be positive')
    if k_skill < 0:
        raise ValueError('k_skill must be non-negative')
    if k_skill > n_skill:
        raise ValueError('k_skill must not exceed n_skill')

def build_pair_features(task_embeddings: torch.Tensor, skill_embeddings: torch.Tensor) -> torch.Tensor:
    return torch.cat([task_embeddings, skill_embeddings, torch.abs(task_embeddings - skill_embeddings), task_embeddings * skill_embeddings], dim=-1)

class SelectorScorer(nn.Module):

    def __init__(self, embedding_dim: int, hidden_dim: int=256, num_layers: int=3, dropout: float=0.1):
        super().__init__()
        input_dim = embedding_dim * 4
        self.config = {'embedding_dim': embedding_dim, 'hidden_dim': hidden_dim, 'num_layers': num_layers, 'dropout': dropout}
        layers: list[nn.Module] = []
        current_dim = input_dim
        for i in range(num_layers - 1):
            next_dim = max(hidden_dim // 2 ** i, 64)
            layers.extend([nn.Linear(current_dim, next_dim), nn.ReLU(), nn.Dropout(dropout)])
            current_dim = next_dim
        layers.append(nn.Linear(current_dim, 1))
        self.mlp = nn.Sequential(*layers)

    def forward(self, task_embeddings: torch.Tensor, skill_embeddings: torch.Tensor) -> dict[str, torch.Tensor]:
        features = build_pair_features(task_embeddings, skill_embeddings)
        logits = self.mlp(features).squeeze(-1)
        probabilities = torch.sigmoid(logits)
        return {'features': features, 'logits': logits, 'probabilities': probabilities}

class CosineScorer(nn.Module):

    def __init__(self, embedding_dim: int=0, init_temperature: float=0.05):
        super().__init__()
        self.config = {'scorer_type': 'cosine', 'embedding_dim': embedding_dim, 'init_temperature': init_temperature}
        self.log_temperature = nn.Parameter(torch.tensor(init_temperature).clamp(min=0.0001).log())

    def forward(self, task_embeddings: torch.Tensor, skill_embeddings: torch.Tensor) -> dict[str, torch.Tensor]:
        cosine_sim = F.cosine_similarity(task_embeddings, skill_embeddings, dim=-1)
        logits = cosine_sim / self.log_temperature.exp()
        probabilities = torch.sigmoid(logits)
        return {'logits': logits, 'probabilities': probabilities}

def pairwise_logistic_loss(positive_logits: torch.Tensor, negative_logits: torch.Tensor) -> torch.Tensor:
    margin = positive_logits - negative_logits
    return torch.log1p(torch.exp(-margin)).mean()

def ports_dual_loss(logits: torch.Tensor, p_skill: torch.Tensor, task_ids: list[str], beta: float=0.5, gamma: float=0.5, lam: float=0.3) -> torch.Tensor:
    device = logits.device
    task_groups: dict[str, list[int]] = {}
    for (idx, tid) in enumerate(task_ids):
        task_groups.setdefault(tid, []).append(idx)
    kl_losses = []
    pref_losses = []
    for (tid, indices) in task_groups.items():
        idx_tensor = torch.tensor(indices, device=device, dtype=torch.long)
        group_logits = logits[idx_tensor]
        group_p_skill = p_skill[idx_tensor]
        log_q = torch.log_softmax(group_p_skill / beta, dim=0)
        log_p = torch.log_softmax(group_logits / gamma, dim=0)
        q = torch.exp(log_q)
        kl = (q * (log_q - log_p)).sum()
        kl_losses.append(kl)
        p = torch.softmax(group_logits / gamma, dim=0)
        p_clamped = p.clamp(1e-07, 1.0 - 1e-07)
        log_odds = torch.log(p_clamped / (1.0 - p_clamped))
        n = len(indices)
        if n > 1:
            odds_diff = log_odds.unsqueeze(0) - log_odds.unsqueeze(1)
            mask = group_p_skill.unsqueeze(0) > group_p_skill.unsqueeze(1)
            if mask.any():
                pair_loss = torch.nn.functional.softplus(-odds_diff[mask]).mean()
                pref_losses.append(pair_loss)
    l_utility = torch.stack(kl_losses).mean() if kl_losses else torch.tensor(0.0, device=device)
    l_pref = torch.stack(pref_losses).mean() if pref_losses else torch.tensor(0.0, device=device)
    return l_utility + lam * l_pref

def listwise_cross_entropy_loss(logits: torch.Tensor, p_skill: torch.Tensor, task_ids: list[str], tau: float=0.5) -> torch.Tensor:
    device = logits.device
    task_groups: dict[str, list[int]] = {}
    for (idx, tid) in enumerate(task_ids):
        task_groups.setdefault(tid, []).append(idx)
    losses = []
    for (tid, indices) in task_groups.items():
        idx_tensor = torch.tensor(indices, device=device, dtype=torch.long)
        group_logits = logits[idx_tensor]
        group_p_skill = p_skill[idx_tensor]
        target = torch.softmax(group_p_skill / tau, dim=0)
        log_pred = torch.log_softmax(group_logits, dim=0)
        loss = -(target * log_pred).sum()
        losses.append(loss)
    return torch.stack(losses).mean() if losses else torch.tensor(0.0, device=device)

def save_selector_checkpoint(path: str, model: SelectorScorer, encoder_name_or_path: str, embedding_dim: int, extra_config: dict[str, Any] | None=None) -> None:
    config = extra_config or {}
    if hasattr(model, 'config'):
        config['mlp_config'] = model.config
    payload = {'model_state_dict': model.state_dict(), 'encoder_name_or_path': encoder_name_or_path, 'embedding_dim': embedding_dim, 'extra_config': config}
    torch.save(payload, path)

def load_selector_checkpoint(path: str, hidden_dim: int | None=None) -> tuple[nn.Module, dict[str, Any]]:
    payload = torch.load(path, map_location='cpu', weights_only=False)
    embedding_dim = int(payload['embedding_dim'])
    extra = payload.get('extra_config', {})
    mlp_config = extra.get('mlp_config', {})
    scorer_type = mlp_config.get('scorer_type', 'mlp')
    if scorer_type == 'cosine':
        model: nn.Module = CosineScorer(embedding_dim=embedding_dim)
        model.load_state_dict(payload['model_state_dict'])
    else:
        _hidden_dim = int(mlp_config.get('hidden_dim', extra.get('hidden_dim', hidden_dim or 128)))
        _num_layers = int(mlp_config.get('num_layers', 2))
        _dropout = float(mlp_config.get('dropout', 0.0))
        model = SelectorScorer(embedding_dim=embedding_dim, hidden_dim=_hidden_dim, num_layers=_num_layers, dropout=_dropout)
        model.load_state_dict(payload['model_state_dict'])
    return (model, payload)
