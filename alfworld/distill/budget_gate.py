from __future__ import annotations
import logging
from typing import Callable
import numpy as np
from pipeline.paths import TASK_TYPES
log = logging.getLogger(__name__)
THRESHOLD_LADDER = [0.9, 0.85, 0.8]
TOKEN_BUDGET = 24000
N_MAX = 30
N_MIN = 12
PER_TYPE_MIN = 2
PER_TYPE_MEDOID_FALLBACK = 5

def cluster_dedup(embeddings: np.ndarray, threshold: float) -> list[list[int]]:
    n = len(embeddings)
    if n == 0:
        return []
    norms = embeddings / (np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-09)
    sim = norms @ norms.T
    visited = [False] * n
    clusters = []
    for i in range(n):
        if visited[i]:
            continue
        c = [i]
        visited[i] = True
        for j in range(i + 1, n):
            if not visited[j] and sim[i, j] >= threshold:
                c.append(j)
                visited[j] = True
        clusters.append(c)
    return clusters

def cluster_medoid(cluster: list[int], embeddings: np.ndarray) -> int:
    if len(cluster) == 1:
        return cluster[0]
    sub = embeddings[cluster]
    sub_n = sub / (np.linalg.norm(sub, axis=1, keepdims=True) + 1e-09)
    centrality = (sub_n @ sub_n.T).sum(axis=1)
    return cluster[int(np.argmax(centrality))]

def _per_type_count_full(skills: list[dict]) -> dict[str, int]:
    out = {tt: 0 for tt in TASK_TYPES}
    for s in skills:
        out[s['source_task_type']] = out.get(s['source_task_type'], 0) + 1
    return out

def _coverage_violated(skills: list[dict]) -> list[str]:
    cnt = _per_type_count_full(skills)
    return [tt for tt in TASK_TYPES if cnt[tt] < PER_TYPE_MIN]

def _enforce_coverage(kept_idx: list[int], raw_skills, embeddings, by_type) -> list[int]:
    out = list(kept_idx)
    out_set = set(out)
    cnt = _per_type_count_full([raw_skills[i] for i in out])
    for tt in TASK_TYPES:
        need = PER_TYPE_MIN - cnt[tt]
        if need <= 0:
            continue
        idx_list = by_type.get(tt, [])
        if not idx_list:
            log.warning(f'  task_type {tt!r} has 0 raw skills — cannot enforce coverage; leaving as 0')
            continue
        sub_emb = embeddings[idx_list]
        sub_clusters = cluster_dedup(sub_emb, 0.8)
        added = 0
        for c in sub_clusters:
            global_idx = idx_list[cluster_medoid(c, sub_emb)]
            if global_idx not in out_set:
                out.append(global_idx)
                out_set.add(global_idx)
                added += 1
                if added >= need:
                    break
    return out

def apply_budget_gate(raw_skills: list[dict], embeddings_8b: np.ndarray, token_count_fn: Callable[[list[dict]], int]) -> tuple[list[dict], dict]:
    assert len(raw_skills) == len(embeddings_8b)
    by_type: dict[str, list[int]] = {}
    for (i, s) in enumerate(raw_skills):
        by_type.setdefault(s['source_task_type'], []).append(i)
    final_idx = None
    final_meta: dict = {}
    for thr in THRESHOLD_LADDER:
        clusters = cluster_dedup(embeddings_8b, thr)
        kept_idx = [cluster_medoid(c, embeddings_8b) for c in clusters]
        kept_idx = _enforce_coverage(kept_idx, raw_skills, embeddings_8b, by_type)
        kept = [raw_skills[i] for i in kept_idx]
        n = len(kept)
        tokens = token_count_fn(kept)
        log.info(f'  threshold={thr} -> N={n}, tokens={tokens}')
        if n <= N_MAX and tokens <= TOKEN_BUDGET:
            final_idx = kept_idx
            final_meta = {'final_dedup_threshold': thr, 'final_pool_size': n, 'full_library_token_count': tokens, 'per_type_count': _per_type_count_full(kept), 'coverage_invariant_satisfied': all((c >= PER_TYPE_MIN for c in _per_type_count_full(kept).values()))}
            break
    if final_idx is None:
        log.warning(f'  threshold ladder exhausted; per-type-medoid fallback')
        fallback_idx = []
        for (tt, idx_list) in by_type.items():
            if not idx_list:
                continue
            sub_emb = embeddings_8b[idx_list]
            sub_clusters = cluster_dedup(sub_emb, 0.8)
            sub_clusters.sort(key=lambda c: -len(c))
            for c in sub_clusters[:PER_TYPE_MEDOID_FALLBACK]:
                fallback_idx.append(idx_list[cluster_medoid(c, sub_emb)])
        kept = [raw_skills[i] for i in fallback_idx]
        tokens = token_count_fn(kept)
        final_idx = fallback_idx
        final_meta = {'final_dedup_threshold': 'per_type_medoid_fallback', 'final_pool_size': len(kept), 'full_library_token_count': tokens, 'per_type_count': _per_type_count_full(kept), 'coverage_invariant_satisfied': all((c >= PER_TYPE_MIN for c in _per_type_count_full(kept).values()))}
    final_pool = [raw_skills[i] for i in final_idx]
    counts = _per_type_count_full(final_pool)
    missing = [tt for tt in TASK_TYPES if counts[tt] < PER_TYPE_MIN]
    n = len(final_pool)
    tokens = final_meta['full_library_token_count']
    if missing:
        raise RuntimeError(f'BUDGET GATE FAIL: types {missing} have <{PER_TYPE_MIN} skills (counts={counts})')
    if n > N_MAX:
        raise RuntimeError(f'BUDGET GATE FAIL: N={n} > N_MAX={N_MAX}')
    if n < N_MIN:
        raise RuntimeError(f'BUDGET GATE FAIL: N={n} < N_MIN={N_MIN}')
    if tokens > TOKEN_BUDGET:
        raise RuntimeError(f'BUDGET GATE FAIL: tokens={tokens} > TOKEN_BUDGET={TOKEN_BUDGET}')
    log.info(f'  budget gate PASSED: meta={final_meta}')
    return (final_pool, final_meta)
