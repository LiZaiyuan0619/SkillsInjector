import argparse
import json
from pathlib import Path
import numpy as np

def load_ranked_skills(input_dir: Path) -> list[dict]:
    path = input_dir / 'skills_ranked.json'
    if not path.exists():
        raise FileNotFoundError(f'skills_ranked.json not found in {input_dir}')
    return json.loads(path.read_text())

def load_similarity_matrix(input_dir: Path) -> np.ndarray:
    path = input_dir / 'similarity_matrix.npy'
    if not path.exists():
        raise FileNotFoundError(f'similarity_matrix.npy not found in {input_dir}')
    return np.load(path)

def build_file_index(parsed_dir: Path) -> dict[str, int]:
    parsed_files = sorted(parsed_dir.glob('skill_*.py'))
    if not parsed_files:
        raise FileNotFoundError(f'No skill_*.py files found in {parsed_dir}')
    return {f.name: i for (i, f) in enumerate(parsed_files)}

def deduplicate_by_name(ranked_skills: list[dict]) -> list[dict]:
    seen = {}
    for skill in ranked_skills:
        name = skill['skill_name']
        if name not in seen or skill['frequency'] > seen[name]['frequency']:
            seen[name] = skill
    return sorted(seen.values(), key=lambda s: -s['frequency'])

def normalize_frequencies(candidates: list[dict]) -> np.ndarray:
    freqs = np.array([c['frequency'] for c in candidates], dtype=float)
    log_freqs = np.log1p(freqs)
    (lo, hi) = (log_freqs.min(), log_freqs.max())
    if hi - lo < 1e-09:
        return np.ones(len(candidates))
    return (log_freqs - lo) / (hi - lo)

def mmr_rerank(candidates: list[dict], sim_sub: np.ndarray, freq_norm: np.ndarray, lam: float) -> list[dict]:
    n = len(candidates)
    selected_indices: list[int] = []
    remaining = set(range(n))
    result: list[dict] = []
    for step in range(n):
        if step == 0:
            best_idx = int(np.argmax(freq_norm))
        else:
            best_score = -float('inf')
            best_idx = -1
            sel_arr = np.array(selected_indices)
            for idx in remaining:
                max_sim = float(np.max(sim_sub[idx, sel_arr]))
                diversity = 1.0 - max_sim
                score = lam * freq_norm[idx] + (1.0 - lam) * diversity
                if score > best_score:
                    best_score = score
                    best_idx = idx
        remaining.discard(best_idx)
        selected_indices.append(best_idx)
        entry = dict(candidates[best_idx])
        entry['rank'] = step
        entry['mmr_score'] = round(lam * freq_norm[best_idx] + (1.0 - lam) * (1.0 - float(np.max(sim_sub[best_idx, selected_indices[:-1]])) if len(selected_indices) > 1 else 1.0), 4)
        entry['freq_norm'] = round(float(freq_norm[best_idx]), 4)
        entry['diversity_score'] = round(1.0 - float(np.max(sim_sub[best_idx, selected_indices[:-1]])) if len(selected_indices) > 1 else 1.0, 4)
        result.append(entry)
    return result

def print_comparison(naive: list[dict], mmr: list[dict], top_n: int=32):
    top_n = min(top_n, len(naive), len(mmr))
    naive_names = [s['skill_name'] for s in naive[:top_n]]
    mmr_names = [s['skill_name'] for s in mmr[:top_n]]
    print(f"\n{'=' * 80}")
    print(f'  RANKING COMPARISON (Top-{top_n})')
    print(f"{'=' * 80}")
    print(f"  {'Rank':>4}  {'Naive (frequency)':40s}  {'MMR (λ-balanced)':40s}")
    print(f"  {'-' * 4}  {'-' * 40}  {'-' * 40}")
    for i in range(top_n):
        n_name = naive_names[i] if i < len(naive_names) else ''
        m_name = mmr_names[i] if i < len(mmr_names) else ''
        marker = ' ' if n_name == m_name else '*'
        print(f'  {i + 1:>4}  {n_name:40s}  {m_name:40s} {marker}')
    print(f'\n  Overlap:')
    for n in [4, 8, 16, 32]:
        if n > top_n:
            break
        naive_set = set(naive_names[:n])
        mmr_set = set(mmr_names[:n])
        overlap = len(naive_set & mmr_set)
        only_naive = naive_set - mmr_set
        only_mmr = mmr_set - naive_set
        print(f'    Top-{n:>3}: {overlap}/{n} shared')
        if only_mmr:
            print(f"            MMR adds:   {', '.join(sorted(only_mmr))}")
            print(f"            Naive adds: {', '.join(sorted(only_naive))}")
    print(f"\n{'=' * 80}")

def print_diversity_stats(mmr: list[dict], sim_sub: np.ndarray, candidate_idx_map: dict[str, int]):
    print(f'\n  Diversity stats (avg pairwise similarity — lower = more diverse):')
    for n in [4, 8, 16, 32, 64]:
        if n > len(mmr):
            break
        indices = [candidate_idx_map[s['file']] for s in mmr[:n]]
        sims = []
        for i in range(len(indices)):
            for j in range(i + 1, len(indices)):
                sims.append(sim_sub[indices[i], indices[j]])
        avg = np.mean(sims) if sims else 0
        print(f'    Top-{n:>3}: avg_sim = {avg:.3f}')

def main():
    parser = argparse.ArgumentParser(description='Step 3.1: Rerank skills using MMR (diversity-aware)')
    parser.add_argument('--input-dir', required=True, help='Dedup output directory (contains skills_ranked.json and similarity_matrix.npy)')
    parser.add_argument('--parsed-dir', required=True, help='Parsed skill directory (for file→index mapping)')
    parser.add_argument('--lambda', dest='lam', type=float, default=0.5, help='Trade-off: 1.0=pure frequency, 0.0=pure diversity (default: 0.5)')
    parser.add_argument('--output', default=None, help='Output file path (default: <input-dir>/skills_reranked_mmr.json)')
    args = parser.parse_args()
    input_dir = Path(args.input_dir)
    parsed_dir = Path(args.parsed_dir)
    output_path = Path(args.output) if args.output else input_dir / 'skills_reranked_mmr.json'
    lam = args.lam
    print(f'Loading data from {input_dir}...')
    ranked_skills = load_ranked_skills(input_dir)
    sim_full = load_similarity_matrix(input_dir)
    file_to_idx = build_file_index(parsed_dir)
    print(f'  skills_ranked.json: {len(ranked_skills)} entries')
    print(f'  similarity_matrix: {sim_full.shape}')
    print(f'  parsed skills: {len(file_to_idx)} files')
    candidates = deduplicate_by_name(ranked_skills)
    print(f'\n  After name-dedup: {len(candidates)} unique skills (removed {len(ranked_skills) - len(candidates)} same-name duplicates)')
    cand_matrix_indices = []
    valid_candidates = []
    for c in candidates:
        idx = file_to_idx.get(c['file'])
        if idx is not None:
            cand_matrix_indices.append(idx)
            valid_candidates.append(c)
        else:
            print(f"  WARNING: file '{c['file']}' not found in parsed dir, skipping")
    candidates = valid_candidates
    cand_matrix_indices = np.array(cand_matrix_indices)
    sim_sub = sim_full[np.ix_(cand_matrix_indices, cand_matrix_indices)]
    print(f'  Candidate sub-matrix: {sim_sub.shape}')
    cand_idx_map = {c['file']: i for (i, c) in enumerate(candidates)}
    freq_norm = normalize_frequencies(candidates)
    print(f"  Frequency range: {min((c['frequency'] for c in candidates))} ~ {max((c['frequency'] for c in candidates))}")
    print(f'  Freq norm range: {freq_norm.min():.3f} ~ {freq_norm.max():.3f}')
    print(f'\nRunning MMR reranking (λ={lam})...')
    mmr_ranked = mmr_rerank(candidates, sim_sub, freq_norm, lam)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(mmr_ranked, indent=2, ensure_ascii=False))
    print(f'\nSaved: {output_path}')
    print(f'  {len(mmr_ranked)} skills, λ={lam}')
    print_comparison(candidates, mmr_ranked)
    print_diversity_stats(mmr_ranked, sim_sub, cand_idx_map)
    print(f'\n  (Naive frequency ranking for reference):')
    for n in [4, 8, 16, 32, 64]:
        if n > len(candidates):
            break
        indices = list(range(n))
        sims = []
        for i in range(len(indices)):
            for j in range(i + 1, len(indices)):
                sims.append(sim_sub[indices[i], indices[j]])
        avg = np.mean(sims) if sims else 0
        print(f'    Top-{n:>3}: avg_sim = {avg:.3f}')
    print(f"\n{'=' * 80}")
    print(f'  Done. Use with step4:')
    print(f'    python temp/step4_run_scaling_experiment.py \\')
    print(f'        --ranked-json {output_path} \\')
    print(f'        --domain <domain> --scales 1 2 4 8 16 32 64')
    print(f"{'=' * 80}")
if __name__ == '__main__':
    main()
