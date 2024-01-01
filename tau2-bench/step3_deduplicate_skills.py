import argparse
import json
import re
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

def extract_skill_text(code: str) -> str:
    name_match = re.search('SKILL_NAME\\s*=\\s*["\\\'](.+?)["\\\']', code)
    desc_match = re.search('SKILL_DESCRIPTION\\s*=\\s*(?:\\(\\s*)?["\\\'](.+?)["\\\']', code, re.DOTALL)
    target_match = re.search('TARGET_PATTERN\\s*=\\s*["\\\'](.+?)["\\\']', code, re.DOTALL)
    doc_match = re.search('"""(.+?)"""', code, re.DOTALL)
    parts = []
    if name_match:
        parts.append(name_match.group(1))
    if desc_match:
        parts.append(desc_match.group(1).strip())
    if target_match:
        parts.append(target_match.group(1).strip())
    if doc_match:
        parts.append(doc_match.group(1).strip())
    return '. '.join(parts) if parts else code[:500]

def load_skills(parsed_dir: Path) -> list[dict]:
    skills = []
    for f in sorted(parsed_dir.glob('skill_*.py')):
        code = f.read_text()
        name_match = re.search('SKILL_NAME\\s*=\\s*["\\\'](.+?)["\\\']', code)
        skill_name = name_match.group(1) if name_match else f.stem
        skills.append({'file': f.name, 'skill_name': skill_name, 'code': code, 'text': extract_skill_text(code)})
    return skills

def count_skill_frequency(raw_dir: Path) -> dict[str, int]:
    freq = {}
    for f in raw_dir.glob('task_*_t*.json'):
        data = json.loads(f.read_text())
        for skill in data.get('skills', []):
            name = skill.get('skill_name', '')
            if name:
                freq[name] = freq.get(name, 0) + 1
    return freq

def compute_embeddings(texts: list[str], model_name: str, device: str) -> np.ndarray:
    print(f'  Loading model: {model_name} on {device}...')
    try:
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer(model_name, device=device)
        embeddings = model.encode(texts, show_progress_bar=True, batch_size=32)
        return np.array(embeddings)
    except ImportError:
        pass
    import torch
    from transformers import AutoModel, AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).to(device)
    model.eval()
    embeddings = []
    batch_size = 16
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        inputs = tokenizer(batch, padding=True, truncation=True, max_length=512, return_tensors='pt').to(device)
        with torch.no_grad():
            outputs = model(**inputs)
            emb = outputs.last_hidden_state[:, 0, :]
        embeddings.append(emb.cpu().numpy())
    return np.vstack(embeddings)

def cluster_by_similarity(sim_matrix: np.ndarray, threshold: float) -> list[list[int]]:
    n = len(sim_matrix)
    assigned = [False] * n
    clusters = []
    neighbor_counts = [(i, np.sum(sim_matrix[i] > threshold)) for i in range(n)]
    neighbor_counts.sort(key=lambda x: -x[1])
    for (idx, _) in neighbor_counts:
        if assigned[idx]:
            continue
        cluster = [idx]
        assigned[idx] = True
        for j in range(n):
            if not assigned[j] and sim_matrix[idx][j] > threshold:
                cluster.append(j)
                assigned[j] = True
        clusters.append(cluster)
    return clusters

def plot_tsne(embeddings: np.ndarray, skills: list[dict], cluster_labels: list[int], output_dir: Path):
    print('  Computing t-SNE...')
    perplexity = min(30, len(embeddings) - 1)
    tsne = TSNE(n_components=2, random_state=42, perplexity=perplexity)
    coords = tsne.fit_transform(embeddings)
    (fig, ax) = plt.subplots(figsize=(14, 10))
    n_clusters = max(cluster_labels) + 1
    cmap = plt.cm.get_cmap('tab20', n_clusters)
    for cid in range(n_clusters):
        mask = np.array(cluster_labels) == cid
        if not mask.any():
            continue
        ax.scatter(coords[mask, 0], coords[mask, 1], c=[cmap(cid)], label=f'C{cid}' if mask.sum() > 1 else None, s=20 + np.array([skills[i]['frequency'] for i in range(len(skills)) if mask[i]]) * 3, alpha=0.7, edgecolors='white', linewidth=0.3)
    annotated = set()
    for (i, s) in enumerate(skills):
        if s.get('is_representative') and cluster_labels[i] not in annotated:
            annotated.add(cluster_labels[i])
            name = s['skill_name']
            if len(name) > 30:
                name = name[:27] + '...'
            ax.annotate(name, (coords[i, 0], coords[i, 1]), fontsize=6, alpha=0.8, xytext=(5, 5), textcoords='offset points')
    ax.set_title(f'Skill Embedding Space (t-SNE) — {n_clusters} clusters from {len(skills)} skills', fontsize=13)
    ax.set_xlabel('t-SNE 1')
    ax.set_ylabel('t-SNE 2')
    if n_clusters <= 20:
        ax.legend(loc='best', fontsize=7, ncol=2, markerscale=0.8)
    plt.tight_layout()
    fig.savefig(output_dir / 'tsne_clusters.png', dpi=150)
    plt.close(fig)
    print(f"  Saved: {output_dir / 'tsne_clusters.png'}")

def plot_similarity_heatmap(sim_matrix: np.ndarray, skills: list[dict], cluster_labels: list[int], output_dir: Path):
    order = sorted(range(len(skills)), key=lambda i: (cluster_labels[i], -skills[i]['frequency']))
    sorted_sim = sim_matrix[np.ix_(order, order)]
    (fig, ax) = plt.subplots(figsize=(12, 10))
    im = ax.imshow(sorted_sim, cmap='YlOrRd', vmin=0, vmax=1, aspect='auto')
    fig.colorbar(im, ax=ax, label='Cosine Similarity', shrink=0.8)
    cluster_ids_sorted = [cluster_labels[i] for i in order]
    boundaries = []
    for i in range(1, len(cluster_ids_sorted)):
        if cluster_ids_sorted[i] != cluster_ids_sorted[i - 1]:
            boundaries.append(i)
    for b in boundaries:
        ax.axhline(y=b - 0.5, color='blue', linewidth=0.5, alpha=0.5)
        ax.axvline(x=b - 0.5, color='blue', linewidth=0.5, alpha=0.5)
    ax.set_title(f'Pairwise Cosine Similarity ({len(skills)} skills, {max(cluster_labels) + 1} clusters)', fontsize=13)
    ax.set_xlabel('Skill index (sorted by cluster)')
    ax.set_ylabel('Skill index (sorted by cluster)')
    plt.tight_layout()
    fig.savefig(output_dir / 'similarity_heatmap.png', dpi=150)
    plt.close(fig)
    print(f"  Saved: {output_dir / 'similarity_heatmap.png'}")

def plot_frequency_distribution(skills: list[dict], cluster_data: list[dict], output_dir: Path):
    (fig, axes) = plt.subplots(1, 2, figsize=(14, 5))
    freqs = [s['frequency'] for s in skills]
    axes[0].hist(freqs, bins=range(1, max(freqs) + 2), color='steelblue', edgecolor='white', alpha=0.8)
    axes[0].set_title('Individual Skill Frequency Distribution')
    axes[0].set_xlabel('Frequency (appearances across tasks × temperatures)')
    axes[0].set_ylabel('Count')
    sizes = [c['size'] for c in cluster_data]
    axes[1].hist(sizes, bins=range(1, max(sizes) + 2), color='coral', edgecolor='white', alpha=0.8)
    axes[1].set_title('Cluster Size Distribution')
    axes[1].set_xlabel('Number of skills in cluster')
    axes[1].set_ylabel('Count')
    plt.tight_layout()
    fig.savefig(output_dir / 'frequency_distribution.png', dpi=150)
    plt.close(fig)
    print(f"  Saved: {output_dir / 'frequency_distribution.png'}")

def main():
    parser = argparse.ArgumentParser(description='Step 3: Deduplicate skills via embedding clustering')
    parser.add_argument('--input-dir', required=True, help='Directory with parsed skill .py files')
    parser.add_argument('--raw-dir', required=True, help='Directory with raw LLM outputs')
    parser.add_argument('--output-dir', required=True, help='Output directory')
    parser.add_argument('--model', default='Qwen/Qwen3-Embedding-8B', help='Embedding model name or path')
    parser.add_argument('--device', default='cuda', choices=['cuda', 'cpu'])
    parser.add_argument('--threshold', type=float, default=0.85, help='Cosine similarity threshold for clustering')
    parser.add_argument('--no-viz', action='store_true', help='Skip visualization generation')
    args = parser.parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    parsed_dir = Path(args.input_dir)
    skills = load_skills(parsed_dir)
    print(f'Loaded {len(skills)} skills from {parsed_dir}')
    if len(skills) == 0:
        print('No skills to deduplicate.')
        return
    raw_dir = Path(args.raw_dir)
    freq = count_skill_frequency(raw_dir)
    for skill in skills:
        skill['frequency'] = freq.get(skill['skill_name'], 1)
    print(f"Frequency range: {min((s['frequency'] for s in skills))} - {max((s['frequency'] for s in skills))}")
    texts = [s['text'] for s in skills]
    embeddings = compute_embeddings(texts, args.model, args.device)
    print(f'Embeddings shape: {embeddings.shape}')
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-08)
    normalized = embeddings / norms
    sim_matrix = normalized @ normalized.T
    sim_off_diag = sim_matrix.copy()
    np.fill_diagonal(sim_off_diag, 0)
    print(f'Similarity matrix: min={sim_matrix.min():.3f}, mean={sim_matrix.mean():.3f}, max off-diag={sim_off_diag.max():.3f}')
    np.save(output_dir / 'similarity_matrix.npy', sim_matrix)
    clusters = cluster_by_similarity(sim_matrix, args.threshold)
    print(f'\nClusters: {len(clusters)} (from {len(skills)} skills, threshold={args.threshold})')
    cluster_data = []
    for (i, cluster_indices) in enumerate(clusters):
        members = []
        for idx in cluster_indices:
            s = skills[idx]
            members.append({'file': s['file'], 'skill_name': s['skill_name'], 'frequency': s['frequency'], 'text_preview': s['text'][:200]})
        members.sort(key=lambda m: -m['frequency'])
        cluster_data.append({'cluster_id': i, 'size': len(cluster_indices), 'representative': members[0]['skill_name'], 'representative_file': members[0]['file'], 'total_frequency': sum((m['frequency'] for m in members)), 'members': members})
    cluster_data.sort(key=lambda c: -c['total_frequency'])
    cluster_json = {'total_skills': len(skills), 'total_clusters': len(clusters), 'threshold': args.threshold, 'model': args.model, 'clusters': cluster_data}
    (output_dir / 'clusters.json').write_text(json.dumps(cluster_json, indent=2, ensure_ascii=False))
    report_lines = [f'# Skill Deduplication Report', f'', f'- **Total skills**: {len(skills)}', f'- **Clusters**: {len(clusters)}', f'- **Threshold**: {args.threshold}', f'- **Model**: {args.model}', f'', f'## Cluster Summary', f'', f'| # | Representative | Size | Total Freq | Members |', f'|---|---|---|---|---|']
    for c in cluster_data:
        member_names = ', '.join((m['skill_name'] for m in c['members'][:5]))
        if len(c['members']) > 5:
            member_names += f" (+{len(c['members']) - 5} more)"
        report_lines.append(f"| {c['cluster_id']} | `{c['representative']}` | {c['size']} | {c['total_frequency']} | {member_names} |")
    report_lines.extend(['', '## Cluster Details', ''])
    for c in cluster_data:
        report_lines.append(f"### Cluster {c['cluster_id']}: `{c['representative']}`")
        report_lines.append(f'')
        for m in c['members']:
            report_lines.append(f"- **{m['skill_name']}** (freq={m['frequency']}, file={m['file']})")
            report_lines.append(f"  {m['text_preview']}")
        report_lines.append('')
    (output_dir / 'cluster_report.md').write_text('\n'.join(report_lines))
    final_skills = []
    for c in cluster_data:
        rep_file = c['representative_file']
        rep_code = (parsed_dir / rep_file).read_text()
        final_skills.append({'rank': len(final_skills), 'skill_name': c['representative'], 'file': rep_file, 'cluster_id': c['cluster_id'], 'cluster_size': c['size'], 'frequency': c['total_frequency'], 'code': rep_code})
    (output_dir / 'skills_ranked.json').write_text(json.dumps(final_skills, indent=2, ensure_ascii=False))
    print(f'\nTop 10 skills by frequency:')
    for s in final_skills[:10]:
        print(f"  #{s['rank']:2d} [{s['frequency']:3d}x] {s['skill_name']} (cluster size={s['cluster_size']})")
    if not args.no_viz:
        print('\nGenerating visualizations...')
        cluster_labels = [0] * len(skills)
        rep_set = set()
        for c in cluster_data:
            for m in c['members']:
                for (idx, s) in enumerate(skills):
                    if s['file'] == m['file']:
                        cluster_labels[idx] = c['cluster_id']
                        if m['skill_name'] == c['representative']:
                            s['is_representative'] = True
                            rep_set.add(idx)
        plot_tsne(embeddings, skills, cluster_labels, output_dir)
        plot_similarity_heatmap(sim_matrix, skills, cluster_labels, output_dir)
        plot_frequency_distribution(skills, cluster_data, output_dir)
    print(f'\nOutput: {output_dir}')
    print(f'  clusters.json      — full cluster data')
    print(f'  cluster_report.md  — human-readable report')
    print(f'  skills_ranked.json — final ranked skill list')
    print(f'  similarity_matrix.npy — pairwise similarity')
    if not args.no_viz:
        print(f'  tsne_clusters.png  — t-SNE cluster visualization')
        print(f'  similarity_heatmap.png — pairwise similarity heatmap')
        print(f'  frequency_distribution.png — frequency & cluster size histograms')
if __name__ == '__main__':
    main()
