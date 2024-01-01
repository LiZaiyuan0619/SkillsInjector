import argparse
import json
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from scipy import stats

def load_report(path):
    with open(path) as f:
        data = json.load(f)
    return data

def detect_num_trials(report):
    for r in report.get('utility_ranking', []):
        pc = r.get('pass_counts', [])
        if pc:
            return len(pc)
    return 3

def shorten_name(name, max_len=35):
    if len(name) <= max_len:
        return name
    return name[:max_len - 2] + '..'

def fig1_utility_ranking_bar(report, outdir, num_trials=5):
    ranking = report['utility_ranking']
    baseline = report['experiment_info']['baseline_pass_rate_mean']
    names = [r['skill_name'] for r in ranking]
    deltas = [r['delta_pass_rate_mean'] * 100 for r in ranking]
    freqs = [r['frequency'] for r in ranking]
    colors = ['#2ecc71' if d > 0 else '#e74c3c' if d < 0 else '#95a5a6' for d in deltas]
    (fig, ax) = plt.subplots(figsize=(14, 24))
    y_pos = np.arange(len(names))
    bars = ax.barh(y_pos, deltas, color=colors, height=0.7, edgecolor='white', linewidth=0.3)
    short_names = [shorten_name(n) for n in names]
    ax.set_yticks(y_pos)
    ax.set_yticklabels([f'{short_names[i]}  (f={freqs[i]})' for i in range(len(names))], fontsize=7, fontfamily='monospace')
    ax.invert_yaxis()
    ax.axvline(x=0, color='black', linewidth=0.8)
    for (i, (d, bar)) in enumerate(zip(deltas, bars)):
        if d >= 0:
            ax.text(d + 0.3, i, f'+{d:.1f}%', va='center', fontsize=6, color='#27ae60')
        else:
            ax.text(d - 0.3, i, f'{d:.1f}%', va='center', fontsize=6, color='#c0392b', ha='right')
    ax.set_xlabel(f'Δ Pass@{num_trials} (percentage points)', fontsize=12)
    domain = report['experiment_info'].get('domain', 'Airline').capitalize()
    ax.set_title(f'Single-Skill Utility Ranking — {domain} (baseline={baseline * 100:.1f}%)\n{len(ranking)} skills, sorted by Δpass@{num_trials} mean', fontsize=14, fontweight='bold')
    ax.xaxis.grid(True, alpha=0.3)
    ax.set_axisbelow(True)
    plt.tight_layout()
    path = os.path.join(outdir, 'fig1_single_skill_utility_ranking.png')
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'  Saved: {path}')

def fig2_freq_vs_utility(report, outdir, num_trials=5):
    ranking = report['utility_ranking']
    freqs = np.array([r['frequency'] for r in ranking])
    deltas = np.array([r['delta_pass_rate_mean'] * 100 for r in ranking])
    names = [r['skill_name'] for r in ranking]
    (fig, ax) = plt.subplots(figsize=(12, 8))
    colors = ['#2ecc71' if d > 0 else '#e74c3c' if d < 0 else '#95a5a6' for d in deltas]
    scatter = ax.scatter(freqs, deltas, c=colors, s=50, alpha=0.7, edgecolors='white', linewidth=0.5, zorder=5)
    ax.set_xscale('log')
    log_freqs = np.log10(freqs)
    (slope, intercept, r_value, p_value, std_err) = stats.linregress(log_freqs, deltas)
    x_line = np.linspace(log_freqs.min(), log_freqs.max(), 100)
    y_line = slope * x_line + intercept
    ax.plot(10 ** x_line, y_line, 'r--', alpha=0.5, linewidth=1.5, label=f'Linear fit (r={r_value:.3f}, p={p_value:.3f})')
    (spearman_r, spearman_p) = stats.spearmanr(freqs, deltas)
    for i in range(min(5, len(names))):
        ax.annotate(shorten_name(names[i], 25), (freqs[i], deltas[i]), textcoords='offset points', xytext=(8, 5), fontsize=7, color='#2c3e50', alpha=0.8)
    for i in range(max(0, len(names) - 3), len(names)):
        ax.annotate(shorten_name(names[i], 25), (freqs[i], deltas[i]), textcoords='offset points', xytext=(8, -8), fontsize=7, color='#c0392b', alpha=0.8)
    ax.axhline(y=0, color='gray', linewidth=0.8, linestyle='--')
    ax.set_xlabel('Frequency (log scale)', fontsize=12)
    ax.set_ylabel(f'Δ Pass@{num_trials} (percentage points)', fontsize=12)
    ax.set_title(f'Frequency vs Single-Skill Utility\nSpearman ρ = {spearman_r:.3f} (p = {spearman_p:.3f})', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    path = os.path.join(outdir, 'fig2_freq_vs_utility_scatter.png')
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'  Saved: {path}')
    return (spearman_r, spearman_p)

def fig3_utility_distribution(report, outdir, num_trials=5):
    ranking = report['utility_ranking']
    deltas = [r['delta_pass_rate_mean'] * 100 for r in ranking]
    (fig, ax) = plt.subplots(figsize=(10, 6))
    bins = np.arange(-6, 20, 1)
    (n, bins_out, patches) = ax.hist(deltas, bins=bins, edgecolor='white', linewidth=0.8, alpha=0.85)
    for (patch, left_edge) in zip(patches, bins_out[:-1]):
        if left_edge >= 0:
            patch.set_facecolor('#2ecc71')
        else:
            patch.set_facecolor('#e74c3c')
    mean_d = np.mean(deltas)
    median_d = np.median(deltas)
    positive = sum((1 for d in deltas if d > 0))
    negative = sum((1 for d in deltas if d < 0))
    neutral = sum((1 for d in deltas if d == 0))
    ax.axvline(mean_d, color='#e67e22', linewidth=2, linestyle='--', label=f'Mean = +{mean_d:.1f}%')
    ax.axvline(median_d, color='#3498db', linewidth=2, linestyle='-.', label=f'Median = +{median_d:.1f}%')
    ax.axvline(0, color='black', linewidth=0.8)
    ax.set_xlabel(f'Δ Pass@{num_trials} (percentage points)', fontsize=12)
    ax.set_ylabel('Number of Skills', fontsize=12)
    domain = report['experiment_info'].get('domain', 'Airline').capitalize()
    ax.set_title(f'Distribution of Single-Skill Utility — {domain}\nPositive: {positive} | Neutral: {neutral} | Negative: {negative}', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    path = os.path.join(outdir, 'fig3_utility_distribution.png')
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'  Saved: {path}')

def fig4_rank_comparison(report, outdir):
    ranking = report['utility_ranking']
    util_rank = {r['skill_name']: i + 1 for (i, r) in enumerate(ranking)}
    freq_sorted = sorted(ranking, key=lambda r: r['frequency'], reverse=True)
    freq_rank = {r['skill_name']: i + 1 for (i, r) in enumerate(freq_sorted)}
    names = [r['skill_name'] for r in ranking]
    n = len(names)
    util_ranks = [util_rank[name] for name in names]
    freq_ranks = [freq_rank[name] for name in names]
    (spearman_r, spearman_p) = stats.spearmanr(util_ranks, freq_ranks)
    (fig, ax) = plt.subplots(figsize=(10, 14))
    highlight_top = 10
    for name in names:
        fr = freq_rank[name]
        ur = util_rank[name]
        rank_diff = abs(fr - ur)
        if ur <= highlight_top:
            color = '#e74c3c' if fr > highlight_top else '#2ecc71'
            alpha = 0.8
            lw = 1.5
        elif fr <= highlight_top:
            color = '#3498db'
            alpha = 0.7
            lw = 1.2
        else:
            color = '#bdc3c7'
            alpha = 0.3
            lw = 0.5
        ax.plot([0, 1], [fr, ur], color=color, alpha=alpha, linewidth=lw)
    for name in names[:10]:
        fr = freq_rank[name]
        ur = util_rank[name]
        short = shorten_name(name, 28)
        ax.text(1.02, ur, f'{ur}. {short}', fontsize=6.5, va='center', color='#2c3e50')
        ax.text(-0.02, fr, f'{fr}.', fontsize=6.5, va='center', ha='right', color='#7f8c8d')
    for r in freq_sorted[:10]:
        name = r['skill_name']
        if util_rank[name] > 10:
            fr = freq_rank[name]
            ur = util_rank[name]
            short = shorten_name(name, 28)
            ax.text(-0.02, fr, f'{fr}. {short}', fontsize=6.5, va='center', ha='right', color='#3498db')
    ax.set_xlim(-0.4, 1.5)
    ax.set_ylim(n + 1, 0)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(['Frequency Rank', 'Utility Rank'], fontsize=13, fontweight='bold')
    ax.set_ylabel('Rank Position', fontsize=12)
    ax.set_title(f'Frequency Rank vs Utility Rank\nSpearman ρ = {spearman_r:.3f} (p = {spearman_p:.4f})\nGreen: top-10 utility & freq | Red: top-10 utility but low freq | Blue: top-10 freq but low utility', fontsize=11, fontweight='bold')
    for y in range(0, n + 1, 10):
        ax.axhline(y, color='#ecf0f1', linewidth=0.5)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    plt.tight_layout()
    path = os.path.join(outdir, 'fig4_freq_vs_util_rank_comparison.png')
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'  Saved: {path}')
    return spearman_r

def fig5_task_impact(report, outdir):
    ranking = report['utility_ranking']
    top_n = min(30, len(ranking))
    subset = ranking[:top_n]
    names = [shorten_name(r['skill_name'], 30) for r in subset]
    improved = [len(r.get('improved_tasks', [])) for r in subset]
    degraded = [-len(r.get('degraded_tasks', [])) for r in subset]
    net = [r.get('net_task_change', 0) for r in subset]
    (fig, ax) = plt.subplots(figsize=(14, 10))
    y_pos = np.arange(top_n)
    ax.barh(y_pos, improved, color='#2ecc71', alpha=0.8, height=0.6, label='Improved tasks')
    ax.barh(y_pos, degraded, color='#e74c3c', alpha=0.8, height=0.6, label='Degraded tasks')
    for i in range(top_n):
        ax.text(max(improved[i], 0) + 0.5, i, f'net={net[i]:+d}', va='center', fontsize=7, color='#2c3e50')
    ax.set_yticks(y_pos)
    ax.set_yticklabels(names, fontsize=7.5, fontfamily='monospace')
    ax.invert_yaxis()
    ax.axvline(0, color='black', linewidth=0.8)
    ax.set_xlabel('Number of Tasks', fontsize=12)
    ax.set_title(f'Per-Skill Task Impact (Top-{top_n} by Utility)\nGreen = improved tasks | Red = degraded tasks', fontsize=13, fontweight='bold')
    ax.legend(fontsize=10, loc='lower right')
    ax.grid(axis='x', alpha=0.3)
    plt.tight_layout()
    path = os.path.join(outdir, 'fig5_task_impact_top30.png')
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'  Saved: {path}')

def print_summary_stats(report, num_trials=5):
    ranking = report['utility_ranking']
    baseline = report['experiment_info']['baseline_pass_rate_mean']
    deltas = [r['delta_pass_rate_mean'] * 100 for r in ranking]
    freqs = [r['frequency'] for r in ranking]
    positive = sum((1 for d in deltas if d > 0))
    negative = sum((1 for d in deltas if d < 0))
    neutral = sum((1 for d in deltas if d == 0))
    (spearman_r, spearman_p) = stats.spearmanr(freqs, deltas)
    print(f"\n{'=' * 70}")
    print(f'  SUMMARY STATISTICS')
    print(f"{'=' * 70}")
    print(f'  Baseline pass@{num_trials} mean: {baseline * 100:.1f}%')
    print(f'  Total skills: {len(ranking)}')
    print(f'  Positive (Δ>0): {positive}  |  Neutral (Δ=0): {neutral}  |  Negative (Δ<0): {negative}')
    print(f'  Mean Δ: +{np.mean(deltas):.2f}%  |  Median Δ: +{np.median(deltas):.2f}%')
    print(f"  Max Δ: +{max(deltas):.1f}% ({ranking[0]['skill_name']})")
    print(f"  Min Δ: {min(deltas):.1f}% ({ranking[-1]['skill_name']})")
    print(f'  Frequency-Utility Spearman ρ: {spearman_r:.3f} (p={spearman_p:.4f})')
    print(f"{'=' * 70}")
    freq_sorted = sorted(ranking, key=lambda r: r['frequency'], reverse=True)
    util_rank = {r['skill_name']: i + 1 for (i, r) in enumerate(ranking)}
    print(f'\n  Top-10 by FREQUENCY → their Utility Rank:')
    for (i, r) in enumerate(freq_sorted[:10]):
        ur = util_rank[r['skill_name']]
        print(f"    freq_rank={i + 1:2d}  freq={r['frequency']:5d}  util_rank={ur:2d}  Δ={r['delta_pass_rate_mean'] * 100:+.1f}%  {r['skill_name']}")
    print(f'\n  Top-10 by UTILITY → their Frequency Rank:')
    freq_rank = {r['skill_name']: i + 1 for (i, r) in enumerate(freq_sorted)}
    for (i, r) in enumerate(ranking[:10]):
        fr = freq_rank[r['skill_name']]
        print(f"    util_rank={i + 1:2d}  Δ={r['delta_pass_rate_mean'] * 100:+.1f}%  freq_rank={fr:2d}  freq={r['frequency']:5d}  {r['skill_name']}")
    print()

def main():
    parser = argparse.ArgumentParser(description='Visualize single-skill utility results')
    parser.add_argument('--report', required=True, help='Path to single_skill_report JSON')
    parser.add_argument('--outdir', default=None, help='Output directory for figures')
    args = parser.parse_args()
    report = load_report(args.report)
    if args.outdir is None:
        args.outdir = os.path.join(os.path.dirname(args.report), 'figures_single_skill')
    os.makedirs(args.outdir, exist_ok=True)
    num_trials = detect_num_trials(report)
    print(f"Loaded report: {len(report['utility_ranking'])} skills, {num_trials} trials")
    print(f'Output directory: {args.outdir}\n')
    print_summary_stats(report, num_trials)
    print('\nGenerating figures...')
    fig1_utility_ranking_bar(report, args.outdir, num_trials)
    fig2_freq_vs_utility(report, args.outdir, num_trials)
    fig3_utility_distribution(report, args.outdir, num_trials)
    fig4_rank_comparison(report, args.outdir)
    fig5_task_impact(report, args.outdir)
    print(f'\nDone! All figures saved to: {args.outdir}')
if __name__ == '__main__':
    main()
