import argparse
import json
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
DEFAULT_NAIVE_DIR = '<MODEL_ROOT>/agent/skills/scaling-organized'
DEFAULT_MMR_DIR = '<MODEL_ROOT>/agent/skills/scaling-organized-mmr'
DEFAULT_BASELINE_DIR = str(Path(__file__).resolve().parent / 'results')
BASELINE_FILENAME_MAP = {'airline': 'baseline_agg3_*.json', 'retail': 'retail_baseline_agg3_*.json', 'telecom': 'telecom_baseline_agg3_*.json'}
DOMAIN_LABELS = {'airline': 'Airline', 'retail': 'Retail', 'telecom': 'Telecom'}
NAIVE_COLOR = '#5C6BC0'
MMR_COLOR = '#E53935'
BASELINE_COLOR = '#757575'

def load_baseline(baseline_dir: Path, domain: str) -> dict | None:
    pattern = BASELINE_FILENAME_MAP.get(domain)
    if not pattern:
        return None
    files = sorted(baseline_dir.glob(pattern))
    if not files:
        return None
    data = json.loads(files[-1].read_text())
    s = data['summary']
    return {'n_skills': 0, 'pass_rate_mean': s.get('pass_rate_mean', s.get('pass_rate', 0)), 'pass_rate_std': s.get('pass_rate_std', 0), 'file': files[-1].name}

def load_scaling_data(data_dir: Path, domains: list[str] | None) -> dict[str, list[dict]]:
    result = {}
    for domain_dir in sorted(data_dir.iterdir()):
        if not domain_dir.is_dir():
            continue
        domain = domain_dir.name
        if domain == 'figures':
            continue
        if domains and domain not in domains:
            continue
        report_files = sorted(domain_dir.glob('scaling_report_*.json'))
        if not report_files:
            continue
        report = json.loads(report_files[-1].read_text())
        result[domain] = report['scaling_summary']
    return result

def make_xaxis_positions(all_ns: list[int]) -> tuple[list[float], list[int]]:
    sorted_ns = sorted(set(all_ns))
    if len(sorted_ns) <= 1:
        return ([0.0], sorted_ns)
    positions = [0.0]
    for i in range(1, len(sorted_ns)):
        (prev, curr) = (sorted_ns[i - 1], sorted_ns[i])
        if prev > 0 and curr > 0:
            ratio = np.log2(curr / prev)
        else:
            ratio = 1.0
        stretch = 1.0 + 0.15 * max(ratio - 1.0, 0)
        positions.append(positions[-1] + stretch)
    return (positions, sorted_ns)

def setup_even_xaxis(ax, all_ns: list[int], xlabel: str='Number of Injected Skills'):
    (positions, sorted_ns) = make_xaxis_positions(all_ns)
    n_to_pos = dict(zip(sorted_ns, positions))
    ax.set_xticks(positions)
    ax.set_xticklabels([str(n) for n in sorted_ns], fontsize=9)
    ax.set_xlabel(xlabel, fontsize=11)
    if len(positions) > 1:
        pad = (positions[-1] - positions[0]) * 0.05
        ax.set_xlim(positions[0] - pad, positions[-1] + pad)
    return n_to_pos

def plot_comparison_curves(naive_data: dict, mmr_data: dict, baselines: dict, output_dir: Path):
    domains = sorted(set(naive_data.keys()) & set(mmr_data.keys()))
    n_domains = len(domains)
    if n_domains == 0:
        print('  WARNING: No common domains found, skipping comparison curves.')
        return
    (fig, axes) = plt.subplots(1, n_domains, figsize=(6 * n_domains, 5.5), squeeze=False)
    axes = axes[0]
    for (idx, domain) in enumerate(domains):
        ax = axes[idx]
        naive_rows = naive_data[domain]
        mmr_rows = mmr_data[domain]
        all_ns = sorted(set([r['n_skills'] for r in naive_rows] + [r['n_skills'] for r in mmr_rows]))
        bl = baselines.get(domain)
        if bl:
            all_ns = sorted(set([0] + all_ns))
        n_to_pos = setup_even_xaxis(ax, all_ns)
        if bl:
            bl_rate = bl['pass_rate_mean'] * 100
            bl_std = bl['pass_rate_std'] * 100
            ax.axhspan(bl_rate - bl_std, bl_rate + bl_std, color=BASELINE_COLOR, alpha=0.1, zorder=0)
            ax.axhline(y=bl_rate, color=BASELINE_COLOR, linestyle=':', linewidth=1.2, alpha=0.6, zorder=1)
            ax.plot(n_to_pos[0], bl_rate, marker='*', color=BASELINE_COLOR, markersize=13, zorder=5, markeredgecolor='white', markeredgewidth=0.8)
            ax.text(n_to_pos[0] + 0.15, bl_rate + 1.2, f'baseline\n{bl_rate:.1f}%', fontsize=8, color=BASELINE_COLOR, va='bottom', ha='left')
        naive_ns = [r['n_skills'] for r in naive_rows]
        naive_pos = [n_to_pos[n] for n in naive_ns]
        naive_means = [r['pass_rate_mean'] * 100 for r in naive_rows]
        naive_stds = [r['pass_rate_std'] * 100 for r in naive_rows]
        ax.errorbar(naive_pos, naive_means, yerr=naive_stds, label='Naive (freq Top-N)', color=NAIVE_COLOR, marker='o', markersize=6, linewidth=2, capsize=4, capthick=1.2, zorder=3)
        mmr_ns = [r['n_skills'] for r in mmr_rows]
        mmr_pos = [n_to_pos[n] for n in mmr_ns]
        mmr_means = [r['pass_rate_mean'] * 100 for r in mmr_rows]
        mmr_stds = [r['pass_rate_std'] * 100 for r in mmr_rows]
        ax.errorbar(mmr_pos, mmr_means, yerr=mmr_stds, label='MMR (diversity-aware)', color=MMR_COLOR, marker='s', markersize=6, linewidth=2, capsize=4, capthick=1.2, linestyle='--', zorder=3)
        ax.set_ylabel('Pass Rate (%)', fontsize=11)
        ax.set_title(DOMAIN_LABELS.get(domain, domain), fontsize=13, fontweight='bold')
        ax.legend(loc='best', fontsize=9, framealpha=0.9)
        ax.grid(True, alpha=0.25)
        ax.yaxis.set_major_formatter(mticker.FormatStrFormatter('%.0f'))
    fig.suptitle('Skill Scaling: Naive (Freq) vs MMR (Diversity-Aware) Selection', fontsize=15, fontweight='bold', y=1.03)
    fig.tight_layout()
    path = output_dir / 'fig_compare_scaling.png'
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'  Saved: {path}')

def plot_delta_bars(naive_data: dict, mmr_data: dict, output_dir: Path):
    domains = sorted(set(naive_data.keys()) & set(mmr_data.keys()))
    n_domains = len(domains)
    if n_domains == 0:
        return
    (fig, axes) = plt.subplots(1, n_domains, figsize=(6 * n_domains, 4.5), squeeze=False)
    axes = axes[0]
    for (idx, domain) in enumerate(domains):
        ax = axes[idx]
        naive_map = {r['n_skills']: r['pass_rate_mean'] * 100 for r in naive_data[domain]}
        mmr_map = {r['n_skills']: r['pass_rate_mean'] * 100 for r in mmr_data[domain]}
        shared_ns = sorted(set(naive_map.keys()) & set(mmr_map.keys()))
        if not shared_ns:
            continue
        deltas = [mmr_map[n] - naive_map[n] for n in shared_ns]
        colors = ['#43A047' if d >= 0 else '#E53935' for d in deltas]
        positions = list(range(len(shared_ns)))
        bars = ax.bar(positions, deltas, color=colors, edgecolor='white', linewidth=1.2, width=0.7, zorder=3)
        for (bar, d) in zip(bars, deltas):
            y_offset = 0.3 if d >= 0 else -0.3
            va = 'bottom' if d >= 0 else 'top'
            ax.text(bar.get_x() + bar.get_width() / 2, d + y_offset, f'{d:+.1f}', ha='center', va=va, fontsize=9, fontweight='bold')
        ax.axhline(y=0, color='black', linewidth=0.8, alpha=0.5)
        ax.set_xticks(positions)
        ax.set_xticklabels([str(n) for n in shared_ns], fontsize=9)
        ax.set_xlabel('Number of Injected Skills', fontsize=11)
        ax.set_ylabel('Δ Pass Rate (MMR − Naive) [%]', fontsize=10)
        ax.set_title(DOMAIN_LABELS.get(domain, domain), fontsize=13, fontweight='bold')
        ax.grid(axis='y', alpha=0.25)
        (ymin, ymax) = ax.get_ylim()
        margin = max(abs(ymin), abs(ymax)) * 0.25
        ax.set_ylim(ymin - margin, ymax + margin)
    fig.suptitle('Δ Pass Rate: MMR − Naive (green = MMR better)', fontsize=15, fontweight='bold', y=1.03)
    fig.tight_layout()
    path = output_dir / 'fig_compare_delta.png'
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'  Saved: {path}')

def plot_trial_comparison(naive_data: dict, mmr_data: dict, baselines: dict, output_dir: Path):
    domains = sorted(set(naive_data.keys()) & set(mmr_data.keys()))
    n_domains = len(domains)
    if n_domains == 0:
        return
    (fig, axes) = plt.subplots(1, n_domains, figsize=(6 * n_domains, 5), squeeze=False)
    axes = axes[0]
    for (idx, domain) in enumerate(domains):
        ax = axes[idx]
        all_ns = sorted(set([r['n_skills'] for r in naive_data[domain]] + [r['n_skills'] for r in mmr_data[domain]]))
        bl = baselines.get(domain)
        if bl:
            all_ns = sorted(set([0] + all_ns))
        n_to_pos = setup_even_xaxis(ax, all_ns)
        if bl:
            bl_rate = bl['pass_rate_mean'] * 100
            ax.axhline(y=bl_rate, color=BASELINE_COLOR, linestyle=':', linewidth=1.2, alpha=0.5)
        for (rows, color, label, offset, marker) in [(naive_data[domain], NAIVE_COLOR, 'Naive', -0.08, 'o'), (mmr_data[domain], MMR_COLOR, 'MMR', +0.08, 's')]:
            plot_pos = []
            plot_means = []
            for r in rows:
                n = r['n_skills']
                pass_counts = r.get('pass_counts', [])
                num_tasks = r.get('num_tasks', 1)
                if not pass_counts:
                    continue
                pos = n_to_pos[n]
                trial_rates = [c / num_tasks * 100 for c in pass_counts]
                jitter = np.random.uniform(offset - 0.04, offset + 0.04, len(trial_rates))
                ax.scatter([pos + j for j in jitter], trial_rates, color=color, marker=marker, alpha=0.35, s=25, zorder=3)
                plot_pos.append(pos)
                plot_means.append(r['pass_rate_mean'] * 100)
            ax.plot(plot_pos, plot_means, color=color, marker=marker, markersize=6, linewidth=1.8, label=label, zorder=4, linestyle='-' if label == 'Naive' else '--')
        ax.set_ylabel('Pass Rate (%)', fontsize=11)
        ax.set_title(DOMAIN_LABELS.get(domain, domain), fontsize=13, fontweight='bold')
        ax.legend(loc='best', fontsize=9)
        ax.grid(True, alpha=0.25)
    fig.suptitle('Trial-Level Variance: Naive vs MMR', fontsize=15, fontweight='bold', y=1.03)
    fig.tight_layout()
    path = output_dir / 'fig_compare_trials.png'
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'  Saved: {path}')

def print_comparison_summary(naive_data: dict, mmr_data: dict, baselines: dict):
    domains = sorted(set(naive_data.keys()) & set(mmr_data.keys()))
    print(f"\n{'=' * 80}")
    print(f'  NAIVE vs MMR COMPARISON SUMMARY')
    print(f"{'=' * 80}")
    for domain in domains:
        bl = baselines.get(domain)
        bl_rate = bl['pass_rate_mean'] * 100 if bl else None
        naive_rows = naive_data[domain]
        mmr_rows = mmr_data[domain]
        naive_means = {r['n_skills']: r['pass_rate_mean'] * 100 for r in naive_rows}
        mmr_means = {r['n_skills']: r['pass_rate_mean'] * 100 for r in mmr_rows}
        naive_stds = {r['n_skills']: r['pass_rate_std'] * 100 for r in naive_rows}
        mmr_stds = {r['n_skills']: r['pass_rate_std'] * 100 for r in mmr_rows}
        shared_ns = sorted(set(naive_means.keys()) & set(mmr_means.keys()))
        print(f'\n  {DOMAIN_LABELS.get(domain, domain).upper()}')
        if bl_rate is not None:
            print(f'  Baseline: {bl_rate:.1f}%')
        print(f"  {'Scale':>6}  {'Naive':>12}  {'MMR':>12}  {'Delta':>8}  {'Winner':>8}")
        print(f"  {'─' * 6}  {'─' * 12}  {'─' * 12}  {'─' * 8}  {'─' * 8}")
        naive_wins = 0
        mmr_wins = 0
        total_delta = 0
        for n in shared_ns:
            nv = naive_means[n]
            mv = mmr_means[n]
            ns = naive_stds[n]
            ms = mmr_stds[n]
            delta = mv - nv
            total_delta += delta
            winner = 'MMR' if delta > 0.5 else 'Naive' if delta < -0.5 else '~tie'
            if delta > 0.5:
                mmr_wins += 1
            elif delta < -0.5:
                naive_wins += 1
            print(f'  {n:>6}  {nv:>5.1f}% ±{ns:>4.1f}  {mv:>5.1f}% ±{ms:>4.1f}  {delta:>+6.1f}%  {winner:>8}')
        avg_delta = total_delta / len(shared_ns) if shared_ns else 0
        naive_best_n = max(shared_ns, key=lambda n: naive_means[n])
        mmr_best_n = max(shared_ns, key=lambda n: mmr_means[n])
        naive_worst_n = min(shared_ns, key=lambda n: naive_means[n])
        mmr_worst_n = min(shared_ns, key=lambda n: mmr_means[n])
        print(f"  {'─' * 6}  {'─' * 12}  {'─' * 12}  {'─' * 8}  {'─' * 8}")
        print(f'  Avg delta: {avg_delta:+.1f}%  |  MMR wins: {mmr_wins}/{len(shared_ns)}  |  Naive wins: {naive_wins}/{len(shared_ns)}')
        print(f'  Naive best: {naive_means[naive_best_n]:.1f}% @ top-{naive_best_n}  |  Naive worst: {naive_means[naive_worst_n]:.1f}% @ top-{naive_worst_n}')
        print(f'  MMR   best: {mmr_means[mmr_best_n]:.1f}% @ top-{mmr_best_n}  |  MMR   worst: {mmr_means[mmr_worst_n]:.1f}% @ top-{mmr_worst_n}')
        for (method_name, means_map) in [('Naive', naive_means), ('MMR', mmr_means)]:
            ns_list = sorted(means_map.keys())
            if len(ns_list) >= 3:
                log_ns = np.log2(ns_list)
                vals = [means_map[n] for n in ns_list]
                slope = np.polyfit(log_ns, vals, 1)[0]
                if slope < -0.5:
                    trend = 'DECLINING'
                elif slope > 0.5:
                    trend = 'INCREASING'
                else:
                    trend = 'FLAT'
                print(f'  {method_name:>5} trend: {trend} (slope={slope:+.2f}%/doubling)')
    print(f"\n{'=' * 80}")

def main():
    parser = argparse.ArgumentParser(description='Step 4.3: Compare Naive vs MMR scaling results')
    parser.add_argument('--naive-dir', default=DEFAULT_NAIVE_DIR, help=f'Organized naive results (default: {DEFAULT_NAIVE_DIR})')
    parser.add_argument('--mmr-dir', default=DEFAULT_MMR_DIR, help=f'Organized MMR results (default: {DEFAULT_MMR_DIR})')
    parser.add_argument('--baseline-dir', default=DEFAULT_BASELINE_DIR, help=f'Baseline results directory (default: {DEFAULT_BASELINE_DIR})')
    parser.add_argument('--output-dir', default=None, help='Output dir for figures (default: <mmr-dir>/comparison)')
    parser.add_argument('--domains', nargs='+', default=None, help='Domains to compare (default: all available)')
    args = parser.parse_args()
    naive_dir = Path(args.naive_dir)
    mmr_dir = Path(args.mmr_dir)
    baseline_dir = Path(args.baseline_dir)
    output_dir = Path(args.output_dir) if args.output_dir else mmr_dir / 'comparison'
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f'Loading Naive data from {naive_dir}...')
    naive_data = load_scaling_data(naive_dir, args.domains)
    print(f'  Found domains: {list(naive_data.keys())}')
    print(f'Loading MMR data from {mmr_dir}...')
    mmr_data = load_scaling_data(mmr_dir, args.domains)
    print(f'  Found domains: {list(mmr_data.keys())}')
    print(f'Loading baselines from {baseline_dir}...')
    baselines = {}
    for domain in set(naive_data.keys()) | set(mmr_data.keys()):
        bl = load_baseline(baseline_dir, domain)
        if bl:
            baselines[domain] = bl
            print(f"  {domain}: {bl['pass_rate_mean'] * 100:.1f}% ({bl['file']})")
    common_domains = sorted(set(naive_data.keys()) & set(mmr_data.keys()))
    if not common_domains:
        print('ERROR: No common domains between naive and MMR data.')
        return
    print(f'\nGenerating comparison figures -> {output_dir}/')
    print('\n[1/3] Scaling curve comparison...')
    plot_comparison_curves(naive_data, mmr_data, baselines, output_dir)
    print('[2/3] Delta bar chart...')
    plot_delta_bars(naive_data, mmr_data, output_dir)
    print('[3/3] Trial-level scatter comparison...')
    plot_trial_comparison(naive_data, mmr_data, baselines, output_dir)
    print_comparison_summary(naive_data, mmr_data, baselines)
    print(f'\nAll figures saved to: {output_dir}/')
if __name__ == '__main__':
    main()
