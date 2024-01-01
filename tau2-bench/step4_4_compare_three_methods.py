import argparse
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
SKILLS_ROOT = '<MODEL_ROOT>/agent/skills'
DEFAULT_NAIVE_DIR = f'{SKILLS_ROOT}/scaling-organized'
DEFAULT_MMR_DIR = f'{SKILLS_ROOT}/scaling-organized-mmr'
DEFAULT_UTILITY_DIR = f'{SKILLS_ROOT}/scaling-organized-utility'
DEFAULT_BASELINE_DIR = str(Path(__file__).resolve().parent / 'results')
BASELINE_FILENAME_MAP = {'airline': 'baseline_agg3_*.json', 'retail': 'retail_baseline_agg3_*.json', 'telecom': 'telecom_baseline_agg3_*.json'}
DOMAIN_LABELS = {'airline': 'Airline', 'retail': 'Retail', 'telecom': 'Telecom'}
METHODS = [{'key': 'naive', 'label': 'Naive (frequency)', 'color': '#5C6BC0', 'marker': 'o', 'ls': '-'}, {'key': 'mmr', 'label': 'MMR (freq + diversity)', 'color': '#E53935', 'marker': 's', 'ls': '--'}, {'key': 'utility', 'label': 'Utility (Δpass@{k})', 'color': '#43A047', 'marker': '^', 'ls': '-.'}]
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
    if not data_dir.exists():
        return result
    for domain_dir in sorted(data_dir.iterdir()):
        if not domain_dir.is_dir() or domain_dir.name == 'figures':
            continue
        domain = domain_dir.name
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

def setup_even_xaxis(ax, all_ns, xlabel='Number of Injected Skills'):
    (positions, sorted_ns) = make_xaxis_positions(all_ns)
    n_to_pos = dict(zip(sorted_ns, positions))
    ax.set_xticks(positions)
    ax.set_xticklabels([str(n) for n in sorted_ns], fontsize=9)
    ax.set_xlabel(xlabel, fontsize=11)
    if len(positions) > 1:
        pad = (positions[-1] - positions[0]) * 0.05
        ax.set_xlim(positions[0] - pad, positions[-1] + pad)
    return n_to_pos

def plot_three_curves(all_data: dict[str, dict], baselines: dict, domains: list[str], output_dir: Path):
    n_domains = len(domains)
    (fig, axes) = plt.subplots(1, n_domains, figsize=(7 * n_domains, 5.5), squeeze=False)
    axes = axes[0]
    for (idx, domain) in enumerate(domains):
        ax = axes[idx]
        all_ns = set()
        for m in METHODS:
            if domain in all_data[m['key']]:
                for r in all_data[m['key']][domain]:
                    all_ns.add(r['n_skills'])
        bl = baselines.get(domain)
        if bl:
            all_ns.add(0)
        all_ns = sorted(all_ns)
        n_to_pos = setup_even_xaxis(ax, all_ns)
        if bl:
            bl_rate = bl['pass_rate_mean'] * 100
            bl_std = bl['pass_rate_std'] * 100
            ax.axhspan(bl_rate - bl_std, bl_rate + bl_std, color=BASELINE_COLOR, alpha=0.08, zorder=0)
            ax.axhline(y=bl_rate, color=BASELINE_COLOR, linestyle=':', linewidth=1.2, alpha=0.6, zorder=1)
            ax.plot(n_to_pos[0], bl_rate, marker='*', color=BASELINE_COLOR, markersize=13, zorder=5, markeredgecolor='white', markeredgewidth=0.8)
            ax.text(n_to_pos[0] + 0.15, bl_rate + 1.0, f'baseline\n{bl_rate:.1f}%', fontsize=8, color=BASELINE_COLOR, va='bottom', ha='left')
        for m in METHODS:
            if domain not in all_data[m['key']]:
                continue
            rows = all_data[m['key']][domain]
            ns = [r['n_skills'] for r in rows]
            pos = [n_to_pos[n] for n in ns]
            means = [r['pass_rate_mean'] * 100 for r in rows]
            stds = [r['pass_rate_std'] * 100 for r in rows]
            ax.errorbar(pos, means, yerr=stds, label=m['label'], color=m['color'], marker=m['marker'], markersize=6, linewidth=2, capsize=4, capthick=1.2, linestyle=m['ls'], zorder=3)
        ax.set_ylabel('Pass Rate (%)', fontsize=11)
        ax.set_title(DOMAIN_LABELS.get(domain, domain), fontsize=13, fontweight='bold')
        ax.legend(loc='best', fontsize=8.5, framealpha=0.9)
        ax.grid(True, alpha=0.25)
        ax.yaxis.set_major_formatter(mticker.FormatStrFormatter('%.0f'))
    fig.suptitle('Skill Scaling: Naive vs MMR vs Utility-Ranked Selection', fontsize=15, fontweight='bold', y=1.03)
    fig.tight_layout()
    path = output_dir / 'fig_three_method_scaling.png'
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'  Saved: {path}')

def plot_delta_from_baseline(all_data: dict[str, dict], baselines: dict, domains: list[str], output_dir: Path):
    n_domains = len(domains)
    (fig, axes) = plt.subplots(1, n_domains, figsize=(7 * n_domains, 5), squeeze=False)
    axes = axes[0]
    bar_width = 0.25
    for (idx, domain) in enumerate(domains):
        ax = axes[idx]
        bl = baselines.get(domain)
        bl_rate = bl['pass_rate_mean'] * 100 if bl else 0
        all_ns_per_method = {}
        for m in METHODS:
            if domain in all_data[m['key']]:
                all_ns_per_method[m['key']] = {r['n_skills']: r['pass_rate_mean'] * 100 for r in all_data[m['key']][domain]}
            else:
                all_ns_per_method[m['key']] = {}
        all_ns = sorted(set().union(*[set(v.keys()) for v in all_ns_per_method.values()]))
        if not all_ns:
            continue
        x = np.arange(len(all_ns))
        for (i, m) in enumerate(METHODS):
            means_map = all_ns_per_method[m['key']]
            deltas = [means_map.get(n, float('nan')) - bl_rate for n in all_ns]
            offset = (i - 1) * bar_width
            bars = ax.bar(x + offset, deltas, bar_width, label=m['label'], color=m['color'], alpha=0.85, edgecolor='white', linewidth=0.8, zorder=3)
            for (bar, d) in zip(bars, deltas):
                if np.isnan(d):
                    continue
                y_off = 0.3 if d >= 0 else -0.3
                va = 'bottom' if d >= 0 else 'top'
                ax.text(bar.get_x() + bar.get_width() / 2, d + y_off, f'{d:+.1f}', ha='center', va=va, fontsize=7, fontweight='bold')
        ax.axhline(y=0, color='black', linewidth=0.8, alpha=0.5)
        ax.set_xticks(x)
        ax.set_xticklabels([str(n) for n in all_ns], fontsize=9)
        ax.set_xlabel('Number of Injected Skills', fontsize=11)
        ax.set_ylabel('Δ Pass Rate from Baseline (%)', fontsize=10)
        ax.set_title(DOMAIN_LABELS.get(domain, domain), fontsize=13, fontweight='bold')
        ax.legend(fontsize=8, loc='best')
        ax.grid(axis='y', alpha=0.25)
        (ymin, ymax) = ax.get_ylim()
        margin = max(abs(ymin), abs(ymax), 3) * 0.3
        ax.set_ylim(ymin - margin, ymax + margin)
    fig.suptitle('Δ Pass Rate from Baseline: Three Selection Methods', fontsize=15, fontweight='bold', y=1.03)
    fig.tight_layout()
    path = output_dir / 'fig_three_method_delta_baseline.png'
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'  Saved: {path}')

def plot_method_summary(all_data: dict[str, dict], baselines: dict, domains: list[str], output_dir: Path):
    (fig, axes) = plt.subplots(1, len(domains), figsize=(6 * len(domains), 4.5), squeeze=False)
    axes = axes[0]
    for (idx, domain) in enumerate(domains):
        ax = axes[idx]
        bl = baselines.get(domain)
        bl_rate = bl['pass_rate_mean'] * 100 if bl else 0
        method_stats = []
        for m in METHODS:
            if domain not in all_data[m['key']]:
                method_stats.append({'label': m['label'], 'color': m['color'], 'avg_delta': 0, 'best': 0, 'worst': 0, 'best_n': 0, 'worst_n': 0})
                continue
            rows = all_data[m['key']][domain]
            means = {r['n_skills']: r['pass_rate_mean'] * 100 for r in rows}
            deltas = {n: v - bl_rate for (n, v) in means.items()}
            avg_d = np.mean(list(deltas.values()))
            best_n = max(deltas, key=deltas.get)
            worst_n = min(deltas, key=deltas.get)
            method_stats.append({'label': m['label'], 'color': m['color'], 'avg_delta': avg_d, 'best': deltas[best_n], 'best_n': best_n, 'worst': deltas[worst_n], 'worst_n': worst_n})
        labels = [ms['label'] for ms in method_stats]
        avg_deltas = [ms['avg_delta'] for ms in method_stats]
        colors = [ms['color'] for ms in method_stats]
        bars = ax.bar(range(len(labels)), avg_deltas, color=colors, alpha=0.85, edgecolor='white', linewidth=1.2, width=0.6)
        for (bar, ms) in zip(bars, method_stats):
            d = ms['avg_delta']
            y_off = 0.3 if d >= 0 else -0.3
            va = 'bottom' if d >= 0 else 'top'
            text = f"avg: {d:+.1f}%\nbest: {ms['best']:+.1f}% @{ms['best_n']}\nworst: {ms['worst']:+.1f}% @{ms['worst_n']}"
            ax.text(bar.get_x() + bar.get_width() / 2, d + y_off, text, ha='center', va=va, fontsize=7.5)
        ax.axhline(y=0, color='black', linewidth=0.8, alpha=0.5)
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels([l.split('(')[0].strip() for l in labels], fontsize=10)
        ax.set_ylabel('Avg Δ from Baseline (%)', fontsize=11)
        ax.set_title(DOMAIN_LABELS.get(domain, domain), fontsize=13, fontweight='bold')
        ax.grid(axis='y', alpha=0.25)
    fig.suptitle('Average Performance by Selection Method', fontsize=15, fontweight='bold', y=1.03)
    fig.tight_layout()
    path = output_dir / 'fig_three_method_summary.png'
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'  Saved: {path}')

def print_summary(all_data: dict[str, dict], baselines: dict, domains: list[str]):
    print(f"\n{'=' * 90}")
    print(f'  THREE-METHOD COMPARISON SUMMARY')
    print(f"{'=' * 90}")
    for domain in domains:
        bl = baselines.get(domain)
        bl_rate = bl['pass_rate_mean'] * 100 if bl else 0
        print(f'\n  {DOMAIN_LABELS.get(domain, domain).upper()}  (baseline: {bl_rate:.1f}%)')
        header = f"  {'Scale':>6}"
        for m in METHODS:
            header += f"  {m['key'].capitalize():>12}"
        header += f"  {'Best':>8}"
        print(header)
        print(f"  {'─' * 6}" + f"  {'─' * 12}" * len(METHODS) + f"  {'─' * 8}")
        means_maps = {}
        for m in METHODS:
            if domain in all_data[m['key']]:
                means_maps[m['key']] = {r['n_skills']: r['pass_rate_mean'] * 100 for r in all_data[m['key']][domain]}
            else:
                means_maps[m['key']] = {}
        all_ns = sorted(set().union(*[set(v.keys()) for v in means_maps.values()]))
        method_wins = {m['key']: 0 for m in METHODS}
        method_deltas = {m['key']: [] for m in METHODS}
        for n in all_ns:
            row = f'  {n:>6}'
            vals = {}
            for m in METHODS:
                v = means_maps[m['key']].get(n)
                if v is not None:
                    delta = v - bl_rate
                    row += f'  {v:>5.1f}%({delta:+.1f})'
                    vals[m['key']] = v
                    method_deltas[m['key']].append(delta)
                else:
                    row += f"  {'N/A':>12}"
            if vals:
                best_key = max(vals, key=vals.get)
                row += f'  {best_key.capitalize():>8}'
                method_wins[best_key] += 1
            print(row)
        print(f"  {'─' * 6}" + f"  {'─' * 12}" * len(METHODS) + f"  {'─' * 8}")
        summary = '  '
        for m in METHODS:
            key = m['key']
            ds = method_deltas[key]
            if ds:
                avg = np.mean(ds)
                summary += f'{key.capitalize()}: avg={avg:+.1f}%, wins={method_wins[key]}/{len(all_ns)}  |  '
            else:
                summary += f'{key.capitalize()}: N/A  |  '
        print(summary.rstrip('  |  '))
        for m in METHODS:
            key = m['key']
            mm = means_maps[key]
            ns_list = sorted(mm.keys())
            if len(ns_list) >= 3:
                log_ns = np.log2(ns_list)
                vals = [mm[n] for n in ns_list]
                slope = np.polyfit(log_ns, vals, 1)[0]
                trend = 'DECLINING' if slope < -0.5 else 'INCREASING' if slope > 0.5 else 'FLAT'
                print(f'  {key.capitalize():>8} trend: {trend} (slope={slope:+.2f}%/doubling)')
    print(f"\n{'=' * 90}")

def main():
    parser = argparse.ArgumentParser(description='Step 4.4: Compare Naive vs MMR vs Utility scaling results')
    parser.add_argument('--naive-dir', default=DEFAULT_NAIVE_DIR)
    parser.add_argument('--mmr-dir', default=DEFAULT_MMR_DIR)
    parser.add_argument('--utility-dir', default=DEFAULT_UTILITY_DIR)
    parser.add_argument('--baseline-dir', default=DEFAULT_BASELINE_DIR)
    parser.add_argument('--output-dir', default=None, help='Output dir (default: <utility-dir>/comparison_three)')
    parser.add_argument('--domains', nargs='+', default=None)
    args = parser.parse_args()
    naive_dir = Path(args.naive_dir)
    mmr_dir = Path(args.mmr_dir)
    utility_dir = Path(args.utility_dir)
    baseline_dir = Path(args.baseline_dir)
    output_dir = Path(args.output_dir) if args.output_dir else utility_dir / 'comparison_three'
    output_dir.mkdir(parents=True, exist_ok=True)
    all_data = {}
    for (m, d) in [('naive', naive_dir), ('mmr', mmr_dir), ('utility', utility_dir)]:
        print(f'Loading {m} data from {d}...')
        all_data[m] = load_scaling_data(d, args.domains)
        print(f'  Found domains: {list(all_data[m].keys())}')
    num_trials = 3
    for method_data in all_data.values():
        for domain_rows in method_data.values():
            for row in domain_rows:
                pc = row.get('pass_counts', [])
                if len(pc) > num_trials:
                    num_trials = len(pc)
    for m in METHODS:
        m['label'] = m['label'].replace('{k}', str(num_trials))
    print(f'  Detected {num_trials} trials')
    all_domains = set()
    for v in all_data.values():
        all_domains.update(v.keys())
    domains = sorted(all_domains)
    if args.domains:
        domains = [d for d in domains if d in args.domains]
    if not domains:
        print('ERROR: No domains found in any dataset.')
        return
    print(f'Loading baselines from {baseline_dir}...')
    baselines = {}
    for domain in domains:
        bl = load_baseline(baseline_dir, domain)
        if bl:
            baselines[domain] = bl
            print(f"  {domain}: {bl['pass_rate_mean'] * 100:.1f}%")
    print(f'\nGenerating comparison figures -> {output_dir}/\n')
    print('[1/3] Three-method scaling curves...')
    plot_three_curves(all_data, baselines, domains, output_dir)
    print('[2/3] Delta-from-baseline grouped bars...')
    plot_delta_from_baseline(all_data, baselines, domains, output_dir)
    print('[3/3] Method summary bars...')
    plot_method_summary(all_data, baselines, domains, output_dir)
    print_summary(all_data, baselines, domains)
    print(f'\nAll figures saved to: {output_dir}/')
if __name__ == '__main__':
    main()
