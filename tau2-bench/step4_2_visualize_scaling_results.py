import argparse
import json
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats as sp_stats
DEFAULT_DATA_DIR = '<MODEL_ROOT>/agent/skills/scaling-organized'
DEFAULT_BASELINE_DIR = str(Path(__file__).resolve().parent / 'results')
BASELINE_FILENAME_MAP = {'airline': 'baseline_agg3_*.json', 'retail': 'retail_baseline_agg3_*.json', 'telecom': 'telecom_baseline_agg3_*.json'}
DOMAIN_COLORS = {'airline': '#2196F3', 'retail': '#FF9800', 'telecom': '#4CAF50'}
DOMAIN_MARKERS = {'airline': 'o', 'retail': 's', 'telecom': '^'}

def load_baseline(baseline_dir: Path, domain: str) -> dict | None:
    pattern = BASELINE_FILENAME_MAP.get(domain)
    if not pattern:
        return None
    files = sorted(baseline_dir.glob(pattern))
    if not files:
        return None
    data = json.loads(files[-1].read_text())
    s = data['summary']
    row = {'n_skills': 0, 'pass_rate_mean': s.get('pass_rate_mean', s.get('pass_rate', 0)), 'pass_rate_std': s.get('pass_rate_std', 0), 'pass_counts': s.get('pass_counts', [s.get('pass_count', 0)]), 'majority_pass_rate': s.get('majority_pass_rate', s.get('pass_rate', 0)), 'majority_pass_count': s.get('majority_pass_count', s.get('pass_count', 0)), 'num_tasks': s.get('num_tasks', 0), 'avg_messages': s.get('avg_messages_mean', s.get('avg_messages', 0)), 'avg_tool_calls': s.get('avg_tool_calls_mean', s.get('avg_tool_calls', 0))}
    per_task = {}
    for r in data.get('results', []):
        per_task[str(r['task_id'])] = r.get('reward', r.get('pass_count', 0) / max(r.get('num_trials', 1), 1))
    return {'summary_row': row, 'per_task': per_task, 'file': files[-1].name}

def load_scaling_reports(data_dir: Path, baseline_dir: Path, domains: list[str] | None) -> dict[str, dict]:
    reports = {}
    for domain_dir in sorted(data_dir.iterdir()):
        if not domain_dir.is_dir():
            continue
        domain = domain_dir.name
        if domains and domain not in domains:
            continue
        report_files = sorted(domain_dir.glob('scaling_report_*.json'))
        if not report_files:
            print(f'  WARNING: No scaling report found for {domain}, skipping')
            continue
        report_path = report_files[-1]
        report = json.loads(report_path.read_text())
        bl = load_baseline(baseline_dir, domain)
        if bl:
            report['scaling_summary'].insert(0, bl['summary_row'])
            for task_entry in report.get('per_task_comparison', []):
                tid = task_entry['task_id']
                if tid in bl['per_task']:
                    task_entry['top0_reward'] = bl['per_task'][tid]
                    task_entry['top0_tools'] = 0
            print(f"  Loaded {domain}: baseline + {len(report['scaling_summary']) - 1} scale points (baseline: {bl['file']})")
        else:
            print(f"  Loaded {domain}: {len(report['scaling_summary'])} scale points (no baseline found)")
        reports[domain] = report
    return reports

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
    ax.set_xlabel(xlabel, fontsize=12)
    if len(positions) > 1:
        pad = (positions[-1] - positions[0]) * 0.04
        ax.set_xlim(positions[0] - pad, positions[-1] + pad)
    return n_to_pos

def plot_scaling_curve(reports: dict[str, dict], output_dir: Path):
    (fig, ax) = plt.subplots(figsize=(10, 6))
    all_ns = sorted(set((r['n_skills'] for report in reports.values() for r in report['scaling_summary'])))
    n_to_pos = setup_even_xaxis(ax, all_ns)
    for (domain, report) in reports.items():
        rows = report['scaling_summary']
        ns = [r['n_skills'] for r in rows]
        positions = [n_to_pos[n] for n in ns]
        means = [r['pass_rate_mean'] * 100 for r in rows]
        stds = [r['pass_rate_std'] * 100 for r in rows]
        color = DOMAIN_COLORS.get(domain, '#666666')
        marker = DOMAIN_MARKERS.get(domain, 'o')
        ax.errorbar(positions, means, yerr=stds, label=f'{domain} (mean ± std)', color=color, marker=marker, markersize=7, linewidth=2, capsize=4, capthick=1.5)
        maj = [r.get('majority_pass_rate', r['pass_rate_mean']) * 100 for r in rows]
        ax.plot(positions, maj, color=color, marker=marker, markersize=5, linewidth=1.2, linestyle='--', alpha=0.6, label=f'{domain} (majority vote)')
        if ns[0] == 0:
            ax.plot(positions[0], means[0], marker='*', color=color, markersize=14, zorder=5, markeredgecolor='white', markeredgewidth=0.8)
    ax.set_ylabel('Pass Rate (%)', fontsize=12)
    ax.set_title('Skill Scaling Experiment: Pass Rate vs. Number of Skills', fontsize=14, fontweight='bold')
    ax.legend(loc='best', fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(bottom=max(0, ax.get_ylim()[0] - 5))
    if 0 in n_to_pos:
        ax.axvline(x=n_to_pos[0], color='gray', linestyle=':', alpha=0.3)
        ax.text(n_to_pos[0], ax.get_ylim()[1] * 0.98, ' baseline', fontsize=8, color='gray', va='top', ha='left')
    fig.tight_layout()
    path = output_dir / 'fig1_scaling_curve.png'
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f'  Saved: {path}')

def plot_per_task_heatmap(reports: dict[str, dict], output_dir: Path):
    for (domain, report) in reports.items():
        comparison = report.get('per_task_comparison', [])
        if not comparison:
            continue
        scales = []
        for key in comparison[0]:
            if key.startswith('top') and key.endswith('_reward'):
                n = int(key.replace('top', '').replace('_reward', ''))
                scales.append(n)
        scales.sort()
        if not scales:
            continue
        task_ids = [t['task_id'] for t in comparison]
        n_tasks = len(task_ids)
        n_scales = len(scales)
        matrix = np.full((n_tasks, n_scales), np.nan)
        for (i, task) in enumerate(comparison):
            for (j, n) in enumerate(scales):
                key = f'top{n}_reward'
                if key in task:
                    matrix[i, j] = task[key]
        if n_scales >= 2:
            degradation = np.nanmean(matrix[:, :2], axis=1) - np.nanmean(matrix[:, -2:], axis=1)
            sort_idx = np.argsort(-degradation)
        else:
            sort_idx = np.arange(n_tasks)
        matrix_sorted = matrix[sort_idx]
        task_ids_sorted = [task_ids[i] for i in sort_idx]
        fig_h = max(6, n_tasks * 0.2 + 2)
        (fig, ax) = plt.subplots(figsize=(max(8, n_scales * 1.2 + 2), fig_h))
        im = ax.imshow(matrix_sorted, aspect='auto', cmap='RdYlGn', vmin=0, vmax=1, interpolation='nearest')
        xlabels = [str(s) if s > 0 else '0\n(base)' for s in scales]
        ax.set_xticks(range(n_scales))
        ax.set_xticklabels(xlabels, fontsize=9)
        ax.set_xlabel('Number of Injected Skills', fontsize=11)
        ax.set_yticks(range(n_tasks))
        ax.set_yticklabels(task_ids_sorted, fontsize=7)
        ax.set_ylabel('Task ID (sorted by degradation)', fontsize=11)
        ax.set_title(f'Per-Task Reward Heatmap — {domain.upper()}\n(green=pass, red=fail, sorted by degradation top→bottom)', fontsize=12, fontweight='bold')
        cbar = fig.colorbar(im, ax=ax, shrink=0.8)
        cbar.set_label('Reward (majority vote)', fontsize=10)
        fig.tight_layout()
        path = output_dir / f'fig2_heatmap_{domain}.png'
        fig.savefig(path, dpi=150)
        plt.close(fig)
        print(f'  Saved: {path}')

def plot_overhead(reports: dict[str, dict], output_dir: Path):
    (fig, (ax1, ax2)) = plt.subplots(1, 2, figsize=(14, 5))
    all_ns = sorted(set((r['n_skills'] for report in reports.values() for r in report['scaling_summary'])))
    for ax in (ax1, ax2):
        n_to_pos = setup_even_xaxis(ax, all_ns)
    for (domain, report) in reports.items():
        rows = report['scaling_summary']
        ns = [r['n_skills'] for r in rows]
        positions = [n_to_pos[n] for n in ns]
        msgs = [r.get('avg_messages', 0) for r in rows]
        tools = [r.get('avg_tool_calls', 0) for r in rows]
        color = DOMAIN_COLORS.get(domain, '#666666')
        marker = DOMAIN_MARKERS.get(domain, 'o')
        ax1.plot(positions, msgs, color=color, marker=marker, markersize=6, linewidth=2, label=domain)
        ax2.plot(positions, tools, color=color, marker=marker, markersize=6, linewidth=2, label=domain)
    for (ax, ylabel, title) in [(ax1, 'Avg Messages per Task', 'Message Overhead'), (ax2, 'Avg Tool Calls per Task', 'Tool Call Overhead')]:
        ax.set_ylabel(ylabel, fontsize=11)
        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.legend(loc='best', fontsize=9)
        ax.grid(True, alpha=0.3)
    fig.suptitle('Overhead vs. Number of Injected Skills', fontsize=14, fontweight='bold', y=1.02)
    fig.tight_layout()
    path = output_dir / 'fig3_overhead.png'
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'  Saved: {path}')

def plot_trial_variance(reports: dict[str, dict], output_dir: Path):
    (fig, ax) = plt.subplots(figsize=(10, 6))
    all_ns = sorted(set((r['n_skills'] for report in reports.values() for r in report['scaling_summary'])))
    n_to_pos = setup_even_xaxis(ax, all_ns)
    for (domain, report) in reports.items():
        rows = report['scaling_summary']
        color = DOMAIN_COLORS.get(domain, '#666666')
        marker = DOMAIN_MARKERS.get(domain, 'o')
        plot_positions = []
        plot_means = []
        for r in rows:
            n = r['n_skills']
            pass_counts = r.get('pass_counts', [])
            num_tasks = r.get('num_tasks', 1)
            if not pass_counts:
                continue
            pos = n_to_pos[n]
            trial_rates = [c / num_tasks * 100 for c in pass_counts]
            jitter = np.random.uniform(-0.08, 0.08, len(trial_rates))
            ax.scatter([pos + j for j in jitter], trial_rates, color=color, marker=marker, alpha=0.4, s=30, zorder=3)
            plot_positions.append(pos)
            plot_means.append(r['pass_rate_mean'] * 100)
        ax.plot(plot_positions, plot_means, color=color, marker=marker, markersize=7, linewidth=2, label=domain, zorder=4)
    ax.set_ylabel('Pass Rate (%)', fontsize=12)
    ax.set_title('Trial-Level Variance: Individual Trials + Mean', fontsize=14, fontweight='bold')
    ax.legend(loc='best', fontsize=10)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    path = output_dir / 'fig4_trial_variance.png'
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f'  Saved: {path}')

def plot_task_stability(reports: dict[str, dict], output_dir: Path):
    for (domain, report) in reports.items():
        comparison = report.get('per_task_comparison', [])
        if not comparison:
            continue
        scales = sorted(set((int(k.replace('top', '').replace('_reward', '')) for k in comparison[0] if k.startswith('top') and k.endswith('_reward'))))
        if len(scales) < 3:
            continue
        rhos = []
        task_ids = []
        for task in comparison:
            rewards = []
            ns_valid = []
            for n in scales:
                key = f'top{n}_reward'
                if key in task and (not np.isnan(task[key])):
                    rewards.append(task[key])
                    ns_valid.append(n)
            if len(ns_valid) < 3:
                continue
            if np.std(rewards) < 1e-09:
                rho = 0.0
            else:
                (rho, _) = sp_stats.spearmanr(ns_valid, rewards)
            rhos.append(rho)
            task_ids.append(task['task_id'])
        if not rhos:
            continue
        rhos = np.array(rhos)
        (fig, (ax1, ax2)) = plt.subplots(1, 2, figsize=(14, 5.5))
        bins = np.linspace(-1, 1, 21)
        colors_hist = []
        for (left, right) in zip(bins[:-1], bins[1:]):
            mid = (left + right) / 2
            if mid < -0.3:
                colors_hist.append('#F44336')
            elif mid > 0.3:
                colors_hist.append('#2196F3')
            else:
                colors_hist.append('#9E9E9E')
        (counts_hist, _, patches) = ax1.hist(rhos, bins=bins, edgecolor='white', linewidth=0.8)
        for (patch, c) in zip(patches, colors_hist):
            patch.set_facecolor(c)
        ax1.axvline(x=0, color='black', linestyle='-', alpha=0.3)
        ax1.axvline(x=-0.3, color='red', linestyle='--', alpha=0.5, linewidth=1)
        ax1.axvline(x=0.3, color='blue', linestyle='--', alpha=0.5, linewidth=1)
        median_rho = np.median(rhos)
        ax1.axvline(x=median_rho, color='black', linestyle='-', linewidth=2, alpha=0.7)
        ax1.text(median_rho + 0.03, ax1.get_ylim()[1] * 0.9, f'median={median_rho:.2f}', fontsize=9, fontweight='bold')
        ax1.set_xlabel('Spearman ρ (N_skills vs. Reward)', fontsize=11)
        ax1.set_ylabel('Number of Tasks', fontsize=11)
        ax1.set_title('Distribution of Per-Task Correlation', fontsize=12, fontweight='bold')
        n_degrade = int(np.sum(rhos < -0.3))
        n_no_trend = int(np.sum(np.abs(rhos) <= 0.3))
        n_improve = int(np.sum(rhos > 0.3))
        always_pass = 0
        always_fail = 0
        variable = 0
        for (i, task) in enumerate(comparison):
            if i >= len(rhos):
                break
            if abs(rhos[i]) > 0.3:
                continue
            rewards = [task.get(f'top{n}_reward', np.nan) for n in scales]
            rewards = [r for r in rewards if not np.isnan(r)]
            if not rewards:
                variable += 1
            elif all((r >= 0.8 for r in rewards)):
                always_pass += 1
            elif all((r <= 0.2 for r in rewards)):
                always_fail += 1
            else:
                variable += 1
        categories = ['Degrades\n(ρ<-0.3)', 'Always\nPass', 'Always\nFail', 'No Trend\n(variable)', 'Improves\n(ρ>0.3)']
        cat_counts = [n_degrade, always_pass, always_fail, variable, n_improve]
        cat_colors = ['#F44336', '#4CAF50', '#333333', '#9E9E9E', '#2196F3']
        bars = ax2.bar(categories, cat_counts, color=cat_colors, edgecolor='white', linewidth=1.5)
        for (bar, count) in zip(bars, cat_counts):
            if count > 0:
                ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3, str(count), ha='center', va='bottom', fontsize=12, fontweight='bold')
        total = len(rhos)
        ax2.set_ylabel('Number of Tasks', fontsize=11)
        ax2.set_title(f'Task Stability Categories ({total} tasks)', fontsize=12, fontweight='bold')
        ax2.grid(axis='y', alpha=0.3)
        fig.suptitle(f'Task Stability Analysis — {domain.upper()}', fontsize=14, fontweight='bold', y=1.02)
        fig.tight_layout()
        path = output_dir / f'fig5_stability_{domain}.png'
        fig.savefig(path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f'  Saved: {path}')

def print_analysis(reports: dict[str, dict]):
    print(f"\n{'=' * 60}")
    print(f'  KEY FINDINGS')
    print(f"{'=' * 60}")
    for (domain, report) in reports.items():
        rows = report['scaling_summary']
        ns = [r['n_skills'] for r in rows]
        means = [r['pass_rate_mean'] for r in rows]
        if ns[0] == 0:
            bl_rate = means[0]
            scale_ns = ns[1:]
            scale_means = means[1:]
        else:
            bl_rate = None
            scale_ns = ns
            scale_means = means
        max_pr = max(scale_means) if scale_means else 0
        min_pr = min(scale_means) if scale_means else 0
        max_n = scale_ns[scale_means.index(max_pr)] if scale_means else 0
        min_n = scale_ns[scale_means.index(min_pr)] if scale_means else 0
        drop = max_pr - min_pr
        print(f'\n  {domain.upper()}:')
        if bl_rate is not None:
            print(f'    Baseline (0 skills): {bl_rate:.1%}')
        print(f'    Best:  {max_pr:.1%} @ top-{max_n}')
        print(f'    Worst: {min_pr:.1%} @ top-{min_n}')
        print(f'    Range (best-worst): {drop:.1%}')
        if len(scale_means) >= 3:
            log_ns = np.log2(scale_ns)
            slope = np.polyfit(log_ns, scale_means, 1)[0]
            if slope < -0.02:
                print(f'    Trend: DECLINING (slope={slope:.4f}/doubling)')
            elif slope > 0.02:
                print(f'    Trend: INCREASING (slope={slope:.4f}/doubling)')
            else:
                print(f'    Trend: FLAT (slope={slope:.4f}/doubling)')
            if bl_rate is not None:
                delta_from_bl = scale_means[-1] - bl_rate
                print(f'    Δ from baseline at max scale: {delta_from_bl:+.1%} (top-{scale_ns[-1]}: {scale_means[-1]:.1%} vs baseline: {bl_rate:.1%})')
    print(f"\n{'=' * 60}")

def main():
    parser = argparse.ArgumentParser(description='Step 4.2: Visualize scaling experiment results')
    parser.add_argument('--data-dir', default=DEFAULT_DATA_DIR, help=f'Organized data directory (default: {DEFAULT_DATA_DIR})')
    parser.add_argument('--baseline-dir', default=DEFAULT_BASELINE_DIR, help=f'Baseline results directory (default: {DEFAULT_BASELINE_DIR})')
    parser.add_argument('--output-dir', default=None, help='Output directory for figures (default: <data-dir>/figures)')
    parser.add_argument('--domains', nargs='+', default=None, help='Domains to visualize (default: all available)')
    args = parser.parse_args()
    data_dir = Path(args.data_dir)
    baseline_dir = Path(args.baseline_dir)
    output_dir = Path(args.output_dir) if args.output_dir else data_dir / 'figures'
    if not data_dir.is_dir():
        print(f'ERROR: Data directory not found: {data_dir}')
        return
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f'Loading scaling reports from {data_dir}...')
    reports = load_scaling_reports(data_dir, baseline_dir, args.domains)
    if not reports:
        print('No scaling reports found.')
        return
    print(f'\nGenerating figures → {output_dir}/')
    print('\n[1/5] Scaling curve...')
    plot_scaling_curve(reports, output_dir)
    print('[2/5] Per-task heatmaps...')
    plot_per_task_heatmap(reports, output_dir)
    print('[3/5] Overhead plots...')
    plot_overhead(reports, output_dir)
    print('[4/5] Trial variance...')
    plot_trial_variance(reports, output_dir)
    print('[5/5] Task stability analysis...')
    plot_task_stability(reports, output_dir)
    print_analysis(reports)
    print(f'\nAll figures saved to: {output_dir}/')
if __name__ == '__main__':
    main()
