import argparse
import json
import os
from pathlib import Path

def main():
    parser = argparse.ArgumentParser(description='Generate utility-ranked skill JSON')
    parser.add_argument('--report', required=True, help='Path to single_skill_report JSON (step5 output)')
    parser.add_argument('--ranked-json', required=True, help='Path to original skills_ranked.json (for code/file fields)')
    parser.add_argument('--outdir', default=None, help='Output directory (default: same as ranked-json)')
    args = parser.parse_args()
    with open(args.report) as f:
        report = json.load(f)
    utility_ranking = report['utility_ranking']
    baseline = report['experiment_info']['baseline_pass_rate_mean']
    with open(args.ranked_json) as f:
        original_ranked = json.load(f)
    original_lookup = {}
    for entry in original_ranked:
        name = entry['skill_name']
        if name not in original_lookup:
            original_lookup[name] = entry
    utility_ranked = []
    for (rank_idx, u_entry) in enumerate(utility_ranking):
        name = u_entry['skill_name']
        if name not in original_lookup:
            print(f'  WARNING: {name} not found in original ranked JSON, skipping')
            continue
        orig = original_lookup[name]
        new_entry = {'rank': rank_idx, 'skill_name': name, 'file': orig['file'], 'cluster_id': orig.get('cluster_id', -1), 'cluster_size': orig.get('cluster_size', 0), 'frequency': orig['frequency'], 'code': orig['code'], 'utility_delta_mean': round(u_entry['delta_pass_rate_mean'], 4), 'utility_pass_rate_mean': round(u_entry['pass_rate_mean'], 4), 'utility_delta_majority': round(u_entry['delta_majority_pass_rate'], 4), 'utility_net_task_change': u_entry['net_task_change']}
        utility_ranked.append(new_entry)
    if args.outdir is None:
        args.outdir = os.path.dirname(args.ranked_json)
    os.makedirs(args.outdir, exist_ok=True)
    outpath = os.path.join(args.outdir, 'skills_reranked_utility.json')
    with open(outpath, 'w') as f:
        json.dump(utility_ranked, f, indent=2, ensure_ascii=False)
    print(f"\n{'=' * 70}")
    print(f'  UTILITY-RANKED SKILL LIST GENERATED')
    print(f"{'=' * 70}")
    print(f'  Source report: {args.report}')
    print(f'  Baseline pass@3 mean: {baseline * 100:.1f}%')
    print(f'  Total skills ranked: {len(utility_ranked)}')
    print(f"  Positive (Δ>0): {sum((1 for e in utility_ranked if e['utility_delta_mean'] > 0))}")
    print(f"  Neutral  (Δ=0): {sum((1 for e in utility_ranked if e['utility_delta_mean'] == 0))}")
    print(f"  Negative (Δ<0): {sum((1 for e in utility_ranked if e['utility_delta_mean'] < 0))}")
    print(f'  Output: {outpath}')
    print(f"{'=' * 70}")
    print(f'\n  Top-10:')
    for e in utility_ranked[:10]:
        print(f"    #{e['rank'] + 1:2d}  Δ={e['utility_delta_mean'] * 100:+5.1f}%  freq={e['frequency']:5d}  {e['skill_name']}")
    print(f'\n  Bottom-5:')
    for e in utility_ranked[-5:]:
        print(f"    #{e['rank'] + 1:2d}  Δ={e['utility_delta_mean'] * 100:+5.1f}%  freq={e['frequency']:5d}  {e['skill_name']}")
    print()
if __name__ == '__main__':
    main()
