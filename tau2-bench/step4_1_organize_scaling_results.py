import argparse
import json
import re
import shutil
from pathlib import Path
DEFAULT_SOURCE = '<MODEL_ROOT>/agent/skills/scaling-data'
DEFAULT_OUTPUT = '<MODEL_ROOT>/agent/skills/scaling-organized'
PAT_TRIAL = re.compile('^(?P<domain>\\w+)_top(?P<n>\\d+)_skills_trial(?P<trial>\\d+)_seed(?P<seed>\\d+)_(?P<ts>\\d{8}_\\d{6})\\.json$')
PAT_AGG = re.compile('^(?P<domain>\\w+)_top(?P<n>\\d+)_skills_agg(?P<k>\\d+)_(?P<ts>\\d{8}_\\d{6})\\.json$')
PAT_REPORT = re.compile('^scaling_report_(?P<domain>\\w+)_(?P<ts>\\d{8}_\\d{6})\\.json$')

def classify_file(filename: str) -> tuple[str, dict] | None:
    m = PAT_TRIAL.match(filename)
    if m:
        return ('trial', m.groupdict())
    m = PAT_AGG.match(filename)
    if m:
        return ('agg', m.groupdict())
    m = PAT_REPORT.match(filename)
    if m:
        return ('report', m.groupdict())
    return None

def compute_destination(file_type: str, meta: dict, filename: str, output_dir: Path) -> Path:
    domain = meta['domain']
    if file_type == 'trial':
        n = meta['n']
        suffix = f"trial{meta['trial']}_seed{meta['seed']}_{meta['ts']}.json"
        return output_dir / domain / 'trials' / f'top{n}' / suffix
    elif file_type == 'agg':
        n = meta['n']
        suffix = f"top{n}_agg{meta['k']}_{meta['ts']}.json"
        return output_dir / domain / 'aggregated' / suffix
    elif file_type == 'report':
        suffix = f"scaling_report_{meta['ts']}.json"
        return output_dir / domain / suffix
    raise ValueError(f'Unknown file_type: {file_type}')

def collect_files(source_dir: Path) -> list[tuple[Path, str, dict]]:
    results = []
    if source_dir.is_dir():
        for f in sorted(source_dir.glob('*.json')):
            classified = classify_file(f.name)
            if classified:
                (file_type, meta) = classified
                results.append((f, file_type, meta))
    return results

def merge_scaling_reports(output_dir: Path):
    for domain_dir in sorted(output_dir.iterdir()):
        if not domain_dir.is_dir() or domain_dir.name in ('figures', 'comparison', 'comparison_three'):
            continue
        reports = sorted(domain_dir.glob('scaling_report_*.json'))
        reports = [r for r in reports if 'merged' not in r.name]
        if len(reports) <= 1:
            continue
        merged_summary = {}
        merged_per_task = {}
        base_report = None
        for rp in reports:
            data = json.load(open(rp))
            if base_report is None:
                base_report = data
            for entry in data.get('scaling_summary', []):
                n = entry['n_skills']
                merged_summary[n] = entry
            for task_entry in data.get('per_task_comparison', []):
                tid = task_entry['task_id']
                if tid not in merged_per_task:
                    merged_per_task[tid] = {'task_id': tid}
                for (k, v) in task_entry.items():
                    if k != 'task_id':
                        merged_per_task[tid][k] = v
        if not base_report or not merged_summary:
            continue
        base_report['scaling_summary'] = sorted(merged_summary.values(), key=lambda x: x['n_skills'])
        if merged_per_task:
            base_report['per_task_comparison'] = sorted(merged_per_task.values(), key=lambda x: int(x['task_id']) if x['task_id'].isdigit() else x['task_id'])
        out_path = domain_dir / 'scaling_report_merged.json'
        with open(out_path, 'w') as f:
            json.dump(base_report, f, indent=2)
        domain = domain_dir.name
        scales = [s['n_skills'] for s in base_report['scaling_summary']]
        n_tasks = len(merged_per_task)
        n_keys = len(next(iter(merged_per_task.values()))) - 1 if merged_per_task else 0
        print(f'  Merged {len(reports)} reports for {domain}: N={scales}, {n_tasks} tasks × {n_keys} columns → {out_path.name}')

def print_summary(files: list[tuple[Path, str, dict]], output_dir: Path):
    domains = {}
    for (src, ftype, meta) in files:
        domain = meta['domain']
        if domain not in domains:
            domains[domain] = {'trial': 0, 'agg': 0, 'report': 0}
        domains[domain][ftype] += 1
    print(f"\n{'=' * 60}")
    print(f'  ORGANIZE SCALING RESULTS')
    print(f"{'=' * 60}")
    print(f'  Output: {output_dir}')
    print()
    for domain in sorted(domains):
        counts = domains[domain]
        total = sum(counts.values())
        scales = set()
        for (_, ftype, meta) in files:
            if meta['domain'] == domain and ftype in ('trial', 'agg'):
                scales.add(int(meta['n']))
        scales_str = ', '.join((str(s) for s in sorted(scales)))
        print(f'  {domain}:')
        print(f'    Scale points: [{scales_str}]')
        print(f"    Trials: {counts['trial']}, Aggregated: {counts['agg']}, Reports: {counts['report']}, Total: {total}")
    print(f'\n  Grand total: {len(files)} files')
    print(f"{'=' * 60}")

def organize(files: list[tuple[Path, str, dict]], output_dir: Path, move: bool, dry_run: bool) -> tuple[int, int]:
    success = 0
    skipped = 0
    action = 'MOVE' if move else 'COPY'
    for (src, ftype, meta) in files:
        dst = compute_destination(ftype, meta, src.name, output_dir)
        if dst.exists():
            if dst.stat().st_size == src.stat().st_size:
                skipped += 1
                if dry_run:
                    print(f'  [SKIP] {src.name} (already exists)')
                continue
        if dry_run:
            print(f'  [{action}] {src.name}')
            print(f'       → {dst}')
            success += 1
            continue
        dst.parent.mkdir(parents=True, exist_ok=True)
        if move:
            shutil.move(str(src), str(dst))
        else:
            shutil.copy2(str(src), str(dst))
        success += 1
    return (success, skipped)

def main():
    parser = argparse.ArgumentParser(description='Step 4.1: Organize scaling experiment results')
    parser.add_argument('--source-dir', default=DEFAULT_SOURCE, help=f'Flat source directory (default: {DEFAULT_SOURCE})')
    parser.add_argument('--output-dir', default=DEFAULT_OUTPUT, help=f'Organized output directory (default: {DEFAULT_OUTPUT})')
    parser.add_argument('--move', action='store_true', help='Move files instead of copying (default: copy)')
    parser.add_argument('--dry-run', action='store_true', help='Preview actions without making changes')
    args = parser.parse_args()
    source_dir = Path(args.source_dir)
    output_dir = Path(args.output_dir)
    if not source_dir.is_dir():
        print(f'ERROR: Source directory not found: {source_dir}')
        return
    files = collect_files(source_dir)
    if not files:
        print('No recognizable files found.')
        return
    print_summary(files, output_dir)
    if args.dry_run:
        print(f"\n--- DRY RUN (no files will be {('moved' if args.move else 'copied')}) ---\n")
    else:
        action = 'Moving' if args.move else 'Copying'
        print(f'\n--- {action} files ---\n')
    (success, skipped) = organize(files, output_dir, args.move, args.dry_run)
    print(f"\nDone: {success} {('would be ' if args.dry_run else '')}processed, {skipped} skipped (already exist)")
    if not args.dry_run:
        print('\nChecking for parallel scaling reports to merge...')
        merge_scaling_reports(output_dir)
    if args.dry_run:
        mode = '--move' if args.move else ''
        print(f"\nTo execute: remove --dry-run flag{(' (with ' + mode + ')' if mode else '')}")
if __name__ == '__main__':
    main()
