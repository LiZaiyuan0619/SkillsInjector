from __future__ import annotations
import argparse
import glob
import json
import logging
from collections import defaultdict
from pathlib import Path
import numpy as np
from scipy import stats
log = logging.getLogger(__name__)
METHODS = ['no-skill', 'random', 'bm25', 'cosine', 'full-library', 'ours']
N_BOOTSTRAP = 10000
SEED = 42

def _expected_keys(expected: list[dict]) -> set[tuple[str, str, int]]:
    return {(e['task_id'], e['method'], int(e['trial_idx'])) for e in expected}

def load_records(jobs_dir: Path, expected: list[dict]) -> dict[str, dict[str, list[float]]]:
    expected_keys = _expected_keys(expected)
    seen: dict[tuple[str, str, int], str] = {}
    extras: list[tuple[str, tuple]] = []
    corrupt: list[tuple[str, str]] = []
    out: dict[str, dict[str, list[float]]] = defaultdict(lambda : defaultdict(list))
    for f in glob.glob(str(jobs_dir / '*/*.json')):
        try:
            r = json.load(open(f))
            spec = r['spec']
            k = (spec['task_id'], spec['method'], int(spec['trial_idx']))
            reward = float(r['reward'])
        except Exception as e:
            corrupt.append((f, str(e)[:200]))
            continue
        if k in seen:
            corrupt.append((f, f'duplicate key already loaded from {seen[k]}'))
            continue
        if k not in expected_keys:
            extras.append((f, k))
            continue
        seen[k] = f
        out[spec['method']][spec['task_id']].append(reward)
    missing = expected_keys - set(seen)
    problems: list[str] = []
    if corrupt:
        problems.append(f'{len(corrupt)} corrupt or duplicate trial JSONs (first 5: {corrupt[:5]})')
    if extras:
        problems.append(f'{len(extras)} extra trial JSONs not in expected_trials.json (stale re-run residue? first 5 keys: {[k for (_, k) in extras[:5]]})')
    if missing:
        problems.append(f'{len(missing)} expected trials have NO JSON on disk (first 5: {sorted(missing)[:5]})')
    if problems:
        raise RuntimeError('aggregator integrity: ' + '; '.join(problems) + f'. jobs_dir={jobs_dir}')
    return out

def task_means(records: dict) -> dict[str, dict[str, float]]:
    return {m: {tid: float(np.mean(rs)) for (tid, rs) in recs.items()} for (m, recs) in records.items()}

def paired_bootstrap_ci(diffs: list[float], alpha: float=0.05, n: int=N_BOOTSTRAP, seed: int=SEED):
    rng = np.random.default_rng(seed)
    diffs_arr = np.array(diffs)
    boot = rng.choice(diffs_arr, size=(n, len(diffs_arr)), replace=True).mean(axis=1)
    return (float(np.percentile(boot, 100 * alpha / 2)), float(np.percentile(boot, 100 * (1 - alpha / 2))))

def holm_bonferroni(p_values: list[float]) -> list[float]:
    m = len(p_values)
    order = sorted(range(m), key=lambda i: p_values[i])
    adj = [0.0] * m
    prev = 0.0
    for (k, idx) in enumerate(order):
        a = min(1.0, max(prev, (m - k) * p_values[idx]))
        adj[idx] = a
        prev = a
    return adj

def check_integrity(jobs_dir: Path, allow_failures: bool) -> list[dict]:
    expected_file = jobs_dir / 'expected_trials.json'
    summary_file = jobs_dir / 'completion_summary.json'
    if not expected_file.exists() or not summary_file.exists():
        raise FileNotFoundError(f'R4: missing expected_trials.json / completion_summary.json in {jobs_dir}')
    expected = json.loads(expected_file.read_text())
    summary = json.loads(summary_file.read_text())
    n_expected = summary['expected']
    n_completed = summary['completed_files']
    n_failed = summary['failed']
    n_missing = summary['missing']
    if n_expected != len(expected):
        raise RuntimeError(f'R4: completion_summary.expected={n_expected} != len(expected_trials.json)={len(expected)}. Inconsistent run artifacts.')
    if n_missing > 0:
        raise RuntimeError(f'R4 INTEGRITY FAIL: {n_missing} missing trial files. This means a worker crashed without writing failure-stub JSON. Investigate before aggregating.')
    if n_completed + n_failed != n_expected:
        raise RuntimeError(f'R4 INTEGRITY FAIL: completed_files={n_completed} + failed={n_failed} != expected={n_expected}. Some trials lost. Investigate worker crashes / orphan stub files.')
    if n_failed > 0 and (not allow_failures):
        raise RuntimeError(f'R4: {n_failed} trials have stop_reason in (env_error/api_error/selection_error). They contribute reward=0 to Tab2; if intentional pass --allow-failures, else investigate.')
    log.info(f'  integrity: expected={n_expected}, completed={n_completed}, failed={n_failed}, missing={n_missing}')
    return expected

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--jobs-dir', required=True)
    p.add_argument('--manifest', default='state/manifest.json')
    p.add_argument('--output-prefix', default='jobs/aggregated')
    p.add_argument('--allow-failures', action='store_true', help="R4: don't error on stop_reason=env_error/api_error/selection_error trials (they still contribute reward=0)")
    args = p.parse_args()
    jobs_dir = Path(args.jobs_dir)
    expected = check_integrity(jobs_dir, args.allow_failures)
    all_tids = sorted({e['task_id'] for e in expected})
    type_of: dict[str, str] = {e['task_id']: e.get('task_type') for e in expected if e.get('task_type') is not None}
    if not type_of:
        manifest = json.loads(Path(args.manifest).read_text())
        type_of = {t['task_id']: t['task_type'] for t in manifest['tasks']}
    records = load_records(jobs_dir, expected)
    means = task_means(records)
    methods_in_run = {e['method'] for e in expected}
    methods_to_report = [m for m in METHODS if m in methods_in_run]
    if 'ours' not in methods_to_report:
        raise RuntimeError(f"aggregator: 'ours' not in this run's methods ({methods_to_report}); Tab2 requires the Ours reference column. Re-run with --methods including ours.")
    print('\n=== Tab2 ALFWorld (overall) ===')
    print(f"{'method':16s} {'pass_rate':10s} {'95% CI vs Ours':25s} {'Holm-adj p':12s}")
    ours_per = [means['ours'][tid] for tid in all_tids]
    rows = []
    p_values: list[float] = []
    for m in methods_to_report:
        per = [means[m][tid] for tid in all_tids]
        rate = float(np.mean(per))
        if m == 'ours':
            rows.append((m, rate, None, None, None, None))
        else:
            diffs = [ours_per[i] - per[i] for i in range(len(all_tids))]
            (lo, hi) = paired_bootstrap_ci(diffs)
            (t_stat, pv) = stats.ttest_rel(ours_per, per)
            p_values.append(float(pv))
            rows.append((m, rate, lo, hi, float(t_stat), float(pv)))
    adj = holm_bonferroni(p_values)
    pi = 0
    for r in rows:
        (m, rate, lo, hi, t, _p) = r
        if lo is None:
            print(f'  {m:14s} {rate:.2%}   (reference)')
        else:
            ap = adj[pi]
            pi += 1
            print(f'  {m:14s} {rate:.2%}   [{lo:+.2%}, {hi:+.2%}]   p_adj={ap:.4f}')
    print('\n=== Appendix C: per task_type ===')
    types = sorted(set(type_of.values()))
    print(f"{'type':35s} " + ' '.join((f'{m[:10]:>10s}' for m in methods_to_report)))
    for tt in types:
        ids = [tid for tid in all_tids if type_of[tid] == tt]
        cells = [tt[:35]]
        for m in methods_to_report:
            r = float(np.mean([means[m][tid] for tid in ids]))
            cells.append(f'{r:.2%}')
        print(f'  {cells[0]:35s} ' + ' '.join((f'{c:>10s}' for c in cells[1:])))
    out = Path(args.output_prefix + '_tab2.json')
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({'method_pass_rates': {m: float(np.mean([means[m][tid] for tid in all_tids])) for m in methods_to_report}, 'n_tasks': len(all_tids), 'n_bootstrap': N_BOOTSTRAP, 'holm_bonferroni_adjusted_p': dict(zip([m for m in methods_to_report if m != 'ours'], adj))}, indent=2))
    log.info(f'wrote {out}')
if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(name)s: %(message)s')
    main()
