import argparse
import json
import sys
from pathlib import Path
from typing import Any
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding='utf-8'))
    if not isinstance(payload, dict):
        raise ValueError(f'Expected JSON object at {path}')
    return payload

def _write_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding='utf-8')

def _write_markdown(path: Path, content: str) -> None:
    path.write_text(content, encoding='utf-8')

def _sort_key(row: dict[str, Any]) -> tuple[str, str, int, int, str]:
    return (str(row['domain']), str(row.get('training_scope', 'single_domain')), int(row.get('budget', 1)), int(row.get('candidate_pool_size', 0) or 0), str(row['method']))

def _discover_online_aggregates(results_root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for aggregate_path in sorted(results_root.glob('*/online/**/aggregate.json')):
        payload = _read_json(aggregate_path)
        rows.append(payload)
    return sorted(rows, key=_sort_key)

def _build_markdown(rows: list[dict[str, Any]]) -> str:
    lines = ['# Benchmark Summary', '', '| Domain | Training Scope | Budget | Candidate Pool Size | Method | Trials | Mean Pass Rate | Std Pass Rate | Max Concurrency |', '| --- | --- | ---: | ---: | --- | ---: | ---: | ---: | ---: |']
    for row in rows:
        lines.append('| ' + ' | '.join([str(row['domain']), str(row.get('training_scope', 'single_domain')), str(int(row.get('budget', 1))), str(int(row.get('candidate_pool_size', 0) or 0)), str(row['method']), str(row['num_trials']), f"{float(row['mean_pass_rate']):.4f}", f"{float(row['std_pass_rate']):.4f}", str(int(row.get('max_concurrency', 1)))]) + ' |')
    lines.append('')
    return '\n'.join(lines)

def summarize_online_results(results_root: str | Path, output_dir: str | Path) -> dict[str, Any]:
    results_root = Path(results_root)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = _discover_online_aggregates(results_root)
    payload = {'num_results': len(rows), 'domains': sorted({str(row['domain']) for row in rows}), 'methods': sorted({str(row['method']) for row in rows}), 'training_scopes': sorted({str(row.get('training_scope', 'single_domain')) for row in rows}), 'budgets': sorted({int(row.get('budget', 1)) for row in rows}), 'candidate_pool_sizes': sorted({int(row.get('candidate_pool_size', 0) or 0) for row in rows}), 'results': rows}
    _write_json(output_dir / 'benchmark_summary.json', payload)
    _write_markdown(output_dir / 'summary.md', _build_markdown(rows))
    return payload

def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description='Summarize multidomain online selector results')
    parser.add_argument('--results-root', required=True)
    parser.add_argument('--output-dir', required=True)
    return parser

def main() -> None:
    parser = _build_arg_parser()
    args = parser.parse_args()
    payload = summarize_online_results(results_root=args.results_root, output_dir=args.output_dir)
    print(json.dumps(payload))
if __name__ == '__main__':
    main()
