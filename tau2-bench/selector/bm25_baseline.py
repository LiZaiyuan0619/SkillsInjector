import argparse
import json
import math
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
from temp.selector.common import resolve_domain_spec
from temp.selector.datasets import load_pair_records
from temp.selector.offline_eval import _load_exported_dataset
_TOKEN_RE = re.compile('[A-Za-z0-9_]+')

def _tokenize(text: str) -> list[str]:
    return [t for t in _TOKEN_RE.findall(text.lower()) if t]

class BM25Okapi:

    def __init__(self, corpus: list[list[str]], k1: float=1.5, b: float=0.75) -> None:
        self.k1 = k1
        self.b = b
        self.corpus_size = len(corpus)
        self.doc_lens = [len(doc) for doc in corpus]
        self.avgdl = sum(self.doc_lens) / max(self.corpus_size, 1)
        self.doc_freqs = [Counter(doc) for doc in corpus]
        self.idf: dict[str, float] = {}
        df: Counter = Counter()
        for doc in corpus:
            for term in set(doc):
                df[term] += 1
        for (term, freq) in df.items():
            self.idf[term] = math.log(1 + (self.corpus_size - freq + 0.5) / (freq + 0.5))

    def scores(self, query: list[str]) -> list[float]:
        out = [0.0] * self.corpus_size
        for term in query:
            idf = self.idf.get(term)
            if idf is None:
                continue
            for (i, freqs) in enumerate(self.doc_freqs):
                f = freqs.get(term, 0)
                if f == 0:
                    continue
                dl = self.doc_lens[i]
                out[i] += idf * (f * (self.k1 + 1)) / (f + self.k1 * (1 - self.b + self.b * dl / self.avgdl))
        return out

def bm25_select(domain: str, data_dir: Path) -> list[dict[str, Any]]:
    spec = resolve_domain_spec(domain)
    (_manifest, tasks_by_id, skill_names, _labels) = _load_exported_dataset(domain=spec.domain, data_dir=data_dir, budget=1, validate_budget=False)
    pair_records = load_pair_records(data_dir)
    task_texts = {r.task_id: r.task_text for r in pair_records}
    skill_texts = {r.skill_name: r.skill_text for r in pair_records}
    corpus = [_tokenize(skill_texts[name]) for name in skill_names]
    bm25 = BM25Okapi(corpus)
    rows: list[dict[str, Any]] = []
    for task_id in sorted(tasks_by_id):
        query_tokens = _tokenize(task_texts[task_id])
        scores = bm25.scores(query_tokens)
        ranked_indices = sorted(range(len(skill_names)), key=lambda i: (-scores[i], skill_names[i]))
        top_idx = ranked_indices[0]
        selected_name = skill_names[top_idx]
        rows.append({'domain': spec.domain, 'task_id': task_id, 'method': 'trained_selector', 'budget': 1, 'candidate_pool_size': None, 'training_scope': 'single_domain', 'selected_skill_name': selected_name, 'selected_skill_names': [selected_name], 'selection_seed': None, 'selection_score': float(scores[top_idx]), 'selection_scores': [float(scores[top_idx])], 'selection_rank': 1, 'selection_source': 'bm25', 'checkpoint_path': None})
    return rows

def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text('\n'.join((json.dumps(row, sort_keys=True) for row in rows)) + ('\n' if rows else ''), encoding='utf-8')

def main() -> None:
    parser = argparse.ArgumentParser(description='BM25 retrieval baseline (top-1)')
    parser.add_argument('--domain', required=True)
    parser.add_argument('--data-dir', required=True)
    parser.add_argument('--output', required=True, help='Output selection JSONL path')
    args = parser.parse_args()
    rows = bm25_select(args.domain, Path(args.data_dir))
    output_path = Path(args.output)
    _write_jsonl(output_path, rows)
    print(json.dumps({'domain': args.domain, 'num_rows': len(rows), 'output': str(output_path), 'selection_source': 'bm25', 'row_method_field': 'trained_selector'}))
if __name__ == '__main__':
    main()
