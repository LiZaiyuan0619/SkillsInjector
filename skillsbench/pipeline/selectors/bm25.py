from __future__ import annotations
import math
import re
from collections import Counter
from pathlib import Path
from ..manifest import Manifest
from ..skill_pool import SkillPool
from .base import Selection, Selector
_TOKEN_RE = re.compile('[A-Za-z0-9]+')

def _tokenize(text: str) -> list[str]:
    return [t.lower() for t in _TOKEN_RE.findall(text)]

class BM25Index:

    def __init__(self, docs: list[list[str]], k1: float=1.5, b: float=0.75):
        self.k1 = k1
        self.b = b
        self.docs = docs
        self.N = len(docs)
        self.doc_len = [len(d) for d in docs]
        self.avgdl = sum(self.doc_len) / self.N if self.N else 0.0
        self.df: dict[str, int] = {}
        self.tf: list[Counter] = [Counter(d) for d in docs]
        for tfd in self.tf:
            for term in tfd:
                self.df[term] = self.df.get(term, 0) + 1
        self.idf = {t: math.log(1 + (self.N - df + 0.5) / (df + 0.5)) for (t, df) in self.df.items()}

    def score(self, query: list[str]) -> list[float]:
        scores = [0.0] * self.N
        for q in query:
            idf = self.idf.get(q)
            if idf is None:
                continue
            for (i, tfd) in enumerate(self.tf):
                f = tfd.get(q, 0)
                if not f:
                    continue
                denom = f + self.k1 * (1 - self.b + self.b * self.doc_len[i] / self.avgdl)
                scores[i] += idf * f * (self.k1 + 1) / denom
        return scores

class BM25Selector(Selector):
    name = 'bm25'

    def __init__(self, k: int=3):
        self.k = k
        self._index: BM25Index | None = None
        self._ids: list[str] = []

    def _build(self, pool: SkillPool) -> None:
        self._ids = sorted(pool.skills.keys())
        docs = [_tokenize(pool.skills[sid].body) for sid in self._ids]
        self._index = BM25Index(docs)

    def _get_query(self, task_id: str, manifest: Manifest) -> list[str]:
        for t in manifest.tasks:
            if t.task_id == task_id:
                instr = (Path(t.task_dir) / 'instruction.md').read_text(errors='replace')
                return _tokenize(instr)
        raise KeyError(f'task_id {task_id!r} not in manifest')

    def select(self, task_id: str, manifest: Manifest, pool: SkillPool, trial_idx: int, seed: int) -> Selection:
        if self._index is None:
            self._build(pool)
        assert self._index is not None
        if not self._ids:
            return Selection(skill_ids=[], method=self.name, task_id=task_id, trial_idx=trial_idx)
        q = self._get_query(task_id, manifest)
        scores = self._index.score(q)
        ranked = sorted(range(len(scores)), key=lambda i: (-scores[i], self._ids[i]))
        k = min(self.k, len(self._ids))
        top_ids = [self._ids[i] for i in ranked[:k]]
        return Selection(skill_ids=top_ids, method=self.name, task_id=task_id, trial_idx=trial_idx, extra={'k': k, 'pool_size': len(self._ids), 'top_scores': [round(scores[i], 4) for i in ranked[:k]]})
