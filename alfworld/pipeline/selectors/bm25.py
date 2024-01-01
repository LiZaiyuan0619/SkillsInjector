import math
import re
from collections import Counter
from .base import Selector, Selection
_TOK = re.compile('[A-Za-z0-9]+')

def _tokenize(text: str) -> list[str]:
    return [t.lower() for t in _TOK.findall(text)]

class BM25Selector(Selector):
    name = 'bm25'

    def __init__(self, pool, k: int=3, k1: float=1.5, b: float=0.75):
        super().__init__(pool)
        self.k = k
        (self.k1, self.b) = (k1, b)
        self._docs = None
        self._idf = None
        self._avgdl = None
        self._doc_len = None
        self._tf = None

    def warmup(self):
        if self._docs is not None:
            return
        self._docs = [_tokenize(s.body) for s in self.pool.skills]
        self._doc_len = [len(d) for d in self._docs]
        N = len(self._docs)
        self._avgdl = sum(self._doc_len) / max(N, 1)
        df: dict[str, int] = {}
        self._tf = [Counter(d) for d in self._docs]
        for tfd in self._tf:
            for t in tfd:
                df[t] = df.get(t, 0) + 1
        self._idf = {t: math.log(1 + (N - d + 0.5) / (d + 0.5)) for (t, d) in df.items()}

    def _score(self, query_tokens: list[str]) -> list[float]:
        scores = [0.0] * len(self._docs)
        for q in query_tokens:
            idf = self._idf.get(q)
            if idf is None:
                continue
            for (i, tfd) in enumerate(self._tf):
                tf = tfd.get(q, 0)
                if tf == 0:
                    continue
                norm = self.k1 * (1 - self.b + self.b * self._doc_len[i] / self._avgdl)
                scores[i] += idf * (tf * (self.k1 + 1)) / (tf + norm)
        return scores

    def select(self, task: dict, query: str, trial_idx: int, seed: int) -> Selection:
        self.warmup()
        scores = self._score(_tokenize(query))
        order = sorted(range(len(scores)), key=lambda i: -scores[i])[:self.k]
        chosen = [self.pool.skills[i] for i in order]
        return Selection(skill_ids=[s.skill_id for s in chosen], method=self.name, task_id=task['task_id'], trial_idx=trial_idx, selected_skills=chosen, extra={'top_scores': [float(scores[i]) for i in order]})
