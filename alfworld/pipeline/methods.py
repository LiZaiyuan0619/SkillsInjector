from __future__ import annotations
from dataclasses import asdict, dataclass
import json
from pathlib import Path
from .paths import STATE_DIR

@dataclass(frozen=True)
class MethodSpec:
    name: str
    display_name: str
    selector_class: str
    k: int | None
    notes: str
METHODS: dict[str, MethodSpec] = {'no-skill': MethodSpec(name='no-skill', display_name='No-skill', selector_class='pipeline.selectors.no_skill.NoSkillSelector', k=0, notes='Floor baseline. Inject zero skills.'), 'random': MethodSpec(name='random', display_name='Random', selector_class='pipeline.selectors.random_sel.RandomSelector', k=3, notes='Per-task-type bucket random K=3 (M1 fix).'), 'bm25': MethodSpec(name='bm25', display_name='BM25', selector_class='pipeline.selectors.bm25.BM25Selector', k=3, notes='Sparse BM25. Query=retrieval_query (post-reset). Corpus=SKILL.md bodies.'), 'cosine': MethodSpec(name='cosine', display_name='Cosine', selector_class='pipeline.selectors.cosine.CosineSelector', k=3, notes='Dense Qwen3-Embedding-0.6B cosine top-3.'), 'full-library': MethodSpec(name='full-library', display_name='Full-library', selector_class='pipeline.selectors.full_library.FullLibrarySelector', k=None, notes='All frozen-pool skills. Pool budget-gate ensures <=24K tokens.'), 'ours': MethodSpec(name='ours', display_name='Ours', selector_class='pipeline.selectors.ours.OursSelector', k=None, notes='Cosine + per-task min-max + tau* threshold + adaptive budget + 235B set-aware rewriter.')}

def save_methods(path: Path | None=None) -> None:
    p = path or STATE_DIR / 'methods.json'
    p.parent.mkdir(parents=True, exist_ok=True)
    payload = {name: asdict(spec) for (name, spec) in METHODS.items()}
    p.write_text(json.dumps(payload, indent=2))

def get_method(name: str) -> MethodSpec:
    if name not in METHODS:
        raise KeyError(f'unknown method {name!r}; available={list(METHODS)}')
    return METHODS[name]
