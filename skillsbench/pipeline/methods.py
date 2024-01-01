from __future__ import annotations
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from .paths import METHODS_PATH

@dataclass(frozen=True)
class MethodSpec:
    name: str
    display_name: str
    selector_class: str
    k: int | None
    notes: str
METHODS: dict[str, MethodSpec] = {'no-skill': MethodSpec(name='no-skill', display_name='No-skill', selector_class='pipeline.selectors.no_skill.NoSkillSelector', k=0, notes='Inject zero skills. Floor baseline.'), 'random': MethodSpec(name='random', display_name='Random', selector_class='pipeline.selectors.random_sel.RandomSelector', k=3, notes='Uniform random sample of K skills from manifest pool.'), 'bm25': MethodSpec(name='bm25', display_name='BM25', selector_class='pipeline.selectors.bm25.BM25Selector', k=3, notes='rank_bm25-style sparse retrieval. Query=instruction.md, corpus=SKILL.md bodies.'), 'cosine': MethodSpec(name='cosine', display_name='Cosine', selector_class='pipeline.selectors.cosine.CosineSelector', k=3, notes='Dense retrieval via Qwen3-Embedding-0.6B. Cached on disk.'), 'full-library': MethodSpec(name='full-library', display_name='Full-library', selector_class='pipeline.selectors.full_library.FullLibrarySelector', k=None, notes='All manifest skills. Cosine-ordered, truncated to token budget; truncated_to logged.'), 'ours': MethodSpec(name='ours', display_name='Ours', selector_class='pipeline.selectors.ours.OursSelector', k=None, notes='tau2-trained selector head + adaptive budget τ* (cross-transfer).')}

def save_methods(path: Path=METHODS_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {name: asdict(spec) for (name, spec) in METHODS.items()}
    path.write_text(json.dumps(payload, indent=2))

def get_method(name: str) -> MethodSpec:
    if name not in METHODS:
        raise KeyError(f'Unknown method: {name}. Known: {list(METHODS)}')
    return METHODS[name]
