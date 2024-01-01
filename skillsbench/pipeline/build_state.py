from __future__ import annotations
import argparse
import logging
import sys
from .manifest import build_manifest, save_manifest
from .methods import save_methods
from .paths import METHODS_PATH
from .skill_pool import build_skill_pool, save_skill_pool

def main(argv: list[str] | None=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument('--with-embeddings', action='store_true', help='Pre-compute Qwen3-Embedding-0.6B vectors for all skills and tasks.')
    args = parser.parse_args(argv if argv is not None else sys.argv[1:])
    logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(name)s: %(message)s')
    print('[1/3] building manifest...')
    m = build_manifest()
    save_manifest(m)
    print(f'      kept {m.n_kept} tasks, excluded {m.n_excluded}')
    print('[2/3] building skill pool...')
    p = build_skill_pool(m)
    save_skill_pool(p)
    print(f'      {p.n_unique_skills} unique skills from {p.n_files_seen} SKILL.md files')
    print('[3/3] saving methods registry...')
    save_methods()
    print(f'      wrote {METHODS_PATH}')
    if args.with_embeddings:
        print('[bonus] pre-computing embeddings (this can take 1-3 min on CPU)...')
        from .selectors.cosine import CosineSelector
        sel = CosineSelector()
        sel.warmup(m, p)
        print(f'        skill emb: {sel.get_skill_emb_view()[1].shape}')
        print(f'        task emb cache: {len(sel._task_emb_cache)} tasks')
    return 0
if __name__ == '__main__':
    sys.exit(main())
