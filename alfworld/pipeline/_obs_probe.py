from __future__ import annotations
import json
import logging
import os
from pathlib import Path
import numpy as np
from pipeline.paths import STATE_DIR, QWEN3_8B_PATH, ALFWORLD_MAPPINGS_TEST, ITEM_ID_MIN, ITEM_ID_MAX
from runner.envserver_client import EnvServerClient
log = logging.getLogger(__name__)

def main():
    raw = json.loads(ALFWORLD_MAPPINGS_TEST.read_text())
    manifest = {'tasks': [{'item_id': int(e['item_id']), 'task_id': e['task_id'], 'task_type': e['task_type'].split('-')[0]} for e in raw if ITEM_ID_MIN <= int(e['item_id']) <= ITEM_ID_MAX]}
    env_url = os.environ.get('AGENTGYM_ENVSERVER', 'http://127.0.0.1:36001')
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(str(QWEN3_8B_PATH))
    obs_tokens = []
    failures = []
    for t in manifest['tasks']:
        iid = t['item_id']
        try:
            with EnvServerClient(base_url=env_url) as env:
                eid = env.create()
                r = env.reset(eid, game=iid, world_type='Text')
                obs = r['observation']
                avail = r.get('available_actions', [])
                obs_with_actions = obs.rstrip() + '\n\nAVAILABLE ACTIONS: ' + ', '.join(avail)
                n_tok = len(tok.encode(obs_with_actions))
                obs_tokens.append({'item_id': iid, 'task_id': t['task_id'], 'task_type': t['task_type'], 'n_tokens': n_tok, 'obs_with_actions': obs_with_actions})
        except Exception as e:
            failures.append({'item_id': iid, 'err': str(e)[:120]})
    counts = np.array([o['n_tokens'] for o in obs_tokens])
    stats = {'n_probed': len(obs_tokens), 'n_failed': len(failures), 'min': int(counts.min()) if len(counts) else 0, 'median': int(np.median(counts)) if len(counts) else 0, 'p95': int(np.percentile(counts, 95)) if len(counts) else 0, 'max': int(counts.max()) if len(counts) else 0, 'mean': int(counts.mean()) if len(counts) else 0}
    log.info(f'observation token stats: {stats}')
    log.info(f"failures: {(failures[:5] if failures else 'none')}")
    out = {'stats': stats, 'per_task': obs_tokens, 'failures': failures}
    out_path = STATE_DIR / 'observation_token_stats.json'
    out_path.write_text(json.dumps(out, indent=2))
    log.info(f'wrote {out_path}')
    return stats
if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(name)s: %(message)s')
    main()
