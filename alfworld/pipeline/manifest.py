from __future__ import annotations
import json
import logging
from collections import Counter
from .paths import ALFWORLD_MAPPINGS_TEST, ITEM_ID_MIN, ITEM_ID_MAX, STATE_DIR, TASK_TYPES
log = logging.getLogger(__name__)

def build_manifest() -> dict:
    raw = json.loads(ALFWORLD_MAPPINGS_TEST.read_text())
    if len(raw) != 200:
        raise RuntimeError(f'mappings_test.json has {len(raw)} entries, expected 200')
    tasks = []
    for entry in raw:
        item_id = int(entry['item_id'])
        if not ITEM_ID_MIN <= item_id <= ITEM_ID_MAX:
            raise RuntimeError(f'item_id {item_id} out of [{ITEM_ID_MIN}, {ITEM_ID_MAX}] — manifest hard invariant violated (spec §6.1)')
        coarse = entry['task_type'].split('-')[0]
        if coarse not in TASK_TYPES:
            raise RuntimeError(f'unknown task_type {coarse!r} in {entry!r}')
        tasks.append({'item_id': item_id, 'task_id': entry['task_id'], 'task_type': coarse, 'task_type_full': entry['task_type']})
    return {'tasks': tasks, 'n_tasks': len(tasks)}

def save_manifest(m: dict, path=None) -> None:
    p = path or STATE_DIR / 'manifest.json'
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(m, indent=2))
    log.info(f"wrote {p} ({m['n_tasks']} tasks)")
if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(name)s: %(message)s')
    m = build_manifest()
    save_manifest(m)
    print(f"task_type distribution: {dict(Counter((t['task_type'] for t in m['tasks'])))}")
