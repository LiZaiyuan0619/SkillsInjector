from __future__ import annotations
import json
import random
from pathlib import Path
from .paths import CONFIGS_DIR, STATE_DIR

def write_task_lists(manifest: dict, configs_dir: Path | None=None, pre_flight_per_type: int=2, seed: int=42) -> None:
    cd = configs_dir or CONFIGS_DIR
    cd.mkdir(parents=True, exist_ok=True)
    tasks = manifest['tasks']
    full_path = cd / 'full_eval_tasks.txt'
    full_path.write_text('\n'.join((str(t['item_id']) for t in tasks)) + '\n')
    rng = random.Random(seed)
    by_type: dict[str, list[int]] = {}
    for t in tasks:
        by_type.setdefault(t['task_type'], []).append(t['item_id'])
    sel: list[int] = []
    for tt in sorted(by_type):
        ids = sorted(by_type[tt])
        rng.shuffle(ids)
        sel.extend(sorted(ids[:pre_flight_per_type]))
    pre_path = cd / 'pre_flight_tasks.txt'
    pre_path.write_text('\n'.join((str(i) for i in sorted(sel))) + '\n')
    print(f'wrote {full_path} ({len(tasks)} ids); {pre_path} ({len(sel)} ids)')
if __name__ == '__main__':
    m = json.loads((STATE_DIR / 'manifest.json').read_text())
    write_task_lists(m)
