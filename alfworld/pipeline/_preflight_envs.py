from __future__ import annotations
import json
import logging
import os
import time
from pathlib import Path
from .paths import STATE_DIR, ITEM_ID_MIN, ITEM_ID_MAX
from runner.envserver_client import EnvServerClient
log = logging.getLogger(__name__)

def main():
    manifest = json.loads((STATE_DIR / 'manifest.json').read_text())
    env_url = os.environ.get('AGENTGYM_ENVSERVER', 'http://127.0.0.1:36001')
    (failures, oks) = ([], [])
    for t in manifest['tasks']:
        iid = t['item_id']
        assert ITEM_ID_MIN <= iid <= ITEM_ID_MAX, f'item_id {iid} out of range'
        try:
            with EnvServerClient(base_url=env_url) as env:
                eid = env.create()
                r = env.reset(eid, game=iid, world_type='Text')
                if 'observation' not in r or not r.get('available_actions'):
                    failures.append((iid, t['task_id'], 'no obs/actions'))
                    continue
                step = env.step(eid, r['available_actions'][0])
                if 'error' in step:
                    failures.append((iid, t['task_id'], f"step err: {step['error']}"))
                    continue
                oks.append(t)
        except Exception as e:
            failures.append((iid, t['task_id'], str(e)[:120]))
    log.info(f'preflight: {len(oks)} ok, {len(failures)} failed')
    if failures:
        new_manifest = {'tasks': oks, 'n_tasks': len(oks), 'excluded': [{'item_id': i, 'task_id': t, 'err': e} for (i, t, e) in failures]}
        backup = STATE_DIR / f'manifest.json.bak.{int(time.time())}'
        backup.write_text((STATE_DIR / 'manifest.json').read_text())
        (STATE_DIR / 'manifest.json').write_text(json.dumps(new_manifest, indent=2))
        log.warning(f'  rewrote manifest with {len(oks)} tasks (excluded {len(failures)}); backup at {backup}')
        from .task_lists import write_task_lists
        write_task_lists(new_manifest)
        log.info(f'  regenerated full_eval_tasks.txt + pre_flight_tasks.txt with new manifest')
if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(name)s: %(message)s')
    main()
