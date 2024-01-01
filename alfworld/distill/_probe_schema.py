from __future__ import annotations
import json
from collections import Counter
from pathlib import Path
from pipeline.paths import AGENTTRAJ_L_ALFWORLD, ALFWORLD_MAPPINGS_TRAIN, STATE_DIR, DOCS_DIR

def parse_task_type_and_id(item_id: str) -> tuple[str, str]:
    (full_type, trial_suffix) = item_id.split('_trial_', 1)
    task_id = 'trial_' + trial_suffix
    coarse_type = full_type.split('-')[0]
    return (coarse_type, task_id)

def _format_conv_sample(convs):
    out = []
    for (i, t) in enumerate(convs[:4]):
        v = (t.get('value') or '')[:200].replace('\n', ' ')
        out.append({'i': i, 'from': t.get('from'), 'loss': t.get('loss'), 'value_excerpt': v})
    return out

def main():
    traj = json.loads(Path(AGENTTRAJ_L_ALFWORLD).read_text())
    mapping = json.loads(Path(ALFWORLD_MAPPINGS_TRAIN).read_text())
    by_type = Counter()
    parse_failures = []
    type_to_task_ids = {}
    for (i, t) in enumerate(traj):
        iid = t.get('item_id')
        if not isinstance(iid, str):
            parse_failures.append({'i': i, 'item_id': iid, 'reason': 'not a string'})
            continue
        try:
            (ct, tid) = parse_task_type_and_id(iid)
            by_type[ct] += 1
            type_to_task_ids.setdefault(ct, []).append(tid)
        except Exception as e:
            parse_failures.append({'i': i, 'item_id': iid, 'reason': str(e)})
    mapping_task_ids = {m['task_id'] for m in mapping}
    traj_task_ids = {tid for ids in type_to_task_ids.values() for tid in ids}
    only_in_traj = traj_task_ids - mapping_task_ids
    only_in_mapping = mapping_task_ids - traj_task_ids
    intersection = traj_task_ids & mapping_task_ids
    artifact = {'trajectory_total': len(traj), 'expected': 2420, 'match': len(traj) == 2420, 'first_entry_keys': sorted(traj[0].keys()) if traj else [], 'first_item_id_sample': traj[0].get('item_id') if traj else None, 'first_item_id_type': type(traj[0].get('item_id')).__name__ if traj else None, 'has_reward_field': 'reward' in (traj[0] if traj else {}), 'has_success_field': 'success' in (traj[0] if traj else {}), 'task_type_distribution': dict(by_type.most_common()), 'parse_failures': parse_failures[:10], 'parse_failures_total': len(parse_failures), 'alignment_with_mappings_train': {'intersection_count': len(intersection), 'only_in_traj_count': len(only_in_traj), 'only_in_mapping_count': len(only_in_mapping), 'fully_aligned': len(only_in_traj) == 0 and len(only_in_mapping) == 0}, 'first_conversations_sample': _format_conv_sample(traj[0]['conversations']) if traj else []}
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    DOCS_DIR.mkdir(parents=True, exist_ok=True)
    (STATE_DIR / 'agenttraj_schema_probe.json').write_text(json.dumps(artifact, indent=2))
    md = ['# AgentTraj-L Schema Probe (Phase 0)\n', f"**Total trajectories:** {artifact['trajectory_total']} (expected 2420, match={artifact['match']})\n", f"**Top-level keys:** {artifact['first_entry_keys']}\n", f"**item_id type:** `{artifact['first_item_id_type']}`", f"  example: `{artifact['first_item_id_sample']}`\n", f"**Has reward field:** {artifact['has_reward_field']}", f"**Has success field:** {artifact['has_success_field']}\n", f'## Task type distribution\n']
    for (k, v) in artifact['task_type_distribution'].items():
        md.append(f'- `{k}`: {v}')
    md.append(f"\n## Parse failures: {artifact['parse_failures_total']}")
    md.append(f'\n## Alignment with mappings_train.json')
    al = artifact['alignment_with_mappings_train']
    for (k, v) in al.items():
        md.append(f'- {k}: {v}')
    (DOCS_DIR / 'agenttraj_schema_probe.md').write_text('\n'.join(md))
    print(json.dumps(artifact, indent=2))
    return artifact
if __name__ == '__main__':
    main()
