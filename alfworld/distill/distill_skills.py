from __future__ import annotations
import argparse
import hashlib
import json
import logging
import random
import re
from pathlib import Path
from pipeline.paths import AGENTTRAJ_L_ALFWORLD, STATE_DIR, TASK_TYPES, ENCODER_8B, QWEN3_8B_PATH
from pipeline.embedding import embed_texts
from runner.llm import LLMConfig, LLMClient
from distill.prompts import DISTILL_SYSTEM, render_distill_user
from distill.budget_gate import apply_budget_gate
from runner.prompt import build_messages
log = logging.getLogger(__name__)
TRAJ_PER_CALL = 3
TRACE_COVERAGE_RATIO = 0.8
TEMPERATURE = 0.7
TOP_P = 0.9
MAX_OUTPUT_TOKENS = 2000
SKILL_BLOCK_RE = re.compile('### Skill:\\s*(?P<name>.+?)\\n\\n\\*\\*Description:\\*\\*\\s*(?P<description>.+?)\\n\\n\\*\\*When to use:\\*\\*\\s*(?P<when>.+?)\\n\\n\\*\\*Steps:\\*\\*\\s*(?P<steps>.+?)(?=\\n---|\\n###\\s*Skill:|\\Z)', re.DOTALL)

def parse_task_type_id(item_id: str) -> tuple[str, str]:
    (full_type, trial_suffix) = item_id.split('_trial_', 1)
    return (full_type.split('-')[0], 'trial_' + trial_suffix)

def stringify_trajectory(entry: dict) -> str:
    convs = entry['conversations']
    lines = []
    for turn in convs[2:]:
        role = 'AGENT' if turn['from'] == 'gpt' else 'ENV'
        v = (turn.get('value') or '').strip()
        lines.append(f'[{role}] {v}')
    return '\n'.join(lines)

def parse_skill_blocks(raw: str, source_task_type: str) -> list[dict]:
    skills = []
    for m in SKILL_BLOCK_RE.finditer(raw):
        name = m.group('name').strip()
        body = f"### Skill: {name}\n\n**Description:** {m.group('description').strip()}\n\n**When to use:** {m.group('when').strip()}\n\n**Steps:** {m.group('steps').strip()}\n"
        sha = hashlib.sha256(body.encode()).hexdigest()[:8]
        slug = re.sub('[^a-z0-9]+', '_', name.lower())[:30].strip('_')
        skills.append({'skill_id': f'{slug}__{sha}', 'display_name': name, 'source_task_type': source_task_type, 'content': body, 'char_count': len(body), 'sha8': sha})
    return skills[:3]

def make_real_token_count_fn():
    import json
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(str(QWEN3_8B_PATH))
    obs_stats_path = STATE_DIR / 'observation_token_stats.json'
    if not obs_stats_path.exists():
        raise FileNotFoundError(f'{obs_stats_path} missing — run Phase 0.3 (pipeline._obs_probe) first. Budget gate cannot bound full-library prompt without real obs distribution (R7).')
    obs_stats = json.loads(obs_stats_path.read_text())
    p95_obs_tokens = obs_stats['stats']['p95']
    max_obs_tokens = obs_stats['stats']['max']
    log.info(f'  budget gate uses P95 obs tokens={p95_obs_tokens} (max={max_obs_tokens}) from Phase 0.3')
    per_task = obs_stats.get('per_task', [])
    if not per_task or 'obs_with_actions' not in (per_task[0] if per_task else {}):
        raise RuntimeError(f'R7/checklist-i: {obs_stats_path} missing per-task obs_with_actions text. Re-run Phase 0.3 with the v1.2.1 _obs_probe.py that persists obs strings.')
    p95_item = min(per_task, key=lambda x: abs(x['n_tokens'] - p95_obs_tokens))
    placeholder_obs_with_actions = p95_item['obs_with_actions']
    log.info(f"  budget gate placeholder = real obs from item_id={p95_item['item_id']} (n_tokens={p95_item['n_tokens']}, P95={p95_obs_tokens})")

    def count(skills: list[dict]) -> int:
        skill_objs = [{'display_name': s['display_name'], 'content': s['content']} for s in skills]
        if '\n\nAVAILABLE ACTIONS: ' in placeholder_obs_with_actions:
            (obs_part, av_part) = placeholder_obs_with_actions.split('\n\nAVAILABLE ACTIONS: ', 1)
            avail_list = [a.strip() for a in av_part.split(',') if a.strip()]
        else:
            (obs_part, avail_list) = (placeholder_obs_with_actions, [])
        msgs = build_messages(skill_objs, obs_part, avail_list, admissible_policy='strict-stock')
        try:
            full_input = tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False)
        except Exception:
            full_input = '\n'.join((f"<|{m['role']}|>\n{m['content']}" for m in msgs))
        return len(tok.encode(full_input))
    return count

def _load_raw_skills_from_cache() -> tuple[list[dict], dict]:
    raw_responses_dir = STATE_DIR / 'distill' / 'raw_responses'
    if not raw_responses_dir.exists():
        raise FileNotFoundError(f'{raw_responses_dir} missing — run --stage=generation first.')
    files = sorted(raw_responses_dir.glob('*.json'))
    if not files:
        raise FileNotFoundError(f'{raw_responses_dir} has no files — run --stage=generation first.')
    raw_skills: list[dict] = []
    audit = {tt: {'calls': 0, 'raw_skills': 0, 'parse_failures': 0} for tt in TASK_TYPES}
    for f in files:
        d = json.loads(f.read_text())
        tt = d['task_type']
        audit[tt]['calls'] += 1
        new = parse_skill_blocks(d['raw_response'], tt)
        if new:
            audit[tt]['raw_skills'] += len(new)
            raw_skills.extend(new)
        else:
            audit[tt]['parse_failures'] += 1
    log.info(f'  reloaded {len(files)} raw_responses from cache')
    log.info(f'  raw_skills total: {len(raw_skills)}; audit: {audit}')
    return (raw_skills, audit)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--max-calls-per-type', type=int, default=None)
    parser.add_argument('--stage', choices=['all', 'generation', 'dedup'], default='all', help='all (default): full pipeline. generation: only Stage A (235B HTTP calls + write raw_responses; no GPU/8B). dedup: skip Stage A, reload raw_responses and run 8B embedding + budget gate (Qwen3-Embedding-8B is loaded locally).')
    args = parser.parse_args()
    random.seed(args.seed)
    if args.dry_run and args.stage == 'dedup':
        raise ValueError('--dry-run incompatible with --stage=dedup (no LLM calls happen in dedup phase)')
    if args.stage == 'dedup':
        log.info('--stage=dedup: skipping Stage A LLM calls, reloading raw_responses from cache')
        (raw_skills, audit) = _load_raw_skills_from_cache()
    else:
        log.info('loading AgentTraj-L alfworld split...')
        traj = json.loads(AGENTTRAJ_L_ALFWORLD.read_text())
        by_type: dict[str, list[dict]] = {}
        for t in traj:
            try:
                (ct, _) = parse_task_type_id(t['item_id'])
                by_type.setdefault(ct, []).append(t)
            except Exception as e:
                log.warning(f'  skip entry: {e}')
        log.info(f'  by_type counts: {dict(((k, len(v)) for (k, v) in by_type.items()))}')
        rewriter: LLMClient | None = None
        if not args.dry_run:
            rew_cfg = LLMConfig.from_env(prefix='SK_REWRITER')
            rewriter = LLMClient(rew_cfg)
        raw_skills: list[dict] = []
        raw_responses_dir = STATE_DIR / 'distill' / 'raw_responses'
        raw_responses_dir.mkdir(parents=True, exist_ok=True)
        failures_file = STATE_DIR / 'distill' / 'parse_failures.jsonl'
        failures_file.parent.mkdir(parents=True, exist_ok=True)
        failures_fh = failures_file.open('w')
        audit = {tt: {'calls': 0, 'raw_skills': 0, 'parse_failures': 0} for tt in TASK_TYPES}
        for tt in TASK_TYPES:
            traces = by_type.get(tt, [])
            if not traces:
                log.warning(f'  no traces for {tt}, skipping')
                continue
            sampled = random.sample(traces, k=int(TRACE_COVERAGE_RATIO * len(traces)))
            n_calls = len(sampled) // TRAJ_PER_CALL
            if args.max_calls_per_type:
                n_calls = min(n_calls, args.max_calls_per_type)
            log.info(f'  {tt}: {len(traces)} traces, {len(sampled)} sampled, {n_calls} calls')
            for i in range(n_calls):
                batch = sampled[i * TRAJ_PER_CALL:(i + 1) * TRAJ_PER_CALL]
                traj_strs = [stringify_trajectory(b) for b in batch]
                user_msg = render_distill_user(tt, traj_strs)
                audit[tt]['calls'] += 1
                if args.dry_run:
                    continue
                try:
                    (text, _u) = rewriter.chat([{'role': 'system', 'content': DISTILL_SYSTEM}, {'role': 'user', 'content': user_msg}], temperature=TEMPERATURE, top_p=TOP_P, max_tokens=MAX_OUTPUT_TOKENS)
                except Exception as e:
                    log.warning(f'  {tt} call {i} LLM error: {e}')
                    continue
                (raw_responses_dir / f'{tt}__{i}.json').write_text(json.dumps({'task_type': tt, 'call_idx': i, 'input_item_ids': [b['item_id'] for b in batch], 'raw_response': text}, indent=2))
                new = parse_skill_blocks(text, tt)
                if not new:
                    audit[tt]['parse_failures'] += 1
                    failures_fh.write(json.dumps({'task_type': tt, 'call_idx': i, 'raw_excerpt': text[:500]}) + '\n')
                    continue
                raw_skills.extend(new)
                audit[tt]['raw_skills'] += len(new)
        failures_fh.close()
        if rewriter is not None:
            rewriter.close()
        log.info(f'=== Stage A (generation) done ===')
        log.info(f'  audit: {audit}')
        log.info(f'  raw_skills total: {len(raw_skills)}')
        if args.dry_run:
            return
        if args.stage == 'generation':
            log.info(f'--stage=generation: stopping before Stage B. Next: re-run with --stage=dedup on a GPU machine.')
            return
    seen = set()
    dedup_pre = []
    for s in raw_skills:
        if s['sha8'] in seen:
            continue
        seen.add(s['sha8'])
        dedup_pre.append(s)
    log.info(f'  pre-budget-gate dedup by sha8: {len(raw_skills)} -> {len(dedup_pre)}')
    log.info('encoding raw skills with Qwen3-Embedding-8B...')
    contents = [s['content'] for s in dedup_pre]
    emb_8b = embed_texts(contents, model_path=ENCODER_8B, cache_dir=STATE_DIR / 'embeddings/skills_8b')
    log.info(f'  emb shape: {emb_8b.shape}')
    log.info('applying budget gate...')
    token_fn = make_real_token_count_fn()
    (final_pool, meta) = apply_budget_gate(dedup_pre, emb_8b, token_fn)
    log.info(f'  final pool: {len(final_pool)}, meta={meta}')
    import shutil
    skills_dir = STATE_DIR / 'skills'
    if skills_dir.exists():
        for child in skills_dir.iterdir():
            if child.is_dir():
                shutil.rmtree(child)
    skills_dir.mkdir(parents=True, exist_ok=True)
    for s in final_pool:
        d = skills_dir / s['skill_id']
        d.mkdir(parents=True, exist_ok=True)
        (d / 'SKILL.md').write_text(s['content'])
    pool_data = {'meta': meta | {'audit': audit}, 'skills': [{k: v for (k, v) in s.items() if k != 'content'} for s in final_pool]}
    (STATE_DIR / 'skill_pool.json').write_text(json.dumps(pool_data, indent=2))
    log.info(f"wrote {STATE_DIR / 'skill_pool.json'}")
if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(name)s: %(message)s')
    main()
