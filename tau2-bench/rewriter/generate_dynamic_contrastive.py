import argparse
import json
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from .utils import get_client, call_llm, parse_json_from_llm, save_jsonl
from .generate_contrastive_v2 import load_all_skills
SYSTEM_MESSAGE = 'You generate concise contrastive scope descriptions for AI agent tools. Output valid JSON only. No markdown fences. No extra explanation.'

def build_dynamic_prompt(target: dict, partners: list[dict]) -> str:
    partner_blocks = []
    for p in partners:
        partner_blocks.append(f"### {p['skill_name']}\nGuidance: {p['prompt_guidance']}\nDescription: {p['docstring']}\nFunction body:\n{p['body_source']}\n")
    partner_context = '\n'.join(partner_blocks)
    target_block = f"Guidance: {target['prompt_guidance']}\nDescription: {target['docstring']}\nFunction body:\n{target['body_source']}\n"
    unlike_parts = []
    for p in partners:
        unlike_parts.append(f"Unlike {p['skill_name']} which <key difference>.")
    unlike_example = ' '.join(unlike_parts)
    return f'''An AI customer service agent will receive {len(partners) + 1} skill-tools simultaneously for a task. It must choose the right one at each step. Your job: add a contrastive scope block to the target skill's guidance to help the agent distinguish it from the other co-present skills.\n\n## The other {len(partners)} skill(s) the agent sees simultaneously:\n\n{partner_context}\n\n## Target skill to augment: "{target['skill_name']}"\n\n{target_block}\n\n## Generate a scope block to APPEND to the target's guidance:\n\n{{"scope_block": "(Scope: <what this skill specifically handles>. {unlike_example})"}}\n\nRules:\n- The scope_block contrasts with ALL {len(partners)} skills listed above\n- Be specific and factual: describe WHAT each skill returns/handles, not behavioral instructions\n- No MUST/NEVER/ALWAYS — use "handles", "returns", "applies to" language\n- Reference each partner skill by name\n- Keep it concise but complete\n\n/no_think'''

def rewrite_one_skill(client, target: dict, partners: list[dict], max_retries: int=3) -> dict | None:
    prompt = build_dynamic_prompt(target, partners)
    for attempt in range(max_retries):
        try:
            raw = call_llm(client, prompt, max_tokens=1024, temperature=0.3, system_message=SYSTEM_MESSAGE)
            parsed = parse_json_from_llm(raw)
            if parsed and isinstance(parsed, dict):
                block = parsed.get('scope_block', '')
                if block:
                    if len(block) > 500:
                        block = block[:497] + '...'
                    new_guidance = target['prompt_guidance'].rstrip()
                    if not new_guidance.endswith('.'):
                        new_guidance += '.'
                    new_guidance += ' ' + block
                    return {'prompt_guidance': new_guidance}
                else:
                    print(f"    [retry] {target['skill_name']}: empty scope_block")
            else:
                raw_preview = (raw or '')[:200].replace('\n', ' ')
                print(f"    [retry] {target['skill_name']}: JSON parse failed, raw={raw_preview}")
            if attempt < max_retries - 1:
                time.sleep(1)
        except Exception as e:
            if attempt < max_retries - 1:
                print(f"    [retry] {target['skill_name']}: {e}")
                time.sleep(2)
            else:
                print(f"    [FAIL] {target['skill_name']}: {e}")
    return None

def rewrite_skill_set(client, skill_set: list[dict]) -> dict[str, dict]:
    results = {}
    for (i, target) in enumerate(skill_set):
        partners = [s for (j, s) in enumerate(skill_set) if j != i]
        rewritten = rewrite_one_skill(client, target, partners)
        if rewritten:
            results[target['skill_name']] = rewritten
    return results

def main():
    parser = argparse.ArgumentParser(description='Dynamic Selection-Aware Contrastive Rewriter (v3a)')
    parser.add_argument('--domain', required=True)
    parser.add_argument('--selection-file', required=True, help='Selection JSONL (e.g., ports_delta_3f_shallow_b2/.../trained_selector.jsonl)')
    parser.add_argument('--api-base', default='http://localhost:32800/v1')
    parser.add_argument('--concurrency', type=int, default=4, help='Max concurrent skill sets being rewritten')
    parser.add_argument('--output', type=str, default=None, help='Output JSONL path (default: auto-generated)')
    args = parser.parse_args()
    print(f"[1/3] Loading skills for '{args.domain}'...")
    all_skills = load_all_skills(args.domain)
    skill_by_name = {s['skill_name']: s for s in all_skills}
    print(f'  Loaded {len(all_skills)} skills')
    print(f'[2/3] Reading selection file...')
    selection_path = Path(args.selection_file)
    with open(selection_path) as f:
        selection_rows = [json.loads(line) for line in f]
    unique_sets: dict[tuple[str, ...], list[str]] = {}
    for row in selection_rows:
        names = tuple(sorted(row['selected_skill_names']))
        task_id = str(row['task_id'])
        if names not in unique_sets:
            unique_sets[names] = []
        unique_sets[names].append(task_id)
    budget = len(next(iter(unique_sets)))
    total_calls = sum((len(names) for names in unique_sets))
    print(f'  B={budget}, {len(selection_rows)} tasks, {len(unique_sets)} unique skill sets, {total_calls} LLM calls needed')
    print(f'[3/3] Generating dynamic contrastive rewrites (concurrency={args.concurrency})...')
    client = get_client(args.api_base)
    output_records = []
    completed = 0
    failed_sets = 0

    def process_one_set(skill_names: tuple[str, ...], task_ids: list[str]):
        skill_set = []
        for name in skill_names:
            if name not in skill_by_name:
                print(f"  [WARN] Skill '{name}' not found, skipping set")
                return None
            skill_set.append(skill_by_name[name])
        rewrites = rewrite_skill_set(client, skill_set)
        if not rewrites:
            return None
        records = []
        for name in skill_names:
            if name in rewrites:
                records.append({'skill_set': list(skill_names), 'skill_name': name, 'task_ids': task_ids, 'mode': 'dynamic_contrastive_v3a', 'rewritten': rewrites[name]})
        return records
    with ThreadPoolExecutor(max_workers=args.concurrency) as executor:
        futures = {executor.submit(process_one_set, names, tids): names for (names, tids) in unique_sets.items()}
        for future in as_completed(futures):
            names = futures[future]
            completed += 1
            try:
                records = future.result()
                if records:
                    output_records.extend(records)
                    print(f'  [{completed}/{len(unique_sets)}] {list(names)} -> {len(records)} rewrites')
                else:
                    failed_sets += 1
                    print(f'  [{completed}/{len(unique_sets)}] {list(names)} FAILED')
            except Exception as e:
                failed_sets += 1
                print(f'  [{completed}/{len(unique_sets)}] {list(names)} ERROR: {e}')
    if args.output:
        output_path = args.output
    else:
        output_dir = Path(__file__).resolve().parent / 'data' / args.domain
        output_dir.mkdir(parents=True, exist_ok=True)
        sel_name = selection_path.parent.parent.name
        output_path = str(output_dir / f'dynamic_v3a_{sel_name}.jsonl')
    save_jsonl(output_path, output_records)
    skill_names_with_rewrites = set((r['skill_name'] for r in output_records))
    unique_guidances = [r['rewritten']['prompt_guidance'] for r in output_records]
    if unique_guidances:
        avg_len = sum((len(g) for g in unique_guidances)) / len(unique_guidances)
        orig_avg = sum((len(skill_by_name[r['skill_name']]['prompt_guidance']) for r in output_records)) / len(output_records)
        print(f'\nGuidance: {orig_avg:.0f} -> {avg_len:.0f} chars (+{avg_len - orig_avg:.0f}, {avg_len / orig_avg:.1f}x)')
    print(f'\nDone: {len(output_records)} rewrites for {completed - failed_sets}/{len(unique_sets)} skill sets ({failed_sets} failed)')
    print(f'Output: {output_path}')
if __name__ == '__main__':
    main()
