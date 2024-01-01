import argparse
import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from .utils import get_client, call_llm, parse_json_from_llm, _extract_skill_fields, _extract_function_parts, save_jsonl
SKILLS_DIR = Path(__file__).resolve().parent.parent / 'skills'
REWRITER_DATA_DIR = Path(__file__).resolve().parent / 'data'
SYSTEM_MESSAGE = 'You generate minimal scope tags for tool disambiguation. Output valid JSON only. No markdown fences. No extra explanation.'

def load_all_skills(domain: str) -> list[dict]:
    skill_dir = SKILLS_DIR / domain
    skills = []
    for py_file in sorted(skill_dir.glob('*.py')):
        if py_file.name.startswith('__'):
            continue
        source = py_file.read_text(encoding='utf-8')
        fields = _extract_skill_fields(source)
        func_parts = _extract_function_parts(source)
        if 'SKILL_NAME' not in fields:
            continue
        skills.append({'skill_name': fields['SKILL_NAME'], 'description': fields.get('SKILL_DESCRIPTION', ''), 'prompt_guidance': fields.get('PROMPT_GUIDANCE', ''), 'target_pattern': fields.get('TARGET_PATTERN', ''), 'docstring': func_parts['docstring'], 'body_source': func_parts['body']})
    return skills

def build_all_skills_context(all_skills: list[dict]) -> str:
    parts = []
    for (i, sk) in enumerate(all_skills, 1):
        parts.append(f"=== Skill {i}/{len(all_skills)}: {sk['skill_name']} ===\n[Guidance]: {sk['prompt_guidance']}\n[Description]: {sk['docstring']}\n[Function Source]:\n{sk['body_source']}\n")
    return '\n'.join(parts)

def build_prompt(target: dict, all_skills: list[dict]) -> str:
    context = build_all_skills_context(all_skills)
    return f'''An AI agent receives multiple skill-tools simultaneously and must choose the right one. For the target skill below, generate a SHORT scope tag to append to its guidance.\n\n## All {len(all_skills)} skills:\n\n{context}\n\n## Target skill: "{target['skill_name']}"\nCurrent guidance: {target['prompt_guidance']}\n\n## Generate a scope tag in this EXACT format:\n\n{{"scope_tag": "(Scope: <what this skill handles, 10-15 words>. Not for: <what to use other skills for, reference 1-2 skill names>.)"}}\n\nRules:\n- The scope_tag MUST be a single parenthetical sentence\n- Maximum 120 characters for the scope_tag value\n- Reference 1-2 similar skills BY NAME for contrast\n- Use factual language only — no MUST/NEVER/ALWAYS\n- The tag will be appended verbatim to the existing guidance'''

def rewrite_one_skill(client, target: dict, all_skills: list[dict], max_retries: int=2) -> dict | None:
    prompt = build_prompt(target, all_skills)
    for attempt in range(max_retries):
        try:
            raw = call_llm(client, prompt, max_tokens=256, temperature=0.3, system_message=SYSTEM_MESSAGE)
            parsed = parse_json_from_llm(raw)
            if parsed and isinstance(parsed, dict):
                tag = parsed.get('scope_tag', '')
                if tag and len(tag) <= 200:
                    new_guidance = target['prompt_guidance'].rstrip()
                    if not new_guidance.endswith('.'):
                        new_guidance += '.'
                    new_guidance += ' ' + tag
                    return {'skill_name': target['skill_name'], 'mode': 'contrastive_v2', 'rewritten': {'prompt_guidance': new_guidance}}
            if attempt < max_retries - 1:
                print(f"  [retry] {target['skill_name']}")
        except Exception as e:
            if attempt < max_retries - 1:
                print(f"  [retry] {target['skill_name']}: {e}")
                time.sleep(2)
            else:
                print(f"  [FAIL] {target['skill_name']}: {e}")
    return None

def main():
    parser = argparse.ArgumentParser(description='Contrastive Skill Rewriter v2 — Minimal Scope Tags')
    parser.add_argument('--domain', required=True)
    parser.add_argument('--api-base', default='http://localhost:32800/v1')
    parser.add_argument('--concurrency', type=int, default=4)
    parser.add_argument('--output', type=str, default=None)
    args = parser.parse_args()
    print(f"[1/2] Loading skills for '{args.domain}'...")
    all_skills = load_all_skills(args.domain)
    print(f'  Loaded {len(all_skills)} skills')
    client = get_client(args.api_base)
    output_path = args.output or str(REWRITER_DATA_DIR / args.domain / 'contrastive_v2_results.jsonl')
    print(f'[2/2] Generating scope tags (concurrency={args.concurrency})...')
    results = []
    failed = []
    with ThreadPoolExecutor(max_workers=args.concurrency) as executor:
        futures = {executor.submit(rewrite_one_skill, client, skill, all_skills): skill['skill_name'] for skill in all_skills}
        for future in as_completed(futures):
            name = futures[future]
            try:
                result = future.result()
                if result:
                    results.append(result)
                    tag = result['rewritten']['prompt_guidance']
                    tag_len = len(tag) - len(next((s['prompt_guidance'] for s in all_skills if s['skill_name'] == name)))
                    print(f'  [{len(results) + len(failed)}/{len(all_skills)}] {name} ✓ (+{tag_len} chars)')
                else:
                    failed.append(name)
                    print(f'  [{len(results) + len(failed)}/{len(all_skills)}] {name} ✗')
            except Exception as e:
                failed.append(name)
                print(f'  [{len(results) + len(failed)}/{len(all_skills)}] {name} ERROR: {e}')
    save_jsonl(output_path, results)
    if results:
        orig_lens = []
        new_lens = []
        for r in results:
            name = r['skill_name']
            orig = next((s['prompt_guidance'] for s in all_skills if s['skill_name'] == name))
            orig_lens.append(len(orig))
            new_lens.append(len(r['rewritten']['prompt_guidance']))
        avg_orig = sum(orig_lens) / len(orig_lens)
        avg_new = sum(new_lens) / len(new_lens)
        avg_added = avg_new - avg_orig
        print(f'\nGuidance: {avg_orig:.0f} → {avg_new:.0f} chars (+{avg_added:.0f}, {avg_new / avg_orig:.1f}x)')
    print(f'Done: {len(results)}/{len(all_skills)} skills tagged ({len(failed)} failed)')
    if failed:
        print(f'Failed: {failed}')
if __name__ == '__main__':
    main()
