import argparse
import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from .utils import get_client, call_llm, parse_json_from_llm, _extract_skill_fields, _extract_function_parts, save_jsonl, load_jsonl
SKILLS_DIR = Path(__file__).resolve().parent.parent / 'skills'
REWRITER_DATA_DIR = Path(__file__).resolve().parent / 'data'
SYSTEM_MESSAGE = 'You are a skill documentation specialist. You add contrastive scope conditions to skill descriptions so that an LLM agent can accurately choose between multiple simultaneously available skills. Output valid JSON only. No markdown fences.'

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

def load_extracted_rules(domain: str) -> dict[str, dict]:
    rules_file = REWRITER_DATA_DIR / domain / 'extracted_rules.jsonl'
    if not rules_file.exists():
        print(f'  [WARN] No extracted rules at {rules_file}')
        return {}
    rules_map = {}
    for record in load_jsonl(rules_file):
        rules_map[record['skill_name']] = record
    return rules_map

def build_all_skills_context(all_skills: list[dict]) -> str:
    parts = []
    for (i, sk) in enumerate(all_skills, 1):
        parts.append(f"=== Skill {i}/{len(all_skills)}: {sk['skill_name']} ===\n[System Prompt Guidance]: {sk['prompt_guidance']}\n[Tool Description]: {sk['docstring']}\n[Function Source (body = return value agent receives after calling)]:\n{sk['body_source']}\n")
    return '\n'.join(parts)

def build_trace_context(rules_data: dict | None) -> str:
    if not rules_data or not rules_data.get('rules'):
        return '(No trace data available for this skill.)'
    parts = [f"Traces used: {rules_data.get('num_success_used', 0)} success, {rules_data.get('num_failure_used', 0)} failure"]
    for r in rules_data.get('rules', []):
        parts.append(f"- Observation: {r.get('rule', '')}\n  Failure scenario: {r.get('failure_pattern', '')}\n  What works: {r.get('correct_behavior', '')}")
    return '\n'.join(parts)

def build_prompt(target: dict, all_skills: list[dict], rules_data: dict | None) -> str:
    context_block = build_all_skills_context(all_skills)
    trace_block = build_trace_context(rules_data)
    return f'''You are optimizing skill tools for an AI customer service agent.\nThe agent receives MULTIPLE skills simultaneously (budget B=2 to 5) and must choose the right skill for each step of a conversation. Your job: rewrite one skill's guidance and docstring so the agent can distinguish it from similar skills.\n\n## All {len(all_skills)} skills in this domain (full agent-visible content):\n\n{context_block}\n\n## Target skill to rewrite: "{target['skill_name']}"\n\nCurrent guidance: {target['prompt_guidance']}\nCurrent docstring: {target['docstring']}\nAddresses failure pattern: {target['target_pattern']}\n\n### Trace-derived insights (from actual agent execution logs):\n{trace_block}\n\n## Instructions\n\nRewrite the **guidance** and **docstring** for "{target['skill_name']}".\n\n### What to ADD — contrastive scope conditions:\n- FACTUAL conditions: "This skill returns X", "Applicable when Y"\n- CONTRAST with similar skills BY NAME: "Differs from [skill_name] which handles Z"\n- NEGATIVE scope: "Not applicable when [condition] — use [other_skill] instead"\n- Use trace insights to inform conditions, but express as FACTS not commands\n\n### What to NEVER include (these patterns are proven harmful):\n- Behavioral directives: "MUST verify", "NEVER proceed", "ALWAYS check"\n- Confirmation commands: "confirm before proceeding", "verify with user first"\n- Generic rules: "follow company policy", "check all conditions"\n- Redundant restating of body content\n\n### Guidance style:\n- 2-4 sentences, concise, factual\n- First sentence: when to call this skill (positive scope)\n- Remaining: how it differs from similar skills (contrastive scope)\n\n### Docstring style:\n- Clear explanation of what this tool returns and when it applies\n- Explicitly differentiate from similar tools by name\n- No length limit — be as clear as needed about contrastive conditions\n\n### Output (JSON, no markdown fences):\n{{"prompt_guidance": "<rewritten guidance>", "docstring": "<rewritten docstring>"}}'''

def rewrite_one_skill(client, target: dict, all_skills: list[dict], rules_map: dict, max_retries: int=2) -> dict | None:
    rules_data = rules_map.get(target['skill_name'])
    prompt = build_prompt(target, all_skills, rules_data)
    for attempt in range(max_retries):
        try:
            raw = call_llm(client, prompt, max_tokens=2048, temperature=0.3, system_message=SYSTEM_MESSAGE)
            parsed = parse_json_from_llm(raw)
            if parsed and isinstance(parsed, dict):
                guidance = parsed.get('prompt_guidance', '')
                docstring = parsed.get('docstring', '')
                if guidance and docstring:
                    return {'skill_name': target['skill_name'], 'mode': 'contrastive_v1', 'rewritten': {'prompt_guidance': guidance, 'docstring': docstring}}
            if attempt < max_retries - 1:
                print(f"  [retry] {target['skill_name']}: invalid output, retrying")
        except Exception as e:
            if attempt < max_retries - 1:
                print(f"  [retry] {target['skill_name']}: {e}")
                time.sleep(2)
            else:
                print(f"  [FAIL] {target['skill_name']}: {e}")
    return None

def main():
    parser = argparse.ArgumentParser(description='Contrastive Skill Rewriter v1')
    parser.add_argument('--domain', required=True)
    parser.add_argument('--api-base', default='http://localhost:32800/v1')
    parser.add_argument('--concurrency', type=int, default=4, help='Number of parallel LLM calls')
    parser.add_argument('--output', type=str, default=None, help='Output JSONL path (default: data/{domain}/contrastive_v1_results.jsonl)')
    args = parser.parse_args()
    print(f"[1/3] Loading skills for domain '{args.domain}'...")
    all_skills = load_all_skills(args.domain)
    print(f'  Loaded {len(all_skills)} skills')
    print(f'[2/3] Loading trace-derived rules...')
    rules_map = load_extracted_rules(args.domain)
    print(f'  Loaded rules for {len(rules_map)} skills')
    client = get_client(args.api_base)
    output_path = args.output or str(REWRITER_DATA_DIR / args.domain / 'contrastive_v1_results.jsonl')
    print(f'[3/3] Rewriting {len(all_skills)} skills (concurrency={args.concurrency})...')
    results = []
    failed = []
    with ThreadPoolExecutor(max_workers=args.concurrency) as executor:
        futures = {executor.submit(rewrite_one_skill, client, skill, all_skills, rules_map): skill['skill_name'] for skill in all_skills}
        for future in as_completed(futures):
            name = futures[future]
            try:
                result = future.result()
                if result:
                    results.append(result)
                    print(f'  [{len(results) + len(failed)}/{len(all_skills)}] {name} ✓')
                else:
                    failed.append(name)
                    print(f'  [{len(results) + len(failed)}/{len(all_skills)}] {name} ✗ (no valid output)')
            except Exception as e:
                failed.append(name)
                print(f'  [{len(results) + len(failed)}/{len(all_skills)}] {name} ERROR: {e}')
    save_jsonl(output_path, results)
    print(f'\nDone: {len(results)}/{len(all_skills)} skills rewritten ({len(failed)} failed)')
    if failed:
        print(f'Failed skills: {failed}')
if __name__ == '__main__':
    main()
