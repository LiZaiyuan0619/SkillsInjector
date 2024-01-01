import argparse
import json
import logging
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from utils import get_client, call_llm_validated, load_skills_from_ranked_json, save_jsonl, REWRITE_SYSTEM_MESSAGE
logging.basicConfig(level=logging.INFO, format='%(asctime)s  %(levelname)-7s  %(message)s', datefmt='%H:%M:%S')
log = logging.getLogger(__name__)
PROJECT_BASE = Path(__file__).resolve().parent.parent.parent

def _ranked_json_path(domain: str) -> Path:
    return PROJECT_BASE / f'temp/generated_skills/{domain}/deduplicated_090/skills_ranked.json'

def _parsed_dir(domain: str) -> Path:
    return PROJECT_BASE / f'temp/generated_skills/{domain}/parsed'

def _output_path(domain: str) -> Path:
    return Path(__file__).resolve().parent / f'data/{domain}/d1_results.jsonl'
D1_PROMPT_TEMPLATE = 'Rewrite the following skill so an LLM agent can:\n1) Decide WHEN to call this skill (and when NOT to)\n2) Understand WHAT it returns\n3) Follow its guidance CORRECTLY after calling\n\n## Original Skill\n\nName: {skill_name}\nDescription: {skill_description}\nPrompt guidance: {prompt_guidance}\nTarget pattern: {target_pattern}\n\nFunction definition:\n```python\n{body}\n```\n\n## Guidelines\n\nInfer (do not output these directly — use them to improve the 3 output fields):\n- When to use vs NOT use this skill\n- Critical rules the agent often ignores\n- Common mistake patterns (e.g., user pressure to deviate from policy)\n- Edge cases and ambiguous scenarios\n\nFor prompt_guidance:\n- State when to call AND when NOT to call (1-3 sentences)\n- Be specific: reference concrete conditions, not vague descriptions\n\nFor docstring:\n- Explain what this skill returns and how the agent should interpret it\n- Keep it concise (2-5 sentences)\n\nFor body_content:\n- Prioritize rules: most critical constraints FIRST\n- Use clear markers for hard constraints (MUST, NEVER, CRITICAL)\n- Add warnings for scenarios where agents commonly make mistakes\n- Keep structured format (bullet points / numbered rules), not prose\n- Preserve the original f-string parameter references if present\n\nOutput ONLY valid JSON (no markdown, no code blocks, no extra text):\n{{"prompt_guidance": "<when to use/not use, 1-3 sentences>", "docstring": "<what this skill does and returns, 2-5 sentences>", "body_content": "<the complete improved function return content>"}}'

def build_d1_prompt(skill: dict) -> str:
    return D1_PROMPT_TEMPLATE.format(skill_name=skill['skill_name'], skill_description=skill['skill_description'], prompt_guidance=skill['prompt_guidance'], target_pattern=skill['target_pattern'], body=skill['body'])
_counter_lock = threading.Lock()
_counter = {'done': 0, 'total': 0, 'ok': 0, 'fail': 0}

def _process_one(client, skill: dict) -> dict | None:
    name = skill['skill_name']
    prompt = build_d1_prompt(skill)
    try:
        parsed = call_llm_validated(client, prompt, system_message=REWRITE_SYSTEM_MESSAGE, max_tokens=4096, temperature=0.3, max_retries=2)
    except Exception as e:
        log.warning('LLM call failed for %s: %s', name, e)
        with _counter_lock:
            _counter['done'] += 1
            _counter['fail'] += 1
            log.info('Progress: %d/%d (ok=%d fail=%d)', _counter['done'], _counter['total'], _counter['ok'], _counter['fail'])
        return None
    if parsed is None:
        log.warning('Validation failed for %s after retries', name)
        with _counter_lock:
            _counter['done'] += 1
            _counter['fail'] += 1
            log.info('Progress: %d/%d (ok=%d fail=%d)', _counter['done'], _counter['total'], _counter['ok'], _counter['fail'])
        return None
    record = {'skill_name': name, 'd0': {'skill_description': skill['skill_description'], 'prompt_guidance': skill['prompt_guidance'], 'target_pattern': skill['target_pattern'], 'docstring': skill['docstring'], 'body': skill['body']}, 'd1': parsed}
    with _counter_lock:
        _counter['done'] += 1
        _counter['ok'] += 1
        log.info('Progress: %d/%d (ok=%d fail=%d)', _counter['done'], _counter['total'], _counter['ok'], _counter['fail'])
    return record

def main():
    parser = argparse.ArgumentParser(description='Generate D1 (general-guideline improvements) for skills.')
    parser.add_argument('--api-base', default='http://localhost:32800/v1', help='SGLang / vLLM API base URL (default: %(default)s)')
    parser.add_argument('--domain', default='airline', help='Domain name (default: %(default)s)')
    parser.add_argument('--concurrency', type=int, default=4, help='Number of parallel LLM calls (default: %(default)s)')
    parser.add_argument('--resume', action='store_true', help='Skip skills already present in the output file')
    args = parser.parse_args()
    ranked_path = _ranked_json_path(args.domain)
    parsed_dir = _parsed_dir(args.domain)
    out_path = _output_path(args.domain)
    log.info('Loading skills from %s', ranked_path)
    skills = load_skills_from_ranked_json(ranked_path, parsed_dir)
    log.info("Loaded %d skills for domain '%s'", len(skills), args.domain)
    existing_records: list[dict] = []
    already_done: set[str] = set()
    if args.resume and out_path.exists():
        with open(out_path, encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                existing_records.append(rec)
                already_done.add(rec['skill_name'])
        log.info('Resume mode: %d skills already processed, will skip them', len(already_done))
    todo = [s for s in skills if s['skill_name'] not in already_done]
    if not todo:
        log.info('All skills already processed. Nothing to do.')
        return
    _counter['total'] = len(todo)
    log.info('Will process %d skills with concurrency=%d', len(todo), args.concurrency)
    client = get_client(api_base=args.api_base)
    new_records: list[dict] = []
    with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        futures = {pool.submit(_process_one, client, s): s for s in todo}
        for fut in as_completed(futures):
            result = fut.result()
            if result is not None:
                new_records.append(result)
    all_records = existing_records + new_records
    save_jsonl(out_path, all_records)
    log.info('=' * 50)
    log.info('Domain:      %s', args.domain)
    log.info('Total skills: %d', len(skills))
    log.info('Processed:   %d (this run)', len(todo))
    log.info('Succeeded:   %d', _counter['ok'])
    log.info('Failed:      %d', _counter['fail'])
    log.info('Output:      %s', out_path)
    log.info('=' * 50)
if __name__ == '__main__':
    main()
