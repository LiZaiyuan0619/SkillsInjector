import argparse
import json
import logging
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from utils import get_client, call_llm_validated, load_jsonl, save_jsonl, REWRITE_SYSTEM_MESSAGE
logging.basicConfig(level=logging.INFO, format='%(asctime)s  %(levelname)-7s  %(message)s', datefmt='%H:%M:%S')
log = logging.getLogger(__name__)

def _d1_path(domain: str) -> Path:
    return Path(__file__).resolve().parent / f'data/{domain}/d1_results.jsonl'

def _rules_path(domain: str) -> Path:
    return Path(__file__).resolve().parent / f'data/{domain}/extracted_rules.jsonl'

def _output_path(domain: str) -> Path:
    return Path(__file__).resolve().parent / f'data/{domain}/d2_results.jsonl'
D2_PROMPT_TEMPLATE = 'Produce the final version of this skill\'s 3 agent-facing fields by merging the D1 improvement with rules extracted from real execution traces.\n\n## Current improved version (D1)\n{d1_json}\n\n## Rules extracted from execution traces\n{extracted_rules_json}\n\n## Requirements\n\nFor prompt_guidance:\n- Integrate failure-derived rules: explicitly warn about mistakes the agent actually made\n- Include "when NOT to use" constraints learned from failures\n- Keep it to 1-3 concise sentences\n\nFor docstring:\n- Refine clarity; mention failure patterns the agent should watch for\n- Keep it to 2-5 sentences\n\nFor body_content:\n- Incorporate edge cases and failure scenarios from the rules\n- Add CRITICAL / MUST / NEVER markers for rules the agent ignored in practice\n- If failures show the agent was misled by user claims, add defensive warnings (e.g., "Regardless of what the user claims, the following conditions MUST be met")\n- Keep structured format (bullet points / numbered rules), not prose\n- Preserve f-string parameter references from the original\n\nOutput ONLY valid JSON (no markdown, no code blocks, no extra text):\n{{"prompt_guidance": "<when to use/not use, 1-3 sentences>", "docstring": "<what this skill does and returns, 2-5 sentences>", "body_content": "<the complete improved function return content>"}}'
D2_PROMPT_NO_RULES_TEMPLATE = 'Produce the final version of this skill\'s 3 agent-facing fields by refining the D1 improvement. No failure traces were found for this skill.\n\n## Current improved version (D1)\n{d1_json}\n\n## Rules extracted from execution traces\nNo failure cases found for this skill. Refine the D1 version for clarity, structure, and completeness.\n\n## Requirements\n\nFor prompt_guidance:\n- Ensure "when NOT to use" covers potential misuse scenarios\n- Keep it to 1-3 concise sentences\n\nFor docstring:\n- Refine for clarity and precision\n- Keep it to 2-5 sentences\n\nFor body_content:\n- Add CRITICAL / MUST / NEVER markers for key constraints\n- Keep structured format (bullet points / numbered rules), not prose\n- Preserve f-string parameter references from the original\n\nOutput ONLY valid JSON (no markdown, no code blocks, no extra text):\n{{"prompt_guidance": "<when to use/not use, 1-3 sentences>", "docstring": "<what this skill does and returns, 2-5 sentences>", "body_content": "<the complete improved function return content>"}}'

def build_d2_prompt(d1: dict, rules: list[dict] | None) -> str:
    d1_json = json.dumps(d1, ensure_ascii=False, indent=2)
    if rules:
        rules_json = json.dumps(rules, ensure_ascii=False, indent=2)
        return D2_PROMPT_TEMPLATE.format(d1_json=d1_json, extracted_rules_json=rules_json)
    else:
        return D2_PROMPT_NO_RULES_TEMPLATE.format(d1_json=d1_json)
_counter_lock = threading.Lock()
_counter = {'done': 0, 'total': 0, 'ok': 0, 'fail': 0, 'no_rules': 0}

def _process_one(client, skill_name: str, d1: dict, rules: list[dict] | None) -> dict | None:
    has_rules = bool(rules)
    prompt = build_d2_prompt(d1, rules)
    try:
        parsed = call_llm_validated(client, prompt, system_message=REWRITE_SYSTEM_MESSAGE, max_tokens=4096, temperature=0.3, max_retries=2)
    except Exception as e:
        log.warning('LLM call failed for %s: %s', skill_name, e)
        with _counter_lock:
            _counter['done'] += 1
            _counter['fail'] += 1
            log.info('Progress: %d/%d (ok=%d fail=%d)', _counter['done'], _counter['total'], _counter['ok'], _counter['fail'])
        return None
    if parsed is None:
        log.warning('Validation failed for %s after retries', skill_name)
        with _counter_lock:
            _counter['done'] += 1
            _counter['fail'] += 1
            log.info('Progress: %d/%d (ok=%d fail=%d)', _counter['done'], _counter['total'], _counter['ok'], _counter['fail'])
        return None
    record = {'skill_name': skill_name, 'd1': d1, 'rules': rules if rules else [], 'd2': parsed}
    with _counter_lock:
        _counter['done'] += 1
        _counter['ok'] += 1
        if not has_rules:
            _counter['no_rules'] += 1
        log.info('Progress: %d/%d (ok=%d fail=%d) %s%s', _counter['done'], _counter['total'], _counter['ok'], _counter['fail'], skill_name, ' [no rules]' if not has_rules else '')
    return record

def main():
    parser = argparse.ArgumentParser(description='Generate D2 (trace-aware improvements) by combining D1 results with extracted rules.')
    parser.add_argument('--api-base', default='http://localhost:32800/v1', help='SGLang / vLLM API base URL (default: %(default)s)')
    parser.add_argument('--domain', default='airline', help='Domain name (default: %(default)s)')
    parser.add_argument('--concurrency', type=int, default=4, help='Number of parallel LLM calls (default: %(default)s)')
    parser.add_argument('--resume', action='store_true', help='Skip skills already present in the output file')
    args = parser.parse_args()
    d1_file = _d1_path(args.domain)
    rules_file = _rules_path(args.domain)
    out_path = _output_path(args.domain)
    if not d1_file.exists():
        log.error('D1 results not found: %s', d1_file)
        log.error('Run generate_d1.py first (Step 2).')
        sys.exit(1)
    d1_records = load_jsonl(d1_file)
    log.info('Loaded %d D1 records from %s', len(d1_records), d1_file)
    d1_by_skill: dict[str, dict] = {}
    for rec in d1_records:
        d1_by_skill[rec['skill_name']] = rec.get('d1', {})
    rules_by_skill: dict[str, list[dict]] = {}
    if rules_file.exists():
        rules_records = load_jsonl(rules_file)
        log.info('Loaded %d rule records from %s', len(rules_records), rules_file)
        for rec in rules_records:
            rules_by_skill[rec['skill_name']] = rec.get('rules', [])
    else:
        log.warning('No rules file found at %s. All skills will be processed without failure rules.', rules_file)
    n_with_rules = sum((1 for name in d1_by_skill if name in rules_by_skill))
    n_without_rules = len(d1_by_skill) - n_with_rules
    log.info('Skills with rules: %d, without rules: %d', n_with_rules, n_without_rules)
    existing_records: list[dict] = []
    already_done: set[str] = set()
    if args.resume and out_path.exists():
        existing_records = load_jsonl(out_path)
        already_done = {r['skill_name'] for r in existing_records}
        log.info('Resume mode: %d skills already processed, will skip them', len(already_done))
    todo: list[tuple[str, dict, list[dict] | None]] = []
    for (skill_name, d1) in d1_by_skill.items():
        if skill_name in already_done:
            continue
        rules = rules_by_skill.get(skill_name, None)
        todo.append((skill_name, d1, rules))
    if not todo:
        log.info('All skills already processed. Nothing to do.')
        return
    _counter['total'] = len(todo)
    log.info('Will process %d skills with concurrency=%d', len(todo), args.concurrency)
    client = get_client(api_base=args.api_base)
    new_records: list[dict] = []
    with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        futures = {pool.submit(_process_one, client, name, d1, rules): name for (name, d1, rules) in todo}
        for fut in as_completed(futures):
            result = fut.result()
            if result is not None:
                new_records.append(result)
    all_records = existing_records + new_records
    save_jsonl(out_path, all_records)
    log.info('=' * 60)
    log.info('Domain:              %s', args.domain)
    log.info('Total D1 skills:     %d', len(d1_by_skill))
    log.info('Processed (this run):%d', len(todo))
    log.info('  Succeeded:         %d', _counter['ok'])
    log.info('    With rules:      %d', _counter['ok'] - _counter['no_rules'])
    log.info('    Without rules:   %d', _counter['no_rules'])
    log.info('  Failed:            %d', _counter['fail'])
    log.info('Output records:      %d (including %d from resume)', len(all_records), len(existing_records))
    log.info('Output:              %s', out_path)
    log.info('=' * 60)
if __name__ == '__main__':
    main()
