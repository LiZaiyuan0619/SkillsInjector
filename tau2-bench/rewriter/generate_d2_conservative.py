import argparse
import json
import logging
import os
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from utils import get_client, call_llm, parse_json_from_llm, load_jsonl, save_jsonl, _extract_skill_fields, _extract_function_parts
logging.basicConfig(level=logging.INFO, format='%(asctime)s  %(levelname)-7s  %(message)s', datefmt='%H:%M:%S')
log = logging.getLogger(__name__)

def _parsed_dir(domain: str) -> Path:
    return Path(__file__).resolve().parent.parent / f'generated_skills/{domain}/parsed'

def _rules_path(domain: str) -> Path:
    return Path(__file__).resolve().parent / f'data/{domain}/extracted_rules.jsonl'

def _output_path(domain: str) -> Path:
    return Path(__file__).resolve().parent / f'data/{domain}/d2_v4_results.jsonl'
SYSTEM_MESSAGE = 'You are a gap-detection specialist for agent tool outputs. You analyze execution traces to find MISSING FACTUAL INFORMATION in tool outputs — information the tool should provide but currently does not.'
D2_CONSERVATIVE_PROMPT = 'You are given the ORIGINAL body content of a skill (the text an agent receives when calling this tool) and rules extracted from real execution traces where agents made mistakes.\n\nYour task: determine if the original body is MISSING any factual information that traces reveal agents needed but didn\'t have. If yes, append ONLY the missing facts. If not, return the original unchanged.\n\n## ORIGINAL body_content\n```\n{original_body}\n```\n\n## Failure patterns from execution traces\n{extracted_rules_json}\n\n## ANALYSIS STEPS (think through these, do not output them)\n\n1. For each failure pattern, ask: "Did the agent fail because the body LACKED specific information,    or because the agent ignored/misapplied information that was already there?"\n2. If the body already covers the relevant policy/rule → the failure is NOT a body gap → skip it.\n3. If the body genuinely lacks a specific fact, edge case, or policy detail → this IS a gap → add it.\n\n## WHAT TO ADD (only if gaps exist)\n\n- Missing policy facts: specific conditions, thresholds, exceptions not mentioned in original\n  Example: "Exception: bookings within 24 hours of purchase are exempt from this restriction."\n- Missing edge cases: scenarios the original doesn\'t cover but traces show agents encounter\n  Example: "If passenger count is 1, cancellation results in full reservation cancellation."\n- Missing escalation paths: when the agent should hand off to human, if not already stated\n  Example: "For disputes about membership level, direct user to loyalty program support."\n\n## WHAT NEVER TO ADD\n\n- Behavioral directives: NO "MUST verify", "NEVER proceed", "ALWAYS check", "confirm before..."\n  (The agent has its own reasoning; behavioral commands in tool output create confusion loops)\n- Redundant information: if the original already says "passenger count cannot be changed",   do NOT add "MUST NOT attempt to change passenger count" — this is the same fact restated.\n- Procedural reminders: NO "call X before Y", "verify Z first" — tool output is information, not workflow.\n- Emphasis markers on existing content: NO adding CRITICAL/MUST/NEVER to things already stated.\n\n## OUTPUT RULES\n\n- Copy original body EXACTLY (preserve all examples, numbers, f-strings like {{reservation_id}}).\n- If gaps found: append "---\\nAdditional policy details:" followed by 1-3 SHORT factual lines.\n- If NO genuine gaps: return the original body UNCHANGED.\n- Max 3 additions. Each must be a NEW FACT not derivable from the original text.\n\nOutput ONLY valid JSON:\n{{"body_content": "<complete body content>", "gaps_found": <true or false>}}'
D2_CONSERVATIVE_NO_RULES_PROMPT = 'No execution traces are available for this skill. Return the original body_content EXACTLY as-is.\n\n## ORIGINAL body_content\n```\n{original_body}\n```\n\nOutput ONLY valid JSON:\n{{"body_content": "<exact copy of the original body_content>", "gaps_found": false}}'

def load_original_skills(parsed_dir: Path) -> dict:
    skills = {}
    for fname in sorted(parsed_dir.glob('*.py')):
        source = fname.read_text(encoding='utf-8')
        fields = _extract_skill_fields(source)
        name = fields.get('SKILL_NAME', '')
        if not name:
            continue
        func_parts = _extract_function_parts(source)
        body_code = func_parts.get('body', '')
        body_text = _extract_return_text(source, name)
        skills[name] = {'body_text': body_text, 'source': source, 'fields': fields}
    return skills

def _extract_return_text(source: str, skill_name: str) -> str:
    import ast
    import textwrap
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return ''
    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        for child in ast.walk(node):
            if isinstance(child, ast.Return) and child.value is not None:
                start = child.lineno - 1
                end = getattr(child, 'end_lineno', None)
                if end is None:
                    end = len(source.splitlines())
                lines = source.splitlines()[start:end]
                return_src = '\n'.join(lines)
                try:
                    arg_names = [a.arg for a in node.args.args]
                    dummy_args = {a: f'{{{a}}}' for a in arg_names}
                    expr_src = return_src.strip()
                    if expr_src.startswith('return'):
                        expr_src = expr_src[6:].strip()
                    result = eval(expr_src, {'__builtins__': {}}, dummy_args)
                    if isinstance(result, str):
                        return result
                except Exception:
                    pass
                expr_start = return_src.find('return') + 6
                return return_src[expr_start:].strip().strip('()')
    return ''
_counter_lock = threading.Lock()
_counter = {'done': 0, 'total': 0, 'ok': 0, 'fail': 0}

def _validate_conservative(parsed: dict | None, original_body: str) -> dict | None:
    if not isinstance(parsed, dict):
        return None
    body = parsed.get('body_content')
    if not isinstance(body, str) or not body.strip():
        return None
    original_start = original_body.strip()[:100]
    if original_start and original_start not in body:
        log.warning("Output doesn't preserve original body start")
    return {'body_content': body, 'gaps_found': parsed.get('gaps_found', None)}

def _process_one(client, skill_name: str, original_body: str, rules: list[dict] | None) -> dict | None:
    has_rules = bool(rules)
    if has_rules:
        rules_json = json.dumps(rules, ensure_ascii=False, indent=2)
        prompt = D2_CONSERVATIVE_PROMPT.format(original_body=original_body, extracted_rules_json=rules_json)
    else:
        prompt = D2_CONSERVATIVE_NO_RULES_PROMPT.format(original_body=original_body)
    for attempt in range(3):
        try:
            raw = call_llm(client, prompt, system_message=SYSTEM_MESSAGE, max_tokens=4096, temperature=0.2)
            parsed = parse_json_from_llm(raw)
            result = _validate_conservative(parsed, original_body)
            if result is not None:
                record = {'skill_name': skill_name, 'original_body': original_body, 'rules': rules if rules else [], 'd2_v4': result}
                with _counter_lock:
                    _counter['done'] += 1
                    _counter['ok'] += 1
                    log.info('Progress: %d/%d (ok=%d fail=%d) %s%s', _counter['done'], _counter['total'], _counter['ok'], _counter['fail'], skill_name, ' [no rules]' if not has_rules else '')
                return record
        except Exception as e:
            log.warning('Attempt %d failed for %s: %s', attempt + 1, skill_name, e)
    with _counter_lock:
        _counter['done'] += 1
        _counter['fail'] += 1
        log.info('Progress: %d/%d (ok=%d fail=%d) FAILED: %s', _counter['done'], _counter['total'], _counter['ok'], _counter['fail'], skill_name)
    return None

def main():
    parser = argparse.ArgumentParser(description='Conservative D2: preserve original body, append trace rules.')
    parser.add_argument('--api-base', default='http://localhost:32800/v1')
    parser.add_argument('--domain', default='airline')
    parser.add_argument('--concurrency', type=int, default=4)
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    parsed_dir = _parsed_dir(args.domain)
    rules_file = _rules_path(args.domain)
    out_path = _output_path(args.domain)
    if not parsed_dir.exists():
        log.error('Parsed skills directory not found: %s', parsed_dir)
        sys.exit(1)
    original_skills = load_original_skills(parsed_dir)
    log.info('Loaded %d original skills from %s', len(original_skills), parsed_dir)
    n_with_body = sum((1 for s in original_skills.values() if s['body_text'].strip()))
    log.info('Skills with extracted body text: %d/%d', n_with_body, len(original_skills))
    rules_by_skill = {}
    if rules_file.exists():
        for rec in load_jsonl(rules_file):
            rules_by_skill[rec['skill_name']] = rec.get('rules', [])
        log.info('Loaded rules for %d skills', len(rules_by_skill))
    existing_records = []
    already_done = set()
    if args.resume and out_path.exists():
        existing_records = load_jsonl(out_path)
        already_done = {r['skill_name'] for r in existing_records}
        log.info('Resuming: %d already done', len(already_done))
    todo = []
    for (skill_name, skill_data) in original_skills.items():
        if skill_name in already_done:
            continue
        body_text = skill_data['body_text']
        if not body_text.strip():
            log.warning('Empty body for %s, skipping', skill_name)
            continue
        rules = rules_by_skill.get(skill_name)
        todo.append((skill_name, body_text, rules))
    if not todo:
        log.info('Nothing to do.')
        return
    _counter['total'] = len(todo)
    log.info('Processing %d skills (concurrency=%d)', len(todo), args.concurrency)
    client = get_client(api_base=args.api_base)
    new_records = []
    with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        futures = {pool.submit(_process_one, client, name, body, rules): name for (name, body, rules) in todo}
        for fut in as_completed(futures):
            result = fut.result()
            if result is not None:
                new_records.append(result)
    all_records = existing_records + new_records
    save_jsonl(out_path, all_records)
    log.info('=' * 60)
    log.info('Domain:       %s', args.domain)
    log.info('Total skills: %d', len(original_skills))
    log.info('Processed:    %d', len(todo))
    log.info('  OK:         %d', _counter['ok'])
    log.info('  Failed:     %d', _counter['fail'])
    log.info('Output:       %s', out_path)
    log.info('=' * 60)
if __name__ == '__main__':
    main()
