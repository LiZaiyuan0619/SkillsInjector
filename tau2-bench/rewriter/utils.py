import ast
import json
import re
import time
from pathlib import Path
from openai import OpenAI

def get_client(api_base: str='http://localhost:32800/v1', api_key: str='not-needed') -> OpenAI:
    return OpenAI(base_url=api_base, api_key=api_key)

def call_llm(client: OpenAI, prompt: str, max_tokens: int=4096, temperature: float=0.3, retries: int=3, system_message: str | None=None) -> str:
    messages = []
    if system_message:
        messages.append({'role': 'system', 'content': system_message})
    messages.append({'role': 'user', 'content': prompt})
    for attempt in range(retries):
        try:
            resp = client.chat.completions.create(model='openai/Qwen/Qwen3-235B-A22B-Instruct-2507', messages=messages, max_tokens=max_tokens, temperature=temperature)
            return resp.choices[0].message.content
        except Exception as e:
            if attempt < retries - 1:
                wait = 2 ** attempt
                print(f'  [retry {attempt + 1}/{retries}] {e}, waiting {wait}s')
                time.sleep(wait)
            else:
                raise
REWRITE_REQUIRED_FIELDS = ('prompt_guidance', 'docstring', 'body_content')
REWRITE_SYSTEM_MESSAGE = 'You are a skill documentation specialist. Your task is to rewrite skill descriptions and body content so an LLM agent can better select and use the skill.'

def validate_rewrite_output(parsed: dict | list | None) -> dict | None:
    if not isinstance(parsed, dict):
        return None
    for field in REWRITE_REQUIRED_FIELDS:
        value = parsed.get(field)
        if not isinstance(value, str) or not value.strip():
            return None
    return parsed

def call_llm_validated(client: OpenAI, prompt: str, *, system_message: str | None=None, max_tokens: int=4096, temperature: float=0.3, max_retries: int=2) -> dict | None:
    for attempt in range(max_retries):
        raw = call_llm(client, prompt, max_tokens=max_tokens, temperature=temperature, system_message=system_message)
        parsed = parse_json_from_llm(raw)
        result = validate_rewrite_output(parsed)
        if result is not None:
            return result
        if attempt < max_retries - 1:
            prompt = f"Your previous output was not valid JSON with the required fields ({', '.join(REWRITE_REQUIRED_FIELDS)}). Please try again.\n\n{prompt}"
    return None

def parse_json_from_llm(text: str) -> dict | list | None:
    text = re.sub('<think>.*?</think>', '', text, flags=re.DOTALL).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    m = re.search('```(?:json)?\\s*\\n?(.*?)\\n?```', text, re.DOTALL)
    if m:
        try:
            return json.loads(m.group(1))
        except json.JSONDecodeError:
            pass
    for (start_char, end_char) in [('{', '}'), ('[', ']')]:
        start = text.find(start_char)
        if start == -1:
            continue
        depth = 0
        for i in range(start, len(text)):
            if text[i] == start_char:
                depth += 1
            elif text[i] == end_char:
                depth -= 1
                if depth == 0:
                    try:
                        return json.loads(text[start:i + 1])
                    except json.JSONDecodeError:
                        break
    return None

def _extract_skill_fields(source: str) -> dict[str, str]:
    module = ast.parse(source)
    wanted = {'SKILL_NAME', 'SKILL_DOMAIN', 'SKILL_DESCRIPTION', 'PROMPT_GUIDANCE', 'TARGET_PATTERN'}
    fields: dict[str, str] = {}
    for node in module.body:
        if not isinstance(node, ast.Assign) or len(node.targets) != 1:
            continue
        target = node.targets[0]
        if not isinstance(target, ast.Name) or target.id not in wanted:
            continue
        if isinstance(node.value, ast.Constant) and isinstance(node.value.value, str):
            fields[target.id] = node.value.value
    return fields

def _extract_function_parts(source: str) -> dict[str, str]:
    result = {'docstring': '', 'body': ''}
    try:
        module = ast.parse(source)
    except SyntaxError:
        return result
    for node in module.body:
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        ds = ast.get_docstring(node)
        if ds:
            result['docstring'] = ds
        lines = source.splitlines()
        if node.end_lineno is not None:
            result['body'] = '\n'.join(lines[node.lineno - 1:node.end_lineno])
        break
    return result

def load_skills_from_ranked_json(ranked_json_path: str | Path, parsed_dir: str | Path) -> list[dict]:
    ranked_json_path = Path(ranked_json_path)
    parsed_dir = Path(parsed_dir)
    with open(ranked_json_path) as f:
        ranked = json.load(f)
    if isinstance(ranked, dict):
        ranked = ranked.get('items', ranked.get('skills', []))
    seen = set()
    skills = []
    for entry in ranked:
        name = entry.get('skill_name', entry.get('name', ''))
        name = name.split('/')[-1] if '/' in name else name
        if name in seen:
            continue
        seen.add(name)
        source = ''
        fname = entry.get('file', '')
        if fname:
            fpath = parsed_dir / fname
            if fpath.exists():
                source = fpath.read_text(encoding='utf-8')
        if not source:
            matches = sorted(parsed_dir.glob(f'*{name}*.py'))
            if matches:
                source = matches[0].read_text(encoding='utf-8')
        if not source:
            source = entry.get('code', '')
        if not source:
            print(f'  [WARN] No source found for skill: {name}')
            continue
        fields = _extract_skill_fields(source)
        func_parts = _extract_function_parts(source)
        skills.append({'skill_name': name, 'skill_description': fields.get('SKILL_DESCRIPTION', ''), 'prompt_guidance': fields.get('PROMPT_GUIDANCE', ''), 'target_pattern': fields.get('TARGET_PATTERN', ''), 'docstring': func_parts['docstring'], 'body': func_parts['body'], 'source': source})
    return skills

def summarize_trace(trajectory: dict, max_parts: int=15) -> str:
    messages = trajectory.get('messages', [])
    parts = []
    for msg in messages:
        role = msg.get('role', 'unknown')
        content = msg.get('content') or ''
        tool_calls = msg.get('tool_calls')
        if role == 'assistant' and tool_calls:
            for tc in tool_calls:
                name = tc.get('name', tc.get('function', {}).get('name', '?'))
                args = tc.get('arguments', tc.get('function', {}).get('arguments', {}))
                if isinstance(args, dict):
                    args_str = json.dumps(args, ensure_ascii=False)
                else:
                    args_str = str(args)
                parts.append(f'[Agent calls] {name}({args_str[:150]})')
        elif role == 'tool':
            content_str = content if isinstance(content, str) else json.dumps(content, ensure_ascii=False)
            parts.append(f'[Tool result] {content_str[:300]}')
        elif role == 'assistant':
            if content:
                parts.append(f'[Agent says] {content[:300]}')
        elif role == 'user':
            if content:
                parts.append(f'[User says] {content[:300]}')
    if len(parts) > max_parts:
        n_head = 4
        n_tail = max_parts - n_head - 1
        parts = parts[:n_head] + [f'... ({len(parts) - n_head - n_tail} steps omitted) ...'] + parts[-n_tail:]
    return '\n'.join(parts)

def save_jsonl(path: str | Path, records: list[dict]):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + '\n')
    print(f'Saved {len(records)} records to {path}')

def load_jsonl(path: str | Path) -> list[dict]:
    records = []
    with open(path, encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records
