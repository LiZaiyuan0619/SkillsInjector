import argparse
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
from temp.selector.common import resolve_domain_spec
from temp.selector.datasets import load_pair_records
from temp.selector.offline_eval import _load_exported_dataset
DEFAULT_API_BASE = 'http://localhost:32800/v1'
PROMPT_TEMPLATE = 'You are a skill router. Given a user task and a list of available skills, select the SINGLE most relevant skill that would best help the agent solve this task.\n\nTask:\n{task_text}\n\nAvailable skills:\n{skill_list}\n\nRespond with ONLY the skill name (exact match from the list above). Do not include quotes, explanations, formatting, or any extra words.'

def _auto_detect_model(api_base: str) -> str:
    from openai import OpenAI
    client = OpenAI(api_key='EMPTY', base_url=api_base)
    models = client.models.list()
    for m in models.data:
        return m.id
    raise RuntimeError(f'no models available at {api_base}')

def _extract_short_description(skill_text: str) -> str:
    for line in skill_text.split('\n'):
        if line.startswith('skill_description:'):
            desc = line.split(':', 1)[1].strip()
            if len(desc) > 300:
                desc = desc[:300] + '...'
            return desc
    head = skill_text[:300]
    return head + ('...' if len(skill_text) > 300 else '')

def _build_skill_list_string(skill_names: list[str], skill_descriptions: dict[str, str]) -> str:
    lines = [f"- {name}: {skill_descriptions.get(name, '').strip()}" for name in skill_names]
    return '\n'.join(lines)

def _extract_skill_name(response: str, valid_names: set[str]) -> str | None:
    response = response.strip().strip('"').strip("'").strip('`').strip()
    if response in valid_names:
        return response
    first_line = response.split('\n', 1)[0].strip().strip('"').strip("'").strip('`').strip('.')
    if first_line in valid_names:
        return first_line
    matches = sorted((name for name in valid_names if name in response), key=len, reverse=True)
    if matches:
        return matches[0]
    return None

def _call_llm_once(client: Any, model: str, task_id: str, task_text: str, skill_list_str: str, valid_names: set[str], max_retries: int) -> tuple[str, str | None, str]:
    prompt = PROMPT_TEMPLATE.format(task_text=task_text, skill_list=skill_list_str)
    last_error = ''
    for attempt in range(max_retries):
        try:
            resp = client.chat.completions.create(model=model, messages=[{'role': 'user', 'content': prompt}], temperature=0.0, max_tokens=64)
            content = (resp.choices[0].message.content or '').strip()
            parsed = _extract_skill_name(content, valid_names)
            if parsed is not None:
                return (task_id, parsed, content)
            last_error = f'unparseable response (attempt {attempt + 1}): {content[:200]!r}'
        except Exception as exc:
            last_error = f'{type(exc).__name__}: {exc}'
            time.sleep(min(1 + attempt * 2, 8))
    return (task_id, None, last_error)

def llm_select(domain: str, data_dir: Path, api_base: str=DEFAULT_API_BASE, model: str | None=None, concurrency: int=8, max_retries: int=3) -> list[dict[str, Any]]:
    from openai import OpenAI
    spec = resolve_domain_spec(domain)
    (_manifest, tasks_by_id, skill_names, _labels) = _load_exported_dataset(domain=spec.domain, data_dir=data_dir, budget=1, validate_budget=False)
    pair_records = load_pair_records(data_dir)
    task_texts = {r.task_id: r.task_text for r in pair_records}
    skill_texts = {r.skill_name: r.skill_text for r in pair_records}
    skill_descriptions = {name: _extract_short_description(skill_texts.get(name, '')) for name in skill_names}
    skill_list_str = _build_skill_list_string(skill_names, skill_descriptions)
    valid_names = set(skill_names)
    client = OpenAI(api_key='EMPTY', base_url=api_base)
    resolved_model = model or _auto_detect_model(api_base)
    print(f'[llm_selector_baseline] domain={spec.domain} model={resolved_model} concurrency={concurrency}', file=sys.stderr)
    results_by_task: dict[str, tuple[str | None, str]] = {}
    with ThreadPoolExecutor(max_workers=concurrency) as executor:
        futures = {executor.submit(_call_llm_once, client, resolved_model, task_id, task_texts[task_id], skill_list_str, valid_names, max_retries): task_id for task_id in sorted(tasks_by_id)}
        for fut in as_completed(futures):
            (task_id, parsed, raw) = fut.result()
            results_by_task[task_id] = (parsed, raw)
    fallback_name = skill_names[0]
    num_fallback = 0
    rows: list[dict[str, Any]] = []
    for task_id in sorted(tasks_by_id):
        (parsed, raw) = results_by_task[task_id]
        if parsed is None:
            num_fallback += 1
            selected = fallback_name
        else:
            selected = parsed
        rows.append({'domain': spec.domain, 'task_id': task_id, 'method': 'trained_selector', 'budget': 1, 'candidate_pool_size': None, 'training_scope': 'single_domain', 'selected_skill_name': selected, 'selected_skill_names': [selected], 'selection_seed': None, 'selection_score': None, 'selection_scores': None, 'selection_rank': 1, 'selection_source': 'llm_as_selector', 'checkpoint_path': None})
    if num_fallback:
        print(f'[llm_selector_baseline] WARNING: {num_fallback}/{len(rows)} tasks fell back to alphabetically-first skill ({fallback_name}) because LLM response was unparseable', file=sys.stderr)
    return rows

def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text('\n'.join((json.dumps(row, sort_keys=True) for row in rows)) + ('\n' if rows else ''), encoding='utf-8')

def main() -> None:
    parser = argparse.ArgumentParser(description='LLM-as-selector baseline (Qwen3-235B, top-1)')
    parser.add_argument('--domain', required=True)
    parser.add_argument('--data-dir', required=True)
    parser.add_argument('--output', required=True, help='Output selection JSONL path')
    parser.add_argument('--api-base', default=DEFAULT_API_BASE)
    parser.add_argument('--model', default=None, help='Model id; auto-detected from /v1/models if omitted')
    parser.add_argument('--concurrency', type=int, default=8)
    parser.add_argument('--max-retries', type=int, default=3)
    args = parser.parse_args()
    rows = llm_select(args.domain, Path(args.data_dir), api_base=args.api_base, model=args.model, concurrency=args.concurrency, max_retries=args.max_retries)
    output_path = Path(args.output)
    _write_jsonl(output_path, rows)
    print(json.dumps({'domain': args.domain, 'num_rows': len(rows), 'output': str(output_path), 'api_base': args.api_base, 'selection_source': 'llm_as_selector', 'row_method_field': 'trained_selector'}))
if __name__ == '__main__':
    main()
