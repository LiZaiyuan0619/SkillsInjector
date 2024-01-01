import argparse
import json
import re
import time
from pathlib import Path
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer
from .generate_contrastive_v2 import load_all_skills
import functools as _ft
print = _ft.partial(print, flush=True)
SYSTEM_MESSAGE = 'You are a skill documentation specialist. Your task is to rewrite skill descriptions and body content so an LLM agent can better select and use the skill.'
_OUTPUT_SCHEMA = '[OUTPUT]\nOutput ONLY valid JSON (no markdown, no code blocks, no extra text):\n{{"prompt_guidance": "<when to use/not use, 1-3 sentences>", "docstring": "<what this skill does and returns, 2-5 sentences>", "body_content": "<the complete improved function return content>"}}'
TRACE_FREE_TEMPLATE = "[INSTRUCTION]\nRewrite the following skill's description and body content so an LLM agent can better select and use it.\n\n[SKILL]\nName: {skill_name}\nDescription: {d0_description}\nPrompt guidance: {d0_guidance}\nTarget pattern: {d0_target_pattern}\nFunction definition:\n{body}\n\n" + _OUTPUT_SCHEMA

def skill_to_d0_fields(skill: dict) -> dict:
    return {'skill_name': skill['skill_name'], 'd0_description': skill.get('description', ''), 'd0_guidance': skill.get('prompt_guidance', ''), 'd0_target_pattern': skill.get('target_pattern', ''), 'body': skill.get('body_source', '')}

def build_prompt(skill: dict) -> str:
    return TRACE_FREE_TEMPLATE.format(**skill_to_d0_fields(skill))

def parse_json_from_llm(text: str) -> dict | None:
    text = re.sub('<think>.*?</think>', '', text, flags=re.DOTALL)
    text = text.strip()
    if text.startswith('```'):
        text = re.sub('^```(?:json)?\\s*\\n?', '', text)
        text = re.sub('\\n?```\\s*$', '', text)
        text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    depth = 0
    start = None
    for (i, ch) in enumerate(text):
        if ch == '{':
            if depth == 0:
                start = i
            depth += 1
        elif ch == '}':
            depth -= 1
            if depth == 0 and start is not None:
                try:
                    return json.loads(text[start:i + 1])
                except json.JSONDecodeError:
                    start = None
    return None

def generate(model, tokenizer, messages: list[dict], max_new_tokens: int, temperature: float) -> str:
    try:
        prompt_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    except TypeError:
        messages = [m if m['role'] != 'user' else {**m, 'content': m['content'].rstrip() + '\n\n/no_think'} for m in messages]
        prompt_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(prompt_text, return_tensors='pt').to(model.device)
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=max_new_tokens, temperature=temperature, do_sample=temperature > 0, top_p=0.9, pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id)
    gen = out[0][inputs['input_ids'].shape[1]:]
    return tokenizer.decode(gen, skip_special_tokens=True)

def rewrite_one_skill(model, tokenizer, skill: dict, max_new_tokens: int, temperature: float, max_retries: int=1) -> dict | None:
    prompt = build_prompt(skill)
    messages = [{'role': 'system', 'content': SYSTEM_MESSAGE}, {'role': 'user', 'content': prompt}]
    for attempt in range(max_retries + 1):
        try:
            raw = generate(model, tokenizer, messages, max_new_tokens, temperature)
        except Exception as e:
            print(f"    [retry {attempt}] {skill['skill_name']}: generate error: {e}")
            continue
        parsed = parse_json_from_llm(raw)
        if parsed and isinstance(parsed, dict) and isinstance(parsed.get('prompt_guidance'), str) and parsed['prompt_guidance'].strip():
            return {'prompt_guidance': parsed.get('prompt_guidance', '').strip(), 'docstring': (parsed.get('docstring', '') or '').strip(), 'body_content': (parsed.get('body_content', '') or '').strip()}
        preview = (raw or '')[:200].replace('\n', ' ')
        print(f"    [retry {attempt}] {skill['skill_name']}: parse failed; raw={preview}")
    return None

def main():
    parser = argparse.ArgumentParser(description='8B LoRA Rewriter — produces --dynamic-rewrite-file compatible JSONL')
    parser.add_argument('--domain', required=True)
    parser.add_argument('--base-model', required=True)
    parser.add_argument('--lora-path', required=True)
    parser.add_argument('--selection-file', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--max-new-tokens', type=int, default=1024)
    parser.add_argument('--temperature', type=float, default=0.3)
    parser.add_argument('--limit-skills', type=int, default=None, help='Smoke-test: only rewrite the first N unique skills')
    args = parser.parse_args()
    print(f"[1/4] Loading D0 skills for domain='{args.domain}'...")
    all_skills = load_all_skills(args.domain)
    skill_by_name = {s['skill_name']: s for s in all_skills}
    print(f'  {len(all_skills)} skills in domain pool')
    print(f'[2/4] Reading selection file {args.selection_file}...')
    with open(args.selection_file) as f:
        selection_rows = [json.loads(line) for line in f if line.strip()]
    unique_sets: dict[tuple, list[str]] = {}
    for row in selection_rows:
        names = tuple(sorted(row['selected_skill_names']))
        unique_sets.setdefault(names, []).append(str(row['task_id']))
    unique_skills = sorted({n for names in unique_sets for n in names})
    if args.limit_skills:
        unique_skills = unique_skills[:args.limit_skills]
    print(f'  {len(selection_rows)} tasks → {len(unique_sets)} unique skill_sets, {len(unique_skills)} unique skills to rewrite' + (f' (LIMITED to first {args.limit_skills})' if args.limit_skills else ''))
    print(f'[3/4] Loading tokenizer + base model...')
    tokenizer = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    base_model = AutoModelForCausalLM.from_pretrained(args.base_model, dtype=torch.bfloat16, trust_remote_code=True, attn_implementation='sdpa', device_map='cuda')
    print(f'  Applying LoRA adapter from {args.lora_path}...')
    model = PeftModel.from_pretrained(base_model, args.lora_path)
    model.eval()
    print(f'[4/4] Rewriting {len(unique_skills)} unique skills...')
    rewrites_by_skill: dict[str, dict] = {}
    t0 = time.time()
    for (idx, name) in enumerate(unique_skills, 1):
        if name not in skill_by_name:
            print(f'  [{idx}/{len(unique_skills)}] {name}: NOT IN DOMAIN POOL, skipping')
            continue
        r = rewrite_one_skill(model, tokenizer, skill_by_name[name], args.max_new_tokens, args.temperature)
        if r:
            rewrites_by_skill[name] = r
            print(f"  [{idx}/{len(unique_skills)}] {name}: OK (guid={len(r['prompt_guidance'])} doc={len(r['docstring'])} body={len(r['body_content'])} chars)")
        else:
            print(f'  [{idx}/{len(unique_skills)}] {name}: FAILED')
    elapsed = time.time() - t0
    print(f'  Rewriting done in {elapsed:.0f}s ({elapsed / max(len(rewrites_by_skill), 1):.1f}s/skill)')
    mode_tag = f'lora_{Path(args.lora_path).name}'
    output_records = []
    for (names, task_ids) in unique_sets.items():
        for name in names:
            if name not in rewrites_by_skill:
                continue
            output_records.append({'skill_set': list(names), 'skill_name': name, 'task_ids': task_ids, 'mode': mode_tag, 'rewritten': rewrites_by_skill[name]})
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open('w', encoding='utf-8') as f:
        for rec in output_records:
            f.write(json.dumps(rec, ensure_ascii=False) + '\n')
    print(f'\nDone: {len(output_records)} records ({len(rewrites_by_skill)}/{len(unique_skills)} skills, {len(unique_sets)} unique sets)')
    print(f'Output: {output_path}')
if __name__ == '__main__':
    main()
