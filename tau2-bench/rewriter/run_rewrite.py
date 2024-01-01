import argparse
import json
import sys
from pathlib import Path
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer
sys.path.insert(0, str(Path(__file__).resolve().parent))
from utils import load_skills_from_ranked_json, load_jsonl, save_jsonl, parse_json_from_llm
PROJECT_BASE = Path(__file__).resolve().parent.parent.parent
DEFAULT_MODEL = '<MODEL_ROOT>/model/Qwen3-8B'
DEFAULT_CHECKPOINT = '<MODEL_ROOT>/agent/skills/rewriter-checkpoints/airline_trace-free+/final'
from utils import REWRITE_SYSTEM_MESSAGE, validate_rewrite_output
SYSTEM_MESSAGE = REWRITE_SYSTEM_MESSAGE
TEMPERATURE = 0.3
TOP_P = 0.9
REPETITION_PENALTY = 1.1
MAX_NEW_TOKENS = 2048
_OUTPUT_SCHEMA = '[OUTPUT]\nOutput ONLY valid JSON (no markdown, no code blocks, no extra text):\n{"prompt_guidance": "<when to use/not use, 1-3 sentences>", "docstring": "<what this skill does and returns, 2-5 sentences>", "body_content": "<the complete improved function return content>"}'

def build_trace_free_input(skill: dict) -> str:
    return f"[INSTRUCTION]\nRewrite the following skill's description and body content so an LLM agent can better select and use it.\n\n[SKILL]\nName: {skill.get('skill_name', '')}\nDescription: {skill.get('skill_description', '')}\nPrompt guidance: {skill.get('prompt_guidance', '')}\nTarget pattern: {skill.get('target_pattern', '')}\nFunction definition:\n{skill.get('body', '')}\n\n{_OUTPUT_SCHEMA}"

def build_trace_based_input(skill: dict, trace_record: dict) -> str:
    success_parts = []
    for (i, t) in enumerate(trace_record.get('success_traces', []), 1):
        summary = t.get('summary', '')
        success_parts.append(f"Case {i} (task_id={t.get('task_id', '?')}):\n{summary}")
    failure_parts = []
    for (i, t) in enumerate(trace_record.get('failure_traces', []), 1):
        summary = t.get('summary', '')
        failure_parts.append(f"Case {i} (task_id={t.get('task_id', '?')}):\n{summary}")
    return f"[INSTRUCTION]\nRewrite the following skill's description and body content so an LLM agent can better select and use it.\n\n[SKILL]\nName: {skill.get('skill_name', '')}\nDescription: {skill.get('skill_description', '')}\nPrompt guidance: {skill.get('prompt_guidance', '')}\nTarget pattern: {skill.get('target_pattern', '')}\nFunction definition:\n{skill.get('body', '')}\n\n[EXECUTION TRACES]\nSuccess cases:\n{(chr(10).join(success_parts) if success_parts else '(none)')}\n\nFailure cases:\n{(chr(10).join(failure_parts) if failure_parts else '(none)')}\n\n{_OUTPUT_SCHEMA}"

def generate(model, tokenizer, user_input: str, device: str) -> str:
    messages = [{'role': 'system', 'content': SYSTEM_MESSAGE}, {'role': 'user', 'content': user_input}]
    prompt_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(prompt_text, return_tensors='pt').to(device)
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=MAX_NEW_TOKENS, temperature=TEMPERATURE, top_p=TOP_P, repetition_penalty=REPETITION_PENALTY, do_sample=True)
    new_tokens = outputs[0][inputs['input_ids'].shape[1]:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True)

def main():
    parser = argparse.ArgumentParser(description='Run the trained Rewriter model for skill description improvement.')
    parser.add_argument('--domain', type=str, default='airline')
    parser.add_argument('--checkpoint', type=str, default=DEFAULT_CHECKPOINT, help='Path to the trained LoRA checkpoint')
    parser.add_argument('--base-model', type=str, default=DEFAULT_MODEL, help='Path to the base model')
    parser.add_argument('--traces-file', type=str, default=None, help='Path to collected_traces.jsonl (for trace-based mode)')
    parser.add_argument('--output', type=str, default=None, help='Output JSONL path')
    parser.add_argument('--device', type=str, default='auto')
    args = parser.parse_args()
    domain = args.domain
    if args.output:
        output_path = Path(args.output)
    else:
        mode_tag = 'trace-based' if args.traces_file else 'trace-free'
        output_path = Path(__file__).resolve().parent / f'data/{domain}/rewritten_skills_{mode_tag}.jsonl'
    ranked_path = PROJECT_BASE / f'temp/generated_skills/{domain}/deduplicated_090/skills_ranked.json'
    parsed_dir = PROJECT_BASE / f'temp/generated_skills/{domain}/parsed'
    skills = load_skills_from_ranked_json(ranked_path, parsed_dir)
    print(f'Loaded {len(skills)} skills')
    traces_by_name = {}
    if args.traces_file:
        traces = load_jsonl(args.traces_file)
        traces_by_name = {t['skill_name']: t for t in traces}
        print(f'Loaded {len(traces)} trace records (trace-based mode)')
    else:
        print('No traces provided (trace-free mode)')
    print(f'Loading base model: {args.base_model}')
    device = args.device
    if device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    tokenizer = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(args.base_model, torch_dtype=torch.bfloat16, trust_remote_code=True, device_map=device)
    print(f'Loading LoRA checkpoint: {args.checkpoint}')
    model = PeftModel.from_pretrained(model, args.checkpoint)
    model.eval()
    results = []
    for (i, skill) in enumerate(skills):
        name = skill['skill_name']
        trace_rec = traces_by_name.get(name)
        if trace_rec:
            user_input = build_trace_based_input(skill, trace_rec)
        else:
            user_input = build_trace_free_input(skill)
        print(f'  [{i + 1}/{len(skills)}] {name}...', end=' ', flush=True)
        raw_output = generate(model, tokenizer, user_input, device)
        parsed = parse_json_from_llm(raw_output)
        validated = validate_rewrite_output(parsed) if parsed else None
        if validated is not None:
            print('OK')
        elif parsed is not None:
            print('WARN: JSON parsed but missing required fields')
            validated = {'raw_output': raw_output, 'partial_parse': parsed}
        else:
            print('WARN: JSON parse failed')
            validated = {'raw_output': raw_output}
        results.append({'skill_name': name, 'mode': 'trace_based' if trace_rec else 'trace_free', 'rewritten': validated})
    save_jsonl(output_path, results)
    print(f'\nDone. {len(results)} skills rewritten → {output_path}')
if __name__ == '__main__':
    main()
