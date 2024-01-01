import argparse
import ast
import json
import re
import sys
from pathlib import Path

def _extract_skill_fields(source: str) -> dict:
    module = ast.parse(source)
    wanted = {'SKILL_NAME', 'PROMPT_GUIDANCE'}
    fields = {}
    for node in module.body:
        if not isinstance(node, ast.Assign) or len(node.targets) != 1:
            continue
        target = node.targets[0]
        if not isinstance(target, ast.Name) or target.id not in wanted:
            continue
        if isinstance(node.value, ast.Constant) and isinstance(node.value.value, str):
            fields[target.id] = node.value.value
    return fields

def _extract_docstring(source: str) -> str:
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return ''
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            ds = ast.get_docstring(node)
            return ds if ds else ''
    return ''

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--domain', default='airline')
    parser.add_argument('--d2-version', default='v3', help="D2 result version: 'v3' for conservative (default)")
    parser.add_argument('--output', default=None, help='Output filename (default: auto-generated)')
    args = parser.parse_args()
    rewriter_dir = Path(__file__).resolve().parent
    parsed_dir = rewriter_dir.parent / f'generated_skills/{args.domain}/parsed'
    d2_file = rewriter_dir / f'data/{args.domain}/d2_{args.d2_version}_results.jsonl'
    if args.output:
        out_path = rewriter_dir / f'data/{args.domain}/{args.output}'
    else:
        out_path = rewriter_dir / f'data/{args.domain}/rewritten_skills_d2{args.d2_version}_body_only.jsonl'
    originals = {}
    for fname in sorted(parsed_dir.glob('*.py')):
        source = fname.read_text(encoding='utf-8')
        fields = _extract_skill_fields(source)
        name = fields.get('SKILL_NAME', '')
        if not name:
            continue
        guidance = fields.get('PROMPT_GUIDANCE', '')
        docstring = _extract_docstring(source)
        originals[name] = {'prompt_guidance': guidance, 'docstring': docstring}
    print(f'Loaded {len(originals)} original skills')
    d2_results = {}
    with open(d2_file) as f:
        for line in f:
            rec = json.loads(line)
            d2_key = f'd2_{args.d2_version}'
            d2_data = rec.get(d2_key, {})
            body = d2_data.get('body_content', '')
            if body.strip():
                d2_results[rec['skill_name']] = body
    print(f'Loaded {len(d2_results)} D2 {args.d2_version} results with body')
    output_records = []
    matched = 0
    for skill_name in sorted(originals.keys()):
        if skill_name not in d2_results:
            continue
        orig = originals[skill_name]
        record = {'skill_name': skill_name, 'mode': f'd2{args.d2_version}_body_only', 'rewritten': {'prompt_guidance': orig['prompt_guidance'], 'docstring': orig['docstring'], 'body_content': d2_results[skill_name]}}
        output_records.append(record)
        matched += 1
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, 'w', encoding='utf-8') as f:
        for rec in output_records:
            f.write(json.dumps(rec, ensure_ascii=False) + '\n')
    print(f'Assembled {matched} skills -> {out_path}')
if __name__ == '__main__':
    main()
