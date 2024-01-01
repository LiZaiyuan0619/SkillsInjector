from __future__ import annotations
import json
import re

class RewriterParseError(Exception):
    pass

def parse_rewriter_response(text: str, input_skills: list[dict]) -> list[dict]:
    m = re.search('```(?:json)?\\s*(\\{.+?\\})\\s*```', text, re.DOTALL)
    blob = m.group(1) if m else text
    try:
        data = json.loads(blob)
    except Exception as e:
        raise RewriterParseError(f'not valid JSON: {e}; raw[:200]={text[:200]!r}')
    if 'rewritten' not in data or not isinstance(data['rewritten'], list):
        raise RewriterParseError(f"missing 'rewritten' list; got keys={list(data)}")
    items = data['rewritten']
    if len(items) != len(input_skills):
        raise RewriterParseError(f'count mismatch: got {len(items)}, expected {len(input_skills)}')
    in_sha8s = {s['sha8'] for s in input_skills}
    out_sha8s = {it.get('sha8') for it in items}
    if in_sha8s != out_sha8s:
        raise RewriterParseError(f'sha8 set mismatch: in={in_sha8s}, out={out_sha8s}')
    required = ['### Skill:', '**Description:**', '**When to use:**', '**Steps:**']
    out_by_sha = {it['sha8']: it for it in items}
    out: list[dict] = []
    for s in input_skills:
        it = out_by_sha[s['sha8']]
        c = it.get('rewritten_content', '')
        if not c or not isinstance(c, str):
            raise RewriterParseError(f"sha8={s['sha8']}: empty rewritten_content")
        for sec in required:
            if sec not in c:
                raise RewriterParseError(f"sha8={s['sha8']}: missing section {sec!r}")
        out.append({'sha8': s['sha8'], 'display_name': it.get('display_name', s['display_name']), 'rewritten_content': c})
    return out
