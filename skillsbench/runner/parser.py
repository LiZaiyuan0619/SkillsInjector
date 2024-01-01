from __future__ import annotations
import json
import re
from dataclasses import dataclass
_ACTION_RE = re.compile('<action>(.*?)</action>', re.DOTALL)
_DONE_RE = re.compile('<done>(.*?)</done>', re.DOTALL)

@dataclass
class ParsedAction:
    kind: str
    cmd: str | None = None
    cwd: str | None = None
    reason: str | None = None
    raw_error: str | None = None

def parse_assistant(text: str) -> ParsedAction:
    done_matches = _DONE_RE.findall(text)
    if done_matches:
        body = done_matches[-1].strip()
        try:
            obj = json.loads(body) if body else {}
            return ParsedAction(kind='done', reason=str(obj.get('reason', '')))
        except json.JSONDecodeError:
            return ParsedAction(kind='done', reason=body[:200])
    action_matches = _ACTION_RE.findall(text)
    if not action_matches:
        return ParsedAction(kind='parse_error', raw_error='No <action>...</action> or <done>...</done> tag found.')
    body = action_matches[-1].strip()
    try:
        obj = json.loads(body)
    except json.JSONDecodeError as e:
        return ParsedAction(kind='parse_error', raw_error=f'Invalid JSON in <action>: {e}. Got: {body[:200]}')
    cmd = obj.get('cmd')
    if not isinstance(cmd, str) or not cmd.strip():
        return ParsedAction(kind='parse_error', raw_error="<action> JSON missing required string field 'cmd'.")
    cwd = obj.get('cwd')
    if cwd is not None and (not isinstance(cwd, str)):
        return ParsedAction(kind='parse_error', raw_error="<action> 'cwd' must be a string if provided.")
    return ParsedAction(kind='action', cmd=cmd, cwd=cwd)
