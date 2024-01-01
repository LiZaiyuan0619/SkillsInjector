from __future__ import annotations
from ._agentgym_vendor import parse_react

def parse_thought_action(text: str) -> tuple[str | None, str | None, str | None]:
    try:
        result = parse_react(text)
    except Exception as e:
        return (None, None, f'parse_react raised: {e}')
    thought = getattr(result, 'thought', None)
    action = getattr(result, 'action', None)
    if action is None or not isinstance(action, str) or action.strip() == '':
        return (thought, None, 'Action: empty or missing')
    lines = action.strip().splitlines()
    first_line = lines[0].strip() if lines else ''
    if not first_line:
        return (thought, None, 'Action: empty after first-line trim')
    return (thought, first_line, None)
