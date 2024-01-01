from __future__ import annotations
import re
from ._agentgym_vendor import REACT_SYSTEM_PROMPT, REACT_ASSISTANT_ACK

def get_base_instruction() -> str:
    return REACT_SYSTEM_PROMPT
ASSISTANT_ACK = REACT_ASSISTANT_ACK

def build_messages(skills: list[dict], initial_observation: str, available_actions: list[str], *, admissible_policy: str='strict-stock') -> list[dict]:
    base = get_base_instruction()
    if skills:
        skill_block = '\n\n## Available Skills\n\nThese skills may help with the task. Read them and use them when appropriate.\n\n'
        for (i, s) in enumerate(skills, 1):
            content = s.get('content') or s.get('body') or ''
            skill_block += f"### Skill {i}: {s['display_name']}\n{content}\n\n"
    else:
        skill_block = ''
    first_user = base + skill_block + '\n\nNow here is your task. Solve it step by step.'
    obs_with_actions = initial_observation.rstrip() + '\n\nAVAILABLE ACTIONS: ' + ', '.join(available_actions)
    return [{'role': 'user', 'content': first_user}, {'role': 'assistant', 'content': ASSISTANT_ACK}, {'role': 'user', 'content': obs_with_actions}]

def append_step_message(messages: list[dict], generated: str, next_obs: str, next_avail: list[str], *, admissible_policy: str='strict-stock') -> None:
    messages.append({'role': 'assistant', 'content': generated})
    if admissible_policy == 'robust-patch':
        next_user = next_obs.rstrip() + '\n\nAVAILABLE ACTIONS: ' + ', '.join(next_avail)
    else:
        next_user = next_obs
    messages.append({'role': 'user', 'content': next_user})

def extract_retrieval_query(initial_observation: str) -> str:
    m = re.search('Your task is to:\\s*(.+?)(?:\\n|$)', initial_observation)
    if m:
        return m.group(1).strip()
    cleaned = initial_observation.split('AVAILABLE ACTIONS:')[0].strip()
    return cleaned[:200]
