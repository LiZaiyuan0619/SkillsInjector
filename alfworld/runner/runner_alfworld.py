from __future__ import annotations
import json
import logging
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Any
from .envserver_client import EnvServerClient, EnvServerError
from .llm import LLMClient
from .alfworld_parser import parse_thought_action
from .prompt import build_messages, append_step_message
log = logging.getLogger(__name__)
MAX_ROUND = 30
TOKEN_CAP = 28000
PARSE_FAIL_LIMIT = 3
INVALID_ACTION_LIMIT = 10

@dataclass
class TrialSpec:
    item_id: int
    task_id: str
    task_type: str
    method: str
    trial_idx: int
    skills: list[dict]
    skill_ids: list[str]
    selection_meta: dict[str, Any] = field(default_factory=dict)
    rewriter_meta: dict[str, Any] = field(default_factory=dict)
    admissible_policy: str = 'strict-stock'
    backend_label: str = 'qwen3-8b-sglang'

@dataclass
class TrialRecord:
    spec: dict[str, Any]
    reward: float
    stop_reason: str
    n_steps: int
    parse_errors: int
    invalid_action_count: int
    invalid_action_streak_max: int
    rewriter_fallback: bool
    duration_s: float
    started_at: str
    messages: list[dict[str, Any]] = field(default_factory=list)
    commands: list[dict[str, Any]] = field(default_factory=list)
    usage_total: dict = field(default_factory=dict)

def _count_tokens(messages: list[dict], tokenizer) -> int:
    try:
        s = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
    except Exception:
        s = '\n'.join((f"<|{m['role']}|>\n{m['content']}" for m in messages))
    return len(tokenizer.encode(s))

def run_trial_loop(spec: TrialSpec, *, env_id: int, initial_observation: str, available_actions: list[str], llm: LLMClient, env: EnvServerClient, tokenizer) -> TrialRecord:
    started = datetime.utcnow().isoformat()
    t0 = time.time()
    r = TrialRecord(spec=asdict(spec) | {'skills': [s.get('skill_id', s.get('sha8', '')) for s in spec.skills]}, reward=0.0, stop_reason='unknown', n_steps=0, parse_errors=0, invalid_action_count=0, invalid_action_streak_max=0, rewriter_fallback=spec.rewriter_meta.get('fallback', False), duration_s=0.0, started_at=started, messages=[], commands=[], usage_total={})
    messages = build_messages(spec.skills, initial_observation, available_actions, admissible_policy=spec.admissible_policy)
    parse_fails = 0
    invalid_streak = 0
    invalid_max = 0
    invalid_total = 0
    total_prompt_tokens = 0
    total_completion_tokens = 0
    for round_idx in range(MAX_ROUND):
        n_tok = _count_tokens(messages, tokenizer)
        if n_tok > TOKEN_CAP:
            r.stop_reason = 'token_cap'
            log.info(f'  [{spec.task_id}] token_cap@round={round_idx} ({n_tok}>{TOKEN_CAP})')
            break
        try:
            temp = 0.0 if spec.trial_idx == 0 else 0.7
            (text, usage) = llm.chat(messages, temperature=temp)
            total_prompt_tokens += usage.get('prompt_tokens', 0)
            total_completion_tokens += usage.get('completion_tokens', 0)
        except Exception as e:
            log.error(f'  [{spec.task_id}] api_error@round={round_idx}: {e}')
            r.stop_reason = 'api_error'
            break
        (thought, action, perr) = parse_thought_action(text)
        if perr:
            parse_fails += 1
            log.warning(f'  [{spec.task_id}] parse_failure {parse_fails}: {perr}')
            messages.append({'role': 'assistant', 'content': text})
            messages.append({'role': 'user', 'content': "Could not parse 'Action:' from your output. Please respond with `Thought: ...\\nAction: <admissible action>`."})
            if parse_fails >= PARSE_FAIL_LIMIT:
                r.stop_reason = 'parse_failure'
                break
            continue
        try:
            step = env.step(env_id, action)
        except EnvServerError as e:
            log.error(f'  [{spec.task_id}] env_error@round={round_idx}: {e}')
            r.stop_reason = 'env_error'
            break
        next_obs = step['observation']
        reward = float(step.get('reward', 0))
        done = bool(step.get('done', False))
        next_avail = step.get('available_actions', [])
        invalid_this_step = next_obs.strip().lower().startswith('nothing happened')
        if invalid_this_step:
            invalid_total += 1
            invalid_streak += 1
            invalid_max = max(invalid_max, invalid_streak)
        else:
            invalid_streak = 0
        r.commands.append({'round': round_idx, 'action': action, 'obs_excerpt': next_obs[:200], 'reward': reward, 'done': done, 'n_avail': len(next_avail), 'invalid_action': invalid_this_step})
        if done:
            r.reward = reward
            r.stop_reason = 'done'
            r.n_steps = round_idx + 1
            append_step_message(messages, text, next_obs, next_avail, admissible_policy=spec.admissible_policy)
            break
        if invalid_streak >= INVALID_ACTION_LIMIT:
            r.stop_reason = 'env_invalid_action'
            break
        append_step_message(messages, text, next_obs, next_avail, admissible_policy=spec.admissible_policy)
    else:
        r.stop_reason = 'max_round'
    r.parse_errors = parse_fails
    r.invalid_action_count = invalid_total
    r.invalid_action_streak_max = invalid_max
    r.n_steps = r.n_steps or len(r.commands)
    r.duration_s = time.time() - t0
    r.messages = messages
    r.usage_total = {'prompt_tokens': total_prompt_tokens, 'completion_tokens': total_completion_tokens}
    return r

def write_trial_record(record: TrialRecord, output_dir: Path) -> Path:
    spec = record.spec
    method = spec['method']
    task_id = spec['task_id']
    trial_idx = spec['trial_idx']
    d = output_dir / method
    d.mkdir(parents=True, exist_ok=True)
    p = d / f'{task_id}__{method}__t{trial_idx}.json'
    p.write_text(json.dumps(asdict(record), indent=2, default=str))
    return p
