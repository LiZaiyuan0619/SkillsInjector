from __future__ import annotations
import logging
from dataclasses import dataclass, field
from .docker_mgr import ShellResult, TaskContainer
from .llm import LLMClient
from .parser import ParsedAction, parse_assistant
logger = logging.getLogger(__name__)
MAX_PARSE_ERRORS = 3

@dataclass
class TrialOutcome:
    stop_reason: str
    n_steps: int
    n_tool_calls: int
    parse_errors: int
    messages: list[dict] = field(default_factory=list)
    commands: list[dict] = field(default_factory=list)
    last_assistant_text: str = ''
    error: str | None = None

def _format_result(res: ShellResult) -> str:
    return f'<result exit_code="{res.exit_code}" duration_s="{res.duration_s}" timed_out="{str(res.timed_out).lower()}">\nstdout:\n{res.stdout}\nstderr:\n{res.stderr}\n</result>'

def _format_parse_error(action: ParsedAction) -> str:
    return f'[parse_error] {action.raw_error}\nPlease retry with EXACTLY one of:\n  <action>{{"cmd": "<bash>", "cwd": "<optional path>"}}</action>\n  <done>{{"reason": "<short>"}}</done>'

def run_react_loop(*, container: TaskContainer, llm: LLMClient, system_prompt: str, max_steps: int) -> TrialOutcome:
    messages: list[dict] = [{'role': 'system', 'content': system_prompt}, {'role': 'user', 'content': 'Begin.'}]
    commands: list[dict] = []
    parse_errors = 0
    n_tool_calls = 0
    last_text = ''
    for step in range(1, max_steps + 1):
        try:
            (assistant_text, _usage) = llm.chat(messages)
        except Exception as e:
            logger.error('LLM call failed at step %d: %s', step, e)
            return TrialOutcome(stop_reason='api_error', n_steps=step - 1, n_tool_calls=n_tool_calls, parse_errors=parse_errors, messages=messages, commands=commands, error=str(e))
        last_text = assistant_text
        messages.append({'role': 'assistant', 'content': assistant_text})
        action = parse_assistant(assistant_text)
        if action.kind == 'done':
            return TrialOutcome(stop_reason='done', n_steps=step, n_tool_calls=n_tool_calls, parse_errors=parse_errors, messages=messages, commands=commands, last_assistant_text=last_text)
        if action.kind == 'parse_error':
            parse_errors += 1
            if parse_errors > MAX_PARSE_ERRORS:
                return TrialOutcome(stop_reason='parse_failure', n_steps=step, n_tool_calls=n_tool_calls, parse_errors=parse_errors, messages=messages, commands=commands, last_assistant_text=last_text, error=action.raw_error)
            messages.append({'role': 'user', 'content': _format_parse_error(action)})
            continue
        n_tool_calls += 1
        try:
            res = container.exec_shell(cmd=action.cmd or '', cwd=action.cwd)
        except Exception as e:
            logger.error('docker exec failed at step %d: %s', step, e)
            return TrialOutcome(stop_reason='exec_error', n_steps=step, n_tool_calls=n_tool_calls, parse_errors=parse_errors, messages=messages, commands=commands, last_assistant_text=last_text, error=str(e))
        commands.append({'step': step, 'cmd': action.cmd, 'cwd': action.cwd, 'exit_code': res.exit_code, 'duration_s': res.duration_s, 'timed_out': res.timed_out})
        messages.append({'role': 'user', 'content': _format_result(res)})
    return TrialOutcome(stop_reason='max_steps', n_steps=max_steps, n_tool_calls=n_tool_calls, parse_errors=parse_errors, messages=messages, commands=commands, last_assistant_text=last_text)
