from __future__ import annotations
import datetime as _dt
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from .agent import run_react_loop
from .docker_mgr import TaskContainer
from .llm import LLMClient, LLMConfig
from .prompt import PROMPT_VERSION, build_system_prompt
logger = logging.getLogger(__name__)

@dataclass
class TrialSpec:
    task_id: str
    task_dir: Path
    method: str
    trial_idx: int
    max_steps: int = 50
    skill_bodies: list[tuple[str, str]] | None = None
    skill_ids: list[str] | None = None
    truncated_to: int | None = None

def _utc_now_iso() -> str:
    return _dt.datetime.now(_dt.timezone.utc).isoformat(timespec='seconds')

def run_trial(spec: TrialSpec, output_dir: Path) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    instruction = (spec.task_dir / 'instruction.md').read_text()
    skill_chars = sum((len(b) for (_, b) in spec.skill_bodies or []))
    cfg = LLMConfig.from_env()
    llm = LLMClient(cfg)
    started = _utc_now_iso()
    record: dict = {'task_id': spec.task_id, 'method': spec.method, 'trial_idx': spec.trial_idx, 'harness': 'minimal_react', 'model': cfg.model, 'backend_label': cfg.backend_label, 'prompt_version': PROMPT_VERSION, 'skill_ids': spec.skill_ids or [], 'skill_chars': skill_chars, 'truncated_to': spec.truncated_to, 'max_steps': spec.max_steps, 'started_at': started}
    container = TaskContainer.for_trial(spec.task_dir, spec.task_id)
    try:
        logger.info('[%s] build image', spec.task_id)
        container.build_image()
        logger.info('[%s] start container %s', spec.task_id, container.container_name)
        container.start()
        system_prompt = build_system_prompt(instruction=instruction, skill_bodies=spec.skill_bodies, max_steps=spec.max_steps)
        outcome = run_react_loop(container=container, llm=llm, system_prompt=system_prompt, max_steps=spec.max_steps)
        logger.info('[%s] agent stop_reason=%s n_steps=%d', spec.task_id, outcome.stop_reason, outcome.n_steps)
        if outcome.stop_reason in ('api_error', 'exec_error'):
            reward = 0
            verifier_log = f'[skipped — stop_reason={outcome.stop_reason}]'
        else:
            try:
                (reward, verifier_log) = container.run_verifier(output_dir / 'trial_artifacts')
            except Exception as e:
                logger.error('[%s] verifier failed: %s', spec.task_id, e)
                reward = 0
                verifier_log = f'[verifier exception]: {e}'
        record.update({'reward': reward, 'stop_reason': outcome.stop_reason, 'n_steps': outcome.n_steps, 'n_tool_calls': outcome.n_tool_calls, 'parse_errors': outcome.parse_errors, 'messages': outcome.messages, 'commands': outcome.commands, 'verifier_log_excerpt': verifier_log, 'agent_error': outcome.error, 'finished_at': _utc_now_iso()})
    finally:
        container.cleanup()
        llm.close()
    fname = f'{spec.task_id}__{spec.method}__t{spec.trial_idx}.json'
    out_path = output_dir / fname
    out_path.write_text(json.dumps(record, indent=2))
    logger.info('wrote %s (reward=%s, stop_reason=%s)', out_path, record.get('reward'), record.get('stop_reason'))
    return record
