from __future__ import annotations
PROMPT_VERSION = 'v1'
_SYSTEM_TEMPLATE = 'You are an autonomous coding agent solving a task in a Linux (Ubuntu 24.04) container. Your only tool is run_shell.\n\n# Action format\n\nOn every turn, end your response with EXACTLY ONE of:\n\n<action>{{"cmd": "<bash command>", "cwd": "<absolute path, optional>"}}</action>\n\nWhen you believe the task is complete and verified, instead output:\n\n<done>{{"reason": "<one-sentence summary>"}}</done>\n\nFree-form reasoning before the tag is allowed and encouraged. Anything after the tag is ignored. JSON must be valid (double quotes, escape literal newlines as \\n inside cmd).\n\n# Tool semantics\n\nrun_shell executes the command via `bash -c "<cmd>"` inside the container. You will receive:\n\n<result exit_code="..." duration_s="...">\nstdout: ...\nstderr: ...\n</result>\n\nEach command has a 120-second timeout (use longer-running operations in the background or split them). Truncated output is marked.\n\n# Working environment\n\nDefault cwd is /root unless you specify otherwise. The task\'s input files are pre-loaded under /root. The /tests directory is NOT accessible during your turn (it appears only after you signal <done>).\n\n# Limits\n\n- You have at most {max_steps} steps. The verifier runs after step {max_steps} or after <done>, whichever comes first.\n- Verify your output before declaring <done>. Reward is binary (0 or 1) and computed by an external pytest suite.\n\n# Task instruction\n\n{instruction}\n{skill_section}'

def _format_skill_section(skill_bodies: list[tuple[str, str]] | None) -> str:
    if not skill_bodies:
        return ''
    parts = ['\n\n# Available skills\n', 'The following skill documents have been pre-selected as potentially relevant. ', 'Treat them as reference material — read what helps, ignore the rest.\n']
    for (name, body) in skill_bodies:
        parts.append(f'\n## skill: {name}\n\n{body}\n')
    return ''.join(parts)

def build_system_prompt(instruction: str, skill_bodies: list[tuple[str, str]] | None, max_steps: int) -> str:
    return _SYSTEM_TEMPLATE.format(max_steps=max_steps, instruction=instruction.strip(), skill_section=_format_skill_section(skill_bodies))
