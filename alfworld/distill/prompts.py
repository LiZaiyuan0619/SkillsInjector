DISTILL_SYSTEM = 'You are an expert at distilling reusable procedural skills from agent trajectories.\n\nGiven several successful expert trajectories that solve household tasks in a text-based environment (ALFWorld), your job is to extract general-purpose skills that an agent can re-use to solve similar tasks.\n\nOutput format: each skill in this exact markdown structure, separated by `---`:\n\n### Skill: <short imperative name>\n\n**Description:** <1-2 sentence summary>\n\n**When to use:** <bullet list of trigger conditions>\n\n**Steps:**\n1. <action template>\n2. <action template>\n3. ...\n\nConstraints:\n- AT MOST 3 skills per call.\n- Skills must be GENERAL (parameterized by object/receptacle), not trajectory-specific.\n- Action vocabulary: go to <recep>, take <obj> from <recep>, put <obj> in/on <recep>, open <recep>, close <recep>, examine <obj>, use <obj>, heat <obj> with <recep>, cool <obj> with <recep>, clean <obj> with <recep>, look.\n'
DISTILL_USER_TEMPLATE = 'Below are 3 successful expert trajectories from ALFWorld task type: **{task_type}**.\n\nExtract up to 3 reusable skills.\n\n---\n\n## Trajectory 1\n{traj_1}\n\n---\n\n## Trajectory 2\n{traj_2}\n\n---\n\n## Trajectory 3\n{traj_3}\n\n---\n\nOutput the extracted skills now.'

def render_distill_user(task_type: str, trajectories: list[str]) -> str:
    if len(trajectories) != 3:
        raise ValueError(f'expected 3 trajectories, got {len(trajectories)}')
    return DISTILL_USER_TEMPLATE.format(task_type=task_type, traj_1=trajectories[0], traj_2=trajectories[1], traj_3=trajectories[2])
REWRITER_PROMPT_VERSION = 'v1.0'
REWRITER_SYSTEM = 'You are an expert at rewriting skill descriptions to disambiguate co-injected skills for a specific task.\n\nGiven:\n- A target task instruction.\n- A set of skills (each with a name, sha8 id, and original description) that have been pre-selected for this task.\n\nYour job: rewrite each skill\'s description (1-3 sentences) so that — when shown TOGETHER as a multi-skill context — they are clearly differentiated for the target task. Preserve names. Preserve the spirit of each skill\'s purpose.\n\nOutput STRICT JSON with this schema:\n\n{\n  "rewritten": [\n    {"sha8": "<input sha8>", "display_name": "<input name>", "rewritten_content": "<your rewritten markdown>"}\n    , ...\n  ]\n}\n\nConstraints:\n- Output exactly len(input_skills) entries.\n- The "sha8" set must EQUAL the input sha8 set.\n- "rewritten_content" must include the four sections: ### Skill: ... / **Description:** / **When to use:** / **Steps:**.\n- Do not output anything outside the JSON object.\n'

def render_rewriter_user(task_query: str, skills: list[dict]) -> str:
    blocks = []
    for s in skills:
        blocks.append(f"## Skill (sha8={s['sha8']}, name={s['display_name']})\n{s['content']}\n")
    return f'Target task: {task_query}\n\nInput skills:\n\n' + '\n---\n'.join(blocks) + '\n\nOutput the JSON now.'
