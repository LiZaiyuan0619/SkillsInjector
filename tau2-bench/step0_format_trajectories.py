import argparse
import json
from pathlib import Path

def format_message(idx: int, msg: dict) -> str:
    role = msg['role'].upper()
    if role == 'TOOL':
        content = msg.get('content', '') or ''
        if len(content) > 1500:
            content = content[:1500] + '\n... [TRUNCATED]'
        error_tag = ' [ERROR]' if msg.get('error') else ''
        return f'[Turn {idx}] TOOL RESPONSE{error_tag}: {content}'
    elif role == 'ASSISTANT':
        tool_calls = msg.get('tool_calls') or []
        content = msg.get('content') or ''
        if tool_calls:
            tc_strs = []
            for tc in tool_calls:
                name = tc['name']
                args = tc.get('arguments', {})
                if isinstance(args, str):
                    tc_strs.append(f'{name}({args})')
                else:
                    args_str = ', '.join((f'{k}="{v}"' if isinstance(v, str) else f'{k}={v}' for (k, v) in args.items()))
                    tc_strs.append(f'{name}({args_str})')
            tc_text = '; '.join(tc_strs)
            if content:
                return f'[Turn {idx}] ASSISTANT: {content}\n           [TOOL CALL] {tc_text}'
            else:
                return f'[Turn {idx}] ASSISTANT: [TOOL CALL] {tc_text}'
        else:
            return f'[Turn {idx}] ASSISTANT: {content}'
    elif role == 'USER':
        content = msg.get('content', '') or ''
        return f'[Turn {idx}] USER: {content}'
    else:
        content = msg.get('content', '') or ''
        return f'[Turn {idx}] {role}: {content}'

def format_single_trial(trial_idx: int, seed: int, result: dict) -> str:
    reward = result['reward']
    outcome = 'PASS' if reward == 1.0 else 'FAIL'
    num_messages = result['num_messages']
    num_tools = result['num_tool_calls']
    header = f'=== Trial {trial_idx} (seed={seed}) | Result: {outcome} | Messages: {num_messages} | Tool Calls: {num_tools} ==='
    traj = result.get('trajectory')
    if not traj:
        return header + '\n[NO TRAJECTORY DATA]\n'
    messages = traj.get('messages', [])
    formatted_msgs = []
    for (i, msg) in enumerate(messages):
        formatted_msgs.append(format_message(i + 1, msg))
    return header + '\n' + '\n'.join(formatted_msgs) + '\n'

def format_task(task_id: str, trial_results: list[dict], seeds: list[int]) -> str:
    task_desc = ''
    user_scenario = ''
    eval_criteria = ''
    for r in trial_results:
        traj = r.get('trajectory')
        if traj:
            td = traj.get('task_description')
            if td:
                purpose = td.get('purpose', '')
                policies = td.get('relevant_policies') or ''
                notes = td.get('notes') or ''
                parts = [f'Purpose: {purpose}']
                if policies:
                    parts.append(f'Relevant Policies: {policies}')
                if notes:
                    parts.append(f'Notes: {notes}')
                task_desc = '\n'.join(parts)
            us = traj.get('user_scenario')
            if us:
                if isinstance(us, dict):
                    instructions = us.get('instructions', {})
                    if isinstance(instructions, dict):
                        reason = instructions.get('reason_for_call', '')
                        known = instructions.get('known_info', '')
                        parts = []
                        if reason:
                            parts.append(f'Reason for Call: {reason}')
                        if known:
                            parts.append(f'Known Info: {known}')
                        user_scenario = '\n'.join(parts)
                    else:
                        user_scenario = str(instructions)
            ec = traj.get('evaluation_criteria')
            if ec:
                eval_criteria = json.dumps(ec, indent=2, ensure_ascii=False)
            break
    rewards = [r['reward'] for r in trial_results]
    pass_count = sum((1 for r in rewards if r == 1.0))
    header_lines = [f"{'=' * 70}", f'TASK {task_id} | Pass: {pass_count}/3 | Rewards: {rewards}', f"{'=' * 70}"]
    if task_desc:
        header_lines.append(f'\n## Task Description\n{task_desc}')
    if user_scenario:
        header_lines.append(f'\n## User Scenario\n{user_scenario}')
    if eval_criteria:
        header_lines.append(f'\n## Evaluation Criteria\n{eval_criteria}')
    header_lines.append('')
    trial_texts = []
    for (i, (result, seed)) in enumerate(zip(trial_results, seeds)):
        trial_texts.append(format_single_trial(i, seed, result))
    return '\n'.join(header_lines) + '\n' + '\n'.join(trial_texts)

def load_trial_data(trial_files: list[str]) -> tuple[list[dict], list[int]]:
    trials = []
    seeds = []
    for f in trial_files:
        data = json.loads(Path(f).read_text())
        trials.append(data)
        seeds.append(data['summary']['seed'])
    return (trials, seeds)

def get_task_ids(trials: list[dict]) -> list[str]:
    ids = set()
    for trial in trials:
        for r in trial['results']:
            ids.add(str(r['task_id']))
    try:
        return sorted(ids, key=lambda x: int(x))
    except ValueError:
        return sorted(ids)

def get_task_results(task_id: str, trials: list[dict]) -> list[dict]:
    results = []
    for trial in trials:
        for r in trial['results']:
            if str(r['task_id']) == task_id:
                results.append(r)
                break
        else:
            results.append({'task_id': task_id, 'reward': 0.0, 'num_messages': 0, 'num_tool_calls': 0, 'tool_sequence': [], 'tool_traces': []})
    return results

def main():
    parser = argparse.ArgumentParser(description='Step 0: Format trajectories for skill generation')
    parser.add_argument('--trial-files', nargs='+', required=True, help='Trial result JSON files (3 files, one per trial)')
    parser.add_argument('--task-ids', nargs='+', default=None, help='Specific task IDs to process (default: all)')
    parser.add_argument('--output-dir', type=str, required=True, help='Output directory for formatted trajectory files')
    args = parser.parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f'Loading {len(args.trial_files)} trial files...')
    (trials, seeds) = load_trial_data(args.trial_files)
    print(f'  Seeds: {seeds}')
    all_task_ids = get_task_ids(trials)
    if args.task_ids:
        task_ids = [tid for tid in args.task_ids if tid in all_task_ids]
        print(f'  Processing {len(task_ids)} specified tasks')
    else:
        task_ids = all_task_ids
        print(f'  Processing all {len(task_ids)} tasks')
    has_trajectory = False
    for r in trials[0]['results'][:1]:
        if 'trajectory' in r:
            has_trajectory = True
    if not has_trajectory:
        print('\n  ERROR: Trial files do not contain trajectory data.')
        print('  Re-run experiments with --save-trajectories flag.')
        return
    skipped = []
    processed = 0
    for task_id in task_ids:
        task_results = get_task_results(task_id, trials)
        missing = [i for (i, r) in enumerate(task_results) if 'trajectory' not in r]
        if missing:
            print(f'  WARNING: Task {task_id} missing trajectory in trial(s) {missing}')
        formatted = format_task(task_id, task_results, seeds)
        estimated_tokens = len(formatted) // 4
        if estimated_tokens > 180000:
            print(f'  SKIP: Task {task_id} estimated {estimated_tokens} tokens (>180K limit)')
            skipped.append({'task_id': task_id, 'estimated_tokens': estimated_tokens})
            continue
        out_file = output_dir / f'task_{task_id}.txt'
        out_file.write_text(formatted, encoding='utf-8')
        processed += 1
    summary = {'total_tasks': len(task_ids), 'processed': processed, 'skipped': skipped, 'seeds': seeds, 'trial_files': args.trial_files}
    summary_file = output_dir / 'format_summary.json'
    summary_file.write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f'\nDone: {processed} tasks formatted, {len(skipped)} skipped')
    print(f'Output: {output_dir}')
    if skipped:
        print(f"Skipped tasks (too long): {[s['task_id'] for s in skipped]}")
if __name__ == '__main__':
    main()
