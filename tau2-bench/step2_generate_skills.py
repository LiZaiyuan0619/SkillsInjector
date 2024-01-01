import argparse
import json
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
import openai
TOOL_LISTS = {'airline': 'Available Tools (15):\nREAD:  list_all_airports, search_direct_flight, search_onestop_flight, get_reservation_details, get_user_details, get_flight_status\nWRITE: book_reservation, cancel_reservation, send_certificate, update_reservation_baggages, update_reservation_flights, update_reservation_passengers\nGENERIC: calculate, transfer_to_human_agents', 'retail': 'Available Tools (16):\nREAD:  find_user_id_by_email, find_user_id_by_name_zip, get_order_details, get_product_details, get_user_details, list_all_product_types\nWRITE: cancel_pending_order, exchange_delivered_order_items, modify_pending_order_address, modify_pending_order_items, modify_pending_order_payment, modify_user_address, return_delivered_order_items\nGENERIC: calculate, transfer_to_human_agents', 'telecom': 'Available Tools - Agent (13):\nREAD:  get_customer_by_id, get_customer_by_name, get_customer_by_phone, get_bills_for_customer, get_data_usage, get_details_by_id\nWRITE: suspend_line, resume_line, refuel_data, send_payment_request, enable_roaming, disable_roaming\nGENERIC: transfer_to_human_agents'}
POSITIVE_SEED = '```python\nSKILL_NAME = "cancellation_guard"\nSKILL_DOMAIN = "airline"\nSKILL_DESCRIPTION = (\n    "Guides the agent through cancellation eligibility checks. "\n    "Addresses the most common failure: agent reversing a correct refusal "\n    "after user pressure."\n)\nPROMPT_GUIDANCE = (\n    "When handling cancellation requests, use the cancellation_guard tool "\n    "to get the eligibility checklist before making any decisions."\n)\nTARGET_PATTERN = "Agent caves under social pressure and approves cancellations that violate policy"\n\n\ndef cancellation_guard(reservation_id: str) -> str:\n    """Get the cancellation eligibility checklist for a reservation.\n\n    Call this tool BEFORE deciding whether to approve or deny a cancellation\n    request. It returns a step-by-step verification procedure.\n\n    Args:\n        reservation_id: The reservation ID the customer wants to cancel.\n\n    Returns:\n        A checklist of verification steps and common pitfalls to avoid.\n    """\n    return (\n        f"CANCELLATION ELIGIBILITY CHECKLIST for {reservation_id}:\\n"\n        "\\n"\n        "⚠ THIS IS A REFERENCE — NO ACTION HAS BEEN PERFORMED.\\n"\n        "\\n"\n        "Step 1: Call get_reservation_details to retrieve the reservation.\\n"\n        "Step 2: Check the \'created_at\' timestamp — within 24 hours?\\n"\n        "Step 3: Check cabin class — only \'business\' qualifies on its own.\\n"\n        "Step 4: Check travel insurance — reason must be health/weather.\\n"\n        "Step 5: Check flight status — airline-cancelled flights qualify.\\n"\n        "\\n"\n        "ELIGIBILITY: At least ONE must be true:\\n"\n        "  a) Booked within last 24 hours\\n"\n        "  b) Business class\\n"\n        "  c) Has insurance AND valid reason\\n"\n        "  d) Flight cancelled by airline\\n"\n        "\\n"\n        "⚠ WARNINGS:\\n"\n        "  - You CANNOT make exceptions for any membership level.\\n"\n        "  - Prior agent promises do NOT override policy.\\n"\n        "  - If NONE of the conditions are met, the answer is NO."\n    )\n```'
NEGATIVE_SEED = '```python\nSKILL_NAME = "action_executor"\nSKILL_DOMAIN = "retail"\nSKILL_DESCRIPTION = "Executes customer requested actions on orders"\nPROMPT_GUIDANCE = "Use action_executor to handle customer order actions"\n\n\ndef action_executor(customer_request: str) -> str:\n    """Execute the customer\'s requested action.\n\n    Args:\n        customer_request: What the customer wants to do.\n\n    Returns:\n        Action execution result.\n    """\n    return (\n        f"Processing action: {customer_request}\\n"\n        "Step 1: Verify user identity\\n"\n        "Step 2: Check order status\\n"\n        "Step 3: Execute the action"\n    )\n```'
INSTRUCTION_ALL_FAIL = "## Analysis Task\nThis task FAILED in all 3 trials. Analyze:\n1. Where did the Agent make wrong decisions? Did all 3 trials fail for the same reason?\n2. What knowledge is the Agent missing? Is it a rule it doesn't know, or one it knows but misapplies?\n3. Could a skill providing the correct rules/knowledge BEFORE the decision point have prevented this error?\n4. Is this failure pattern generalizable — might other similar tasks also encounter it?\n\nDesign skill(s) to fill the Agent's knowledge gap."
INSTRUCTION_ALL_PASS = '## Analysis Task\nThis task PASSED in all 3 trials. Analyze:\n1. What key domain knowledge did the Agent demonstrate?\n2. Which of these insights are inferred from policy (not explicitly stated)?\n3. Could this implicit knowledge be forgotten or misapplied in harder scenarios?\n4. Did the Agent succeed "by luck" (right outcome but wrong reasoning)?\n\nDesign skill(s) to solidify these success patterns for consistency in harder tasks.\nIf the Agent\'s success is entirely based on explicit policy rules with nothing special,\noutput ONLY the text "NO_SKILL_NEEDED" instead of generating a skill.'
INSTRUCTION_MIXED = '## Analysis Task\nThis task has MIXED results — some trials passed, some failed. Compare:\n1. At which decision point did the successful and failed trajectories diverge?\n2. What did the successful trajectory do right that the failed one missed?\n3. Was the difference caused by different user phrasing, or different Agent reasoning?\n4. Can a skill ensure the Agent always takes the successful path?\n\nDesign skill(s) to bridge the gap between success and failure.'

def build_prompt(domain: str, policy_text: str, formatted_trajectory: str, category: str) -> str:
    tool_list = TOOL_LISTS.get(domain, '')
    if category == 'ALL_FAIL':
        type_instruction = INSTRUCTION_ALL_FAIL
    elif category == 'ALL_PASS':
        type_instruction = INSTRUCTION_ALL_PASS
    else:
        type_instruction = INSTRUCTION_MIXED
    prompt = f'''## Role\nYou are an Agent Skill designer. Your task is to analyze an Agent's conversation\ntrajectories and design Skills that help the Agent perform tasks better.\n\n## Domain Policy (what the Agent already knows)\n{policy_text}\n\n## {tool_list}\n\n{type_instruction}\n\n## Conversation Trajectories\n{formatted_trajectory}\n\n## Skill Format Reference\n\n### Positive Example (verified effective, +8pp improvement)\n{POSITIVE_SEED}\n↑ This skill works because it encodes specific policy rules the Agent forgets under pressure,\nuses reference semantics in naming (guard/checklist), and won't be confused with action tools.\n\n### Negative Example (verified harmful, -8 tasks)\n{NEGATIVE_SEED}\n↑ This skill is harmful because:\n1. Name uses action semantics (executor) — Agent thinks calling it = performing the action\n2. Content is generic process advice the Agent already knows — no new information\n3. 33% of degradation: Agent called skill and stopped, never calling the real API\n\n## Constraints\n1. Skills MUST be reference/information tools — they CANNOT replace real action tools\n2. Names MUST use reference semantics: rules/guide/checklist/reference/policy/criteria\n   Names MUST NOT use action semantics: executor/router/handler/processor/manager\n3. Skills MUST provide domain-specific knowledge the Agent doesn't know,\n   or critical rules the Agent knows but forgets under pressure\n4. Skills MUST NOT be generic process advice (e.g., "verify user identity first")\n5. Function return value MUST include "⚠ THIS IS A REFERENCE — NO ACTION HAS BEEN PERFORMED"\n6. Each skill MUST target a specific pattern observed in the trajectories\n7. SKILL_DOMAIN must be "{domain}"\n\n## Output Format\nOutput each skill as a Python code block. Format:\n\n```python\nSKILL_NAME = "descriptive_name_here"\nSKILL_DOMAIN = "{domain}"\nSKILL_DESCRIPTION = "One-sentence description of what this skill provides"\nPROMPT_GUIDANCE = "When to call this skill"\nTARGET_PATTERN = "Specific pattern this skill addresses"\n\ndef descriptive_name_here(query: str) -> str:\n    """Docstring describing what this reference tool provides.\n\n    Args:\n        query: Brief description of the situation.\n    Returns:\n        Reference information (not an action).\n    """\n    return (\n        f"REFERENCE for {{query}}:\\n"\n        "⚠ THIS IS A REFERENCE — NO ACTION HAS BEEN PERFORMED.\\n"\n        "\\n"\n        "Specific domain knowledge content...\\n"\n    )\n```\n\nGenerate 1-3 skills based on the number of distinct patterns you find.\nDo NOT generate low-quality skills just to fill a quota.\nIf this is an ALL_PASS task with nothing special, output "NO_SKILL_NEEDED".\n'''
    return prompt

def parse_skills_from_response(response_text: str, domain: str) -> list[dict]:
    pattern = '```python\\s*\\n(.*?)```'
    matches = re.findall(pattern, response_text, re.DOTALL)
    skills = []
    for code in matches:
        code = code.strip()
        name_match = re.search('SKILL_NAME\\s*=\\s*["\\\'](.+?)["\\\']', code)
        desc_match = re.search('SKILL_DESCRIPTION\\s*=\\s*(?:\\(\\s*)?["\\\'](.+?)["\\\']', code, re.DOTALL)
        target_match = re.search('TARGET_PATTERN\\s*=\\s*["\\\'](.+?)["\\\']', code, re.DOTALL)
        guidance_match = re.search('PROMPT_GUIDANCE\\s*=\\s*(?:\\(\\s*)?["\\\'](.+?)["\\\']', code, re.DOTALL)
        if not desc_match:
            desc_match = re.search('SKILL_DESCRIPTION\\s*=\\s*\\(\\s*["\\\'](.+?)["\\\']', code, re.DOTALL)
        skill_name = name_match.group(1) if name_match else None
        if not skill_name:
            continue
        func_match = re.search('def\\s+(\\w+)\\s*\\(', code)
        if not func_match:
            continue
        skills.append({'skill_name': skill_name, 'description': desc_match.group(1).strip() if desc_match else '', 'target_pattern': target_match.group(1).strip() if target_match else '', 'prompt_guidance': guidance_match.group(1).strip() if guidance_match else '', 'function_name': func_match.group(1), 'code': code})
    return skills

def call_llm(client: openai.OpenAI, model: str, prompt: str, temperature: float) -> str:
    response = client.chat.completions.create(model=model, messages=[{'role': 'user', 'content': prompt}], temperature=temperature, max_tokens=4096)
    return response.choices[0].message.content or ''

def main():
    parser = argparse.ArgumentParser(description='Step 2: Generate skills from trajectories via LLM')
    parser.add_argument('--domain', required=True, choices=['airline', 'retail', 'telecom'])
    parser.add_argument('--formatted-dir', required=True, help='Directory with formatted trajectory .txt files')
    parser.add_argument('--classification', required=True, help='Task classification JSON from step1')
    parser.add_argument('--policy-file', required=True, help='Domain policy .md file')
    parser.add_argument('--output-dir', required=True, help='Output directory')
    parser.add_argument('--task-ids', nargs='+', default=None, help='Specific task IDs (default: all)')
    parser.add_argument('--temperatures', nargs='+', type=float, default=[0.0, 0.3, 0.5, 0.7, 1.0], help='Temperature values for generation')
    parser.add_argument('--concurrency', '-j', type=int, default=1)
    parser.add_argument('--api-base', default='http://localhost:32800/v1')
    parser.add_argument('--model', default='openai/Qwen/Qwen3-235B-A22B-Instruct-2507')
    args = parser.parse_args()
    output_dir = Path(args.output_dir)
    raw_dir = output_dir / 'raw'
    parsed_dir = output_dir / 'parsed'
    raw_dir.mkdir(parents=True, exist_ok=True)
    parsed_dir.mkdir(parents=True, exist_ok=True)
    policy_text = Path(args.policy_file).read_text()
    classification = json.loads(Path(args.classification).read_text())
    formatted_dir = Path(args.formatted_dir)
    if args.task_ids:
        task_ids = args.task_ids
    else:
        task_ids = sorted(classification['classifications'].keys(), key=lambda x: int(x) if x.isdigit() else x)
    print(f'Domain: {args.domain}')
    print(f'Tasks: {len(task_ids)}')
    print(f'Temperatures: {args.temperatures}')
    print(f'Total LLM calls: {len(task_ids) * len(args.temperatures)}')
    print(f'Concurrency: {args.concurrency}')
    client = openai.OpenAI(base_url=args.api_base, api_key='not-needed')
    work_items = []
    for task_id in task_ids:
        traj_file = formatted_dir / f'task_{task_id}.txt'
        if not traj_file.exists():
            print(f'  WARNING: {traj_file} not found, skipping task {task_id}')
            continue
        cat_info = classification['classifications'].get(task_id)
        if not cat_info:
            print(f'  WARNING: No classification for task {task_id}, skipping')
            continue
        category = cat_info['category']
        formatted_trajectory = traj_file.read_text()
        for temp in args.temperatures:
            work_items.append({'task_id': task_id, 'category': category, 'temperature': temp, 'formatted_trajectory': formatted_trajectory})
    print(f'\nTotal work items: {len(work_items)}')
    lock = threading.Lock()
    counters = {'completed': 0, 'skills_found': 0, 'no_skill': 0, 'errors': 0}
    all_skills = []

    def process_item(item):
        task_id = item['task_id']
        category = item['category']
        temp = item['temperature']
        raw_file = raw_dir / f'task_{task_id}_t{temp}.json'
        if raw_file.exists():
            with lock:
                counters['completed'] += 1
                existing = json.loads(raw_file.read_text())
                print(f"  [SKIP] Task {task_id} t={temp} (already exists, {len(existing.get('skills', []))} skills)")
            return existing.get('skills', [])
        prompt = build_prompt(domain=args.domain, policy_text=policy_text, formatted_trajectory=item['formatted_trajectory'], category=category)
        try:
            response_text = call_llm(client, args.model, prompt, temp)
            if 'NO_SKILL_NEEDED' in response_text and category == 'ALL_PASS':
                skills = []
                with lock:
                    counters['no_skill'] += 1
            else:
                skills = parse_skills_from_response(response_text, args.domain)
            raw_output = {'task_id': task_id, 'category': category, 'temperature': temp, 'response': response_text, 'skills': skills, 'timestamp': datetime.now().isoformat()}
            raw_file.write_text(json.dumps(raw_output, indent=2, ensure_ascii=False))
            with lock:
                counters['completed'] += 1
                counters['skills_found'] += len(skills)
                total = len(work_items)
                print(f"  [{counters['completed']}/{total}] Task {task_id} t={temp} [{category}]: {len(skills)} skill(s)")
            return skills
        except Exception as e:
            with lock:
                counters['errors'] += 1
                counters['completed'] += 1
                print(f'  [ERROR] Task {task_id} t={temp}: {e}')
            return []
    start_time = time.time()
    if args.concurrency <= 1:
        for item in work_items:
            skills = process_item(item)
            all_skills.extend(skills)
    else:
        with ThreadPoolExecutor(max_workers=args.concurrency) as executor:
            futures = {executor.submit(process_item, item): item for item in work_items}
            for future in as_completed(futures):
                skills = future.result()
                all_skills.extend(skills)
    elapsed = time.time() - start_time
    skill_counter = 0
    seen_names = set()
    for skill in all_skills:
        name = skill['skill_name']
        if name in seen_names:
            name = f'{name}_{skill_counter}'
        seen_names.add(name)
        skill_file = parsed_dir / f'skill_{skill_counter:04d}_{name}.py'
        skill_file.write_text(skill['code'])
        skill_counter += 1
    log = {'domain': args.domain, 'total_tasks': len(task_ids), 'temperatures': args.temperatures, 'total_calls': len(work_items), 'completed': counters['completed'], 'errors': counters['errors'], 'no_skill_needed': counters['no_skill'], 'total_skills_generated': len(all_skills), 'unique_skill_files': skill_counter, 'elapsed_seconds': round(elapsed, 1), 'timestamp': datetime.now().isoformat()}
    log_file = output_dir / 'generation_log.json'
    log_file.write_text(json.dumps(log, indent=2, ensure_ascii=False))
    print(f"\n{'=' * 60}")
    print(f"  Completed: {counters['completed']}/{len(work_items)}")
    print(f"  Errors: {counters['errors']}")
    print(f"  NO_SKILL_NEEDED: {counters['no_skill']}")
    print(f'  Total skills generated: {len(all_skills)}')
    print(f'  Unique skill files: {skill_counter}')
    print(f'  Elapsed: {elapsed:.0f}s ({elapsed / 60:.1f}min)')
    print(f'  Output: {output_dir}')
    print(f"{'=' * 60}")
if __name__ == '__main__':
    main()
