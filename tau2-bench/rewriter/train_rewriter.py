import argparse
import json
import os
import random
import sys
from pathlib import Path
import torch
from datasets import Dataset
from peft import LoraConfig, TaskType, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer, DataCollatorForSeq2Seq
DEFAULT_MODEL = '<MODEL_ROOT>/model/Qwen3-8B'
DEFAULT_OUTPUT_DIR = '<MODEL_ROOT>/agent/skills/rewriter-checkpoints'
DEFAULT_LORA_RANK = 64
DEFAULT_LORA_ALPHA = 128
DEFAULT_LR = 5e-05
DEFAULT_BATCH_SIZE = 1
DEFAULT_GRAD_ACCUM = 8
DEFAULT_MAX_LENGTH = 8192
DEFAULT_WARMUP_RATIO = 0.1
DEFAULT_SEED = 42
CURRICULUM_RATIOS = (0.1, 0.9)
LORA_TARGET_MODULES = ['q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj']
SYSTEM_MESSAGE = 'You are a skill documentation specialist. Your task is to rewrite skill descriptions and body content so an LLM agent can better select and use the skill.'

def load_training_data(path: str | Path) -> tuple[list[dict], list[dict]]:
    trace_based = []
    trace_free = []
    with open(path, encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            if record['type'] == 'trace_based':
                trace_based.append(record)
            else:
                trace_free.append(record)
    return (trace_based, trace_free)

def build_curriculum_dataset(trace_based: list[dict], trace_free: list[dict], trace_free_ratio: float, seed: int) -> list[dict]:
    rng = random.Random(seed)
    tf_by_name = {r['skill_name']: r for r in trace_free}
    samples = []
    for tb_record in trace_based:
        skill_name = tb_record['skill_name']
        tf_record = tf_by_name.get(skill_name)
        if tf_record is None:
            samples.append(tb_record)
            continue
        if rng.random() < trace_free_ratio:
            samples.append(tf_record)
        else:
            samples.append(tb_record)
    tb_names = {r['skill_name'] for r in trace_based}
    for tf_record in trace_free:
        if tf_record['skill_name'] not in tb_names:
            samples.append(tf_record)
    rng.shuffle(samples)
    return samples

def format_chat_messages(record: dict) -> list[dict]:
    return [{'role': 'system', 'content': SYSTEM_MESSAGE}, {'role': 'user', 'content': record['input']}, {'role': 'assistant', 'content': record['output']}]

def tokenize_chat(messages: list[dict], tokenizer, max_length: int) -> dict:
    full_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
    prompt_messages = messages[:-1]
    prompt_text = tokenizer.apply_chat_template(prompt_messages, tokenize=False, add_generation_prompt=True)
    full_tokens = tokenizer(full_text, truncation=True, max_length=max_length, padding=False, return_tensors=None)
    prompt_tokens = tokenizer(prompt_text, truncation=True, max_length=max_length, padding=False, return_tensors=None)
    prompt_len = len(prompt_tokens['input_ids'])
    input_ids = full_tokens['input_ids']
    attention_mask = full_tokens['attention_mask']
    labels = [-100] * prompt_len + input_ids[prompt_len:]
    labels = labels[:len(input_ids)]
    return {'input_ids': input_ids, 'attention_mask': attention_mask, 'labels': labels}

def prepare_hf_dataset(samples: list[dict], tokenizer, max_length: int, stage_name: str='') -> Dataset:
    tokenized_records = []
    skipped = 0
    for record in samples:
        messages = format_chat_messages(record)
        tok = tokenize_chat(messages, tokenizer, max_length)
        prompt_len = sum((1 for l in tok['labels'] if l == -100))
        output_len = len(tok['labels']) - prompt_len
        if output_len < 50:
            skipped += 1
            continue
        tokenized_records.append(tok)
    if skipped > 0:
        print(f'  [{stage_name}] Skipped {skipped} samples (too long, output truncated)')
    total_lens = [len(r['input_ids']) for r in tokenized_records]
    output_lens = [len(r['labels']) - sum((1 for l in r['labels'] if l == -100)) for r in tokenized_records]
    print(f'  [{stage_name}] Samples: {len(tokenized_records)}')
    if total_lens:
        print(f'  [{stage_name}] Total tokens: mean={sum(total_lens) / len(total_lens):.0f}, max={max(total_lens)}, min={min(total_lens)}')
        print(f'  [{stage_name}] Output tokens: mean={sum(output_lens) / len(output_lens):.0f}, max={max(output_lens)}, min={min(output_lens)}')
    return Dataset.from_list(tokenized_records)

def main():
    parser = argparse.ArgumentParser(description='Train the Rewriter model with curriculum learning (Trace-Free+).')
    parser.add_argument('--domain', type=str, default='airline')
    parser.add_argument('--training-data', type=str, default=None, help='Path to training_data.jsonl')
    parser.add_argument('--model-path', type=str, default=DEFAULT_MODEL, help=f'Base model path (default: {DEFAULT_MODEL})')
    parser.add_argument('--output-dir', type=str, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument('--mode', type=str, default='trace-free+', choices=['trace-free+', 'trace-free', 'trace-based'], help='Training mode: trace-free+ (curriculum), trace-free (only trace-free), trace-based (only trace-based)')
    parser.add_argument('--lora-rank', type=int, default=DEFAULT_LORA_RANK)
    parser.add_argument('--lora-alpha', type=int, default=DEFAULT_LORA_ALPHA)
    parser.add_argument('--lr', type=float, default=DEFAULT_LR)
    parser.add_argument('--batch-size', type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument('--grad-accum', type=int, default=DEFAULT_GRAD_ACCUM)
    parser.add_argument('--max-length', type=int, default=DEFAULT_MAX_LENGTH)
    parser.add_argument('--epochs-per-stage', type=int, default=1, help='Epochs per curriculum stage (paper uses 1, total 2)')
    parser.add_argument('--seed', type=int, default=DEFAULT_SEED)
    parser.add_argument('--curriculum-ratios', type=str, default='0.1,0.9', help='Trace-free ratios for stage 1 and 2, comma-separated')
    args = parser.parse_args()
    ratios = tuple((float(x) for x in args.curriculum_ratios.split(',')))
    assert len(ratios) == 2, 'Need exactly 2 curriculum ratios'
    if args.training_data:
        data_path = Path(args.training_data)
    else:
        data_path = Path(__file__).resolve().parent / f'data/{args.domain}/training_data.jsonl'
    output_dir = Path(args.output_dir) / f'{args.domain}_{args.mode}'
    print('=' * 60)
    print(f'  Rewriter Training — {args.mode}')
    print('=' * 60)
    print(f'  Domain:          {args.domain}')
    print(f'  Mode:            {args.mode}')
    print(f'  Model:           {args.model_path}')
    print(f'  Training data:   {data_path}')
    print(f'  Output:          {output_dir}')
    print(f'  LoRA:            rank={args.lora_rank}, alpha={args.lora_alpha}')
    print(f'  LR:              {args.lr}')
    print(f'  Eff. batch size: {args.batch_size * args.grad_accum}')
    print(f'  Max length:      {args.max_length}')
    print(f'  Curriculum:      {ratios}')
    print(f'  Epochs/stage:    {args.epochs_per_stage}')
    print(f'  Seed:            {args.seed}')
    print('=' * 60)
    print('\n[1/4] Loading training data...')
    (trace_based, trace_free) = load_training_data(data_path)
    print(f'  trace_based: {len(trace_based)}, trace_free: {len(trace_free)}')
    print('\n[2/4] Loading tokenizer and model...')
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    print('\n[3/4] Building curriculum datasets...')
    if args.mode == 'trace-free+':
        stage1_samples = build_curriculum_dataset(trace_based, trace_free, trace_free_ratio=ratios[0], seed=args.seed)
        stage2_samples = build_curriculum_dataset(trace_based, trace_free, trace_free_ratio=ratios[1], seed=args.seed + 1)
        s1_tf = sum((1 for s in stage1_samples if s['type'] == 'trace_free'))
        s1_tb = sum((1 for s in stage1_samples if s['type'] == 'trace_based'))
        s2_tf = sum((1 for s in stage2_samples if s['type'] == 'trace_free'))
        s2_tb = sum((1 for s in stage2_samples if s['type'] == 'trace_based'))
        print(f'  Stage 1: {len(stage1_samples)} samples (trace_based={s1_tb}, trace_free={s1_tf})')
        print(f'  Stage 2: {len(stage2_samples)} samples (trace_based={s2_tb}, trace_free={s2_tf})')
    elif args.mode == 'trace-free':
        stage1_samples = trace_free[:]
        stage2_samples = None
        print(f'  Trace-free only: {len(stage1_samples)} samples')
    elif args.mode == 'trace-based':
        stage1_samples = trace_based[:]
        stage2_samples = None
        print(f'  Trace-based only: {len(stage1_samples)} samples')
    print('\n  Tokenizing Stage 1...')
    stage1_dataset = prepare_hf_dataset(stage1_samples, tokenizer, args.max_length, 'Stage 1')
    if stage2_samples:
        print('  Tokenizing Stage 2...')
        stage2_dataset = prepare_hf_dataset(stage2_samples, tokenizer, args.max_length, 'Stage 2')
    print('\n[4/4] Loading model with LoRA...')
    model = AutoModelForCausalLM.from_pretrained(args.model_path, dtype=torch.bfloat16, trust_remote_code=True, attn_implementation='sdpa')
    model.config.use_cache = False
    lora_config = LoraConfig(task_type=TaskType.CAUSAL_LM, r=args.lora_rank, lora_alpha=args.lora_alpha, lora_dropout=0.05, target_modules=LORA_TARGET_MODULES, bias='none')
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, padding=True, pad_to_multiple_of=8)

    def make_training_args(stage_output_dir: str, num_epochs: int, dataset_size: int) -> TrainingArguments:
        return TrainingArguments(output_dir=stage_output_dir, num_train_epochs=num_epochs, per_device_train_batch_size=args.batch_size, gradient_accumulation_steps=args.grad_accum, learning_rate=args.lr, lr_scheduler_type='cosine', warmup_ratio=DEFAULT_WARMUP_RATIO, bf16=True, logging_steps=1, save_strategy='epoch', save_total_limit=2, seed=args.seed, gradient_checkpointing=True, gradient_checkpointing_kwargs={'use_reentrant': False}, report_to='none', remove_unused_columns=False, dataloader_pin_memory=True)
    if args.mode == 'trace-free+':
        print(f"\n{'=' * 60}")
        print(f'  STAGE 1: {ratios[0] * 100:.0f}% trace-free, {(1 - ratios[0]) * 100:.0f}% trace-based')
        print(f"{'=' * 60}")
        stage1_args = make_training_args(str(output_dir / 'stage1'), num_epochs=args.epochs_per_stage, dataset_size=len(stage1_dataset))
        trainer = Trainer(model=model, args=stage1_args, train_dataset=stage1_dataset, data_collator=data_collator)
        trainer.train()
        trainer.save_model(str(output_dir / 'stage1' / 'checkpoint-final'))
        print(f"\n{'=' * 60}")
        print(f'  STAGE 2: {ratios[1] * 100:.0f}% trace-free, {(1 - ratios[1]) * 100:.0f}% trace-based')
        print(f"{'=' * 60}")
        stage2_args = make_training_args(str(output_dir / 'stage2'), num_epochs=args.epochs_per_stage, dataset_size=len(stage2_dataset))
        trainer = Trainer(model=model, args=stage2_args, train_dataset=stage2_dataset, data_collator=data_collator)
        trainer.train()
        trainer.save_model(str(output_dir / 'final'))
    else:
        total_epochs = args.epochs_per_stage * 2
        print(f'\n  Training {args.mode} for {total_epochs} epochs...')
        training_args = make_training_args(str(output_dir), num_epochs=total_epochs, dataset_size=len(stage1_dataset))
        trainer = Trainer(model=model, args=training_args, train_dataset=stage1_dataset, data_collator=data_collator)
        trainer.train()
        trainer.save_model(str(output_dir / 'final'))
    config_record = {'domain': args.domain, 'mode': args.mode, 'model_path': args.model_path, 'lora_rank': args.lora_rank, 'lora_alpha': args.lora_alpha, 'lr': args.lr, 'effective_batch_size': args.batch_size * args.grad_accum, 'max_length': args.max_length, 'curriculum_ratios': ratios, 'epochs_per_stage': args.epochs_per_stage, 'num_trace_based': len(trace_based), 'num_trace_free': len(trace_free), 'seed': args.seed}
    config_path = output_dir / 'training_config.json'
    config_path.parent.mkdir(parents=True, exist_ok=True)
    with open(config_path, 'w') as f:
        json.dump(config_record, f, indent=2)
    print(f"\n{'=' * 60}")
    print(f'  Training complete!')
    print(f"  Final checkpoint:  {output_dir / 'final'}")
    print(f'  Config saved to:   {config_path}')
    print(f"{'=' * 60}")
if __name__ == '__main__':
    main()
