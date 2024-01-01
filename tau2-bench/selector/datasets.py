import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any
import torch

@dataclass(frozen=True)
class PairRecord:
    task_id: str
    skill_name: str
    task_text: str
    skill_text: str
    k_base: int
    n_base: int
    k_skill: int
    n_skill: int
    p_base: float
    p_skill: float
    delta: float

@dataclass(frozen=True)
class PairwiseRecord:
    task_id: str
    positive_skill_name: str
    negative_skill_name: str
    positive_delta: float
    negative_delta: float

def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding='utf-8'))
    if not isinstance(payload, dict):
        raise ValueError(f'Expected JSON object at {path}')
    return payload

def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding='utf-8').splitlines():
        rows.append(json.loads(line))
    return rows

def load_export_artifacts(data_dir: str | Path) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    data_dir = Path(data_dir)
    return (_read_json(data_dir / 'manifest.json'), _read_jsonl(data_dir / 'tasks.jsonl'), _read_jsonl(data_dir / 'skills.jsonl'), _read_jsonl(data_dir / 'labels.jsonl'))

def load_pair_records(data_dir: str | Path) -> list[PairRecord]:
    (_manifest, task_rows, skill_rows, label_rows) = load_export_artifacts(data_dir)
    task_meta = {str(row['task_id']): {'task_text': str(row['task_text']), 'k_base': int(row['k_base']), 'n_base': int(row['n_base'])} for row in task_rows}
    skill_meta = {str(row['skill_name']): {'skill_text': str(row['skill_text'])} for row in skill_rows}
    records: list[PairRecord] = []
    for row in label_rows:
        task_id = str(row['task_id'])
        skill_name = str(row['skill_name'])
        task = task_meta[task_id]
        skill = skill_meta[skill_name]
        k_base = int(task['k_base'])
        n_base = int(task['n_base'])
        k_skill = int(row['k_skill'])
        n_skill = int(row['n_skill'])
        p_base = float(k_base) / float(n_base)
        p_skill = float(k_skill) / float(n_skill)
        records.append(PairRecord(task_id=task_id, skill_name=skill_name, task_text=str(task['task_text']), skill_text=str(skill['skill_text']), k_base=k_base, n_base=n_base, k_skill=k_skill, n_skill=n_skill, p_base=p_base, p_skill=p_skill, delta=p_skill - p_base))
    return records

def load_pair_records_multi(data_dirs: list[str | Path]) -> tuple[list[PairRecord], list[str]]:
    records: list[PairRecord] = []
    source_domains: list[str] = []
    for data_dir in data_dirs:
        (manifest, task_rows, skill_rows, label_rows) = load_export_artifacts(data_dir)
        domain = str(manifest['domain'])
        source_domains.append(domain)
        task_meta = {f"{domain}::{row['task_id']}": {'task_text': str(row['task_text']), 'k_base': int(row['k_base']), 'n_base': int(row['n_base'])} for row in task_rows}
        skill_meta = {f"{domain}/{row['skill_name']}": {'skill_text': str(row['skill_text'])} for row in skill_rows}
        for row in label_rows:
            task_id = f"{domain}::{row['task_id']}"
            skill_name = f"{domain}/{row['skill_name']}"
            task = task_meta[task_id]
            skill = skill_meta[skill_name]
            k_base = int(task['k_base'])
            n_base = int(task['n_base'])
            k_skill = int(row['k_skill'])
            n_skill = int(row['n_skill'])
            p_base = float(k_base) / float(n_base)
            p_skill = float(k_skill) / float(n_skill)
            records.append(PairRecord(task_id=task_id, skill_name=skill_name, task_text=str(task['task_text']), skill_text=str(skill['skill_text']), k_base=k_base, n_base=n_base, k_skill=k_skill, n_skill=n_skill, p_base=p_base, p_skill=p_skill, delta=p_skill - p_base))
    return (records, sorted(set(source_domains)))

def build_pairwise_records(records: list[PairRecord]) -> list[PairwiseRecord]:
    grouped: dict[str, list[PairRecord]] = defaultdict(list)
    for record in records:
        grouped[record.task_id].append(record)
    pairwise: list[PairwiseRecord] = []
    for (task_id, task_records) in grouped.items():
        ordered = sorted(task_records, key=lambda item: (-item.delta, item.skill_name))
        for positive in ordered:
            for negative in ordered:
                if positive.skill_name == negative.skill_name:
                    continue
                if positive.delta <= negative.delta:
                    continue
                pairwise.append(PairwiseRecord(task_id=task_id, positive_skill_name=positive.skill_name, negative_skill_name=negative.skill_name, positive_delta=positive.delta, negative_delta=negative.delta))
    return pairwise

def build_pair_batch(records: list[PairRecord], task_embeddings: dict[str, torch.Tensor], skill_embeddings: dict[str, torch.Tensor]) -> dict[str, Any]:
    return {'task_ids': [record.task_id for record in records], 'skill_names': [record.skill_name for record in records], 'task_embeddings': torch.stack([task_embeddings[record.task_id] for record in records]), 'skill_embeddings': torch.stack([skill_embeddings[record.skill_name] for record in records]), 'k_skill': torch.tensor([record.k_skill for record in records], dtype=torch.float32), 'n_skill': torch.tensor([record.n_skill for record in records], dtype=torch.float32), 'p_skill': torch.tensor([record.p_skill for record in records], dtype=torch.float32), 'delta': torch.tensor([record.delta for record in records], dtype=torch.float32)}
