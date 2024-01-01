from __future__ import annotations
import hashlib
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable
import numpy as np
from .paths import EMBEDDINGS_DIR, QWEN3_EMBEDDING_PATH
logger = logging.getLogger(__name__)

def _hash16(text: str) -> str:
    return hashlib.sha256(text.encode('utf-8')).hexdigest()[:16]

@dataclass
class EmbedderConfig:
    model_path: Path = QWEN3_EMBEDDING_PATH
    max_length: int = 8192
    batch_size: int = 8
    cache_dir: Path = EMBEDDINGS_DIR
    pooling: str = 'last_token'

    def fingerprint(self) -> str:
        sig = f'{self.model_path}|{self.pooling}|{self.max_length}|L2|fp32-saved'
        return hashlib.sha256(sig.encode('utf-8')).hexdigest()[:8]

class QwenEmbedder:

    def __init__(self, cfg: EmbedderConfig | None=None):
        self.cfg = cfg or EmbedderConfig()
        self.cfg.cache_dir.mkdir(parents=True, exist_ok=True)
        self._tokenizer = None
        self._model = None
        self._dim: int | None = None
        self._device: str = ''
        self._cfg_fp = self.cfg.fingerprint()

    @property
    def dim(self) -> int:
        if self._dim is None:
            self._lazy_load()
        assert self._dim is not None
        return self._dim

    def _lazy_load(self) -> None:
        if self._model is not None:
            return
        import torch
        from transformers import AutoModel, AutoTokenizer
        self._device = 'cuda' if torch.cuda.is_available() else 'cpu'
        dtype = torch.float16 if self._device == 'cuda' else torch.float32
        logger.info('loading encoder: %s (device=%s, dtype=%s, cfg_fp=%s)', self.cfg.model_path, self._device, dtype, self._cfg_fp)
        self._tokenizer = AutoTokenizer.from_pretrained(str(self.cfg.model_path), trust_remote_code=True)
        self._model = AutoModel.from_pretrained(str(self.cfg.model_path), trust_remote_code=True, dtype=dtype).to(self._device)
        self._model.eval()
        self._dim = int(self._model.config.hidden_size)
        logger.info('encoder ready (dim=%d)', self._dim)

    def _cache_path_for(self, text: str) -> Path:
        return self.cfg.cache_dir / f'{self._cfg_fp}_{_hash16(text)}.npy'

    def embed_texts(self, texts: list[str], use_cache: bool=True) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        out = np.zeros((len(texts), 0), dtype=np.float32)
        cached: list[np.ndarray | None] = []
        if use_cache:
            for t in texts:
                p = self._cache_path_for(t)
                if p.is_file():
                    cached.append(np.load(p))
                else:
                    cached.append(None)
        else:
            cached = [None] * len(texts)
        miss_idx = [i for (i, c) in enumerate(cached) if c is None]
        if miss_idx:
            self._lazy_load()
            import torch
            assert self._tokenizer is not None and self._model is not None
            batch = max(self.cfg.batch_size, 32 if getattr(self, '_device', 'cpu') == 'cuda' else self.cfg.batch_size)
            for start in range(0, len(miss_idx), batch):
                batch_idx = miss_idx[start:start + batch]
                batch_texts = [texts[i] for i in batch_idx]
                inputs = self._tokenizer(batch_texts, padding=True, truncation=True, max_length=self.cfg.max_length, return_tensors='pt')
                inputs = {k: v.to(self._device) for (k, v) in inputs.items()}
                with torch.no_grad():
                    h = self._model(**inputs).last_hidden_state
                attn = inputs['attention_mask']
                seq_lens = attn.sum(dim=1) - 1
                last = h[torch.arange(h.size(0)), seq_lens]
                last = torch.nn.functional.normalize(last, p=2, dim=-1)
                arr = last.float().cpu().numpy().astype(np.float32)
                for (j, i) in enumerate(batch_idx):
                    cached[i] = arr[j]
                    if use_cache:
                        np.save(self._cache_path_for(texts[i]), arr[j])
        out = np.stack([c for c in cached], axis=0)
        return out
