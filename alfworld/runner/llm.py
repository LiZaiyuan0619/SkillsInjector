from __future__ import annotations
import logging
import os
import time
from dataclasses import dataclass
import httpx
log = logging.getLogger(__name__)

@dataclass
class LLMConfig:
    base_url: str
    api_key: str
    model: str
    backend_label: str
    default_temperature: float = 0.0
    default_top_p: float = 1.0
    default_max_tokens: int = 2048

    @classmethod
    def from_env(cls, prefix: str='SK_RUNNER') -> 'LLMConfig':
        base_url = os.environ[f'{prefix}_BASE_URL']
        api_key = os.environ.get(f'{prefix}_API_KEY', '')
        model = os.environ[f'{prefix}_MODEL']
        label = os.environ.get(f'{prefix}_BACKEND_LABEL', 'sglang')
        return cls(base_url=base_url, api_key=api_key, model=model, backend_label=label)

class LLMClient:

    def __init__(self, cfg: LLMConfig, timeout: float=60.0, retry: int=1):
        self.cfg = cfg
        self.retry = retry
        self._client = httpx.Client(base_url=cfg.base_url, timeout=timeout, trust_env=False, headers={'Authorization': f'Bearer {cfg.api_key}'} if cfg.api_key else {})

    def close(self):
        self._client.close()

    def __enter__(self):
        return self

    def __exit__(self, *a):
        self.close()

    def chat(self, messages: list[dict], *, temperature: float | None=None, top_p: float | None=None, max_tokens: int | None=None) -> tuple[str, dict]:
        body = {'model': self.cfg.model, 'messages': messages, 'temperature': temperature if temperature is not None else self.cfg.default_temperature, 'top_p': top_p if top_p is not None else self.cfg.default_top_p, 'max_tokens': max_tokens if max_tokens is not None else self.cfg.default_max_tokens, 'chat_template_kwargs': {'enable_thinking': False}}
        last_err = None
        for attempt in range(self.retry + 1):
            t0 = time.time()
            try:
                r = self._client.post('/chat/completions', json=body)
                r.raise_for_status()
                d = r.json()
                text = d['choices'][0]['message']['content']
                usage = d.get('usage', {}) | {'latency_s': time.time() - t0, 'backend_label': self.cfg.backend_label}
                return (text, usage)
            except Exception as e:
                last_err = e
                log.warning(f'  LLM attempt {attempt + 1} failed: {e}')
                time.sleep(0.5)
        raise RuntimeError(f'LLM failed after {self.retry + 1} attempts: {last_err}')
