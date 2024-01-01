from __future__ import annotations
import logging
import os
import random
import time
from dataclasses import dataclass
import httpx
logger = logging.getLogger(__name__)

@dataclass
class LLMConfig:
    base_url: str
    api_key: str
    model: str
    backend_label: str
    temperature: float = 0.3
    max_tokens: int = 4096
    request_timeout: float = 600.0

    @classmethod
    def from_env(cls) -> 'LLMConfig':
        base_url = os.environ.get('SK_RUNNER_BASE_URL', '').rstrip('/')
        api_key = os.environ.get('SK_RUNNER_API_KEY', '')
        model = os.environ.get('SK_RUNNER_MODEL', '')
        label = os.environ.get('SK_RUNNER_BACKEND_LABEL', 'unknown')
        if not (base_url and model):
            raise SystemExit("Missing env: SK_RUNNER_BASE_URL and SK_RUNNER_MODEL are required. Optionally set SK_RUNNER_API_KEY (default empty), SK_RUNNER_BACKEND_LABEL (default 'unknown').")
        return cls(base_url=base_url, api_key=api_key, model=model, backend_label=label)

class LLMClient:
    _RETRY_BACKOFF_S = (5, 15, 30, 60, 120, 180)

    def __init__(self, cfg: LLMConfig):
        self._cfg = cfg
        self._client = httpx.Client(timeout=cfg.request_timeout, trust_env=False)

    def chat(self, messages: list[dict]) -> tuple[str, dict]:
        url = f'{self._cfg.base_url}/chat/completions'
        headers = {'Content-Type': 'application/json'}
        if self._cfg.api_key:
            headers['Authorization'] = f'Bearer {self._cfg.api_key}'
        body = {'model': self._cfg.model, 'messages': messages, 'temperature': self._cfg.temperature, 'max_tokens': self._cfg.max_tokens}
        last_err: Exception | None = None
        max_attempts = 1 + len(self._RETRY_BACKOFF_S)
        for attempt in range(max_attempts):
            try:
                t0 = time.time()
                r = self._client.post(url, headers=headers, json=body)
                r.raise_for_status()
                data = r.json()
                msg = data['choices'][0]['message']['content']
                usage = data.get('usage', {})
                usage['latency_s'] = round(time.time() - t0, 2)
                return (msg, usage)
            except (httpx.HTTPError, KeyError, IndexError) as e:
                last_err = e
                logger.warning('LLM call attempt %d/%d failed: %s', attempt + 1, max_attempts, e)
                if attempt < len(self._RETRY_BACKOFF_S):
                    base = self._RETRY_BACKOFF_S[attempt]
                    time.sleep(base * random.uniform(0.75, 1.25))
        raise RuntimeError(f'LLM call failed after {max_attempts} attempts: {last_err}')

    def close(self) -> None:
        self._client.close()
