from __future__ import annotations
import logging
import os
import time
import httpx
log = logging.getLogger(__name__)
DEFAULT_BASE = os.environ.get('AGENTGYM_ENVSERVER', 'http://localhost:36001')

class EnvServerError(Exception):
    pass

class EnvServerClient:

    def __init__(self, base_url: str | None=None, timeout: float=60.0):
        self.base = base_url or DEFAULT_BASE
        self._client = httpx.Client(base_url=self.base, timeout=timeout, trust_env=False)

    def close(self):
        self._client.close()

    def __enter__(self):
        return self

    def __exit__(self, *a):
        self.close()

    def create(self) -> int:
        r = self._post('/create', {})
        if 'error' in r:
            raise EnvServerError(r['error'])
        return r['id']

    def reset(self, env_id: int, game: int, world_type: str='Text') -> dict:
        r = self._post('/reset', {'id': env_id, 'game': game, 'world_type': world_type})
        if 'error' in r:
            raise EnvServerError(r['error'])
        return r

    def step(self, env_id: int, action: str) -> dict:
        r = self._post('/step', {'id': env_id, 'action': action})
        if 'error' in r:
            raise EnvServerError(r['error'])
        return r

    def _post(self, path: str, body: dict, retry: int=1) -> dict:
        last_err = None
        for attempt in range(retry + 1):
            try:
                r = self._client.post(path, json=body)
                r.raise_for_status()
                return r.json()
            except Exception as e:
                last_err = e
                log.warning(f'  EnvServer {path} attempt {attempt + 1} failed: {e}')
                time.sleep(0.5)
        raise EnvServerError(f'{path} failed after {retry + 1} attempts: {last_err}')
