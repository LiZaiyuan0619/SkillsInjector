from __future__ import annotations
import logging
import os
import shutil
import subprocess
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
_BUILD_ENV = {**os.environ, 'DOCKER_BUILDKIT': '1'}
logger = logging.getLogger(__name__)
STDOUT_CAP = 4000
STDERR_CAP = 2000
DEFAULT_CMD_TIMEOUT = 120

@dataclass
class ShellResult:
    exit_code: int
    stdout: str
    stderr: str
    truncated_stdout: bool
    truncated_stderr: bool
    timed_out: bool
    duration_s: float

class TaskContainer:

    def __init__(self, task_dir: Path, container_name: str, image_tag: str):
        self.task_dir = task_dir.resolve()
        self.container_name = container_name
        self.image_tag = image_tag
        self._started = False

    @classmethod
    def for_trial(cls, task_dir: Path, task_id: str) -> 'TaskContainer':
        suffix = uuid.uuid4().hex[:8]
        return cls(task_dir=task_dir, container_name=f'skbench-{task_id}-{suffix}', image_tag=f'skbench-{task_id}:latest')

    def build_image(self) -> None:
        check = subprocess.run(['docker', 'image', 'inspect', self.image_tag], capture_output=True, text=True, timeout=10)
        if check.returncode == 0:
            return
        env_dir = self.task_dir / 'environment'
        if not (env_dir / 'Dockerfile').is_file():
            raise FileNotFoundError(f'No Dockerfile at {env_dir}/Dockerfile')
        cmd = ['docker', 'build', '-t', self.image_tag, str(env_dir)]
        logger.info('docker build %s (on-demand fallback)', self.image_tag)
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=1800, env=_BUILD_ENV)
        if r.returncode != 0:
            raise RuntimeError(f'docker build failed (rc={r.returncode}): {r.stderr[-2000:]}')

    def start(self) -> None:
        cmd = ['docker', 'run', '-d', '--rm', '--name', self.container_name, '--memory', '4g', '--cpus', '2', self.image_tag, 'tail', '-f', '/dev/null']
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
        if r.returncode != 0:
            raise RuntimeError(f'docker run failed: {r.stderr[-1000:]}')
        self._started = True

    def exec_shell(self, cmd: str, cwd: str | None=None, timeout: int=DEFAULT_CMD_TIMEOUT) -> ShellResult:
        argv = ['docker', 'exec']
        if cwd:
            argv.extend(['-w', cwd])
        argv.extend([self.container_name, 'bash', '-c', cmd])
        t0 = time.time()
        timed_out = False
        try:
            r = subprocess.run(argv, capture_output=True, text=True, encoding='utf-8', errors='replace', timeout=timeout)
            exit_code = r.returncode
            stdout = r.stdout or ''
            stderr = r.stderr or ''
        except subprocess.TimeoutExpired as e:
            timed_out = True
            exit_code = -1
            stdout = (e.stdout or b'').decode('utf-8', errors='replace') if isinstance(e.stdout, bytes) else e.stdout or ''
            stderr = (e.stderr or b'').decode('utf-8', errors='replace') if isinstance(e.stderr, bytes) else e.stderr or ''
            stderr += f'\n[command timed out after {timeout}s]'
        duration = round(time.time() - t0, 2)
        orig_stdout_len = len(stdout)
        orig_stderr_len = len(stderr)
        truncated_stdout = orig_stdout_len > STDOUT_CAP
        truncated_stderr = orig_stderr_len > STDERR_CAP
        if truncated_stdout:
            stdout = stdout[:STDOUT_CAP] + f'\n...[truncated, total {orig_stdout_len} chars]'
        if truncated_stderr:
            stderr = stderr[:STDERR_CAP] + f'\n...[truncated, total {orig_stderr_len} chars]'
        return ShellResult(exit_code=exit_code, stdout=stdout, stderr=stderr, truncated_stdout=truncated_stdout, truncated_stderr=truncated_stderr, timed_out=timed_out, duration_s=duration)

    def inject_skills(self, skill_paths: list[Path], target_dir: str='/skills') -> None:
        if not skill_paths:
            return
        self.exec_shell(f'mkdir -p {target_dir}')
        for sp in skill_paths:
            r = subprocess.run(['docker', 'cp', str(sp), f'{self.container_name}:{target_dir}/'], capture_output=True, text=True, timeout=30)
            if r.returncode != 0:
                logger.warning('docker cp skill %s failed: %s', sp, r.stderr)

    def run_verifier(self, output_dir: Path) -> tuple[int, str]:
        tests_dir = self.task_dir / 'tests'
        if not tests_dir.is_dir():
            raise FileNotFoundError(f'No tests dir at {tests_dir}')
        r = subprocess.run(['docker', 'cp', str(tests_dir) + '/.', f'{self.container_name}:/tests/'], capture_output=True, text=True, timeout=60)
        if r.returncode != 0:
            raise RuntimeError(f'docker cp tests failed: {r.stderr[-500:]}')
        self.exec_shell('mkdir -p /logs/verifier')
        verifier_res = self.exec_shell('bash /tests/test.sh', cwd='/root', timeout=900)
        reward_res = self.exec_shell('cat /logs/verifier/reward.txt 2>/dev/null || echo MISSING')
        reward_text = reward_res.stdout.strip()
        if reward_text == '1':
            reward = 1
        elif reward_text == '0':
            reward = 0
        else:
            reward = 0
            logger.warning('reward.txt missing or unparseable: %r', reward_text)
        output_dir.mkdir(parents=True, exist_ok=True)
        subprocess.run(['docker', 'cp', f'{self.container_name}:/logs/verifier', str(output_dir / 'verifier')], capture_output=True, text=True, timeout=60)
        log_excerpt = verifier_res.stdout[-3000:] + '\n---STDERR---\n' + verifier_res.stderr[-1000:]
        return (reward, log_excerpt)

    def cleanup(self) -> None:
        if not self._started:
            return
        subprocess.run(['docker', 'rm', '-f', self.container_name], capture_output=True, text=True, timeout=30)
        self._started = False

    def __enter__(self) -> 'TaskContainer':
        return self

    def __exit__(self, *_exc) -> None:
        self.cleanup()
