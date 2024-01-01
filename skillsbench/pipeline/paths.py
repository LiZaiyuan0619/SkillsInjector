from __future__ import annotations
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
SKILLSBENCH_ROOT = Path('<SKILLSBENCH_ROOT>')
TASKS_DIR = SKILLSBENCH_ROOT / 'tasks'
STATE_DIR = PROJECT_ROOT / 'state'
MANIFEST_PATH = STATE_DIR / 'manifest.json'
SKILL_POOL_PATH = STATE_DIR / 'skill_pool.json'
METHODS_PATH = STATE_DIR / 'methods.json'
SELECTIONS_DIR = STATE_DIR / 'selections'
EMBEDDINGS_DIR = STATE_DIR / 'embeddings'
JOBS_DIR = PROJECT_ROOT / 'jobs'
RESULTS_DIR = PROJECT_ROOT / 'results'
QWEN3_EMBEDDING_PATH = Path('<MODEL_ROOT>/model/Qwen3-Embedding-0.6B')
EXCLUDED_TASKS = frozenset({'gh-repo-analytics', 'mhc-layer-impl', 'pedestrian-traffic-counting', 'pg-essay-to-audiobook', 'scheduling-email-assistant', 'speaker-diarization-subtitles', 'trend-anomaly-causal-inference', 'video-filler-word-remover', 'video-tutorial-indexer', 'azure-bgp-oscillation-route-leak', 'latex-formula-extraction', 'earthquake-phase-association', 'fix-build-agentops', 'fix-build-google-auto', 'fix-druid-loophole-cve', 'glm-lake-mendota', 'python-scala-translation', 'suricata-custom-exfil', 'taxonomy-tree-merge'})
