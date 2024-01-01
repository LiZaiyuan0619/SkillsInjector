from pathlib import Path
REPO_ROOT = Path('<REPO_ROOT>')
STATE_DIR = REPO_ROOT / 'state'
CONFIGS_DIR = REPO_ROOT / 'configs'
JOBS_DIR = REPO_ROOT / 'jobs'
CACHE_DIR = REPO_ROOT / 'cache'
DOCS_DIR = REPO_ROOT / 'docs'
AGENTGYM_ROOT = Path('<AGENTGYM_ROOT>')
ALFWORLD_MAPPINGS_TEST = AGENTGYM_ROOT / 'agentenv-alfworld/configs/mappings_test.json'
ALFWORLD_MAPPINGS_TRAIN = AGENTGYM_ROOT / 'agentenv-alfworld/configs/mappings_train.json'
AGENTTRAJ_L_ALFWORLD = Path('<DATA_ROOT>/agent/AgentGym_reproduce/data/AgentTraj-L/alfworld_train.json')
ITEM_ID_MIN = 2420
ITEM_ID_MAX = 2619
TASK_TYPES = ['pick_and_place_simple', 'pick_two_obj_and_place', 'pick_clean_then_place_in_recep', 'pick_heat_then_place_in_recep', 'pick_cool_then_place_in_recep', 'look_at_obj_in_light']
ENCODER_0_6B = Path('<MODEL_ROOT>/model/Qwen3-Embedding-0.6B')
ENCODER_8B = Path('<MODEL_ROOT>/model/Qwen3-Embedding-8B')
QWEN3_8B_PATH = Path('<MODEL_ROOT>/model/Qwen3-8B')
