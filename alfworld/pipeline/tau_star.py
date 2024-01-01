from __future__ import annotations
import json
from pathlib import Path
from .paths import STATE_DIR

def load_tau_star(path: Path | None=None) -> float:
    p = path or STATE_DIR / 'tau_star.json'
    if not p.exists():
        raise FileNotFoundError(f'{p} missing. Generate it first (Task 1.0):\n  populate domain_values dict with tau2 calibrated thresholds\n  then dump JSON with median field.')
    data = json.loads(p.read_text())
    if 'median' not in data:
        raise ValueError(f"{p} has no 'median' field; got keys={list(data)}")
    domain_values = data.get('domain_values', {})
    if not isinstance(domain_values, dict) or len(domain_values) != 3:
        raise ValueError(f'R10: {p} domain_values must have exactly 3 tau2 domains (airline/retail/telecom); got {len(domain_values)} entries: {list(domain_values)}')
    median = float(data['median'])
    if not 0.0 < median < 1.0:
        raise ValueError(f'R10: tau_star.median={median} out of (0, 1)')
    return median

def write_tau_star(domain_values: dict[str, float], notes: str='', path: Path | None=None) -> None:
    p = path or STATE_DIR / 'tau_star.json'
    import statistics
    from datetime import datetime
    median = statistics.median(domain_values.values())
    payload = {'source': 'tau2 calibration', 'domain_values': domain_values, 'median': median, 'created_at': datetime.utcnow().isoformat(), 'notes': notes or 'Cosine variant threshold transfer (spec §1.1)'}
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(payload, indent=2))
    print(f'wrote {p}: median={median} from {domain_values}')
