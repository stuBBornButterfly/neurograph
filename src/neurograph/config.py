import subprocess
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]


def load_config(name: str = "paths") -> dict:
    with open(REPO_ROOT / "configs" / f"{name}.yaml") as f:
        return yaml.safe_load(f)


def git_hash() -> str:
    out = subprocess.run(["git", "-C", str(REPO_ROOT), "rev-parse", "--short", "HEAD"],
                         capture_output=True, text=True)
    return out.stdout.strip() or "unknown"