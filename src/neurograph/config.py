import os
import subprocess
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]


def local_paths(node):
    # local runs (not Kaggle): NEUROGRAPH_INPUT_ROOT replaces /kaggle/input in every input glob
    root = os.environ.get("NEUROGRAPH_INPUT_ROOT")
    if isinstance(node, dict):
        return {k: local_paths(v) for k, v in node.items()}
    if isinstance(node, list):
        return [local_paths(v) for v in node]
    if root and isinstance(node, str) and node.startswith("/kaggle/input"):
        return root + node[len("/kaggle/input"):]
    return node


def load_config(name: str = "paths") -> dict:
    with open(REPO_ROOT / "configs" / f"{name}.yaml") as f:
        cfg = local_paths(yaml.safe_load(f))
    if name == "paths":  # NEUROGRAPH_BIDS_DIR / NEUROGRAPH_DERIVATIVES_DIR replace the Kaggle paths
        for key in cfg["paths"]:
            cfg["paths"][key] = os.environ.get(f"NEUROGRAPH_{key.upper()}", cfg["paths"][key])
    return cfg


def git_hash() -> str:
    out = subprocess.run(["git", "-C", str(REPO_ROOT), "rev-parse", "--short", "HEAD"],
                         capture_output=True, text=True)
    return out.stdout.strip() or "unknown"
