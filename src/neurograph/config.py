import os
import subprocess
from pathlib import Path
from urllib.parse import urlparse

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]


def dotenv() -> dict:
    # KEY=value lines of the repo's private .env (gitignored), if there is one
    f = REPO_ROOT / ".env"
    if not f.exists():
        return {}
    pairs = (l.split("=", 1) for l in f.read_text(encoding="utf-8").splitlines() if "=" in l and not l.startswith("#"))
    return {k.strip(): v.strip().strip("'\"") for k, v in pairs}


def site() -> dict:
    # the university's name and website come from private settings (environment or .env), never from the repo:
    # SITE_NAME as the prompts say it, SITE_SHORT (also the URL slug, lower-cased), SITE_URL (the public website)
    env = {**dotenv(), **os.environ}
    url = env.get("SITE_URL", "https://www.university.example").rstrip("/")
    short = env.get("SITE_SHORT", "university")
    host = urlparse(url).hostname or ""
    return {"name": env.get("SITE_NAME", "university"), "short": short, "slug": short.lower(), "url": url,
            "domain": host[4:] if host.startswith("www.") else host}


def expand_site(node, s: dict):
    # ${SITE_URL}, ${SITE_DOMAIN}, ${SITE_SLUG}, ${SITE_SHORT}, ${SITE_NAME} in config strings
    if isinstance(node, dict):
        return {k: expand_site(v, s) for k, v in node.items()}
    if isinstance(node, list):
        return [expand_site(v, s) for v in node]
    if isinstance(node, str) and "${SITE_" in node:
        for k, v in s.items():
            node = node.replace("${SITE_" + k.upper() + "}", v)
    return node


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
        cfg = expand_site(local_paths(yaml.safe_load(f)), site())
    if name == "paths":  # NEUROGRAPH_BIDS_DIR / NEUROGRAPH_DERIVATIVES_DIR replace the Kaggle paths
        for key in cfg["paths"]:
            cfg["paths"][key] = os.environ.get(f"NEUROGRAPH_{key.upper()}", cfg["paths"][key])
    return cfg


def git_hash() -> str:
    out = subprocess.run(["git", "-C", str(REPO_ROOT), "rev-parse", "--short", "HEAD"],
                         capture_output=True, text=True)
    return out.stdout.strip() or "unknown"
