import argparse
import shutil
import time
from pathlib import Path

from neurograph.config import REPO_ROOT, git_hash, load_config

PACKAGES = ["agent", "app", "graph", "ingest", "literature"]  # what the assistant imports (no EEG code)
CONFIGS = ["app", "rag", "policy_graph", "ingest_urls", "benchmark", "paths"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(REPO_ROOT / "dist" / "space"))
    args = ap.parse_args()
    t0 = time.perf_counter()
    der = Path(load_config("paths")["paths"]["derivatives_dir"])
    out = Path(args.out)
    if out.exists():
        shutil.rmtree(out)
    shutil.copytree(REPO_ROOT / "space", out)
    for pkg in PACKAGES:
        shutil.copytree(REPO_ROOT / "src" / "neurograph" / pkg, out / "src" / "neurograph" / pkg,
                        ignore=shutil.ignore_patterns("__pycache__"))
    for f in ["__init__.py", "config.py"]:
        shutil.copy(REPO_ROOT / "src" / "neurograph" / f, out / "src" / "neurograph" / f)
    (out / "configs").mkdir()
    for c in CONFIGS:
        shutil.copy(REPO_ROOT / "configs" / f"{c}.yaml", out / "configs" / f"{c}.yaml")
    (out / "app_data").mkdir()
    shutil.copy(der / "institutional" / "chunks.jsonl", out / "app_data" / "chunks.jsonl")  # PII-redacted at intake
    shutil.copy(der / "app" / "graph_snapshot.json", out / "app_data" / "graph_snapshot.json")
    (out / "app_data" / "BUILD.txt").write_text(f"built from commit {git_hash()}\n")

    files = [p for p in out.rglob("*") if p.is_file()]
    print(f"== Hugging Face Space folder | commit {git_hash()} ==")
    print(f"files {len(files)} | size {sum(p.stat().st_size for p in files) / 1e6:.1f} MB | time {time.perf_counter() - t0:.1f} s")
    print(f"saved: {out}\nupload: huggingface-cli upload <user>/<space> {out} . --repo-type space")


if __name__ == "__main__":
    main()
