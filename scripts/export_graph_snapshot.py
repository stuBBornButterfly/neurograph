import argparse
import json
import time
from pathlib import Path

from neurograph.agent.snapshot import export
from neurograph.config import git_hash, load_config
from neurograph.graph.client import get_driver


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=None, help="default: <derivatives>/app/graph_snapshot.json")
    args = ap.parse_args()
    t0 = time.perf_counter()
    out = Path(args.out) if args.out else Path(load_config("paths")["paths"]["derivatives_dir"]) / "app" / "graph_snapshot.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    driver = get_driver()
    snap = export(driver, git_hash())
    driver.close()
    out.write_text(json.dumps(snap, indent=1, ensure_ascii=False, default=str), encoding="utf-8")

    print(f"== graph snapshot for the demo app | commit {git_hash()} ==")
    print(f"services {len(snap['services'])} | rules {sum(map(len, snap['rules'].values()))} on {len(snap['rules'])} "
          f"sections | study queries {len(snap['study'])} | verified claims {len(snap['claims'])} | "
          f"literature comparisons {len(snap['comparisons'])}")
    print(f"size {out.stat().st_size / 1e3:.0f} kB | time {time.perf_counter() - t0:.0f} s | saved: {out}")


if __name__ == "__main__":
    main()
