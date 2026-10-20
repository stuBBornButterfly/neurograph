import argparse
import json
import time
from pathlib import Path

import pandas as pd

from neurograph.config import REPO_ROOT, git_hash, load_config
from neurograph.ingest.index_qdrant import HybridIndex


def first_hit(ranked: list[int], chunks: list[dict], sources: list[str]) -> int | None:
    # rank (1-based) of the first chunk whose page title matches an accepted source
    for r, i in enumerate(ranked, 1):
        if any(s.lower() in chunks[i]["title"].lower() for s in sources):
            return r
    return None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default=None, help="default: <derivatives>/institutional")
    args = ap.parse_args()
    t0 = time.perf_counter()
    cfg = load_config("rag")
    der = Path(load_config("paths")["paths"]["derivatives_dir"])
    inst = Path(args.dir) if args.dir else der / "institutional"
    chunks = [json.loads(l) for l in (inst / "chunks.jsonl").read_text(encoding="utf-8").splitlines()]
    e, q, r = cfg["embedding"], cfg["qdrant"], cfg["retrieval"]
    index = HybridIndex(chunks, inst / Path(q["path"]).name, q["collection"], e["model"], e["query_prefix"])
    dim = index.build(e["batch_size"])
    t_build = time.perf_counter() - t0

    gold = [json.loads(l) for l in (REPO_ROOT / cfg["gold"]).read_text().splitlines()]
    depth = max(r["eval_k"])
    methods = {"dense (BGE)": lambda s: index.dense(s, depth), "keyword (BM25)": lambda s: index.keyword(s, depth),
               "hybrid (RRF)": lambda s: index.hybrid(s, depth, r["rrf_k"])}
    rows = []
    for g in gold:
        for name, search in methods.items():
            rows.append({"id": g["id"], "method": name, "rank": first_hit(search(g["question"]), chunks, g["sources"])})
    res = pd.DataFrame(rows)
    res.to_csv(inst / "retrieval_eval.csv", index=False)

    print(f"== step 6.2 hybrid index (Qdrant + BM25) | commit {git_hash()} ==")
    print(f"chunks {len(chunks)} | model {e['model']} ({dim}-dim) | collection '{q['collection']}' on disk | "
          f"build {t_build:.0f} s")
    print(f"\nretrieval on {len(gold)} hand-written student questions (hit = a chunk from the right page)")
    print(f"{'method':16s} " + " ".join(f"{'hit@' + str(k):>6s}" for k in r["eval_k"]) + f" {'MRR':>6s}")
    for name in methods:
        ranks = res.loc[res.method == name, "rank"]
        hits = " ".join(f"{(ranks <= k).mean():6.2f}" for k in r["eval_k"])
        mrr = ranks.map(lambda x: 1 / x if pd.notna(x) else 0).mean()
        print(f"{name:16s} {hits} {mrr:6.3f}")
    missed = res[(res.method == "hybrid (RRF)") & (res["rank"].isna() | (res["rank"] > 3))]["id"].tolist()
    print(f"hybrid misses at top 3: {missed or 'none'}")

    probe = next(g for g in gold if g["id"] == "r09")
    print(f"\nexample: \"{probe['question']}\"")
    for k, i in enumerate(index.hybrid(probe["question"], 3, r["rrf_k"]), 1):
        print(f"  {k}. {chunks[i]['title'][:45]} | {chunks[i]['text'][:90].replace(chr(10), ' / ')}")
    print(f"\ntime {time.perf_counter() - t0:.0f} s | saved: {inst / Path(q['path']).name} | retrieval_eval.csv")


if __name__ == "__main__":
    main()
