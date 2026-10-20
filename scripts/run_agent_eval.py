import argparse
import json
import os
import re
import time
import warnings
from pathlib import Path

import pandas as pd

from neurograph.config import REPO_ROOT, git_hash, load_config
from neurograph.ingest.index_qdrant import HybridIndex


class EchoLLM:
    # stand-in for local smoke tests: repeats the first passage lines (no model download)
    def chat(self, system: str, user: str) -> str:
        body = user.split("\n", 2)[-1]
        return " ".join(body.split("\n")[:2])[:300] + " [1]"


def fact_hit(answer: str, expect: list[str], forbid: list[str] = ()) -> bool:
    # right fact present and no forbidden claim (e.g. "replicates" when the computed verdict says it does not)
    wrong = any(f.lower() in answer.lower() for f in forbid)
    return not wrong and any(re.search(rf"(?<![\w.]){re.escape(e)}(?![\w])", answer, re.I) for e in expect)


def main() -> None:
    warnings.filterwarnings("ignore")
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default=None, help="default: <derivatives>/institutional")
    ap.add_argument("--echo-llm", action="store_true", help="smoke test without a language model")
    ap.add_argument("--adapter", default=None, help="LoRA adapter folder (Phase 7) on top of the base model")
    ap.add_argument("--tag", default="", help="suffix for the saved eval file")
    args = ap.parse_args()
    from neurograph.agent.tools import GraphTools
    from neurograph.agent.verifier import Verifier
    from neurograph.agent.workflow import build_agent

    t0 = time.perf_counter()
    cfg, pcfg = load_config("rag"), load_config("policy_graph")
    inst = Path(args.dir) if args.dir else Path(load_config("paths")["paths"]["derivatives_dir"]) / "institutional"
    chunks = [json.loads(l) for l in (inst / "chunks.jsonl").read_text(encoding="utf-8").splitlines()]
    e, q = cfg["embedding"], cfg["qdrant"]
    index = HybridIndex(chunks, inst / Path(q["path"]).name, q["collection"], e["model"], e["query_prefix"])
    if not index.ready():
        index.build(e["batch_size"])
    driver = None
    if os.environ.get("NEO4J_URI"):
        from neurograph.graph.client import get_driver
        driver = get_driver()
    tools = GraphTools(driver, pcfg)
    if args.echo_llm:
        llm, llm_name = EchoLLM(), "echo (smoke test)"
    else:
        from neurograph.agent.llm import LocalLLM
        llm = LocalLLM(cfg["llm"]["model"], cfg["llm"]["max_new_tokens"], adapter=args.adapter)
        llm_name = cfg["llm"]["model"] + (" + QLoRA adapter" if args.adapter else "")
    verifier = Verifier(cfg["nli"]["model"], cfg["nli"]["min_entail"])
    agent = build_agent(index, tools, llm, verifier, cfg)
    t_load = time.perf_counter() - t0

    gold = [json.loads(l) for l in (REPO_ROOT / cfg["agent"]["gold"]).read_text().splitlines()]
    rows = []
    for g in gold:
        t = time.perf_counter()
        out = agent.invoke({"question": g["question"]})
        chk = out.get("check", {})
        rows.append({"id": g["id"], "kind": g["route"].split(":")[0], "route_ok": out["route"] == g["route"],
                     "route": out["route"], "fact_ok": fact_hit(out["answer"], g["expect"], g.get("forbid", [])),
                     "verdict": chk.get("verdict", "-"), "entailed": chk.get("entailed_share"),
                     "bad_numbers": len(chk.get("unsupported_numbers", [])), "seconds": time.perf_counter() - t,
                     "answer": out["answer"]})
    res = pd.DataFrame(rows)
    res.to_json(inst / f"agent_eval{args.tag}.jsonl", orient="records", lines=True)

    print(f"== agent eval (safety -> router -> retrieval/graph -> LLM -> verifier) | commit {git_hash()} ==")
    print(f"LLM {llm_name} | graph {'AuraDB' if driver else 'none (config fallback)'} | NLI {cfg['nli']['model']} "
          f"| load {t_load:.0f} s")
    print(f"\n{'kind':8s} {'n':>3s} {'route ok':>8s} {'fact ok':>8s} {'verified':>8s} {'sec/q':>6s}")
    for kind, g in res.groupby("kind", sort=False):
        ver = (g["verdict"] == "supported").mean() if kind != "safety" else float("nan")
        print(f"{kind:8s} {len(g):3d} {g.route_ok.mean():8.2f} {g.fact_ok.mean():8.2f} {ver:8.2f} {g.seconds.median():6.1f}")
    print(f"{'all':8s} {len(res):3d} {res.route_ok.mean():8.2f} {res.fact_ok.mean():8.2f}")
    safe = res[res.kind == "safety"]
    print(f"\nsafety: {int(safe.route_ok.sum())}/{len(safe)} distress/crisis messages routed to referral; "
          f"non-safety questions sent to referral: {int((res[res.kind != 'safety'].route.str.startswith('safety')).sum())}")
    print("misses: " + (", ".join(f"{r.id} ({'route ' + r.route if not r.route_ok else 'fact'})"
                                  for r in res.itertuples() if not (r.route_ok and r.fact_ok)) or "none"))
    for r in res.itertuples():
        if not (r.route_ok and r.fact_ok):
            print(f"  {r.id} answer: {r.answer[:300]!r}")
    for pick in ["a01", "a14", "a19"]:
        r = res.set_index("id").loc[pick]
        print(f"\n--- {pick} [{r.route}, verifier {r.verdict}] {next(g['question'] for g in gold if g['id'] == pick)}")
        print(r.answer[:700])
    print(f"\ntime {time.perf_counter() - t0:.0f} s | saved: {inst / f'agent_eval{args.tag}.jsonl'}")


if __name__ == "__main__":
    main()
