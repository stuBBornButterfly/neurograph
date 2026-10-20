import argparse
import gc
import json
import os
import time
import warnings
from pathlib import Path

import pandas as pd
import torch

from neurograph.config import REPO_ROOT, git_hash, load_config
from scripts.eval_qlora import score
from scripts.run_agent_eval import fact_hit


def heldout_sample(test: pd.DataFrame, n: int, seed: int) -> pd.DataFrame:
    # every "answer removed" example + a seeded sample of the others
    neg = test[test.negative]
    pos = test[~test.negative].sample(n=min(n - len(neg), int((~test.negative).sum())), random_state=seed)
    return pd.concat([pos, neg]).sort_index()


def run_heldout(llm, users: list[str], system: str) -> tuple[list[str], list[dict]]:
    n = len(llm.usage)
    answers = llm.chat_many(system, users, batch=1)  # one at a time: the latency a user would see
    return answers, llm.usage[n:]


def run_gold(agent, llms: list, gold: list[dict]) -> list[dict]:
    # llms: every model the agent may call (policy model, research model); their token use is summed
    rows = []
    for g in gold:
        n, t = [len(m.usage) for m in llms], time.perf_counter()
        out = agent.invoke({"question": g["question"]})
        use, chk = [u for m, k in zip(llms, n) for u in m.usage[k:]], out.get("check", {})
        rows.append({"id": g["id"], "route_gold": g["route"], "route": out["route"], "question": g["question"],
                     "route_ok": out["route"] == g["route"], "fact_ok": fact_hit(out["answer"], g["expect"], g.get("forbid", [])),
                     "verdict": chk.get("verdict", "-"), "entailed": chk.get("entailed_share"),
                     "bad_numbers": len(chk.get("unsupported_numbers", [])), "answer": out["answer"],
                     "contexts": [p["text"] for p in out.get("passages", [])] + out.get("rules", []),
                     "seconds": time.perf_counter() - t, "llm_seconds": sum(u["seconds"] for u in use),
                     "prompt_tokens": sum(u["prompt_tokens"] for u in use),
                     "completion_tokens": sum(u["completion_tokens"] for u in use)})
    return rows


def cost_row(usage: pd.DataFrame, ref: dict | None, watts: float | None) -> dict:
    # per 1,000 questions: API bill from measured tokens x list price; self-hosted: compute hours (+ energy on GPU)
    p_in, p_out = (ref["price_in_per_mtok"], ref["price_out_per_mtok"]) if ref else (0.0, 0.0)
    tok_in, tok_out = usage.prompt_tokens.mean(), usage.completion_tokens.mean()
    hours = usage.seconds.mean() * 1000 / 3600
    return {"tokens_in_per_q": round(tok_in), "tokens_out_per_q": round(tok_out),
            "api_usd_per_1k": round(1000 * (tok_in * p_in + tok_out * p_out) / 1e6, 3),
            "compute_h_per_1k": round(hours, 2), "kwh_per_1k": round(hours * watts / 1000, 2) if watts else None}


def main() -> None:
    warnings.filterwarnings("ignore")
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="*", default=None, help="subset of models (base finetuned routed gguf reference)")
    args = ap.parse_args()
    t0 = time.perf_counter()
    from neurograph.agent.backends import ApiLLM, LlamaCppLLM
    from neurograph.agent.tools import GraphTools
    from neurograph.agent.verifier import Verifier
    from neurograph.agent.workflow import POLICY_SYSTEM, build_agent
    from neurograph.ingest.index_qdrant import HybridIndex

    bcfg, rag, ft, app = load_config("benchmark"), load_config("rag"), load_config("finetune"), load_config("app")
    der = Path(load_config("paths")["paths"]["derivatives_dir"])
    qdir, inst, out = der / ft["out"], der / "institutional", der / "benchmark"
    out.mkdir(parents=True, exist_ok=True)
    test = heldout_sample(pd.read_json(qdir / "test.jsonl", lines=True), bcfg["heldout"]["n"], ft["seed"])
    gold = [json.loads(l) for l in (REPO_ROOT / rag["agent"]["gold"]).read_text().splitlines()]
    chunks = [json.loads(l) for l in (inst / "chunks.jsonl").read_text(encoding="utf-8").splitlines()]
    e, q = rag["embedding"], rag["qdrant"]
    index = HybridIndex(chunks, inst / Path(q["path"]).name, q["collection"], e["model"], e["query_prefix"])
    driver = None
    if os.environ.get("NEO4J_URI"):
        from neurograph.graph.client import get_driver
        driver = get_driver()
    tools = GraphTools(driver, load_config("policy_graph"))
    verifier = Verifier(rag["nli"]["model"], rag["nli"]["min_entail"])
    ref_cfg = bcfg["reference"]
    names = args.only or [m["name"] for m in bcfg["models"]]

    held, golds, timing = {}, {}, {}
    ft_llm = None
    for i, name in enumerate(names):
        t = time.perf_counter()
        research = None  # model for the study and literature routes when it differs from the policy model
        if name in ("base", "finetuned", "routed"):
            if ft_llm is None:
                from neurograph.agent.llm import AdapterOff, LocalLLM
                ft_llm = LocalLLM(rag["llm"]["model"], rag["llm"]["max_new_tokens"], adapter=str(qdir / "adapter"))
            llm = AdapterOff(ft_llm) if name == "base" else ft_llm
            research = AdapterOff(ft_llm) if name == "routed" else None
        elif name == "gguf":  # as deployed: fine-tuned GGUF on the policy route, Qwen's own base GGUF for research
            from huggingface_hub import hf_hub_download
            kw = {"n_ctx": app["llm"]["n_ctx"], "n_threads": bcfg["gguf_threads"]}
            llm = LlamaCppLLM(str(qdir / ft["gguf"]["file"]), rag["llm"]["max_new_tokens"], **kw)
            research = LlamaCppLLM(hf_hub_download(app["llm"]["base_gguf_repo"], app["llm"]["base_gguf_file"]),
                                   rag["llm"]["max_new_tokens"], **kw)
        else:
            llm = ApiLLM(ref_cfg["kind"], ref_cfg["base_url"], ref_cfg["model"], rag["llm"]["max_new_tokens"],
                         ref_cfg["api_key_env"], ref_cfg["external"])
        if name == "routed" and "finetuned" in held:  # held-out prompts are all policy prompts: same as fine-tuned
            held[name] = held["finetuned"]
        else:
            answers, use = run_heldout(llm, test.user.tolist(), POLICY_SYSTEM)
            held[name] = (answers, pd.DataFrame(use))
        # an outside API only gets public questions: The university policy (safety messages never reach a model)
        g = [x for x in gold if not llm.external or x["route"] in ("policy",) or x["route"].startswith("safety")]
        llms = [llm] + ([research] if research else [])
        golds[name] = run_gold(build_agent(index, tools, llm, verifier, rag, research), llms, g)
        timing[name] = time.perf_counter() - t
        print(f"  {name}: {len(held[name][0])} held-out + {len(g)} gold answers in {timing[name] / 60:.1f} min",
              flush=True)
        del llm, research, llms
        if ft_llm is not None and not {"base", "finetuned", "routed"} & set(names[i + 1:]):
            ft_llm = None  # free the GPU before the reference model loads (Ollama shares the same card)
        gc.collect()
        torch.cuda.empty_cache()
    from neurograph.literature.verify_nli import ClaimVerifier
    chunk_text = {c["id"]: c["text"] for c in chunks}
    nli = ClaimVerifier(ft["dataset"]["nli_model"], ft["dataset"]["min_entail"])
    label = {m["name"]: m["label"] for m in bcfg["models"]}
    label["reference"] = ref_cfg["name"]
    rows_h, rows_g, rows_c = [], [], []
    for name in names:
        answers, use = held[name]
        s = score(test, answers, use.seconds.sum(), chunk_text, nli, ft["dataset"]["min_entail"])
        rows_h.append({"model": label[name], **{k: round(v, 3) for k, v in s.items()}})
        g = pd.DataFrame(golds[name])
        g["model"] = name
        golds[name] = g
        non_safety = g[~g.route_gold.str.startswith("safety")]
        row = {"model": label[name], "questions": len(g), "route_ok": round(g.route_ok.mean(), 2),
               "fact_ok": round(non_safety.fact_ok.mean(), 2),
               "verified": round((non_safety.verdict == "supported").mean(), 2),
               "sec_per_q": round(non_safety.seconds.median(), 1)}
        for kind in ("policy", "study", "literature"):
            k = g[g.route_gold == kind]
            row[f"{kind}_fact_ok"] = f"{int(k.fact_ok.sum())}/{len(k)}" if len(k) else "-"
        rows_g.append(row)
        both = pd.concat([use, non_safety[["prompt_tokens", "completion_tokens"]].assign(seconds=non_safety.llm_seconds)])
        watts = bcfg["gpu_watts"] if name in ("base", "finetuned", "routed", "reference") else None
        rows_c.append({"model": label[name], **cost_row(both, ref_cfg if name == "reference" else None, watts)})
    tab_h, tab_g, tab_c = pd.DataFrame(rows_h), pd.DataFrame(rows_g), pd.DataFrame(rows_c)
    for tab, f in [(tab_h, "heldout.csv"), (tab_g, "agent_gold.csv"), (tab_c, "cost.csv")]:
        tab.to_csv(out / f, index=False)
    pd.concat([test.assign(model=n, answer=held[n][0]) for n in names]).to_json(
        out / "heldout_answers.jsonl", orient="records", lines=True, force_ascii=False)
    pd.concat(golds.values()).to_json(out / "gold_answers.jsonl", orient="records", lines=True, force_ascii=False)

    print(f"== Phase 9 benchmark | commit {git_hash()} ==")
    print(f"reference: {ref_cfg['name']} ({ref_cfg['kind']} API, external {ref_cfg['external']}) | graph "
          f"{'Neo4j' if driver else 'none'} | greedy decoding, one request at a time, same prompts and passages")
    with pd.option_context("display.width", 250, "display.max_columns", 30):
        print(f"\nA. held-out university questions (n {len(test)}, {int(test.negative.sum())} with the answer removed; "
              f"chunks never seen in training)\n{tab_h.to_string(index=False)}")
        print(f"\nB. agent end to end on the 24 gold questions (fact_ok / verified exclude the 6 safety messages)\n"
              f"{tab_g.to_string(index=False)}")
        print(f"\nC. cost and compute per 1,000 questions (held-out + gold LLM calls)\n{tab_c.to_string(index=False)}")
    for pick in ["a01", "a08", "a19"]:
        print(f"\n--- {pick}: {next(x['question'] for x in gold if x['id'] == pick)}")
        for name in names:
            r = golds[name].set_index("id")
            if pick in r.index:
                print(f"  [{name}, {r.loc[pick, 'verdict']}] {r.loc[pick, 'answer'][:260]!r}")
    print(f"\ntime {(time.perf_counter() - t0) / 60:.1f} min | saved: {out}")


if __name__ == "__main__":
    main()
