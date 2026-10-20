import argparse
import json
import os
import re
import time
import warnings
from pathlib import Path

import pandas as pd

from neurograph.agent.replication import VALUES_QUERY, literature_vs_ours
from neurograph.config import git_hash, load_config
from neurograph.literature import extract as ex
from neurograph.literature.fetch_pubmed import fetch, search
from neurograph.literature.verify_nli import ClaimVerifier


class RuleLLM:
    # stand-in for local smoke tests (no model download): first band/region/direction word wins
    def chat(self, system: str, user: str) -> str:
        s = user.lower()
        band = next((b for b in ex.BANDS if b in s), "alpha")
        region = next((r for r in ex.REGIONS if r[:5] in s), "global")
        d = "lower" if re.search(r"lower|reduc|decreas|less", s) else "no_difference" if "differ" in s else "higher"
        return json.dumps({"claim": True, "band": band, "measure": "asymmetry" if "asymmetr" in s else "power",
                           "region": region, "direction": d})


def main() -> None:
    warnings.filterwarnings("ignore")
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default=None, help="default: <derivatives>/literature")
    ap.add_argument("--refetch", action="store_true")
    ap.add_argument("--rule-llm", action="store_true", help="smoke test without a language model")
    args = ap.parse_args()
    t0 = time.perf_counter()
    cfg = load_config("literature")
    out = Path(args.dir) if args.dir else Path(load_config("paths")["paths"]["derivatives_dir"]) / "literature"
    out.mkdir(parents=True, exist_ok=True)

    abs_path = out / "abstracts.jsonl"
    if args.refetch or not abs_path.exists():
        arts = fetch(cfg["pubmed"], search(cfg["pubmed"]))
        abs_path.write_text("\n".join(json.dumps(a) for a in arts), encoding="utf-8")
    arts = [json.loads(l) for l in abs_path.read_text(encoding="utf-8").splitlines()]
    cands = ex.candidates(arts, cfg["extract"])

    if args.rule_llm:
        llm, llm_name = RuleLLM(), "rule-based stand-in (smoke test)"
    else:
        from neurograph.agent.llm import LocalLLM
        rag = load_config("rag")["llm"]
        llm, llm_name = LocalLLM(rag["model"], cfg["extract"]["max_new_tokens"]), rag["model"]
    claims = []
    for c in cands:
        parsed = ex.parse_claim(llm.chat(ex.EXTRACT_SYSTEM, f"Sentence: {c['sentence']}"))
        if parsed:
            claims.append({**c, **parsed, "id": f"{c['pmid']}-{c['sent_idx']}", "condition": cfg["condition"],
                           "hypothesis": ex.hypothesis(parsed),
                           "biomarkers": ex.biomarkers(parsed, cfg["regions"], cfg["measures"])})
    t_extract = time.perf_counter() - t0

    nli = ClaimVerifier(cfg["nli"]["model"], cfg["nli"]["min_entail"])
    for c, p in zip(claims, nli.entail([(c["sentence"], c["hypothesis"]) for c in claims])):
        c["entail"], c["verified"] = p, p >= cfg["nli"]["min_entail"]
    (out / "claims.jsonl").write_text("\n".join(json.dumps(c) for c in claims), encoding="utf-8")
    df = pd.DataFrame(claims)
    ver = df[df.verified] if len(df) else df

    print(f"== step 6.4-6.6 literature graph | commit {git_hash()} ==")
    print(f"PubMed abstracts {len(arts)} (years {min(a['year'] or 9999 for a in arts)}-{max(a['year'] or 0 for a in arts)}) "
          f"| candidate sentences {len(cands)} from {len({c['pmid'] for c in cands})} papers")
    print(f"LLM {llm_name} -> claims in schema {len(df)} | NLI {cfg['nli']['model']} verified {len(ver)} "
          f"({len(ver) / max(len(df), 1):.0%}) from {ver.pmid.nunique() if len(ver) else 0} papers | extract {t_extract:.0f} s")
    if len(ver):
        print("\nverified claims by biomarker (direction in depression vs controls):")
        tab = ver.assign(key=ver.band + " " + ver.measure + " " + ver.region).pivot_table(
            index="key", columns="direction", values="id", aggfunc="count", fill_value=0)
        print(tab.to_string())
        print("\nexamples (verified):")
        for r in ver.sort_values("entail", ascending=False).head(3).itertuples():
            print(f"  PMID {r.pmid} [{r.band} {r.measure} {r.region} {r.direction}, entail {r.entail}] {r.sentence[:160]}")
    rej = df[~df.verified] if len(df) else df
    if len(rej):
        print("examples (rejected by NLI):")
        for r in rej.head(2).itertuples():
            print(f"  PMID {r.pmid} [{r.hypothesis}] entail {r.entail} | {r.sentence[:140]}")

    if os.environ.get("NEO4J_URI"):
        from neurograph.graph import load_literature as ll
        from neurograph.graph.client import get_driver, run, write_batches
        driver = get_driver()
        for stmt in ll.SCHEMA:
            run(driver, stmt)
        run(driver, ll.RESET)
        papers = [{k: a[k] for k in ("pmid", "title", "year", "journal")} for a in arts
                  if a["pmid"] in set(df.pmid if len(df) else [])]
        write_batches(driver, ll.PAPERS, papers, 500)
        write_batches(driver, ll.CLAIMS, claims, 500)
        n = run(driver, """MATCH (c:Claim) OPTIONAL MATCH (c)-[a:ABOUT]->(:Biomarker)
                           RETURN count(DISTINCT c) AS claims, count(a) AS about""")[0]
        n_pap = run(driver, "MATCH (p:Paper) RETURN count(p) AS n")[0]["n"]
        print(f"\ngraph: Paper {n_pap} | Claim {n['claims']} | ABOUT -> shared Biomarker nodes {n['about']}")
        rows = literature_vs_ours(ver.to_dict("records"), lambda names: run(driver, VALUES_QUERY, names=names),
                                  cfg["regions"])
        rep = pd.DataFrame(rows)
        rep.to_csv(out / "literature_vs_ours.csv", index=False)
        print("\nreplication: published direction vs our eyes-closed data (Welch t-test computed in Python)")
        cols = ["band", "measure", "region", "papers", "lit_higher", "lit_lower", "lit_no_diff", "diff", "cohens_d", "p", "verdict"]
        with pd.option_context("display.width", 250, "display.max_colwidth", 70):
            print(rep[[c for c in cols if c in rep]].to_string(index=False))
        driver.close()
    print(f"\ntime {time.perf_counter() - t0:.0f} s | saved: {out / 'claims.jsonl'}")


if __name__ == "__main__":
    main()
