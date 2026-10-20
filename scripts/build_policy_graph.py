import argparse
import json
import time
from pathlib import Path

from neurograph.config import git_hash, load_config
from neurograph.graph import load_policy as lp
from neurograph.graph.client import get_driver, run, write_batches


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default=None, help="default: <derivatives>/institutional")
    args = ap.parse_args()
    t0 = time.perf_counter()
    cfg = load_config("policy_graph")
    inst = Path(args.dir) if args.dir else Path(load_config("paths")["paths"]["derivatives_dir"]) / "institutional"
    docs = [json.loads(l) for l in (inst / "docs.jsonl").read_text().splitlines()]
    chunks = [json.loads(l) for l in (inst / "chunks.jsonl").read_text(encoding="utf-8").splitlines()]
    rules = lp.rule_rows(cfg["rules"], chunks)  # fails loudly if a quote is not on the page
    refers = lp.refers_rows(chunks, cfg["services"])

    driver = get_driver()
    for stmt in lp.SCHEMA:
        run(driver, stmt)
    run(driver, lp.RESET)
    n = {"documents": write_batches(driver, lp.DOCUMENTS, docs, 500),
         "sections": write_batches(driver, lp.SECTIONS, lp.section_rows(chunks), 500),
         "services": write_batches(driver, lp.SERVICES, cfg["services"], 500),
         "refers_to": write_batches(driver, lp.REFERS, refers, 500),
         "rules": write_batches(driver, lp.RULES, rules, 500)}

    print(f"== step 6.3 university policy graph | commit {git_hash()} ==")
    print("rows sent: " + " | ".join(f"{k} {v}" for k, v in n.items()))
    counts = run(driver, """MATCH (n) WHERE n:Document OR n:Policy OR n:Service OR n:Rule OR n:Condition
                            RETURN labels(n)[0] AS label, count(*) AS n ORDER BY label""")
    print("nodes: " + " | ".join(f"{r['label']} {r['n']}" for r in counts))
    print("\nwho helps with what (Service)-[:SUPPORTS]->(Condition):")
    for r in run(driver, """MATCH (s:Service)-[:SUPPORTS]->(c:Condition)
                            RETURN s.name AS service, s.location AS location, collect(c.name) AS conditions"""):
        print(f"  {r['service']} ({r['location']}): {', '.join(r['conditions'])}")
    print("\nrules with their source sentence (Document)-[:HAS_SECTION]->(Policy)-[:STATES]->(Rule):")
    for r in run(driver, """MATCH (d:Document)-[:HAS_SECTION]->(:Policy)-[:STATES]->(r:Rule)
                            RETURN r.subject AS subject, r.condition AS cond, r.value AS value, d.title AS page
                            ORDER BY r.id"""):
        print(f"  {r['subject']:17s} if {r['cond'][:48]:48s} -> {r['value'][:55]}")
    total = run(driver, "MATCH (n) RETURN count(n) AS n")[0]["n"]
    print(f"\nwhole graph now {total} nodes (study + policy) | time {time.perf_counter() - t0:.0f} s")
    driver.close()


if __name__ == "__main__":
    main()
