import json
from datetime import datetime, timezone
from pathlib import Path

from neurograph.agent.replication import FAA_QUERY, VALUES_QUERY, faa_facts, literature_vs_ours
from neurograph.agent.safety import SERVICE_QUERY
from neurograph.agent.tools import (FAA_IDX, REGIONS, literature_params, read_only, study_facts, study_index,
                                    top_comparison_facts)
from neurograph.graph.client import run
from neurograph.graph.queries import QUERIES

ALL_CLAIMS = """MATCH (p:Paper)-[:REPORTS]->(c:Claim {verified: true})
RETURN p.pmid AS pmid, p.year AS year, c.text AS text, c.band AS band, c.measure AS measure,
       c.region AS region, c.direction AS direction, c.entail AS entail
ORDER BY c.entail DESC"""
ALL_RULES = """MATCH (p:Policy)-[:STATES]->(r:Rule)
RETURN p.id AS policy, r.subject AS s, r.condition AS c, r.value AS v, r.quote AS q ORDER BY r.id"""


def export(driver, git: str) -> dict:
    # what the agent reads from Neo4j, frozen for the demo Space: aggregates and computed facts only,
    # never the per-participant values behind them (FAA and literature comparisons are computed here)
    rules: dict[str, list[str]] = {}
    for r in run(driver, read_only(ALL_RULES)):
        rules.setdefault(r["policy"], []).append(f"Rule ({r['s']}): if {r['c']} -> {r['v']}. Source: \"{r['q']}\"")
    study = []
    for idx, q in enumerate(QUERIES):
        rows = run(driver, read_only(q["cypher"]))
        facts = study_facts(idx, rows) + (faa_facts(run(driver, read_only(FAA_QUERY))) if idx == FAA_IDX else [])
        study.append({"title": q["title"], "rows": rows, "facts": facts})
    claims = run(driver, read_only(ALL_CLAIMS))
    comparisons = literature_vs_ours(claims, lambda names: run(driver, read_only(VALUES_QUERY), names=names), REGIONS)
    return {"created": datetime.now(timezone.utc).isoformat(timespec="seconds"), "git": git,
            "services": run(driver, read_only(SERVICE_QUERY)), "rules": rules, "study": study,
            "claims": claims, "comparisons": comparisons}


def claim_matches(c: dict, p: dict) -> bool:
    # same filter as CLAIM_QUERY in graph/load_literature.py
    return ((p["band"] is None or c["band"] == p["band"]) and (p["region"] is None or c["region"] in (p["region"], "global"))
            and (p["measure"] is None or c["measure"] == p["measure"]))


class SnapshotTools:
    # same interface as GraphTools, answered from an exported snapshot (no database needed)
    def __init__(self, path: Path):
        self.snap = json.loads(Path(path).read_text(encoding="utf-8"))

    def services(self) -> list[dict]:
        return self.snap["services"]

    def rules_for(self, chunk_ids: list[str]) -> list[str]:
        return [r for i in chunk_ids for r in self.snap["rules"].get(i, [])]

    def study(self, question: str) -> tuple[str, list[dict], list[str]]:
        s = self.snap["study"][study_index(question)]
        return s["title"], s["rows"], s["facts"]

    def literature(self, question: str, k: int = 6) -> tuple[list[dict], list[str]]:
        p = literature_params(question)
        claims = [c for c in self.snap["claims"] if claim_matches(c, p)]
        keys = {(c["band"], c["measure"], c["region"]) if c["measure"] == "power" else ("alpha", "asymmetry", "frontal")
                for c in claims}
        rows = [r for r in self.snap["comparisons"] if (r["band"], r["measure"], r["region"]) in keys]
        return claims[:k], top_comparison_facts(rows)
