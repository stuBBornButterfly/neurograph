import re

from neurograph.agent.replication import FAA_QUERY, faa_facts
from neurograph.agent.safety import SERVICE_QUERY
from neurograph.graph.client import run
from neurograph.graph.queries import QUERIES

WRITE = re.compile(r"\b(CREATE|MERGE|DELETE|DETACH|SET|REMOVE|DROP|LOAD\s+CSV|FOREACH)\b|CALL\s+dbms|CALL\s+apoc", re.I)

# study-graph questions map to the 10 hand-written queries (index into QUERIES); first match wins
STUDY_ROUTES = [(r"asymmetr|\bfaa\b|replicat", 6), (r"risk tier|tier", 9), (r"drive|important|weight|biomarkers? matter", 8),
                (r"best model|auc|accura|perform|baseline", 7), (r"\balpha\b", 5), (r"reject|artifact|qc|noisy", 3),
                (r"bdi|stai|score|questionnaire", 1), (r"participant|subject|how many people|sex|women|men", 0)]
STUDY_WORDS = re.compile(r"\beeg\b|biomarker|alpha|theta|beta|gamma|delta|asymmetr|\bfaa\b|participant|"
                         r"our (?:data|study|model)|dataset|\bauc\b|risk tier|replicat|ds003478", re.I)


def read_only(cypher: str) -> str:
    if WRITE.search(cypher):
        raise PermissionError("write or admin Cypher is not allowed in the agent")
    return cypher


class GraphTools:
    # everything the agent reads from Neo4j; driver=None uses config data (local tests without a database)
    def __init__(self, driver=None, policy_cfg: dict | None = None):
        self.driver, self.cfg = driver, policy_cfg or {}

    def services(self) -> list[dict]:
        if self.driver:
            return run(self.driver, read_only(SERVICE_QUERY))
        return [{"name": s["name"], "location": s["location"], "contact": s["contact_public"], "url": s["url"]}
                for s in self.cfg.get("services", []) if {"depression", "anxiety", "stress"} & set(s["supports"])]

    def rules_for(self, chunk_ids: list[str]) -> list[str]:
        if not self.driver:
            return []
        rows = run(self.driver, read_only("""MATCH (p:Policy)-[:STATES]->(r:Rule) WHERE p.id IN $ids
                                             RETURN r.subject AS s, r.condition AS c, r.value AS v, r.quote AS q"""),
                   ids=chunk_ids)
        return [f"Rule ({r['s']}): if {r['c']} -> {r['v']}. Source: \"{r['q']}\"" for r in rows]

    def study(self, question: str) -> tuple[str, list[dict], list[str]]:
        idx = next((i for pat, i in STUDY_ROUTES if re.search(pat, question, re.I)), 4)
        q = QUERIES[idx]
        rows = run(self.driver, read_only(q["cypher"])) if self.driver else []
        facts = study_facts(idx, rows)
        if idx == FAA_IDX and self.driver:
            facts += faa_facts(run(self.driver, read_only(FAA_QUERY)))
        return q["title"], rows, facts


FAA_IDX, TIER_IDX, PARTICIPANT_IDX = 6, 9, 0


def study_facts(idx: int, rows: list[dict]) -> list[str]:
    # totals the small LLM would otherwise have to add up itself, as plain sentences
    if idx == PARTICIPANT_IDX and rows:
        by = {g: sum(r["n"] for r in rows if r["group"] == g) for g in sorted({r["group"] for r in rows})}
        fem = {g: sum(r["n"] for r in rows if r["group"] == g and r["sex"] == "F") for g in by}
        return [f"The EEG study has {sum(by.values())} participants in total."] + \
            [f"Group {g}: {n} participants ({fem[g]} female)." for g, n in by.items()]
    if idx == TIER_IDX and rows:
        return [f"The {r['tier']} risk tier has {r['ctl'] + r['dep']} participants ({r['ctl']} controls, "
                f"{r['dep']} depression group)." for r in rows]
    return []


def rows_text(rows: list[dict]) -> str:
    return "\n".join(", ".join(f"{k}: {v}" for k, v in r.items()) for r in rows)
