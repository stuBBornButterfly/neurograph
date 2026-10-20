import hashlib
import re


def squash(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def rule_rows(rules: list[dict], chunks: list[dict]) -> list[dict]:
    # attach each curated rule to the chunk that contains its quote verbatim; refuse unquotable rules
    rows = []
    for i, r in enumerate(rules):
        hit = next((c for c in chunks if c["url"] == r["url"] and squash(r["quote"]) in squash(c["text"])), None)
        if hit is None:
            raise ValueError(f"rule '{r['subject']}': quote not found verbatim in {r['url']}")
        rows.append({"id": f"rule-{i:02d}", "policy": hit["id"], "subject": r["subject"], "condition": r["condition"],
                     "value": r["value"], "quote": squash(r["quote"])})
    return rows


def refers_rows(chunks: list[dict], services: list[dict]) -> list[dict]:
    return [{"policy": c["id"], "service": s["id"]} for c in chunks for s in services
            if any(k.lower() in c["text"].lower() for k in s["keywords"])]


def section_rows(chunks: list[dict]) -> list[dict]:
    return [{"id": c["id"], "url": c["url"], "title": c["title"], "chunk": c["chunk"], "text": c["text"],
             "text_hash": hashlib.sha256(c["text"].encode()).hexdigest()[:16]} for c in chunks]


SCHEMA = [
    "CREATE CONSTRAINT document_url IF NOT EXISTS FOR (n:Document) REQUIRE n.url IS UNIQUE",
    "CREATE CONSTRAINT policy_id IF NOT EXISTS FOR (n:Policy) REQUIRE n.id IS UNIQUE",
    "CREATE CONSTRAINT service_id IF NOT EXISTS FOR (n:Service) REQUIRE n.id IS UNIQUE",
    "CREATE CONSTRAINT rule_id IF NOT EXISTS FOR (n:Rule) REQUIRE n.id IS UNIQUE",
    "CREATE CONSTRAINT condition_name IF NOT EXISTS FOR (n:Condition) REQUIRE n.name IS UNIQUE",
]

RESET = """MATCH (n) WHERE n:Document OR n:Policy OR n:Service OR n:Rule
CALL (n) { DETACH DELETE n } IN TRANSACTIONS OF 5000 ROWS"""

DOCUMENTS = """UNWIND $rows AS row
MERGE (d:Document {url: row.url})
SET d.title = row.title, d.group = row.group, d.retrieved_at = row.retrieved_at, d.sha256 = row.sha256"""

SECTIONS = """UNWIND $rows AS row
MATCH (d:Document {url: row.url})
MERGE (p:Policy {id: row.id})
SET p.title = row.title, p.chunk = row.chunk, p.text = row.text, p.text_hash = row.text_hash
MERGE (d)-[:HAS_SECTION]->(p)"""

SERVICES = """UNWIND $rows AS row
MERGE (s:Service {id: row.id})
SET s.name = row.name, s.office = row.office, s.location = row.location,
    s.contact_public = row.contact_public, s.url = row.url
WITH s, row
UNWIND row.supports AS cond
MERGE (c:Condition {name: cond})
MERGE (s)-[:SUPPORTS]->(c)"""

REFERS = """UNWIND $rows AS row
MATCH (p:Policy {id: row.policy}), (s:Service {id: row.service})
MERGE (p)-[:REFERS_TO]->(s)"""

RULES = """UNWIND $rows AS row
MATCH (p:Policy {id: row.policy})
MERGE (r:Rule {id: row.id})
SET r.subject = row.subject, r.condition = row.condition, r.value = row.value, r.quote = row.quote
MERGE (p)-[:STATES]->(r)"""
