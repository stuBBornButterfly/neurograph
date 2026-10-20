import os
from pathlib import Path

from neo4j import Driver, GraphDatabase


def get_driver() -> Driver:
    # credentials come from the environment (Kaggle Secrets / .env), never from code or configs
    uri, user, pw = (os.environ.get(k) for k in ("NEO4J_URI", "NEO4J_USERNAME", "NEO4J_PASSWORD"))
    if not (uri and user and pw):
        raise RuntimeError("set NEO4J_URI, NEO4J_USERNAME and NEO4J_PASSWORD (Kaggle: Add-ons -> Secrets)")
    driver = GraphDatabase.driver(uri, auth=(user, pw))
    driver.verify_connectivity()
    return driver


def run(driver: Driver, query: str, **params) -> list[dict]:
    with driver.session() as s:  # auto-commit, so CALL ... IN TRANSACTIONS also works
        return [r.data() for r in s.run(query, **params)]


def write_batches(driver: Driver, query: str, rows: list[dict], batch_size: int) -> int:
    # query must start with UNWIND $rows AS row
    for k in range(0, len(rows), batch_size):
        run(driver, query, rows=rows[k:k + batch_size])
    return len(rows)


def apply_schema(driver: Driver, path: Path) -> int:
    text = "\n".join(l for l in path.read_text().splitlines() if not l.strip().startswith("//"))
    stmts = [s.strip() for s in text.split(";") if s.strip()]
    for s in stmts:
        run(driver, s)
    return len(stmts)
