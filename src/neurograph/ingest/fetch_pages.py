import hashlib
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import requests


def approved_urls(cfg: dict) -> list[tuple[str, str]]:
    # (group, url) pairs; refuses to run until the list is approved
    if cfg.get("status") != "approved":
        raise RuntimeError("configs/ingest_urls.yaml is not approved yet (status must be 'approved')")
    return [(group, url) for group, urls in cfg["pages"].items() for url in urls]


def fetch_all(pairs: list[tuple[str, str]], raw_dir: Path, crawl: dict) -> list[dict]:
    # polite: one request at a time, fixed delay, identifying user agent; saves bytes + a manifest row
    raw_dir.mkdir(parents=True, exist_ok=True)
    session = requests.Session()
    session.headers["User-Agent"] = crawl["user_agent"]
    rows = []
    for k, (group, url) in enumerate(pairs):
        if k:
            time.sleep(crawl["delay_s"])
        r = session.get(url, timeout=30)
        body = r.content
        sha = hashlib.sha256(body).hexdigest()
        kind = "pdf" if "pdf" in r.headers.get("Content-Type", "") or url.lower().endswith(".pdf") else "html"
        path = raw_dir / f"{sha[:16]}.{kind}"
        if r.ok:
            path.write_bytes(body)
        rows.append({"url": url, "group": group, "status": r.status_code, "kind": kind, "bytes": len(body),
                     "sha256": sha, "file": path.name if r.ok else None,
                     "retrieved_at": datetime.now(timezone.utc).isoformat(timespec="seconds")})
    (raw_dir / "manifest.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    return rows
