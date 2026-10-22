import argparse
import json
import re
import time
from pathlib import Path

import yaml

from neurograph.config import REPO_ROOT, git_hash, load_config
from neurograph.ingest.chunk import chunk_lines
from neurograph.ingest.clean import html_text, pdf_text
from neurograph.ingest.fetch_pages import approved_urls, fetch_all
from neurograph.ingest.redact_pii import AWARD, HONORIFIC, build_analyzer, redact

EMAIL = re.compile(r"[\w.+-]+@[\w-]+\.[\w.]+")


def leaks(text: str, keep: list[str]) -> list[str]:
    # anything that still looks like a personal email or an honorific + name after redaction
    emails = [e for e in EMAIL.findall(text) if e.lower() not in [k.lower() for k in keep]]
    return emails + [n for n in re.findall(HONORIFIC, text) if not re.search(AWARD, n)]  # named awards are kept


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default="/tmp/pages_raw", help="raw pages (contain PII, never indexed or saved)")
    ap.add_argument("--out", default=None, help="default: <derivatives>/institutional")
    ap.add_argument("--refetch", action="store_true")
    args = ap.parse_args()
    t0 = time.perf_counter()
    cfg = yaml.safe_load((REPO_ROOT / "configs" / "ingest_urls.yaml").read_text())
    raw = Path(args.raw)
    out = Path(args.out or Path(load_config("paths")["paths"]["derivatives_dir"]) / "institutional")
    (out / "clean").mkdir(parents=True, exist_ok=True)

    skipped = {s["url"] for s in cfg.get("skipped", [])}
    pairs = [(g, u) for g, u in approved_urls(cfg) if u not in skipped]
    if args.refetch or not (raw / "manifest.jsonl").exists():
        fetch_all(pairs, raw, cfg["crawl"])
    manifest = [json.loads(l) for l in (raw / "manifest.jsonl").read_text().splitlines()]
    manifest = [m for m in manifest if m["url"] in {u for _, u in pairs} and m["file"]]

    engine = build_analyzer(cfg["spacy_model"])
    pii, keep, cl, ck = cfg["pii"], cfg["pii"]["keep_office_contacts"], cfg["clean"], cfg["chunk"]
    docs, chunks, all_leaks = [], [], []
    for m in manifest:
        data = (raw / m["file"]).read_bytes()
        title, text = pdf_text(data) if m["kind"] == "pdf" else html_text(data, cl["start_prefix"], cl["stop_markers"])
        text, counts = redact(text, engine, pii["redact"], keep)
        title, _ = redact(title, engine, pii["redact"], keep)
        slug = m["sha256"][:12]
        (out / "clean" / f"{slug}.txt").write_text(text, encoding="utf-8")
        parts = chunk_lines(text, ck["max_words"], ck["overlap_words"])
        for i, part in enumerate(parts):
            chunks.append({"id": f"{slug}-{i:03d}", "url": m["url"], "title": title, "group": m["group"],
                           "sha256": m["sha256"], "retrieved_at": m["retrieved_at"], "chunk": i, "text": part})
        found = leaks(text, keep)
        all_leaks += found
        docs.append({"url": m["url"], "title": title, "group": m["group"], "kind": m["kind"],
                     "sha256": m["sha256"], "retrieved_at": m["retrieved_at"], "words": len(text.split()),
                     "chunks": len(parts), "redacted": counts, "leaks": len(found)})
    (out / "docs.jsonl").write_text("\n".join(json.dumps(d) for d in docs) + "\n")
    (out / "chunks.jsonl").write_text("\n".join(json.dumps(c, ensure_ascii=False) for c in chunks) + "\n",
                                      encoding="utf-8")

    print(f"== step 6.1 intake: fetch -> clean -> PII redaction -> chunks | commit {git_hash()} ==")
    print(f"approved {len(approved_urls(cfg))} | skipped {len(skipped)} | ingested {len(docs)} "
          f"(html {sum(d['kind'] == 'html' for d in docs)}, pdf {sum(d['kind'] == 'pdf' for d in docs)})")
    print(f"\n{'group':17s} {'words':>6s} {'chunks':>6s}  {'PERSON':>6s} {'EMAIL':>5s} {'PHONE':>5s}  title")
    for d in docs:
        r = d["redacted"]
        print(f"{d['group']:17s} {d['words']:6d} {d['chunks']:6d}  {r.get('PERSON', 0):6d} {r.get('EMAIL_ADDRESS', 0):5d} "
              f"{r.get('PHONE_NUMBER', 0):5d}  {d['title'][:48]}")
    tot = {k: sum(d["redacted"].get(k, 0) for d in docs) for k in ["PERSON", "EMAIL_ADDRESS", "PHONE_NUMBER"]}
    words = [len(c["text"].split()) for c in chunks]
    print(f"\ntotal words {sum(d['words'] for d in docs)} | chunks {len(chunks)} (words per chunk median "
          f"{sorted(words)[len(words) // 2]}, max {max(words)}) | redacted {tot}")
    print(f"PII leak check after redaction: {len(all_leaks)} suspicious strings (should be 0)")
    print(f"office contacts kept on purpose: {keep}")
    print(f"time {time.perf_counter() - t0:.0f} s | saved: {out}/docs.jsonl | chunks.jsonl | clean/*.txt")


if __name__ == "__main__":
    main()
