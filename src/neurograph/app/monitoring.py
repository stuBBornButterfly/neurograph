import json
import math
import os
import threading
from datetime import datetime, timezone
from pathlib import Path
from statistics import median

from neurograph.config import load_config


class InteractionLog:
    # one JSON line per answer and per feedback vote; questions are PII-redacted, distress/crisis messages not stored
    def __init__(self, folder: Path, hf_dataset: str | None = None, every_min: int = 10):
        folder.mkdir(parents=True, exist_ok=True)
        self.path = folder / "interactions.jsonl"
        self.lock = threading.Lock()
        self.scheduler = None
        if hf_dataset and os.environ.get("HF_TOKEN"):  # free Spaces lose files on restart: push the log to a dataset
            from huggingface_hub import CommitScheduler
            self.scheduler = CommitScheduler(repo_id=hf_dataset, repo_type="dataset", folder_path=folder,
                                             every=every_min, private=True, allow_patterns=["*.jsonl"])
            self.lock = self.scheduler.lock
        from neurograph.ingest.redact_pii import build_analyzer
        ing = load_config("ingest_urls")
        self.engine, self.pii = build_analyzer(ing["spacy_model"]), ing["pii"]

    def redact(self, text: str) -> str:
        from neurograph.ingest.redact_pii import redact
        return redact(text, self.engine, self.pii["redact"], self.pii["keep_office_contacts"])[0]

    def write(self, rec: dict) -> None:
        rec = {"ts": datetime.now(timezone.utc).isoformat(timespec="seconds"), **rec}
        with self.lock:
            with self.path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    def answer(self, question: str, res: dict, llm: str) -> None:
        sensitive = res["route"].startswith("safety")
        self.write({"type": "answer", "id": res["id"], "route": res["route"], "llm": llm,
                    "question": None if sensitive else self.redact(question),
                    "answer": None if sensitive else res["answer"], "verdict": res["verdict"],
                    "entailed_share": res["entailed_share"], "unsupported_numbers": len(res["unsupported_numbers"]),
                    "prompt_tokens": res["prompt_tokens"], "completion_tokens": res["completion_tokens"],
                    "seconds": res["seconds"]})

    def feedback(self, answer_id: str, vote: str, comment: str) -> None:
        self.write({"type": "feedback", "id": answer_id, "vote": vote, "comment": self.redact(comment) if comment else ""})

    def records(self) -> list[dict]:
        if not self.path.exists():
            return []
        with self.lock:
            return [json.loads(l) for l in self.path.read_text(encoding="utf-8").splitlines() if l.strip()]

    def stats(self) -> dict:
        recs = self.records()
        ans = [r for r in recs if r["type"] == "answer"]
        fb = [r for r in recs if r["type"] == "feedback"]
        checked = [r for r in ans if r["verdict"] in ("supported", "unverified")]
        secs = sorted(r["seconds"] for r in ans if not r["route"].startswith("safety"))  # referrals take ~0.1 s
        routes: dict[str, int] = {}
        for r in ans:
            routes[r["route"]] = routes.get(r["route"], 0) + 1
        return {"answers": len(ans), "routes": routes,
                "verified_share": round(sum(r["verdict"] == "supported" for r in checked) / len(checked), 2) if checked else None,
                "seconds_median": round(median(secs), 1) if secs else None,
                "seconds_p95": secs[max(0, math.ceil(0.95 * len(secs)) - 1)] if secs else None,  # nearest rank
                "feedback_up": sum(r["vote"] == "up" for r in fb), "feedback_down": sum(r["vote"] == "down" for r in fb)}
