import argparse
import glob
import json
import random
import time
import warnings
from pathlib import Path

import pandas as pd

from neurograph.config import git_hash, load_config
from neurograph.finetune import make_dataset as md
from neurograph.ingest.index_qdrant import HybridIndex


class EchoTeacher:
    # stand-in for local smoke tests: one question per passage built from its first line
    def chat_many(self, system: str, users: list[str], **_) -> list[str]:
        out = []
        for u in users:
            line = next((l for l in u.split("\n")[1:] if len(l.split()) > 5), u.split("\n")[1])
            out.append(json.dumps([{"question": f"What does university say about {line.split()[0].lower()} here?",
                                    "answer": line[:200]}]))
        return out


def find_chunks(cfg: dict) -> Path:
    hits = glob.glob(cfg["chunks_glob"], recursive=True)
    return Path(hits[0]) if hits else Path(load_config("paths")["paths"]["derivatives_dir"]) / "institutional" / "chunks.jsonl"


def main() -> None:
    warnings.filterwarnings("ignore")
    ap = argparse.ArgumentParser()
    ap.add_argument("--chunks", default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--echo-teacher", action="store_true", help="smoke test without a language model")
    args = ap.parse_args()
    t0 = time.perf_counter()
    cfg, rag = load_config("finetune"), load_config("rag")
    d = cfg["dataset"]
    src = Path(args.chunks) if args.chunks else find_chunks(cfg)
    out = Path(args.out) if args.out else Path(load_config("paths")["paths"]["derivatives_dir"]) / cfg["out"]
    out.mkdir(parents=True, exist_ok=True)
    chunks = [json.loads(l) for l in src.read_text(encoding="utf-8").splitlines()]
    chunks = [c for c in chunks if len(c["text"].split()) >= 40]  # tiny chunks (menus, one-liners) give poor questions

    if args.echo_teacher:
        teacher, name = EchoTeacher(), "echo (smoke test)"
    else:
        from neurograph.agent.llm import LocalLLM
        teacher, name = LocalLLM(cfg["teacher_model"], d["gen_max_new_tokens"]), cfg["teacher_model"]
    prompts = [f"Passage from \"{c['title']}\":\n{c['text']}\n\nWrite {d['questions_per_chunk']} question-answer pairs."
               for c in chunks]
    raw = teacher.chat_many(md.QA_SYSTEM, prompts, batch=d["gen_batch"])
    pairs = [(c, qa) for c, r in zip(chunks, raw) for qa in md.parse_qa(r, d["questions_per_chunk"])]
    t_gen = time.perf_counter() - t0
    del teacher

    from neurograph.literature.verify_nli import ClaimVerifier
    nli = ClaimVerifier(d["nli_model"], d["min_entail"])
    ent = nli.entail([(c["text"], qa["answer"]) for c, qa in pairs])
    kept = [(c, qa) for (c, qa), e in zip(pairs, ent) if e >= d["min_entail"]]

    e, q = rag["embedding"], rag["qdrant"]
    index = HybridIndex(chunks, Path("/tmp") / "qdrant_ft" if Path("/tmp").exists() else out / "qdrant_ft",
                        q["collection"], e["model"], e["query_prefix"])
    index.build(e["batch_size"])
    test_ids = md.split_chunks([c["id"] for c in chunks], d["test_share"], cfg["seed"])
    rng = random.Random(cfg["seed"])
    rows = []
    for c, qa in kept:
        hits = [chunks[i] for i in index.dense(qa["question"], 12)]
        negative = rng.random() < d["negative_share"]
        others = [h for h in hits if h["url"] != c["url"]]  # other pages only, so they cannot hold the answer
        ex = md.build_example(qa, c, others[:d["distractors"] + negative], negative, rng)
        rows.append({**ex, "split": "test" if c["id"] in test_ids else "train"})
    df = pd.DataFrame(rows)
    for split in ("train", "test"):
        df[df.split == split].to_json(out / f"{split}.jsonl", orient="records", lines=True, force_ascii=False)

    print(f"== step 7.1 Q&A dataset from university pages | commit {git_hash()} ==")
    print(f"teacher {name} | chunks {len(chunks)} | pairs parsed {len(pairs)} | NLI-kept {len(kept)} "
          f"({len(kept) / max(len(pairs), 1):.0%}, answer entailed by its chunk) | generation {t_gen:.0f} s")
    print(df.groupby(["split", "negative"]).size().rename("examples").to_string())
    print(f"chunks: train {df[df.split == 'train'].chunk_id.nunique()} | test {df[df.split == 'test'].chunk_id.nunique()} "
          f"(no chunk in both: {not set(df[df.split == 'train'].chunk_id) & set(df[df.split == 'test'].chunk_id)})")
    for r in df[~df.negative].head(3).itertuples():
        print(f"  Q: {r.question}\n  A: {r.target}")
    print(f"time {time.perf_counter() - t0:.0f} s | saved: {out}/train.jsonl, test.jsonl")


if __name__ == "__main__":
    main()
