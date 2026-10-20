import time
from pathlib import Path

import pandas as pd

from neurograph.config import REPO_ROOT, git_hash, load_config
from scripts.run_ragas import prompt_passages

RATINGS = ["correct (2 yes / 1 partly / 0 no)", "grounded (2 / 1 / 0)", "safe_and_appropriate (1 / 0)",
           "helpful (1-5)", "notes"]
# 50 answers: 6 safety messages + per model 3 policy, 2 study, 1 literature gold answers + 4 held-out + 1 "answer removed"
PER_MODEL = {"policy": 3, "study": 2, "literature": 1}


def take(df: pd.DataFrame, n: int, seed: int) -> pd.DataFrame:
    return df.sample(n=min(n, len(df)), random_state=seed)


def main() -> None:
    t0 = time.perf_counter()
    b = load_config("benchmark")
    a, seed = b["audit"], b["audit"]["seed"]
    bench = Path(load_config("paths")["paths"]["derivatives_dir"]) / "benchmark"
    # "routed" re-uses the fine-tuned answers (policy) and the base answers (research): rating it again adds nothing
    gold = pd.read_json(bench / "gold_answers.jsonl", lines=True).query("model != 'routed'")
    held = pd.read_json(bench / "heldout_answers.jsonl", lines=True).query("model != 'routed'")

    parts = [take(gold[gold.route_gold.str.startswith("safety") & (gold.model == "gguf")], 6, seed)]
    for m in gold.model.unique():
        g = gold[gold.model == m]
        parts += [take(g[g.route_gold == k], n, seed) for k, n in PER_MODEL.items()]
    gold_pick = pd.concat(parts)
    rows = [{"set": "gold", "model": r.model, "item": r.id, "route": r.route_gold, "question": r.question,
             "evidence": "\n---\n".join(r.contexts) if len(r.contexts) else "(fixed referral text; contacts from the graph)",
             "reference": "", "answer": r.answer} for r in gold_pick.itertuples()]
    for m in held.model.unique():
        h = held[held.model == m]
        for r in pd.concat([take(h[~h.negative], 4, seed), take(h[h.negative], 1, seed)]).itertuples():
            rows.append({"set": "held-out" + (" (answer removed)" if r.negative else ""), "model": m, "item": r.chunk_id,
                         "route": "policy", "question": r.question, "evidence": "\n---\n".join(prompt_passages(r.user)),
                         "reference": "" if r.negative else r.reference, "answer": r.answer})
    sheet = pd.DataFrame(rows).sample(frac=1, random_state=seed).reset_index(drop=True)  # shuffled: blind to model
    sheet.insert(0, "audit_id", [f"h{k + 1:02d}" for k in range(len(sheet))])
    out = REPO_ROOT / a["out"]
    sheet.drop(columns=["model", "item"]).assign(**{c: "" for c in RATINGS}).to_csv(out, index=False)
    sheet[["audit_id", "model", "set", "item"]].to_csv(out.with_name(out.stem + "_key.csv"), index=False)

    print(f"== Phase 9 human-audit sheet | commit {git_hash()} ==")
    print(f"{len(sheet)} answers, shuffled; the model is hidden in the sheet and listed in {out.stem}_key.csv")
    print(sheet.groupby(["model", "set"]).size().unstack(fill_value=0).to_string())
    print(f"time {time.perf_counter() - t0:.1f} s | saved: {out}")


if __name__ == "__main__":
    main()
