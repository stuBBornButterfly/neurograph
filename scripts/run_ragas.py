import argparse
import asyncio
import math
import re
import time
import warnings
from pathlib import Path

import pandas as pd

from neurograph.config import git_hash, load_config

PASSAGE = re.compile(r"\n\n(?=\[\d+\] )")


def clean_answer(text: str) -> str:
    # the claims the model made: no citation marks, no verifier note or source list appended by the agent
    text = re.split(r"\n\n\(Note: I could not fully verify|\n\nSources: ", text)[0]
    return re.sub(r"\s*\[\d+\]", "", text).strip()


def prompt_passages(user: str) -> list[str]:
    body = user.rsplit("\n\nQuestion: ", 1)[0]
    return [re.sub(r"^\[\d+\] ", "", p) for p in PASSAGE.split(body) if p.strip()]


def samples(bench: Path, heldout_n: int, seed: int) -> pd.DataFrame:
    gold = pd.read_json(bench / "gold_answers.jsonl", lines=True)
    gold = gold[gold.route_gold.isin(["policy", "study", "literature"])]
    rows = [{"set": "gold", "model": r.model, "id": r.id, "route": r.route_gold, "question": r.question,
             "answer": clean_answer(r.answer), "contexts": r.contexts, "reference": None} for r in gold.itertuples()]
    held = pd.read_json(bench / "heldout_answers.jsonl", lines=True)
    held = held[~held.negative & (held.model != "routed")]  # routed = fine-tuned on these (policy) prompts
    keep = held.drop_duplicates("question").sample(n=min(heldout_n, held.question.nunique()), random_state=seed).question
    for r in held[held.question.isin(set(keep))].itertuples():
        rows.append({"set": "heldout", "model": r.model, "id": r.chunk_id, "route": "policy", "question": r.question,
                     "answer": clean_answer(r.answer), "contexts": prompt_passages(r.user), "reference": r.reference})
    return pd.DataFrame(rows)


async def score_all(df: pd.DataFrame, metrics: dict, parallel: int) -> list[dict]:
    sem = asyncio.Semaphore(parallel)

    async def one(r) -> dict:
        out = {}
        async with sem:
            for name, (metric, kw) in metrics.items():
                args = {"user_input": r.question, "response": r.answer or ".", "retrieved_contexts": r.contexts or ["-"],
                        "reference": r.reference}
                if name == "context_precision" and (r.reference is None or pd.isna(r.reference)):  # None -> NaN in pandas
                    continue  # needs a reference answer (held-out set only)
                try:
                    res = await metric.ascore(**{k: args[k] for k in kw})
                    out[name] = float(res.value) if res.value is not None else math.nan
                except Exception as e:  # a judge reply that does not parse counts as missing, and is reported
                    out[name], out[f"{name}_error"] = math.nan, type(e).__name__
        return out

    return await asyncio.gather(*(one(r) for r in df.itertuples()))


def main() -> None:
    warnings.filterwarnings("ignore")
    ap = argparse.ArgumentParser()
    ap.add_argument("--parallel", type=int, default=4)
    ap.add_argument("--limit", type=int, default=None, help="smoke test: score only the first N samples")
    args = ap.parse_args()
    t0 = time.perf_counter()
    from openai import AsyncOpenAI
    from ragas.embeddings import embedding_factory
    from ragas.llms import llm_factory
    from ragas.metrics.collections import AnswerRelevancy, ContextPrecisionWithReference, Faithfulness

    b, ft = load_config("benchmark"), load_config("finetune")
    j = b["judge"]
    bench = Path(load_config("paths")["paths"]["derivatives_dir"]) / "benchmark"
    df = samples(bench, j["heldout_n"], ft["seed"])
    df = df.head(args.limit) if args.limit else df
    llm = llm_factory(j["model"], client=AsyncOpenAI(base_url=j["base_url"], api_key="local"), temperature=0,
                      max_tokens=2048)
    emb = embedding_factory("huggingface", model=j["embeddings"])
    metrics = {"faithfulness": (Faithfulness(llm=llm), ["user_input", "response", "retrieved_contexts"]),
               "answer_relevancy": (AnswerRelevancy(llm=llm, embeddings=emb), ["user_input", "response"]),
               "context_precision": (ContextPrecisionWithReference(llm=llm), ["user_input", "reference", "retrieved_contexts"])}
    scores = pd.DataFrame(asyncio.run(score_all(df, metrics, args.parallel)), index=df.index)
    res = pd.concat([df.drop(columns=["contexts"]), scores.reindex(columns=list(dict.fromkeys([*metrics, *scores.columns])))],
                    axis=1)  # a metric with no score at all still gets a column
    res.to_json(bench / "ragas_scores.jsonl", orient="records", lines=True, force_ascii=False)
    agg = res.groupby(["set", "model"], sort=False)[list(metrics)].agg(["mean", "std", "count"]).round(3)
    agg.to_csv(bench / "ragas.csv")

    print(f"== Phase 9 RAGAS ({j['model']} as judge, {j['embeddings']} embeddings) | commit {git_hash()} ==")
    print(f"samples {len(df)}: gold questions (policy, study, literature routes) + {j['heldout_n']} held-out questions "
          f"per model | context_precision needs a reference answer, so held-out only")
    with pd.option_context("display.width", 250):
        for name in metrics:
            t = agg[name].assign(cell=lambda x: x["mean"].map("{:.2f}".format) + " ± " + x["std"].map("{:.2f}".format)
                                 + " (n " + x["count"].astype(int).astype(str) + ")")["cell"].unstack(0)
            print(f"\n{name}:\n{t.to_string()}")
    errs = [c for c in res.columns if c.endswith("_error")]
    print("\njudge failures (no score): " + (", ".join(f"{c[:-6]} {int(res[c].notna().sum())}" for c in errs) or "none"))
    print(f"time {(time.perf_counter() - t0) / 60:.1f} min | saved: {bench / 'ragas.csv'}")


if __name__ == "__main__":
    main()
