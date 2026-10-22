import argparse
import glob
import json
import re
import time
import warnings
from pathlib import Path

import pandas as pd

from neurograph.agent.workflow import POLICY_SYSTEM
from neurograph.config import git_hash, load_config
from neurograph.finetune.make_dataset import REFUSAL, token_f1


def score(df: pd.DataFrame, answers: list[str], secs: float, chunk_text: dict, nli, min_entail: float) -> dict:
    pos, neg = df[~df.negative], df[df.negative]
    a_pos = [a for a, n in zip(answers, df.negative) if not n]
    a_neg = [a for a, n in zip(answers, df.negative) if n]
    clean = [re.sub(r"\[\d\]", "", a).strip() for a in a_pos]
    ent = nli.entail([(chunk_text[c], a or ".") for c, a in zip(pos.chunk_id, clean)])
    return {"token_f1": sum(token_f1(a, r) for a, r in zip(clean, pos.reference)) / len(pos),
            "faithful": sum(e >= min_entail for e in ent) / len(pos),
            "cites_right_passage": sum(f"[{int(p)}]" in a for a, p in zip(a_pos, pos.gold_pos)) / len(pos),  # read as float
            "wrongly_refuses": sum(bool(REFUSAL.search(a)) for a in a_pos) / len(pos),
            "says_not_in_passages": sum(bool(REFUSAL.search(a)) for a in a_neg) / max(len(neg), 1),
            "words": sum(len(a.split()) for a in answers) / len(answers), "sec_per_answer": secs / len(answers)}


def main() -> None:
    warnings.filterwarnings("ignore")
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default=None, help="default: <derivatives>/qlora")
    ap.add_argument("--chunks", default=None)
    ap.add_argument("--base-model", default=None, help="smoke test: a smaller model than the config's")
    args = ap.parse_args()
    t0 = time.perf_counter()
    cfg = load_config("finetune")
    cfg["base_model"] = args.base_model or cfg["base_model"]
    out = Path(args.dir) if args.dir else Path(load_config("paths")["paths"]["derivatives_dir"]) / cfg["out"]
    test = pd.read_json(out / "test.jsonl", lines=True)
    src = [args.chunks] if args.chunks else glob.glob(cfg["chunks_glob"], recursive=True)
    src = Path(src[0]) if src else Path(load_config("paths")["paths"]["derivatives_dir"]) / "institutional" / "chunks.jsonl"
    chunk_text = {c["id"]: c["text"] for c in map(json.loads, src.read_text(encoding="utf-8").splitlines())}

    from neurograph.agent.llm import LocalLLM
    from neurograph.literature.verify_nli import ClaimVerifier
    llm = LocalLLM(cfg["base_model"], cfg["eval"]["max_new_tokens"], adapter=str(out / "adapter"))
    users = test.user.tolist()
    res, answers = {}, {}
    for name in ("base", "fine-tuned"):
        t = time.perf_counter()
        if name == "base":
            with llm.model.disable_adapter():  # same weights without the LoRA add-ons = the base model
                answers[name] = llm.chat_many(POLICY_SYSTEM, users, batch=cfg["eval"]["batch"])
        else:
            answers[name] = llm.chat_many(POLICY_SYSTEM, users, batch=cfg["eval"]["batch"])
        res[name] = time.perf_counter() - t
    del llm
    nli = ClaimVerifier(cfg["dataset"]["nli_model"], cfg["dataset"]["min_entail"])
    table = pd.DataFrame({n: score(test, answers[n], res[n], chunk_text, nli, cfg["dataset"]["min_entail"])
                          for n in answers}).T.round(3)
    table.to_csv(out / "eval.csv")
    test.assign(base=answers["base"], finetuned=answers["fine-tuned"]).to_json(
        out / "eval_answers.jsonl", orient="records", lines=True, force_ascii=False)

    print(f"== step 7.3 base vs QLoRA on held-out chunks | commit {git_hash()} ==")
    print(f"test examples {len(test)} ({int(test.negative.sum())} with the answer removed) from "
          f"{test.chunk_id.nunique()} chunks never seen in training | same prompt, greedy decoding")
    print(table.to_string())
    for k in range(2):
        r = test[~test.negative].iloc[k]
        i = test.index.get_loc(r.name)
        print(f"\nQ: {r.question}\n  reference: {r.reference}\n  base: {answers['base'][i][:250]}"
              f"\n  fine-tuned: {answers['fine-tuned'][i][:250]}")
    write_card(out, cfg, table, len(test))
    print(f"\ntime {time.perf_counter() - t0:.0f} s | saved: {out / 'eval.csv'}, {out / 'model_card.md'}")


def gpu_name() -> str:
    import torch
    return torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"


def write_card(out: Path, cfg: dict, table: pd.DataFrame, n_test: int) -> None:
    lc, tc = cfg["lora"], cfg["train"]
    rows = "\n".join(f"| {m} | {table.loc['base', m]} | {table.loc['fine-tuned', m]} |" for m in table.columns)
    n_train = sum(1 for _ in (out / "train.jsonl").open(encoding="utf-8"))
    (out / "model_card.md").write_text(f"""# NeuroGraph university assistant adapter (QLoRA)

LoRA adapter for **{cfg['base_model']}** that answers university student questions from retrieved
passages, cites the passage it used, and says so when the passages do not contain the answer.

- **Training data:** {n_train} question-answer examples written by {cfg['teacher_model']} (self-hosted) from 19 public
  university pages after PII redaction; each answer was kept only if an NLI model ({cfg['dataset']['nli_model']}) found it
  entailed by its source chunk. {cfg['dataset']['negative_share']:.0%} of examples have the right passage removed and
  teach the model to say the passages do not answer the question.
- **Held-out test:** {n_test} examples from {cfg['dataset']['test_share']:.0%} of chunks that never appear in training.
- **Method:** QLoRA, 4-bit NF4 base, LoRA r {lc['r']}, alpha {lc['alpha']}, dropout {lc['dropout']} on
  {', '.join(lc['targets'])}; {tc['epochs']} epochs, lr {tc['lr']}, effective batch {tc['batch'] * tc['grad_accum']};
  one {gpu_name()}.

| metric (held-out) | base | fine-tuned |
|---|---|---|
{rows}

`token_f1`: word overlap with the reference answer. `faithful`: answer entailed by the right chunk (NLI).
`cites_right_passage`: cites the passage that holds the answer. `wrongly_refuses`: says "not in the passages" when it is.
`says_not_in_passages`: correct refusal when the answer was removed.

**Limits:** the questions are synthetic and written by another model, so they are cleaner than real student
questions; the test pages are the same 19 pages (different chunks); the university pages change over time. Not for
medical, psychological or legal advice. Research prototype.
""", encoding="utf-8")


if __name__ == "__main__":
    main()
