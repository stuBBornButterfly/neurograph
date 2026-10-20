import argparse
import json
import math
import time
import warnings
from pathlib import Path

import torch
from transformers import (AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, Trainer, TrainingArguments,
                          set_seed)

from neurograph.agent.workflow import POLICY_SYSTEM
from neurograph.config import git_hash, load_config


def encode(tok, ex: dict, max_len: int) -> dict | None:
    # loss only on the answer: prompt tokens get label -100
    msgs = [{"role": "system", "content": POLICY_SYSTEM}, {"role": "user", "content": ex["user"]}]
    text = tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False)
    prompt = tok(text, add_special_tokens=False)["input_ids"]  # same tokens as LocalLLM sees at inference
    answer = tok(ex["target"] + "<|im_end|>", add_special_tokens=False)["input_ids"]
    ids = list(prompt) + answer
    if len(ids) > max_len:
        return None  # dropped, never cut: a cut context could lose the answer
    return {"input_ids": ids, "labels": [-100] * len(prompt) + answer}


def collate(pad_id: int):
    def fn(batch: list[dict]) -> dict:
        n = max(len(b["input_ids"]) for b in batch)
        ids = torch.full((len(batch), n), pad_id)
        labels = torch.full((len(batch), n), -100)
        mask = torch.zeros((len(batch), n), dtype=torch.long)
        for i, b in enumerate(batch):
            k = len(b["input_ids"])
            ids[i, :k], labels[i, :k], mask[i, :k] = torch.tensor(b["input_ids"]), torch.tensor(b["labels"]), 1
        return {"input_ids": ids, "labels": labels, "attention_mask": mask}
    return fn


def main() -> None:
    warnings.filterwarnings("ignore")
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default=None, help="default: <derivatives>/qlora")
    ap.add_argument("--base-model", default=None, help="smoke test: a smaller model than the config's")
    ap.add_argument("--max-steps", type=int, default=-1, help="smoke test: stop early")
    args = ap.parse_args()
    t0 = time.perf_counter()
    cfg = load_config("finetune")
    cfg["base_model"] = args.base_model or cfg["base_model"]
    lc, tc = cfg["lora"], cfg["train"]
    out = Path(args.dir) if args.dir else Path(load_config("paths")["paths"]["derivatives_dir"]) / cfg["out"]
    set_seed(cfg["seed"])
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

    tok = AutoTokenizer.from_pretrained(cfg["base_model"])
    data = {s: [json.loads(l) for l in (out / f"{s}.jsonl").read_text(encoding="utf-8").splitlines()]
            for s in ("train", "test")}
    enc = {s: [e for e in (encode(tok, x, tc["max_len"]) for x in rows) if e] for s, rows in data.items()}

    cuda = torch.cuda.is_available()
    quant = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True,
                               bnb_4bit_compute_dtype=torch.float16) if cuda else None
    model = AutoModelForCausalLM.from_pretrained(cfg["base_model"], quantization_config=quant,
                                                 device_map={"": 0} if cuda else None,
                                                 dtype=torch.float16 if cuda else torch.float32)
    if cuda:
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
    model = get_peft_model(model, LoraConfig(r=lc["r"], lora_alpha=lc["alpha"], lora_dropout=lc["dropout"],
                                             target_modules=lc["targets"], task_type="CAUSAL_LM"))
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_all = sum(p.numel() for p in model.parameters())

    steps = math.ceil(len(enc["train"]) / (tc["batch"] * tc["grad_accum"])) * tc["epochs"]
    targs = TrainingArguments(output_dir=str(out / "trainer"), num_train_epochs=tc["epochs"], max_steps=args.max_steps,
                              learning_rate=tc["lr"], per_device_train_batch_size=tc["batch"],
                              per_device_eval_batch_size=tc["batch"], gradient_accumulation_steps=tc["grad_accum"],
                              warmup_steps=max(1, round(tc["warmup_ratio"] * steps)), lr_scheduler_type="cosine", logging_steps=5,
                              eval_strategy="epoch" if args.max_steps < 0 else "no", save_strategy="no",
                              fp16=cuda, optim="paged_adamw_8bit" if cuda else "adamw_torch", report_to="none",
                              seed=cfg["seed"], remove_unused_columns=False, gradient_checkpointing=cuda)
    trainer = Trainer(model=model, args=targs, train_dataset=enc["train"], eval_dataset=enc["test"],
                      data_collator=collate(tok.pad_token_id))
    trainer.train()
    model.save_pretrained(out / "adapter")
    tok.save_pretrained(out / "adapter")
    logs = trainer.state.log_history
    (out / "train_log.json").write_text(json.dumps(logs, indent=1))

    print(f"== step 7.2 QLoRA fine-tune | commit {git_hash()} ==")
    print(f"base {cfg['base_model']} in 4-bit NF4 | LoRA r {lc['r']}, alpha {lc['alpha']} on {len(lc['targets'])} "
          f"projections | trainable {n_train / 1e6:.1f} M of {n_all / 1e6:.0f} M parameters ({n_train / n_all:.2%})")
    print(f"examples train {len(enc['train'])} / test {len(enc['test'])} (dropped as too long: "
          f"{len(data['train']) - len(enc['train'])} / {len(data['test']) - len(enc['test'])}) | epochs {tc['epochs']} "
          f"| effective batch {tc['batch'] * tc['grad_accum']} | steps {trainer.state.global_step}")
    train_loss = [l["loss"] for l in logs if "loss" in l]
    if train_loss:
        print(f"train loss first {train_loss[0]:.3f} -> last {train_loss[-1]:.3f}")
    for l in logs:
        if "eval_loss" in l:
            print(f"  epoch {l['epoch']:.0f}: held-out loss {l['eval_loss']:.3f}")
    if cuda:
        print(f"peak GPU memory {torch.cuda.max_memory_allocated() / 1e9:.1f} GB")
    print(f"time {time.perf_counter() - t0:.0f} s | saved: {out / 'adapter'}")


if __name__ == "__main__":
    main()
