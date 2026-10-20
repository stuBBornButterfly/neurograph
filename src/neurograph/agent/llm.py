import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig


def gpu_gb() -> float:
    return sum(torch.cuda.get_device_properties(i).total_memory for i in range(torch.cuda.device_count())) / 1e9


class LocalLLM:
    # self-hosted chat model (no outside API); greedy decoding so answers are reproducible; optional LoRA adapter
    def __init__(self, model_name: str, max_new_tokens: int, adapter: str | None = None, device_map=None,
                 load_4bit: bool = False):
        self.tok = AutoTokenizer.from_pretrained(model_name, padding_side="left")
        cuda = torch.cuda.is_available()
        quant = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True,
                                   bnb_4bit_compute_dtype=torch.float16) if load_4bit and cuda else None
        self.model = AutoModelForCausalLM.from_pretrained(model_name, dtype=torch.float16 if cuda else torch.float32,
                                                          device_map=device_map or ("auto" if cuda else None),
                                                          quantization_config=quant)
        if adapter:
            from peft import PeftModel
            self.model = PeftModel.from_pretrained(self.model, adapter)
        self.model.eval()
        self.max_new_tokens, self.external = max_new_tokens, False
        self.usage: list[dict] = []  # tokens and seconds per answer (a batch's time is split evenly)

    def prompt(self, system: str, user: str) -> str:
        msgs = [{"role": "system", "content": system}, {"role": "user", "content": user}]
        return self.tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False)

    def chat_many(self, system: str, users: list[str], batch: int = 8, max_new_tokens: int | None = None) -> list[str]:
        out = []
        for k in range(0, len(users), batch):
            enc = self.tok([self.prompt(system, u) for u in users[k:k + batch]], return_tensors="pt", padding=True,
                           add_special_tokens=False).to(self.model.device)
            t = time.perf_counter()
            with torch.no_grad():
                gen = self.model.generate(**enc, max_new_tokens=max_new_tokens or self.max_new_tokens, do_sample=False,
                                          pad_token_id=self.tok.pad_token_id)
            new_ids = gen[:, enc["input_ids"].shape[1]:]
            secs = (time.perf_counter() - t) / len(new_ids)
            for p_len, ids in zip(enc["attention_mask"].sum(dim=1).tolist(), new_ids.tolist()):
                n = next((i + 1 for i, x in enumerate(ids) if x in (self.tok.eos_token_id, self.tok.pad_token_id)), len(ids))
                self.usage.append({"prompt_tokens": p_len, "completion_tokens": n, "seconds": secs})
            out += self.tok.batch_decode(new_ids, skip_special_tokens=True)
        return [o.strip() for o in out]

    def chat(self, system: str, user: str) -> str:
        return self.chat_many(system, [user], batch=1)[0]


class AdapterOff:
    # the loaded fine-tuned model with its LoRA add-ons switched off = the base model (same weights, same code)
    def __init__(self, llm: LocalLLM):
        self.llm, self.usage, self.external = llm, [], False

    def chat_many(self, system: str, users: list[str], batch: int = 1, max_new_tokens: int | None = None) -> list[str]:
        n = len(self.llm.usage)
        with self.llm.model.disable_adapter():
            out = self.llm.chat_many(system, users, batch, max_new_tokens)
        self.usage += self.llm.usage[n:]
        del self.llm.usage[n:]
        return out

    def chat(self, system: str, user: str) -> str:
        return self.chat_many(system, [user])[0]
