import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


class LocalLLM:
    # self-hosted chat model (no outside API); greedy decoding so answers are reproducible; optional LoRA adapter
    def __init__(self, model_name: str, max_new_tokens: int, adapter: str | None = None, device_map=None):
        self.tok = AutoTokenizer.from_pretrained(model_name, padding_side="left")
        cuda = torch.cuda.is_available()
        self.model = AutoModelForCausalLM.from_pretrained(model_name, dtype=torch.float16 if cuda else torch.float32,
                                                          device_map=device_map or ("auto" if cuda else None))
        if adapter:
            from peft import PeftModel
            self.model = PeftModel.from_pretrained(self.model, adapter)
        self.model.eval()
        self.max_new_tokens = max_new_tokens

    def prompt(self, system: str, user: str) -> str:
        msgs = [{"role": "system", "content": system}, {"role": "user", "content": user}]
        return self.tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False)

    def chat_many(self, system: str, users: list[str], batch: int = 8, max_new_tokens: int | None = None) -> list[str]:
        out = []
        for k in range(0, len(users), batch):
            enc = self.tok([self.prompt(system, u) for u in users[k:k + batch]], return_tensors="pt", padding=True,
                           add_special_tokens=False).to(self.model.device)
            with torch.no_grad():
                gen = self.model.generate(**enc, max_new_tokens=max_new_tokens or self.max_new_tokens, do_sample=False,
                                          pad_token_id=self.tok.pad_token_id)
            out += self.tok.batch_decode(gen[:, enc["input_ids"].shape[1]:], skip_special_tokens=True)
        return [o.strip() for o in out]

    def chat(self, system: str, user: str) -> str:
        return self.chat_many(system, [user], batch=1)[0]
