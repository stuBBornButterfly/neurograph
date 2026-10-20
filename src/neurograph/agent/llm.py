import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


class LocalLLM:
    # self-hosted chat model (no outside API); greedy decoding so answers are reproducible
    def __init__(self, model_name: str, max_new_tokens: int):
        self.tok = AutoTokenizer.from_pretrained(model_name)
        dtype = torch.float16 if torch.cuda.is_available() else torch.float32
        self.model = AutoModelForCausalLM.from_pretrained(model_name, dtype=dtype,
                                                          device_map="auto" if torch.cuda.is_available() else None)
        self.max_new_tokens = max_new_tokens

    def chat(self, system: str, user: str) -> str:
        msgs = [{"role": "system", "content": system}, {"role": "user", "content": user}]
        ids = self.tok.apply_chat_template(msgs, add_generation_prompt=True, return_tensors="pt",
                                           return_dict=True).to(self.model.device)
        with torch.no_grad():
            out = self.model.generate(**ids, max_new_tokens=self.max_new_tokens, do_sample=False)
        return self.tok.decode(out[0, ids["input_ids"].shape[1]:], skip_special_tokens=True).strip()
