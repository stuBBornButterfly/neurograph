import os
import time

import requests


class ApiLLM:
    # chat model behind an HTTP API: a self-hosted Ollama server or any OpenAI-compatible service (commercial APIs)
    # external=True marks an outside service: callers must only send public text (university pages, benchmark questions)
    def __init__(self, kind: str, base_url: str, model: str, max_new_tokens: int, api_key_env: str | None = None,
                 external: bool = False, timeout: int = 300, seed: int = 97):
        if kind not in ("ollama", "openai"):
            raise ValueError(f"unknown API kind {kind!r} (ollama | openai)")
        self.kind, self.base_url, self.model, self.external = kind, base_url.rstrip("/"), model, external
        self.max_new_tokens, self.timeout, self.seed = max_new_tokens, timeout, seed
        self.key = os.environ.get(api_key_env, "") if api_key_env else ""  # from the environment, never from configs
        if api_key_env and not self.key:
            raise RuntimeError(f"set {api_key_env} (Kaggle / Hugging Face Secrets or a local .env)")
        self.usage: list[dict] = []

    def _post(self, msgs: list[dict], max_new_tokens: int) -> tuple[str, int, int]:
        if self.kind == "ollama":  # native API: thinking off, greedy, fixed seed
            r = requests.post(f"{self.base_url}/api/chat", timeout=self.timeout, json={
                "model": self.model, "messages": msgs, "stream": False, "think": False,
                "options": {"temperature": 0, "num_predict": max_new_tokens, "seed": self.seed}})
            r.raise_for_status()
            d = r.json()
            return d["message"]["content"], d.get("prompt_eval_count", 0), d.get("eval_count", 0)
        r = requests.post(f"{self.base_url}/chat/completions", timeout=self.timeout,
                          headers={"Authorization": f"Bearer {self.key}"} if self.key else {},
                          json={"model": self.model, "messages": msgs, "temperature": 0, "max_tokens": max_new_tokens})
        r.raise_for_status()
        d = r.json()
        u = d.get("usage") or {}
        return d["choices"][0]["message"]["content"] or "", u.get("prompt_tokens", 0), u.get("completion_tokens", 0)

    def chat_many(self, system: str, users: list[str], batch: int = 1, max_new_tokens: int | None = None) -> list[str]:
        out = []
        for u in users:  # one request at a time: latency per answer is what the benchmark reports
            t = time.perf_counter()
            text, p_tok, c_tok = self._post([{"role": "system", "content": system}, {"role": "user", "content": u}],
                                            max_new_tokens or self.max_new_tokens)
            self.usage.append({"prompt_tokens": p_tok, "completion_tokens": c_tok, "seconds": time.perf_counter() - t})
            out.append(text.strip())
        return out

    def chat(self, system: str, user: str) -> str:
        return self.chat_many(system, [user])[0]


class LlamaCppLLM:
    # GGUF model on CPU through llama.cpp (the demo Space); the chat template comes from the GGUF file
    def __init__(self, model_path: str, max_new_tokens: int, n_ctx: int = 4096, n_threads: int | None = None):
        from llama_cpp import Llama
        self.llm = Llama(model_path=model_path, n_ctx=n_ctx, n_threads=n_threads or os.cpu_count(), seed=97,
                         verbose=False)
        self.max_new_tokens, self.external = max_new_tokens, False
        self.usage: list[dict] = []

    def chat_many(self, system: str, users: list[str], batch: int = 1, max_new_tokens: int | None = None) -> list[str]:
        out = []
        for u in users:
            t = time.perf_counter()
            r = self.llm.create_chat_completion([{"role": "system", "content": system}, {"role": "user", "content": u}],
                                                temperature=0, max_tokens=max_new_tokens or self.max_new_tokens)
            self.usage.append({"prompt_tokens": r["usage"]["prompt_tokens"],
                               "completion_tokens": r["usage"]["completion_tokens"],
                               "seconds": time.perf_counter() - t})
            out.append(r["choices"][0]["message"]["content"].strip())
        return out

    def chat(self, system: str, user: str) -> str:
        return self.chat_many(system, [user])[0]
