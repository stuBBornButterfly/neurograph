import json
import os
import tempfile
import threading
import time
import uuid
from pathlib import Path

from neurograph.config import REPO_ROOT, load_config

DISCLAIMER = ("Research prototype. Not a diagnostic tool and not a counsellor. In an emergency in Bangladesh call 999; "
              "The university students can talk to the psychosocial counsellors (Department of Students' Welfare, Room 301).")


def setting(value, env: str):
    return os.environ.get(env) or value


def app_dir(cfg: dict, key: str, env: str) -> Path:
    d = Path(setting(cfg[key]["dir"], env))
    return d if d.is_absolute() else REPO_ROOT / d


def make_llm(cfg: dict, rag: dict):
    # (policy model, study/literature model, description); the QLoRA adapter is trained on university policy prompts only,
    # so by default the research routes use the base model
    c = cfg["llm"]
    backend, base_research = setting(c["backend"], "LLM_BACKEND"), setting(c["research_model"], "RESEARCH_MODEL") == "base"
    if backend == "llamacpp":
        from huggingface_hub import hf_hub_download
        from neurograph.agent.backends import LlamaCppLLM
        path = setting(c["gguf_path"], "GGUF_PATH")
        if not path:
            repo = setting(c["gguf_repo"], "GGUF_REPO")
            if not repo:
                raise RuntimeError("set GGUF_REPO (Hugging Face model repo with the GGUF) or GGUF_PATH (local file)")
            path = hf_hub_download(repo, setting(c["gguf_file"], "GGUF_FILE"))
        llm = LlamaCppLLM(path, c["max_new_tokens"], n_ctx=c["n_ctx"])
        if not base_research:
            return llm, llm, f"{Path(path).name} (llama.cpp, CPU)"
        base = setting(c["base_gguf_path"], "BASE_GGUF_PATH") or hf_hub_download(c["base_gguf_repo"], c["base_gguf_file"])
        return (llm, LlamaCppLLM(base, c["max_new_tokens"], n_ctx=c["n_ctx"]),
                f"{Path(path).name} for university questions, {Path(base).name} for research (llama.cpp, CPU)")
    if backend == "transformers":
        from neurograph.agent.llm import AdapterOff, LocalLLM
        adapter = setting(c["adapter"], "ADAPTER") or None
        llm = LocalLLM(rag["llm"]["model"], c["max_new_tokens"], adapter=adapter)
        name = rag["llm"]["model"] + (" + QLoRA adapter" if adapter else "")
        if adapter and base_research:
            return llm, AdapterOff(llm), name + " (adapter on university questions only)"
        return llm, llm, name
    if backend == "api":
        from neurograph.agent.backends import ApiLLM
        r = load_config("benchmark")["reference"]
        if r["external"]:  # the study route sends results derived from participant data to the model
            raise RuntimeError("the demo serves study answers, so it only runs on a self-hosted model")
        llm = ApiLLM(r["kind"], r["base_url"], r["model"], c["max_new_tokens"], r["api_key_env"])
        return llm, llm, r["name"]
    raise ValueError(f"unknown LLM backend {backend!r}")


def make_tools(cfg: dict, data: Path):
    mode = setting(cfg["graph"], "GRAPH")
    if mode == "neo4j" or (mode == "auto" and os.environ.get("NEO4J_URI")):
        from neurograph.agent.tools import GraphTools
        from neurograph.graph.client import get_driver
        try:
            return GraphTools(get_driver(), load_config("policy_graph")), "Neo4j (live)"
        except Exception as e:  # e.g. a paused AuraDB Free instance: answer from the snapshot instead
            if mode == "neo4j":
                raise
            print(f"Neo4j not reachable ({type(e).__name__}), using the snapshot")
    from neurograph.agent.snapshot import SnapshotTools
    snap = SnapshotTools(data / "graph_snapshot.json")
    return snap, f"graph snapshot of {snap.snap['created'][:10]}"


class Assistant:
    # the same LangGraph agent as the evaluation (safety -> router -> retrieval/graph -> LLM -> verifier)
    def __init__(self):
        from neurograph.agent.verifier import Verifier
        from neurograph.agent.workflow import build_agent
        from neurograph.ingest.index_qdrant import HybridIndex
        t0 = time.perf_counter()
        cfg, rag = load_config("app"), load_config("rag")
        data = app_dir(cfg, "data", "APP_DATA")
        chunks = [json.loads(l) for l in (data / "chunks.jsonl").read_text(encoding="utf-8").splitlines()]
        e, q = rag["embedding"], rag["qdrant"]
        self.index = HybridIndex(chunks, Path(tempfile.gettempdir()) / "neurograph_qdrant", q["collection"],
                                 e["model"], e["query_prefix"])
        if not self.index.ready():
            self.index.build(e["batch_size"])
        self.tools, self.graph_name = make_tools(cfg, data)
        self.llm, self.research_llm, self.llm_name = make_llm(cfg, rag)
        self.llms = [self.llm] if self.research_llm is self.llm else [self.llm, self.research_llm]
        self.agent = build_agent(self.index, self.tools, self.llm, Verifier(rag["nli"]["model"], rag["nli"]["min_entail"]),
                                 rag, self.research_llm)
        self.lock = threading.Lock()  # one answer at a time: llama.cpp and the GPU model are not thread-safe
        self.info = {"llm": self.llm_name, "graph": self.graph_name, "chunks": len(chunks),
                     "startup_s": round(time.perf_counter() - t0, 1)}

    def ask(self, question: str) -> dict:
        t = time.perf_counter()
        with self.lock:
            n_usage = [len(m.usage) for m in self.llms]
            out = self.agent.invoke({"question": question})
            usage = [u for m, k in zip(self.llms, n_usage) for u in m.usage[k:]]
        chk = out.get("check", {})
        return {"id": uuid.uuid4().hex[:12], "route": out["route"], "answer": out["answer"],
                "verdict": chk.get("verdict", "referral" if out["route"].startswith("safety") else "-"),
                "entailed_share": chk.get("entailed_share"), "unsupported_numbers": chk.get("unsupported_numbers", []),
                "sources": sorted({p["url"] for p in out.get("passages", []) if p.get("url", "").startswith("http")}),
                "prompt_tokens": sum(u["prompt_tokens"] for u in usage),
                "completion_tokens": sum(u["completion_tokens"] for u in usage),
                "seconds": round(time.perf_counter() - t, 2)}
