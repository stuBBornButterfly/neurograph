from contextlib import asynccontextmanager
from typing import Literal

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from neurograph.app.monitoring import InteractionLog
from neurograph.app.service import DISCLAIMER, Assistant, app_dir, setting
from neurograph.config import load_config

state: dict = {}


@asynccontextmanager
async def lifespan(_: FastAPI):
    cfg = load_config("app")
    state["log"] = InteractionLog(app_dir(cfg, "logs", "LOG_DIR"), setting(cfg["logs"]["hf_dataset"], "LOG_DATASET"),
                                  cfg["logs"]["push_minutes"])
    state["assistant"] = Assistant()  # loads the index, graph, LLM and NLI verifier once
    yield
    if state["log"].scheduler:
        state["log"].scheduler.trigger()  # last push of the log before shutdown


app = FastAPI(title="NeuroGraph student assistant", version="1.0", lifespan=lifespan,
              description="Answers from a university's public pages and the NeuroGraph knowledge graph, checked before they are "
                          "shown. " + DISCLAIMER)


class Question(BaseModel):
    question: str = Field(min_length=2, max_length=1000)


class Vote(BaseModel):
    id: str = Field(min_length=6, max_length=32)
    vote: Literal["up", "down"]
    comment: str = Field("", max_length=1000)


@app.get("/health")
def health() -> dict:
    a = state.get("assistant")
    return {"status": "ok" if a else "loading", **(a.info if a else {}), "disclaimer": DISCLAIMER}


@app.post("/ask")
def ask(q: Question) -> dict:  # plain def: FastAPI runs it in a worker thread, the model call blocks
    if "assistant" not in state:
        raise HTTPException(503, "still loading")
    res = state["assistant"].ask(q.question.strip())
    state["log"].answer(q.question, res, state["assistant"].llm_name)
    return res


@app.post("/feedback")
def feedback(v: Vote) -> dict:
    state["log"].feedback(v.id, v.vote, v.comment)
    return {"ok": True}


@app.get("/stats")
def stats() -> dict:
    return state["log"].stats()
