import json
from typing import TypedDict

from langgraph.graph import END, StateGraph

from neurograph.agent import safety
from neurograph.agent.tools import STUDY_WORDS

POLICY_SYSTEM = ("You are the the university student information assistant. Answer ONLY from the numbered "
                 "passages and rules below. Cite passages like [1]. Keep numbers exactly as written. If the passages "
                 "do not contain the answer, say you don't know and suggest the relevant university office. Be brief "
                 "(at most 5 sentences). You do not give medical, psychological or legal advice.")
STUDY_SYSTEM = ("You explain results from the NeuroGraph EEG research database. Use ONLY the query results given; "
                "copy numbers exactly; do not add statistics of your own. Say this is research data, not a diagnosis. "
                "At most 4 sentences.")


class State(TypedDict, total=False):
    question: str
    route: str
    passages: list[dict]
    rules: list[str]
    rows: list[dict]
    answer: str
    check: dict


def build_agent(index, tools, llm, verifier, cfg: dict):
    k = cfg["agent"]["context_k"]

    def safety_node(s: State) -> State:
        level = safety.check(s["question"])
        if level:
            return {"route": f"safety:{level}", "answer": safety.referral(level, tools.services())}
        return {"route": "study" if STUDY_WORDS.search(s["question"]) else "policy"}

    def policy_node(s: State) -> State:
        ids = index.dense(s["question"], k)
        passages = [index.chunks[i] for i in ids]
        rules = tools.rules_for([p["id"] for p in passages])
        context = "\n\n".join(f"[{n}] {p['title']}\n{p['text']}" for n, p in enumerate(passages, 1))
        if rules:
            context += "\n\nGraph rules:\n" + "\n".join(rules)
        answer = llm.chat(POLICY_SYSTEM, f"{context}\n\nQuestion: {s['question']}")
        return {"passages": passages, "rules": rules, "answer": answer}

    def study_node(s: State) -> State:
        title, rows = tools.study(s["question"])
        context = f"Query: {title}\nResults: {json.dumps(rows, default=str)}"
        answer = llm.chat(STUDY_SYSTEM, f"{context}\n\nQuestion: {s['question']}")
        return {"rows": rows, "passages": [{"title": title, "text": json.dumps(rows, default=str), "url": "neo4j"}],
                "answer": answer}

    def verify_node(s: State) -> State:
        check = verifier.check(s["answer"], [p["text"] for p in s.get("passages", [])], s.get("rules", []))
        answer = s["answer"]
        if check["verdict"] != "supported":
            answer += "\n\n(Note: I could not fully verify this answer against the sources; please check the linked pages.)"
        sources = sorted({p["url"] for p in s.get("passages", []) if p.get("url", "").startswith("http")})
        if sources:
            answer += "\n\nSources: " + " | ".join(sources[:3])
        return {"answer": answer, "check": check}

    g = StateGraph(State)
    for name, fn in [("safety", safety_node), ("policy", policy_node), ("study", study_node), ("verify", verify_node)]:
        g.add_node(name, fn)
    g.set_entry_point("safety")
    g.add_conditional_edges("safety", lambda s: s["route"] if s["route"] in ("policy", "study") else "end",
                            {"policy": "policy", "study": "study", "end": END})
    g.add_edge("policy", "verify")
    g.add_edge("study", "verify")
    g.add_edge("verify", END)
    return g.compile()
