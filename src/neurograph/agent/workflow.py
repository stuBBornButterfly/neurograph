import re
from typing import TypedDict

from langgraph.graph import END, StateGraph

from neurograph.agent import safety
from neurograph.agent.tools import LIT_WORDS, STUDY_WORDS, rows_text

POLICY_SYSTEM = ("You are the the university student information assistant. Answer ONLY from the numbered "
                 "passages and rules below. Cite passages like [1]. Keep numbers exactly as written. If the passages "
                 "do not contain the answer, say you don't know and suggest the relevant university office. Be brief "
                 "(at most 5 sentences). You do not give medical, psychological or legal advice.")
STUDY_SYSTEM = ("You explain results from the NeuroGraph EEG research database. Use ONLY the query results and the "
                "facts computed in Python; copy numbers exactly; do not add statistics of your own. If a verdict is "
                "given, state that verdict and never call a result significant or replicated unless the verdict says "
                "so. Say this is research data, not a diagnosis. At most 4 sentences.")
LIT_SYSTEM = ("You summarise published EEG research for depression. Use ONLY the listed claims and the facts computed "
              "in Python. Cite every claim you use with its PMID, like (PMID 12345678). Copy numbers exactly. If a verdict about our own data is "
              "given, state it and never call a result replicated unless the verdict says so. Say this is research, "
              "not a diagnosis. At most 5 sentences.")


class State(TypedDict, total=False):
    question: str
    route: str
    passages: list[dict]
    rules: list[str]
    rows: list[dict]
    answer: str
    check: dict


def expand(question: str, expansions: dict[str, str]) -> str:
    extra = [words for pat, words in expansions.items() if re.search(pat, question, re.I)]
    return question + (" " + " ".join(extra) if extra else "")


def build_agent(index, tools, llm, verifier, cfg: dict):
    k = cfg["agent"]["context_k"]

    def safety_node(s: State) -> State:
        level = safety.check(s["question"])
        if level:
            return {"route": f"safety:{level}", "answer": safety.referral(level, tools.services())}
        q = s["question"]
        return {"route": "literature" if LIT_WORDS.search(q) else "study" if STUDY_WORDS.search(q) else "policy"}

    def policy_node(s: State) -> State:
        ids = index.dense(expand(s["question"], cfg["agent"].get("query_expansions", {})), k)
        passages = [index.chunks[i] for i in ids]
        rules = tools.rules_for([p["id"] for p in passages])
        context = "\n\n".join(f"[{n}] {p['title']}\n{p['text']}" for n, p in enumerate(passages, 1))
        if rules:
            context += "\n\nGraph rules:\n" + "\n".join(rules)
        answer = llm.chat(POLICY_SYSTEM, f"{context}\n\nQuestion: {s['question']}")
        return {"passages": passages, "rules": rules, "answer": answer}

    def study_node(s: State) -> State:
        title, rows, facts = tools.study(s["question"])
        table = rows_text(rows)
        context = f"Query: {title}\nResults:\n{table}"
        if facts:
            context += "\n\nFacts computed in Python:\n" + "\n".join(facts)
        answer = llm.chat(STUDY_SYSTEM, f"{context}\n\nQuestion: {s['question']}")
        # each fact is its own short passage so the NLI check compares like with like
        passages = [{"title": title, "text": table, "url": "neo4j"}] + \
            [{"title": "fact", "text": f, "url": "neo4j"} for f in facts]
        return {"rows": rows, "passages": passages, "answer": answer}

    def literature_node(s: State) -> State:
        claims, facts = tools.literature(s["question"])
        lines = [f"- (PMID {c['pmid']}, {c['year']}) {c['text']}" for c in claims]  # no [n], so it cites PMIDs
        context = "Published claims (NLI-verified):\n" + ("\n".join(lines) or "none found")
        if facts:
            context += "\n\nFacts computed in Python:\n" + "\n".join(facts)
        answer = llm.chat(LIT_SYSTEM, f"{context}\n\nQuestion: {s['question']}")
        passages = [{"title": f"PMID {c['pmid']}", "text": f"PMID {c['pmid']} ({c['year']}): {c['text']}",
                     "url": f"https://pubmed.ncbi.nlm.nih.gov/{c['pmid']}/"} for c in claims] + \
            [{"title": "fact", "text": f, "url": "neo4j"} for f in facts]
        return {"passages": passages, "answer": answer}

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
    for name, fn in [("safety", safety_node), ("policy", policy_node), ("study", study_node),
                     ("literature", literature_node), ("verify", verify_node)]:
        g.add_node(name, fn)
    g.set_entry_point("safety")
    g.add_conditional_edges("safety", lambda s: s["route"] if s["route"] in ("policy", "study", "literature") else "end",
                            {"policy": "policy", "study": "study", "literature": "literature", "end": END})
    g.add_edge("policy", "verify")
    g.add_edge("study", "verify")
    g.add_edge("literature", "verify")
    g.add_edge("verify", END)
    return g.compile()
