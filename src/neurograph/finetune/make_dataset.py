import json
import random
import re

from neurograph.config import site

SITE = site()  # the university's name comes from private settings
ARTICLE = "an" if SITE["short"][:1] in "AEIOU" else "a"
QA_SYSTEM = (
    f"You write questions that {ARTICLE} {SITE['short']} student might ask, each answerable ONLY from the passage "
    "given. Reply with a JSON list only, no prose: "
    '[{"question": "...", "answer": "..."}]. Rules: answers are 1-2 sentences and copy numbers, grades, fees, '
    "rooms and office names exactly as written in the passage; no questions about people, names or redacted "
    "text like [PERSON]; no yes/no questions; each question must make sense without seeing the passage.")
NOT_IN_PASSAGES = (f"The passages above do not answer this. Please check the official {SITE['short']} website or ask "
                   f"the relevant {SITE['short']} office.")
REFUSAL = re.compile(r"do(?:es)? not (?:answer|say|mention|contain|cover|provide)|don't know|do not know|"
                     r"not (?:mentioned|stated|specified|covered)|no information|cannot find|can't find", re.I)


def parse_qa(raw: str, n_max: int) -> list[dict]:
    # keep well-formed pairs only; the model's output is never trusted blindly
    m = re.search(r"\[.*\]", raw, re.S)
    try:
        items = json.loads(m.group(0)) if m else []
    except json.JSONDecodeError:
        return []
    out = []
    for it in items if isinstance(items, list) else []:
        if isinstance(it, dict) and isinstance(it.get("question"), str) and isinstance(it.get("answer"), str):
            q, a = it["question"].strip(), it["answer"].strip()
            if 4 <= len(q.split()) <= 40 and 2 <= len(a.split()) <= 80 and "[PERSON]" not in q + a:
                out.append({"question": q, "answer": a})
    return out[:n_max]


def split_chunks(chunk_ids: list[str], test_share: float, seed: int) -> set[str]:
    # held out by chunk: test answers come from text the model never trained on
    ids = sorted(chunk_ids)
    random.Random(seed).shuffle(ids)
    return set(ids[:round(len(ids) * test_share)])


def neighbours(chunk: dict, by_pos: dict) -> list[dict]:
    # the chunks just before and after on the same page (they share ~40 words with this one)
    return [by_pos[k] for k in ((chunk["url"], chunk["chunk"] - 1), (chunk["url"], chunk["chunk"] + 1)) if k in by_pos]


def context(passages: list[dict]) -> str:
    # same layout as the agent's policy prompt, so the adapter works inside the agent unchanged
    return "\n\n".join(f"[{n}] {p['title']}\n{p['text']}" for n, p in enumerate(passages, 1))


def build_example(qa: dict, gold: dict, distractors: list[dict], negative: bool, rng: random.Random) -> dict:
    if negative:
        passages, target, pos = distractors, NOT_IN_PASSAGES, None
    else:
        pos = rng.randrange(len(distractors) + 1)
        passages = distractors[:pos] + [gold] + distractors[pos:]
        target, pos = f"{qa['answer'].rstrip()} [{pos + 1}]", pos + 1
    return {"chunk_id": gold["id"], "question": qa["question"], "reference": qa["answer"], "negative": negative,
            "gold_pos": pos, "user": f"{context(passages)}\n\nQuestion: {qa['question']}", "target": target}


def token_f1(pred: str, ref: str) -> float:
    p, r = re.findall(r"\w+", pred.lower()), re.findall(r"\w+", ref.lower())
    common = sum(min(p.count(w), r.count(w)) for w in set(p))
    if not p or not r or not common:
        return 0.0
    prec, rec = common / len(p), common / len(r)
    return 2 * prec * rec / (prec + rec)
