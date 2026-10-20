import re

# checked before any model call; matches are deliberately broad (a missed crisis is worse than an extra referral)
CRISIS = re.compile(r"suicid|kill (?:my ?self|me)|end (?:my|it all)|take my (?:own )?life|self[- ]?harm|hurt(?:ing)? my ?self|"
                    r"cut(?:ting)? my ?self|don'?t want to (?:live|be alive)|no reason to live|better off dead|"
                    r"want to die|overdose|আত্মহত্যা|মরে যেতে", re.I)
DISTRESS = re.compile(r"depress|anxi|panic|stress|overwhelm|hopeless|lonely|can'?t cope|can'?t sleep|"
                      r"burn(?:ed|t)? ?out|feel(?:ing)? (?:so |very |really )?(?:sad|low|empty|worthless|down)|"
                      r"mental health|cry(?:ing)? all|breakdown|traumati", re.I)

SERVICE_QUERY = """MATCH (s:Service)-[:SUPPORTS]->(c:Condition)
WHERE c.name IN ['depression', 'anxiety', 'stress']
RETURN DISTINCT s.name AS name, s.location AS location, s.contact_public AS contact, s.url AS url"""


FIRST_PERSON = re.compile(r"\b(?:i|i'?m|i'?ve|i'?ll|me|my|myself)\b", re.I)


def check(message: str) -> str | None:
    if CRISIS.search(message):  # always, even if phrased about others
        return "crisis"
    if DISTRESS.search(message) and FIRST_PERSON.search(message):  # "my anxiety", not "depression in our data"
        return "distress"
    return None


def referral(level: str, services: list[dict]) -> str:
    # fixed wording, contacts come from the graph (Service nodes), never from the language model
    where = "\n".join(f"- {s['name']}, {s['location']}, {s['contact']} ({s['url']})" for s in services) or \
        "- university Department of Students' Welfare"
    if level == "crisis":
        return ("I'm really sorry you're feeling this way. You deserve support right now, and you don't have to "
                "handle this alone.\n\nIf you are in immediate danger or might act on these thoughts, call **999** "
                "(Bangladesh national emergency) or go to the nearest hospital emergency department now, and if you "
                "can, tell someone you trust who is nearby.\n\nAt university you can talk in confidence to the "
                f"psychosocial counsellors:\n{where}\n\nI'm an information assistant, not a counsellor, so I can't "
                "give the support you need, but these people can.")
    return ("That sounds hard, and it's good that you reached out. Many students go through stress like this, "
            "and talking to someone often helps.\n\nThe university has free psychosocial counselling for students:\n"
            f"{where}\n\nI can't give medical or psychological advice, but I can help with questions about "
            "university rules, deadlines or services if that would take some pressure off.")
