import re

from sentence_transformers import CrossEncoder

from neurograph.literature.verify_nli import windows

NUMBER = re.compile(r"(?<![\w.])\d+(?:\.\d+)?%?")


def sentences(text: str) -> list[str]:
    parts = re.split(r"(?<=[.!?])\s+|\n+", text)
    return [p.strip() for p in parts if len(p.split()) >= 4]


def unsupported_numbers(answer: str, evidence: str) -> list[str]:
    # every number in the answer must appear in the passages or graph rules it was given
    found = set(NUMBER.findall(evidence))
    return [n for n in NUMBER.findall(answer) if n not in found and not re.fullmatch(r"\[?\d\]?", n)]


UP = {"higher", "greater", "larger", "more", "increased", "elevated"}
DOWN = {"lower", "smaller", "less", "decreased", "reduced"}
OUR_DIRECTION = re.compile(r"In our data(?: \(eyes closed\))? the depression group is (higher|lower)", re.I)
DEP_CLAIM = re.compile(r"depress(?:ion|ed) group\b([^.;]{0,40}?)\b(" + "|".join(sorted(UP | DOWN)) + r")\b", re.I)
ABOUT_OUR_DATA = re.compile(r"depression group|\b(?:our|this) (?:eeg )?(?:data|study|sample|dataset)\b", re.I)
ABOUT_LITERATURE = re.compile(r"literature|published|\bpapers?\b|\bPMID|stud(?:y|ies) (?:report|show|found|find)", re.I)


def direction_conflicts(answer: str, evidence: str) -> list[str]:
    # the direction of each group difference in our data is computed in Python ("In our data ... the depression group
    # is higher/lower ..."); an answer sentence about our depression group that states the opposite is flagged.
    # NLI often misses such a swap. Only checked when every computed direction in the evidence agrees.
    dirs = {d.lower() for d in OUR_DIRECTION.findall(evidence)}
    if len(dirs) != 1:
        return []
    truth = dirs.pop()
    out = []
    for sent in sentences(answer):
        if not ABOUT_OUR_DATA.search(sent) or ABOUT_LITERATURE.search(sent):
            continue
        for between, word in DEP_CLAIM.findall(sent):
            if re.search(r"\bnot\b|n't\b|\bno\b", between, re.I):
                continue
            said = "higher" if word.lower() in UP else "lower"
            if said != truth:
                out.append(sent)
                break
    return out


class Verifier:
    # answer check against the evidence: numbers must match; each sentence must be entailed by some passage
    def __init__(self, nli_model: str, min_entail: float):
        self.nli = CrossEncoder(nli_model)
        labels = {v.lower(): int(k) for k, v in self.nli.model.config.id2label.items()}
        self.entail_idx, self.min_entail = labels["entailment"], min_entail

    def check(self, answer: str, passages: list[str], rules: list[str]) -> dict:
        evidence = "\n".join(passages + rules)
        sents = sentences(re.sub(r"\[\d+\]", "", answer))
        if not passages + rules:  # nothing to check against: never "supported"
            return {"verdict": "unverified", "entailed_share": 0.0, "sentences": len(sents),
                    "unsupported_numbers": unsupported_numbers(answer, ""), "direction_conflicts": []}
        supported = 0
        premises = [w for p in passages + rules for w in windows(p)]  # sentence windows, not whole chunks
        for s in sents:
            probs = self.nli.predict([(w, s) for w in premises], apply_softmax=True, batch_size=64)
            supported += int(max(row[self.entail_idx] for row in probs) >= self.min_entail)
        bad_numbers, flipped = unsupported_numbers(answer, evidence), direction_conflicts(answer, evidence)
        share = supported / len(sents) if sents else 1.0
        verdict = "supported" if share >= 0.5 and not bad_numbers and not flipped else "unverified"
        return {"verdict": verdict, "entailed_share": round(share, 2), "sentences": len(sents),
                "unsupported_numbers": bad_numbers, "direction_conflicts": flipped}
