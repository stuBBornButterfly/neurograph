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
                    "unsupported_numbers": unsupported_numbers(answer, "")}
        supported = 0
        premises = [w for p in passages + rules for w in windows(p)]  # sentence windows, not whole chunks
        for s in sents:
            probs = self.nli.predict([(w, s) for w in premises], apply_softmax=True, batch_size=64)
            supported += int(max(row[self.entail_idx] for row in probs) >= self.min_entail)
        bad_numbers = unsupported_numbers(answer, evidence)
        share = supported / len(sents) if sents else 1.0
        verdict = "supported" if share >= 0.5 and not bad_numbers else "unverified"
        return {"verdict": verdict, "entailed_share": round(share, 2), "sentences": len(sents),
                "unsupported_numbers": bad_numbers}
