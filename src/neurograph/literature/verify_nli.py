import re

from sentence_transformers import CrossEncoder


def windows(text: str, size: int = 3) -> list[str]:
    # NLI models are trained on sentence pairs: a whole chunk with tables reads as "contradiction" for everything,
    # so a statement is checked against every run of 1..size consecutive sentences or table rows
    segs = [s.strip() for s in re.split(r"(?<=[.!?])\s+|\n+", text) if len(s.split()) >= 3]
    return [" ".join(segs[i:i + n]) for n in range(1, size + 1) for i in range(len(segs) - n + 1)] or [text]


class ClaimVerifier:
    # a claim counts as verified only if its restated hypothesis is entailed by the text it came from
    def __init__(self, model: str, min_entail: float):
        self.nli = CrossEncoder(model)
        labels = {v.lower(): int(k) for k, v in self.nli.model.config.id2label.items()}
        self.idx, self.min_entail = labels["entailment"], min_entail

    def entail(self, pairs: list[tuple[str, str]]) -> list[float]:
        # best entailment over the premise's sentence windows
        flat, owner = [], []
        for k, (premise, hyp) in enumerate(pairs):
            for w in windows(premise):
                flat.append((w, hyp))
                owner.append(k)
        probs = self.nli.predict(flat, apply_softmax=True, batch_size=64) if flat else []
        best = [0.0] * len(pairs)
        for k, p in zip(owner, probs):
            best[k] = max(best[k], float(p[self.idx]))
        return [round(b, 3) for b in best]
