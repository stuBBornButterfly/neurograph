from sentence_transformers import CrossEncoder


class ClaimVerifier:
    # a claim counts as verified only if its restated hypothesis is entailed by the abstract sentence it came from
    def __init__(self, model: str, min_entail: float):
        self.nli = CrossEncoder(model)
        labels = {v.lower(): int(k) for k, v in self.nli.model.config.id2label.items()}
        self.idx, self.min_entail = labels["entailment"], min_entail

    def entail(self, pairs: list[tuple[str, str]]) -> list[float]:
        probs = self.nli.predict(pairs, apply_softmax=True, batch_size=32)
        return [round(float(p[self.idx]), 3) for p in probs]
