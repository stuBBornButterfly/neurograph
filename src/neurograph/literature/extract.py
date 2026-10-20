import json
import re

# fixed claim schema (like LLMGraphTransformer with allowed nodes/relations, but plain JSON so the output is checkable)
EXTRACT_SYSTEM = (
    "You extract one EEG finding from a sentence of a research abstract. Reply with JSON only, no prose. "
    'Schema: {"claim": true|false, "band": "delta"|"theta"|"alpha"|"beta"|"gamma", "measure": "power"|"asymmetry", '
    '"region": "frontal"|"central"|"temporal"|"parietal"|"occipital"|"posterior"|"global", '
    '"direction": "higher"|"lower"|"no_difference"}. '
    "direction = how the depressed group compares with healthy controls. "
    'Set "claim": false if the sentence does not compare depressed people with healthy controls on EEG band power '
    "or frontal alpha asymmetry (e.g. connectivity, coherence, treatment effects, remitted or at-risk groups, "
    "prediction accuracy, other disorders, methods). "
    'Use "global" when no scalp region is named.')
BANDS = ("delta", "theta", "alpha", "beta", "gamma")
REGIONS = ("frontal", "central", "temporal", "parietal", "occipital", "posterior", "global")
POSTERIOR = ["parietal", "occipital"]
DIRECTIONS = ("higher", "lower", "no_difference")


def sentences(text: str) -> list[str]:
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+(?=[A-Z(])", text) if len(s.split()) >= 6]


def candidates(abstracts: list[dict], cfg: dict) -> list[dict]:
    band = re.compile("|".join(cfg["band_words"]), re.I)
    group, comp = re.compile(cfg["group_words"], re.I), re.compile(cfg["compare_words"], re.I)
    other = re.compile(cfg.get("other_measures", "(?!x)x"), re.I)
    out = []
    for a in abstracts:
        for k, s in enumerate(sentences(a["abstract"])):
            if other.search(s) and not re.search(r"power|asymmetr", s, re.I):
                continue
            if band.search(s) and group.search(s) and comp.search(s):
                out.append({"pmid": a["pmid"], "sent_idx": k, "sentence": s})
    return out


def parse_claim(raw: str) -> dict | None:
    # keep only well-formed claims inside the fixed schema; anything else is dropped, not guessed
    m = re.search(r"\{.*\}", raw, re.S)
    try:
        c = json.loads(m.group(0)) if m else None
    except json.JSONDecodeError:
        return None
    if not c or c.get("claim") is not True:
        return None
    if c.get("band") not in BANDS or c.get("region") not in REGIONS or c.get("direction") not in DIRECTIONS \
            or c.get("measure") not in ("power", "asymmetry"):
        return None
    if c["measure"] == "asymmetry":
        c.update(band="alpha", region="frontal")
    return {k: c[k] for k in ("band", "measure", "region", "direction")}


def hypothesis(c: dict) -> str:
    # the claim restated as a plain sentence; NLI must find it entailed by the source sentence
    if c["measure"] == "asymmetry":
        return ("Frontal alpha asymmetry did not differ between depressed patients and healthy controls."
                if c["direction"] == "no_difference" else "Depressed patients showed altered frontal alpha asymmetry.")
    where = "" if c["region"] == "global" else f" over {c['region']} regions"
    if c["direction"] == "no_difference":
        return f"{c['band'].capitalize()} power{where} did not differ between depressed patients and healthy controls."
    return f"Depressed patients showed {c['direction']} {c['band']} power{where} than healthy controls."


def biomarkers(c: dict, regions: list[str], measures: list[str]) -> list[str]:
    # graph names from Phase 5: {band}_{measure}_{region}; FAA claims link to the F4-F3 pair
    if c["measure"] == "asymmetry":
        return ["faa_F4_F3"]
    regs = regions if c["region"] == "global" else POSTERIOR if c["region"] == "posterior" else [c["region"]]
    return [f"{c['band']}_{m}_{r}" for m in measures for r in regs]
