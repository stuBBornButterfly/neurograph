import re

import pandas as pd

FEATURE_RE = re.compile(r"^(closed|open)_(delta|theta|alpha|beta|gamma)_(abs|rel|de)_(\w+)$")
FAA_RE = re.compile(r"^(closed|open)_faa_([A-Z0-9]+)_([A-Z0-9]+)$")
UNITS = {"abs_power": "log10 µV²", "rel_power": "share of 1-45 Hz power", "diff_entropy": "nats",
         "faa": "ln(µV²) right - left", "iaf": "Hz"}


def parse_feature(col: str, measures: dict[str, str]) -> dict | None:
    # feature column -> biomarker + condition; None for bookkeeping columns
    m = FEATURE_RE.match(col)
    if m:
        cond, band, meas, region = m.groups()
        name = f"{band}_{measures[meas]}_{region}"
        return {"condition": cond, "biomarker": name, "band": band, "region": region,
                "measure": measures[meas], "description": f"{band} {measures[meas].replace('_', ' ')}, {region} region"}
    m = FAA_RE.match(col)
    if m:
        cond, r, l = m.groups()
        return {"condition": cond, "biomarker": f"faa_{r}_{l}", "band": "alpha", "region": "frontal",
                "measure": "faa", "description": f"frontal alpha asymmetry ln(alpha {r}) - ln(alpha {l})"}
    if col == "iaf_hz":
        return {"condition": "closed", "biomarker": "iaf", "band": "alpha", "region": "occipital",
                "measure": "iaf", "description": "individual alpha peak frequency (O1/OZ/O2, eyes closed)"}
    return None


def code(sub: str) -> str:
    return f"sub-{sub}"


def clean(v):
    return None if pd.isna(v) else v  # Neo4j: null instead of NaN


def participant_rows(labels: pd.DataFrame, dataset: str) -> list[dict]:
    return [{"code": code(r["sub"]), "dataset": dataset, "group": r["group"],
             "sex": "F" if r["sex"] == 1 else "M", "dep_severity": clean(r["dep_severity"])}
            for _, r in labels.iterrows()]


def assessment_rows(labels: pd.DataFrame, stai_high: float) -> list[dict]:
    rows = []
    for _, r in labels.iterrows():
        rows.append({"id": f"{code(r['sub'])}|BDI", "code": code(r["sub"]), "instrument": "BDI-II",
                     "score": float(r["BDI"]), "band": clean(r["bdi_ii_tier"])})
        if pd.notna(r["STAI"]):
            rows.append({"id": f"{code(r['sub'])}|STAI", "code": code(r["sub"]), "instrument": "STAI",
                         "score": float(r["STAI"]), "band": "high" if r["STAI"] >= stai_high else "not high"})
    return rows


def recording_rows(qc: pd.DataFrame) -> list[dict]:
    rows = []
    for _, r in qc.iterrows():
        c, rec = code(r["subject"]), f"{code(r['subject'])}|run-{r['run']}"
        kept = int(r["epochs_kept.closed"]) + int(r["epochs_kept.open"])
        rows.append({"code": c, "session": f"{c}|ses-1", "recording": rec,
                     "rec": {"run": r["run"], "sfreq": float(r["sfreq"]), "n_channels": int(r["n_channels"]),
                             "epochs_closed": int(r["epochs_kept.closed"]), "epochs_open": int(r["epochs_kept.open"]),
                             "clean_duration_s": 2.0 * kept},
                     "run_id": f"{rec}|{r['commit']}",
                     "run": {"git_hash": r["commit"], "ica_removed": int(r["ica_n_excluded"]),
                             "ica_brain": int(r["ica_n_brain"]), "epochs_rejected_pct": float(r["pct_rejected"]),
                             "n_bad_channels": int(r["n_bads"]), "bad_channels": str(r["bads"]) if pd.notna(r["bads"]) else "",
                             "eog_swapped": bool(r["eog_swapped"]),
                             "qc_flags": str(r["flags"]) if pd.notna(r["flags"]) else ""}})
    return rows


def biomarker_rows(columns: list[str], measures: dict[str, str]) -> list[dict]:
    seen = {}
    for col in columns:
        p = parse_feature(col, measures)
        if p and p["biomarker"] not in seen:
            seen[p["biomarker"]] = {"name": p["biomarker"], "band": p["band"], "region": p["region"],
                                    "measure": p["measure"], "unit": UNITS[p["measure"]],
                                    "description": p["description"]}
    return list(seen.values())


def feature_rows(feats: pd.DataFrame, run: str, measures: dict[str, str]) -> list[dict]:
    parsed = {c: parse_feature(c, measures) for c in feats.columns}
    rows = []
    for _, r in feats.iterrows():
        rec = f"{code(r['sub'])}|run-{run}"
        for col, p in parsed.items():
            if p and pd.notna(r[col]):
                rows.append({"id": f"{code(r['sub'])}|{p['condition']}|{p['biomarker']}", "recording": rec,
                             "biomarker": p["biomarker"], "condition": p["condition"], "value": float(r[col])})
    return rows


def model_rows(results: pd.DataFrame, splits: dict[str, str]) -> list[dict]:
    rows = []
    for _, r in results.iterrows():
        mid = f"{r['step']}|{r['model']}"
        rows.append({"id": mid, "name": r["model"], "step": r["step"],
                     "split": splits.get(r["step"], splits["default"]), "metric_id": f"{mid}|auc",
                     "mean": float(r["auc"]), "sd": float(r["sd"]) if pd.notna(r["sd"]) else 0.0,
                     "n_folds": int(r["folds"])})
    return rows


def weight_rows(weights: pd.DataFrame, model_id: str, measures: dict[str, str]) -> list[dict]:
    rows = []
    for col, r in weights.iterrows():
        p = parse_feature(col, measures)
        if p:
            rows.append({"model": model_id, "biomarker": p["biomarker"], "condition": p["condition"],
                         "mean": float(r["mean"]), "sd": float(r["sd"]), "same_sign": float(r["same_sign"])})
    return rows


def risk_rows(risk: pd.DataFrame, models: dict[str, str]) -> list[dict]:
    rows = []
    for suffix, model in models.items():
        for _, r in risk.iterrows():
            rows.append({"id": f"{code(r['sub'])}|{model}", "code": code(r["sub"]), "model": model,
                         "p_dep": float(r[f"p_dep{suffix}"]), "tier": clean(r[f"tier{suffix}"])})
    return rows


RESET = """MATCH (n) WHERE any(l IN labels(n) WHERE l IN $labels)
CALL (n) { DETACH DELETE n } IN TRANSACTIONS OF 5000 ROWS"""

DATASET = "MERGE (d:Dataset {id: $id}) SET d.version = $version, d.source = 'OpenNeuro'"

PARTICIPANTS = """UNWIND $rows AS row
MATCH (d:Dataset {id: row.dataset})
MERGE (p:Participant {code: row.code})
SET p.group = row.group, p.sex = row.sex, p.dep_severity = row.dep_severity
MERGE (p)-[:IN_DATASET]->(d)"""

ASSESSMENTS = """UNWIND $rows AS row
MATCH (p:Participant {code: row.code})
MERGE (a:Assessment {id: row.id})
SET a.instrument = row.instrument, a.score = row.score, a.band = row.band
MERGE (p)-[:HAS_ASSESSMENT]->(a)"""

RECORDINGS = """UNWIND $rows AS row
MATCH (p:Participant {code: row.code})
MERGE (s:Session {id: row.session})
MERGE (p)-[:HAS_SESSION]->(s)
MERGE (r:Recording {id: row.recording})
SET r += row.rec
MERGE (s)-[:HAS_RECORDING]->(r)
MERGE (pr:PipelineRun {run_id: row.run_id})
SET pr += row.run
MERGE (r)-[:PROCESSED_BY]->(pr)"""

BANDS = """UNWIND $rows AS row
MERGE (f:FrequencyBand {name: row.name}) SET f.fmin = row.fmin, f.fmax = row.fmax"""

REGIONS = """UNWIND $rows AS row
MERGE (g:BrainRegion {name: row.name}) SET g.channels = row.channels"""

BIOMARKERS = """UNWIND $rows AS row
MERGE (b:Biomarker {name: row.name})
SET b.measure = row.measure, b.unit = row.unit, b.description = row.description
WITH b, row
MATCH (f:FrequencyBand {name: row.band})
MATCH (g:BrainRegion {name: row.region})
MERGE (b)-[:IN_BAND]->(f)
MERGE (b)-[:AT_REGION]->(g)"""

FEATURES = """UNWIND $rows AS row
MATCH (r:Recording {id: row.recording})
MATCH (b:Biomarker {name: row.biomarker})
MERGE (v:FeatureValue {id: row.id})
SET v.value = row.value, v.condition = row.condition
MERGE (r)-[:HAS_FEATURE]->(v)
MERGE (v)-[:OF]->(b)"""

MODELS = """UNWIND $rows AS row
MERGE (m:Model {id: row.id}) SET m.name = row.name, m.step = row.step
MERGE (sp:Split {name: row.split})
MERGE (m)-[:EVALUATED_ON]->(sp)
MERGE (x:Metric {id: row.metric_id})
SET x.name = 'auc', x.mean = row.mean, x.sd = row.sd, x.n_folds = row.n_folds
MERGE (m)-[:HAS_METRIC]->(x)"""

WEIGHTS = """UNWIND $rows AS row
MATCH (m:Model {id: row.model})
MATCH (b:Biomarker {name: row.biomarker})
MERGE (m)-[w:WEIGHTS {condition: row.condition}]->(b)
SET w.mean = row.mean, w.sd = row.sd, w.same_sign = row.same_sign"""

RISKS = """UNWIND $rows AS row
MATCH (p:Participant {code: row.code})
MERGE (e:RiskEstimate {id: row.id})
SET e.model = row.model, e.p_dep = row.p_dep, e.tier = row.tier
MERGE (p)-[:HAS_RISK]->(e)"""
