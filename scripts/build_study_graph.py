import glob
import json
import time
from pathlib import Path

import pandas as pd

from neurograph.config import REPO_ROOT, git_hash, load_config
from neurograph.graph import load_study as ls
from neurograph.graph.client import apply_schema, get_driver, run, write_batches


def find(pattern: str, what: str) -> Path:
    hits = sorted(glob.glob(pattern, recursive=True))
    if not hits:
        raise FileNotFoundError(f"{what}: nothing matches {pattern}; attach the notebook that made it")
    return Path(hits[0])


def main() -> None:
    t0 = time.perf_counter()
    paths, g, f, study = load_config("paths"), load_config("graph"), load_config("features"), load_config("study")
    eeg = load_config("eeg")
    src = {k: find(p, k) for k, p in g["inputs"].items()}
    der = src["features"].parent.parent
    labels = pd.read_csv(der / "labels.csv", dtype={"sub": str})
    feats = pd.read_csv(src["features"], dtype={"sub": str})
    regions = json.loads((der / "features" / "regions.json").read_text())
    qc = pd.read_csv(src["qc_table"], dtype={"subject": str, "run": str})
    qc = qc[qc["subject"].isin(labels["sub"])]
    risk = pd.read_csv(src["risk_tiers"], dtype={"sub": str})
    weights = pd.read_csv(src["lr_weights"], index_col=0)
    results = pd.read_csv(src["results"])
    meas, dataset = g["measures"], paths["dataset"]["id"]

    driver = get_driver()
    print(f"== step 5.1 study graph | commit {git_hash()} | {driver.get_server_info().agent} ==")
    n_con = apply_schema(driver, REPO_ROOT / "src" / "neurograph" / "graph" / "schema.cypher")
    if g["reset"]:
        run(driver, ls.RESET, labels=g["study_labels"])
    run(driver, ls.DATASET, id=dataset, version=str(paths["dataset"]["version"]))
    bs = g["batch_size"]
    loaded = {
        "participants": write_batches(driver, ls.PARTICIPANTS, ls.participant_rows(labels, dataset), bs),
        "assessments": write_batches(driver, ls.ASSESSMENTS, ls.assessment_rows(labels, study["stai_high"]), bs),
        "recordings": write_batches(driver, ls.RECORDINGS, ls.recording_rows(qc), bs),
        "bands": write_batches(driver, ls.BANDS, [{"name": b, "fmin": lo, "fmax": hi}
                                                  for b, (lo, hi) in f["bands"].items()], bs),
        "regions": write_batches(driver, ls.REGIONS, [{"name": r, "channels": c} for r, c in regions.items()], bs),
        "biomarkers": write_batches(driver, ls.BIOMARKERS, ls.biomarker_rows(list(feats.columns), meas), bs),
    }
    t1 = time.perf_counter()
    loaded["feature values"] = write_batches(driver, ls.FEATURES, ls.feature_rows(feats, eeg["main_run"], meas), bs)
    t_feat = time.perf_counter() - t1
    loaded["models"] = write_batches(driver, ls.MODELS, ls.model_rows(results, g["splits"]), bs)
    loaded["model weights"] = write_batches(driver, ls.WEIGHTS, ls.weight_rows(
        weights, f"{g['weights_step']}|{g['weights_model']}", meas), bs)
    loaded["risk estimates"] = write_batches(driver, ls.RISKS, ls.risk_rows(risk, g["risk_models"]), bs)

    print(f"constraints {n_con} | reset {g['reset']}")
    for k, v in src.items():
        print(f"  input {k:11s} {str(v).replace('/kaggle/input/notebooks/', '')}")
    print("rows sent: " + " | ".join(f"{k} {v}" for k, v in loaded.items()) + f" (features took {t_feat:.0f} s)")
    nodes = run(driver, "MATCH (n) RETURN labels(n)[0] AS label, count(*) AS n ORDER BY label")
    rels = run(driver, "MATCH ()-[r]->() RETURN type(r) AS type, count(*) AS n ORDER BY type")
    print("\nnodes: " + " | ".join(f"{r['label']} {r['n']}" for r in nodes))
    print("relationships: " + " | ".join(f"{r['type']} {r['n']}" for r in rels))
    total_n, total_r = sum(r["n"] for r in nodes), sum(r["n"] for r in rels)
    print(f"total {total_n} nodes / {total_r} relationships (AuraDB Free limit 200,000 / 400,000)")
    orphans = run(driver, "MATCH (v:FeatureValue) WHERE NOT (v)-[:OF]->(:Biomarker) RETURN count(v) AS n")[0]["n"]
    print(f"checks: feature values without a biomarker {orphans} (should be 0)")
    driver.close()
    print(f"time {time.perf_counter() - t0:.0f} s")


if __name__ == "__main__":
    main()
