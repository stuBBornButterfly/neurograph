import glob
import logging
import os
import warnings
from pathlib import Path

os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")  # keep the printed summary short
import mlflow  # noqa: E402
import pandas as pd  # noqa: E402

from neurograph.config import REPO_ROOT, git_hash, load_config  # noqa: E402


def find_files(pattern: str, local_dir: Path, suffix: str) -> dict[str, str]:
    # earlier notebooks' outputs (attached as inputs), overridden by anything made in this session
    files = {Path(p).name: p for p in sorted(glob.glob(pattern, recursive=True))}
    files.update({p.name: str(p) for p in local_dir.glob(f"*{suffix}")})
    return files


def run_name(df: pd.DataFrame) -> pd.Series:
    if "layout" in df:
        return df["layout"] + "/" + df["model"]
    return df["model"] if "model" in df else pd.Series("eegnet", index=df.index)


def main() -> None:
    warnings.filterwarnings("ignore")
    logging.getLogger("alembic").setLevel(logging.WARNING)
    logging.getLogger("mlflow").setLevel(logging.WARNING)
    paths, m = load_config("paths"), load_config("models")
    ml = m["mlflow"]
    der = Path(paths["paths"]["derivatives_dir"])
    tables = find_files(ml["folds_glob"], der / "models", "_folds.csv")
    figures = find_files(ml["figures_glob"], REPO_ROOT / paths["paths"]["figures_dir"], ".png")

    db = der / "mlflow.db"
    mlflow.set_tracking_uri(f"sqlite:///{db}")
    exp = mlflow.get_experiment_by_name(ml["experiment"])
    exp_id = exp.experiment_id if exp else mlflow.create_experiment(
        ml["experiment"], artifact_location=(der / "mlartifacts").resolve().as_uri())

    print(f"== step 4.6 MLflow log | commit {git_hash()} | store {db} ==")
    rows = []
    for table, (step, fig) in ml["steps"].items():
        if table not in tables:
            print(f"  missing {table} -> {step} skipped (attach the notebook that made it)")
            continue
        df = pd.read_csv(tables[table])
        df["run"] = run_name(df)
        with mlflow.start_run(experiment_id=exp_id, run_name=step) as parent:
            mlflow.log_params({"step": step, "fold_table": table, "logged_from_commit": git_hash()})
            mlflow.log_artifact(tables[table])
            if fig in figures:
                mlflow.log_artifact(figures[fig])
            for name, g in df.groupby("run", sort=False):
                with mlflow.start_run(experiment_id=exp_id, run_name=f"{step} | {name}", nested=True):
                    mlflow.log_params({"step": step, "model": name, "n_folds": len(g)})
                    metrics = {f"{col}_mean": float(g[col].mean()) for col in
                               ["auc", "bal_acc", "brier", "auc_epoch", "auc_lr"] if col in g}
                    metrics["auc_sd"] = float(g["auc"].std())
                    mlflow.log_metrics(metrics)
                    for i, v in enumerate(g["auc"]):
                        mlflow.log_metric("auc_fold", float(v), step=i)
                    rows.append({"step": step, "model": name, "folds": len(g),
                                 "auc": metrics["auc_mean"], "sd": metrics["auc_sd"]})
        print(f"  {step:26s} {df['run'].nunique():2d} model runs | figure {'✓' if fig in figures else 'missing'}"
              f" | parent run {parent.info.run_id[:8]}")

    res = pd.DataFrame(rows)
    res.to_csv(der / "models" / "all_results.csv", index=False)
    print(f"\n{'step':26s} {'model':30s} {'folds':>5s}  AUC mean ± SD")
    for _, r in res.iterrows():
        print(f"{r['step']:26s} {r['model']:30s} {r['folds']:5d}  {r['auc']:.3f} ± {r['sd']:.3f}")
    n_runs = len(mlflow.search_runs(experiment_ids=[exp_id]))
    print(f"\nruns in experiment '{ml['experiment']}': {n_runs} | saved: {db} | {der / 'mlartifacts'} | "
          f"models/all_results.csv")
    print(f"view later: download the notebook output, then  mlflow ui --backend-store-uri sqlite:///mlflow.db")


if __name__ == "__main__":
    main()
