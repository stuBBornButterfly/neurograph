import numpy as np
from scipy import stats

# classic finding (Henriques & Davidson 1991; Thibodeau et al. 2006 meta-analysis): depression shows relatively
# less left-frontal activity = more left alpha, so ln(F4) - ln(F3) is LOWER in depression than in controls
FAA_QUERY = """MATCH (p:Participant)-[:HAS_SESSION]->()-[:HAS_RECORDING]->()-[:HAS_FEATURE]->(v:FeatureValue {condition: 'closed'})
      -[:OF]->(:Biomarker {name: 'faa_F4_F3'})
RETURN p.group AS group, v.value AS value"""


def compare_groups(ctl: np.ndarray, dep: np.ndarray, expect_dep_lower: bool) -> dict:
    # statistics are computed here, never by the language model
    t = stats.ttest_ind(dep, ctl, equal_var=False)
    sd = np.sqrt((ctl.var(ddof=1) + dep.var(ddof=1)) / 2)
    diff = dep.mean() - ctl.mean()
    same_dir = diff < 0 if expect_dep_lower else diff > 0
    if same_dir and t.pvalue < 0.05:
        verdict = "replicated: same direction as the published finding and significant"
    elif same_dir:
        verdict = "not replicated: same direction as the published finding but not significant"
    elif t.pvalue < 0.05:
        verdict = "not replicated: the difference is significant but in the opposite direction to the published finding"
    else:
        verdict = "not replicated: the difference is in the opposite direction to the published finding and not significant"
    return {"n_ctl": len(ctl), "n_dep": len(dep), "mean_ctl": round(float(ctl.mean()), 3),
            "mean_dep": round(float(dep.mean()), 3), "diff": round(float(diff), 3),
            "cohens_d": round(float(diff / sd), 2), "p": round(float(t.pvalue), 3), "verdict": verdict}


def faa_facts(rows: list[dict]) -> list[str]:
    vals = {g: np.array([r["value"] for r in rows if r["group"] == g and r["value"] is not None]) for g in ("CTL", "DEP")}
    r = compare_groups(vals["CTL"], vals["DEP"], expect_dep_lower=True)
    return [f"Frontal alpha asymmetry ln(F4) minus ln(F3), eyes closed: controls mean {r['mean_ctl']} (n {r['n_ctl']}), "
            f"depression group mean {r['mean_dep']} (n {r['n_dep']}).",
            "The published finding expects a lower asymmetry score in depression than in controls.",
            f"In our data the depression group is {'lower' if r['diff'] < 0 else 'higher'} by {abs(r['diff'])} "
            f"(Cohen's d {r['cohens_d']}, Welch t-test p {r['p']}).",
            f"Verdict computed in Python: {r['verdict']}."]
