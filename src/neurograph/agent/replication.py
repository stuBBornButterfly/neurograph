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


VALUES_QUERY = """MATCH (p:Participant)-[:HAS_SESSION]->()-[:HAS_RECORDING]->()-[:HAS_FEATURE]->(v:FeatureValue {condition: 'closed'})
      -[:OF]->(b:Biomarker) WHERE b.name IN $names
RETURN p.code AS code, p.group AS group, avg(v.value) AS value"""


def claim_key(c: dict) -> tuple[str, str, str]:
    return (c["band"], c["measure"], c["region"]) if c["measure"] == "power" else ("alpha", "asymmetry", "frontal")


def our_biomarkers(key: tuple[str, str, str], regions: list[str]) -> list[str]:
    band, measure, region = key
    if measure == "asymmetry":
        return ["faa_F4_F3"]
    regs = regions if region == "global" else ["parietal", "occipital"] if region == "posterior" else [region]
    return [f"{band}_abs_power_{r}" for r in regs]  # global / posterior = mean of those regions


def literature_vs_ours(claims: list[dict], fetch_values, regions: list[str]) -> list[dict]:
    # claims: verified literature claims; fetch_values(names) -> rows {group, value} from the study graph
    out = []
    for key in sorted({claim_key(c) for c in claims}):
        cs = [c for c in claims if claim_key(c) == key]
        n = {d: sum(c["direction"] == d for c in cs) for d in ("higher", "lower", "no_difference")}
        rows = fetch_values(our_biomarkers(key, regions))
        vals = {g: np.array([r["value"] for r in rows if r["group"] == g and r["value"] is not None]) for g in ("CTL", "DEP")}
        if key[1] == "asymmetry":
            expect = "lower"  # published convention: ln(F4) - ln(F3) lower in depression
        else:
            expect = "higher" if n["higher"] > n["lower"] else "lower" if n["lower"] > n["higher"] else None
        row = {"band": key[0], "measure": key[1], "region": key[2], "papers": len({c["pmid"] for c in cs}),
               "lit_higher": n["higher"], "lit_lower": n["lower"], "lit_no_diff": n["no_difference"],
               "expected_in_dep": expect or "mixed", "pmids": sorted({c["pmid"] for c in cs})}
        if len(vals["CTL"]) > 2 and len(vals["DEP"]) > 2:
            r = compare_groups(vals["CTL"], vals["DEP"], expect_dep_lower=expect != "higher")
            if expect is None:
                r["verdict"] = "literature is mixed, nothing to replicate"
            row.update({k: r[k] for k in ("mean_ctl", "mean_dep", "diff", "cohens_d", "p", "verdict")})
        out.append(row)
    return out


def comparison_facts(row: dict) -> list[str]:
    what = "frontal alpha asymmetry" if row["measure"] == "asymmetry" else f"{row['band']} power ({row['region']})"
    facts = [f"Published claims on {what} in depression vs controls: {row['lit_higher']} report higher, "
             f"{row['lit_lower']} lower, {row['lit_no_diff']} no difference ({row['papers']} papers)."]
    if "verdict" in row:
        facts.append(f"In our data (eyes closed) the depression group is {'lower' if row['diff'] < 0 else 'higher'} "
                     f"by {abs(row['diff'])} (Cohen's d {row['cohens_d']}, Welch t-test p {row['p']}).")
        facts.append(f"Verdict computed in Python: {row['verdict']}.")
    return facts


def faa_facts(rows: list[dict]) -> list[str]:
    vals = {g: np.array([r["value"] for r in rows if r["group"] == g and r["value"] is not None]) for g in ("CTL", "DEP")}
    r = compare_groups(vals["CTL"], vals["DEP"], expect_dep_lower=True)
    return [f"Frontal alpha asymmetry ln(F4) minus ln(F3), eyes closed: controls mean {r['mean_ctl']} (n {r['n_ctl']}), "
            f"depression group mean {r['mean_dep']} (n {r['n_dep']}).",
            "The published finding expects a lower asymmetry score in depression than in controls.",
            f"In our data the depression group is {'lower' if r['diff'] < 0 else 'higher'} by {abs(r['diff'])} "
            f"(Cohen's d {r['cohens_d']}, Welch t-test p {r['p']}).",
            f"Verdict computed in Python: {r['verdict']}."]
