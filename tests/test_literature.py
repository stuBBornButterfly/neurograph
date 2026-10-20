import numpy as np

from neurograph.agent.replication import comparison_facts, literature_vs_ours
from neurograph.literature.extract import biomarkers, candidates, hypothesis, parse_claim

CFG = {"band_words": {"alpha": "alpha", "theta": "theta", "asymmetr": "alpha"}, "group_words": "depress|MDD",
       "compare_words": "higher|lower|reduc|increase"}
REGIONS = ["frontal", "central", "temporal", "parietal", "occipital"]


def test_parse_claim_keeps_only_schema():
    assert parse_claim('{"claim": true, "band": "theta", "measure": "power", "region": "frontal", "direction": "higher"}') \
        == {"band": "theta", "measure": "power", "region": "frontal", "direction": "higher"}
    assert parse_claim('{"claim": false}') is None
    assert parse_claim('{"claim": true, "band": "mu", "measure": "power", "region": "frontal", "direction": "higher"}') is None
    assert parse_claim("not json") is None


def test_claim_links_to_shared_biomarkers():
    c = {"band": "alpha", "measure": "power", "region": "posterior", "direction": "lower"}
    assert biomarkers(c, REGIONS, ["abs_power"]) == ["alpha_abs_power_parietal", "alpha_abs_power_occipital"]
    assert hypothesis(c) == "Depressed patients showed lower alpha power over posterior regions than healthy controls."
    assert biomarkers({**c, "measure": "asymmetry"}, REGIONS, ["abs_power"]) == ["faa_F4_F3"]


def test_candidates_need_band_group_and_comparison():
    arts = [{"pmid": "1", "abstract": "Depressed patients showed higher frontal theta power than controls at rest. "
                                      "We recorded resting EEG in sixty adults for ten minutes."}]
    assert [c["sentence"][:9] for c in candidates(arts, CFG)] == ["Depressed"]


def test_literature_vs_ours_tests_our_data_in_python():
    rng = np.random.default_rng(0)
    claims = [{"pmid": str(k), "band": "theta", "measure": "power", "region": "frontal", "direction": "higher"} for k in range(3)]
    data = [{"group": "CTL", "value": v} for v in rng.normal(0, 1, 80)] + [{"group": "DEP", "value": v} for v in rng.normal(1, 1, 50)]
    row = literature_vs_ours(claims, lambda names: data, REGIONS)[0]
    assert row["papers"] == 3 and row["expected_in_dep"] == "higher" and row["verdict"].startswith("replicated")
    assert "3 report higher" in comparison_facts(row)[0]
