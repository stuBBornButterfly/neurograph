import numpy as np

from neurograph.agent.replication import compare_groups, faa_facts
from neurograph.agent.tools import PARTICIPANT_IDX, TIER_IDX, study_facts


def test_participant_total_is_summed_in_python():
    rows = [{"group": "CTL", "sex": "F", "n": 40}, {"group": "CTL", "sex": "M", "n": 35},
            {"group": "DEP", "sex": "F", "n": 33}, {"group": "DEP", "sex": "M", "n": 12}]
    facts = study_facts(PARTICIPANT_IDX, rows)
    assert "120 participants in total" in facts[0] and "Group DEP: 45 participants (33 female)." in facts


def test_tier_counts_add_both_groups():
    rows = [{"tier": "high", "ctl": 0, "dep": 2, "mean_p": 0.7}]
    assert study_facts(TIER_IDX, rows) == ["The high risk tier has 2 participants (0 controls, 2 depression group)."]


def test_verdict_directions():
    rng = np.random.default_rng(0)
    ctl = rng.normal(0, 1, 200)
    assert compare_groups(ctl, rng.normal(-1, 1, 200), True)["verdict"].startswith("replicated")
    assert "opposite direction" in compare_groups(ctl, rng.normal(1, 1, 200), True)["verdict"]
    assert compare_groups(ctl, ctl + 0.001, False)["verdict"].startswith("not replicated: same direction")


def test_faa_facts_say_not_replicated_when_dep_is_higher():
    rng = np.random.default_rng(1)
    rows = [{"group": "CTL", "value": v} for v in rng.normal(-0.03, 0.1, 75)] + \
           [{"group": "DEP", "value": v} for v in rng.normal(0.05, 0.1, 45)]
    facts = faa_facts(rows)
    assert "higher" in facts[2] and "not replicated" in facts[-1]


def test_verifier_flags_a_swapped_direction_about_our_data():
    from neurograph.agent.verifier import direction_conflicts
    facts = "In our data the depression group is higher by 0.073 (Cohen's d 0.45, Welch t-test p 0.016)."
    wrong = ("Our EEG data does not replicate the published finding of higher frontal alpha asymmetry in depression. "
             "The depression group shows a lower asymmetry score, with a significant difference.")
    right = ("The depression group had a higher mean of 0.046. Published studies report a lower score in "
             "depressed patients, so the finding is not replicated.")
    assert direction_conflicts(wrong, facts) == ["The depression group shows a lower asymmetry score, with a significant difference."]
    assert direction_conflicts(right, facts) == []
    assert direction_conflicts("The depression group is not lower than controls here.", facts) == []
    mixed = facts + " In our data (eyes closed) the depression group is lower by 0.05."
    assert direction_conflicts(wrong, mixed) == []  # directions disagree across biomarkers: no check
