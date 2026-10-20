import json

import pytest

from neurograph.agent.backends import ApiLLM
from neurograph.agent.snapshot import SnapshotTools, claim_matches
from neurograph.finetune.make_dataset import neighbours
from scripts.run_ragas import clean_answer, prompt_passages

SNAP = {"created": "2026-10-21T00:00:00+00:00", "git": "abc",
        "services": [{"name": "Psychosocial counselling", "location": "Room 301", "contact": "swd@university.example", "url": "u"}],
        "rules": {"p-001": ["Rule (probation): if CGPA below 2.25 -> probation."]},
        "study": [{"title": f"q{i}", "rows": [{"i": i}], "facts": [f"fact {i}"]} for i in range(10)],
        "claims": [{"pmid": "1", "year": 2020, "text": "t1", "band": "alpha", "measure": "power", "region": "frontal",
                    "direction": "lower", "entail": 0.9},
                   {"pmid": "2", "year": 2021, "text": "t2", "band": "theta", "measure": "power", "region": "global",
                    "direction": "higher", "entail": 0.8},
                   {"pmid": "3", "year": 2022, "text": "t3", "band": "alpha", "measure": "asymmetry", "region": "frontal",
                    "direction": "lower", "entail": 0.7}],
        "comparisons": [{"band": "alpha", "measure": "power", "region": "frontal", "papers": 1, "lit_higher": 0,
                         "lit_lower": 1, "lit_no_diff": 0, "diff": -0.1, "cohens_d": -0.2, "p": 0.3,
                         "verdict": "not replicated: same direction as the published finding but not significant"},
                        {"band": "alpha", "measure": "asymmetry", "region": "frontal", "papers": 1, "lit_higher": 0,
                         "lit_lower": 1, "lit_no_diff": 0, "diff": 0.07, "cohens_d": 0.4, "p": 0.04,
                         "verdict": "not replicated: the difference is significant but in the opposite direction"}]}


@pytest.fixture()
def snap(tmp_path):
    p = tmp_path / "graph_snapshot.json"
    p.write_text(json.dumps(SNAP))
    return SnapshotTools(p)


def test_snapshot_routes_like_the_live_graph(snap):
    assert snap.study("Does our EEG data replicate frontal alpha asymmetry?")[0] == "q6"  # FAA query
    assert snap.study("How many participants are in the high risk tier?")[0] == "q9"
    assert snap.rules_for(["p-001", "p-999"]) == SNAP["rules"]["p-001"]


def test_snapshot_literature_filters_claims_and_comparisons(snap):
    claims, facts = snap.literature("What do published studies say about frontal alpha asymmetry?")
    assert [c["pmid"] for c in claims] == ["3"] and "Verdict computed in Python" in facts[-1]
    claims, _ = snap.literature("Does the literature report more theta power over frontal sites?")
    assert [c["pmid"] for c in claims] == ["2"]  # 'global' claims count for every region
    assert claim_matches(SNAP["claims"][0], {"band": None, "region": None, "measure": None})


def test_neighbour_chunks_share_a_page():
    chunks = [{"id": f"a-{i}", "url": "a", "chunk": i} for i in range(3)] + [{"id": "b-0", "url": "b", "chunk": 0}]
    by_pos = {(c["url"], c["chunk"]): c for c in chunks}
    assert [n["id"] for n in neighbours(chunks[1], by_pos)] == ["a-0", "a-2"]
    assert neighbours(chunks[3], by_pos) == []


def test_api_key_must_come_from_the_environment(monkeypatch):
    monkeypatch.delenv("BENCH_TEST_KEY", raising=False)
    with pytest.raises(RuntimeError):
        ApiLLM("openai", "https://example.invalid/v1", "m", 10, api_key_env="BENCH_TEST_KEY", external=True)
    with pytest.raises(ValueError):
        ApiLLM("grpc", "x", "m", 10)


def test_ragas_inputs_drop_agent_notes_and_split_passages():
    ans = "Probation starts below 2.25 [1].\n\n(Note: I could not fully verify this answer.)\n\nSources: https://x"
    assert clean_answer(ans) == "Probation starts below 2.25."
    user = "[1] Grades\nCGPA rules here.\n\n[2] FAQ\nOnline classes.\n\nQuestion: What is probation?"
    assert prompt_passages(user) == ["Grades\nCGPA rules here.", "FAQ\nOnline classes."]


def test_log_redacts_questions_and_skips_distress(tmp_path):
    from neurograph.app.monitoring import InteractionLog
    log = InteractionLog(tmp_path)
    res = {"id": "abc123", "route": "policy", "answer": "a", "verdict": "supported", "entailed_share": 1.0,
           "unsupported_numbers": [], "prompt_tokens": 5, "completion_tokens": 3, "seconds": 1.0}
    log.answer("My number is 01712-345678, when is the add/drop period?", res, "llm")
    log.answer("I want to kill myself", {**res, "id": "def456", "route": "safety:crisis"}, "llm")
    log.feedback("abc123", "up", "")
    recs = log.records()
    assert "01712" not in recs[0]["question"] and "<PHONE>" in recs[0]["question"]
    assert recs[1]["question"] is None and recs[1]["answer"] is None
    s = log.stats()
    assert s["answers"] == 2 and s["feedback_up"] == 1 and s["routes"] == {"policy": 1, "safety:crisis": 1}
