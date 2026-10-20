import pytest

from neurograph.agent.safety import check, referral
from neurograph.agent.tools import read_only


@pytest.mark.parametrize("msg", ["I want to kill myself", "I don't want to live anymore",
                                 "sometimes I think about hurting myself", "thinking about suicide"])
def test_crisis(msg):
    assert check(msg) == "crisis"


@pytest.mark.parametrize("msg", ["I'm so stressed I can't sleep", "my anxiety is getting worse",
                                 "I've been feeling really low"])
def test_distress(msg):
    assert check(msg) == "distress"


@pytest.mark.parametrize("msg", ["What CGPA puts me on probation?", "Does our EEG data replicate FAA in depression?",
                                 "How do I activate my email?"])
def test_normal_questions_not_flagged(msg):
    assert check(msg) is None


def test_crisis_referral_has_emergency_number_and_counsellors():
    text = referral("crisis", [{"name": "Psychosocial counselling", "location": "Room 301",
                                "contact": "swd@university.example", "url": "https://www.university.example"}])
    assert "999" in text and "Room 301" in text and "not a counsellor" in text


@pytest.mark.parametrize("cypher", ["MATCH (n) DETACH DELETE n", "CREATE (n:X)", "MATCH (n) SET n.x = 1",
                                    "CALL dbms.components()", "MERGE (a:A)"])
def test_write_cypher_rejected(cypher):
    with pytest.raises(PermissionError):
        read_only(cypher)


def test_read_cypher_allowed():
    assert read_only("MATCH (p:Participant) RETURN count(p)")
