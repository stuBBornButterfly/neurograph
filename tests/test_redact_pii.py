import pytest

from neurograph.ingest.redact_pii import build_analyzer, redact

ENTITIES = ["PERSON", "EMAIL_ADDRESS", "PHONE_NUMBER"]
KEEP = ["swd@example.edu", "09666775577"]


@pytest.fixture(scope="module")
def engine():
    return build_analyzer("en_core_web_sm")


def test_staff_directory_names_and_emails_removed(engine):
    text = ("Farzana Akter Srabony\nPsychosocial Counsellor\nRoom No: 301\nEmail: farzana.srabony@example.edu\n"
            "Dr. Anisur Rahman\nProctor, University\nExt: 256\nEmail: anis@example.edu")
    out, counts = redact(text, engine, ENTITIES, KEEP)
    assert "Srabony" not in out and "Anisur" not in out and "@example.edu" not in out
    assert "Psychosocial Counsellor" in out and "Room No: 301" in out and "Proctor, University" in out  # roles stay
    assert counts == {"PERSON": 2, "EMAIL_ADDRESS": 2}


def test_bangladeshi_mobile_removed_office_line_kept(engine):
    out, _ = redact("Call 01712-345678 or the hotline 09666775577.", engine, ENTITIES, KEEP)
    assert "01712" not in out and "<PHONE>" in out and "09666775577" in out


def test_office_email_kept(engine):
    out, _ = redact("Write to swd@example.edu for an appointment.", engine, ENTITIES, KEEP)
    assert "swd@example.edu" in out


def test_rules_text_untouched(engine):
    text = ("Students may drop a course during the add/drop period. Dean's List requires a CGPA of 3.80. "
            "Your professor may invite you to Google Classroom. Letter Grade B+ equals 3.25.")
    out, counts = redact(text, engine, ENTITIES, KEEP)
    assert out == text and counts == {}


def test_named_scholarship_kept(engine):
    text = "The Rowshan Ara Begum Scholarship and the Dr. Ameerul Haque Scholarship are awarded each year."
    out, _ = redact(text, engine, ENTITIES, KEEP)
    assert out == text
