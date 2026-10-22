import re

from presidio_analyzer import AnalyzerEngine, Pattern, PatternRecognizer, RecognizerResult
from presidio_analyzer.nlp_engine import NlpEngineProvider

from neurograph.config import site

TAGS = {"PERSON": "<PERSON>", "EMAIL_ADDRESS": "<EMAIL>", "PHONE_NUMBER": "<PHONE>"}
# names after an honorific ("Dr. Farida Begum", "Md. Abu Syeed"); spaCy's small model misses many Bangla names
HONORIFIC = r"\b(?:Dr|Md|Mr|Mrs|Ms|Prof|Engr)\.?[ ]?[A-Z][a-zA-Z]+(?:\.?[ ][A-Z][a-zA-Z]+){0,4}"  # one line only
BD_PHONE = r"(?:\+?880[\s-]?|\b0)1[3-9]\d{2}[\s-]?\d{6}\b|\b0\d{4}[\s-]?\d{6}\b"


def build_analyzer(model: str) -> AnalyzerEngine:
    nlp = NlpEngineProvider(nlp_configuration={"nlp_engine_name": "spacy",
                                               "models": [{"lang_code": "en", "model_name": model}]}).create_engine()
    engine = AnalyzerEngine(nlp_engine=nlp, supported_languages=["en"])
    flags = re.DOTALL | re.MULTILINE  # case-sensitive: Presidio ignores case by default ("Dr" would match "drop")
    engine.registry.add_recognizer(PatternRecognizer("PERSON", patterns=[Pattern("honorific", HONORIFIC, 0.85)],
                                                     name="honorific_name", global_regex_flags=flags))
    engine.registry.add_recognizer(PatternRecognizer("PHONE_NUMBER", patterns=[Pattern("bd", BD_PHONE, 0.8)],
                                                     name="bd_phone", global_regex_flags=flags))
    return engine


ROLE = re.compile(r"Office|Room|Ext|Department|Dept|Registrar|Officer|Counsel|Proctor|Professor|Director|Advisor|"
                  r"Head|Nurse|Lecturer|Dean|Manager|Assistant|Asst|Coordinator|Science|Engineering|University|"
                  r"Welfare|Medical|Center|Centre|Policy|Committee|" + re.escape(site()["short"]))


def directory_names(text: str) -> list[RecognizerResult]:
    # staff directories: a short capitalised line followed within 4 lines by "Email:" is a person's name
    out, pos, lines = [], 0, text.split("\n")
    for i, line in enumerate(lines):
        words = line.split()
        if (2 <= len(words) <= 5 and all(w[0].isupper() for w in words if w[0].isalpha())
                and not re.search(r"\d|:|@", line) and not ROLE.search(line)
                and any(l.startswith("Email") for l in lines[i + 1:i + 7])):
            out.append(RecognizerResult("PERSON", pos, pos + len(line), 0.9))
        pos += len(line) + 1
    return out


NOT_NAME = re.compile(r"Grade|Academic|Calendar|Scholarship|Program|Medal|\bApp\b|Online|Helpdesk|Email|Account|"
                      r"Address|Google|Drive|Fighters|Report|Dismissal|Present|Taka|Merit|Course|Semester|Exam|"
                      r"Bangladesh|Dhaka|Board|Council|Trust|Fund|Award|Prize|Library|Club|Hall|Types|Directive|"
                      r"Principle|Rule|Quorum|Retire|Graduate|Name\b|Requirement|Guideline|Handling|Improvement|"
                      r"Prohibited|Suite|Desk|Activating|Profile|Registration")
AWARD = r"(?:Scholarship|Fund|Award|Trust|Prize|Foundation)\b"
NAMED_AWARD = re.compile(r"^[^\n]{0,40}?" + AWARD)  # award named after a donor follows on the same line


def trim_role(text: str, start: int, end: int) -> int:
    # "Dr. Ahmed Wasif Reza Dean" -> stop before the first role word or line break
    value = text[start:end]
    cut = min([m.start() for m in [ROLE.search(value)] if m] + [value.find("\n") if "\n" in value else len(value)])
    return start + len(value[:cut].rstrip(" ,."))


def plausible_name(value: str) -> bool:
    # spaCy's small model tags many capitalised phrases as PERSON; accept only name-shaped ones
    words = value.split()
    return (2 <= len(words) <= 5 and all(w[0].isupper() for w in words) and "\n" not in value
            and not re.search(r"\d", value) and not ROLE.search(value) and not NOT_NAME.search(value))


def redact(text: str, engine: AnalyzerEngine, entities: list[str], keep: list[str],
           threshold: float = 0.5) -> tuple[str, dict[str, int]]:
    found = engine.analyze(text=text, language="en", entities=entities, score_threshold=threshold)
    found += [r for r in directory_names(text) if r.entity_type in entities]
    hits = []
    for r in found:
        start, end, value = r.start, r.end, text[r.start:r.end]
        if any(k.lower() in value.lower() for k in keep):  # role-based office contacts stay
            continue
        if r.entity_type == "PERSON":
            if re.search(AWARD, value) or NAMED_AWARD.match(text[end:end + 60]):  # "Sofia Khatun Scholarship" = award
                continue
            source = (r.recognition_metadata or {}).get("recognizer_name", "")
            if source == "SpacyRecognizer" and not plausible_name(value):
                continue
            end = trim_role(text, start, end)
            if end - start < 3:
                continue
        hits.append((start, end, r.entity_type))
    spans, counts = [], {}
    for start, end, kind in sorted(hits, key=lambda h: (h[0], -(h[1] - h[0]))):
        if spans and start < spans[-1][1]:  # overlapping hits: merge
            spans[-1] = (spans[-1][0], max(spans[-1][1], end), spans[-1][2])
            continue
        spans.append((start, end, kind))
    for start, end, kind in reversed(spans):
        text = text[:start] + TAGS[kind] + text[end:]
        counts[kind] = counts.get(kind, 0) + 1
    return text, counts
