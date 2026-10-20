import time
import xml.etree.ElementTree as ET

import requests


def search(cfg: dict) -> list[str]:
    r = requests.get(cfg["esearch"], params={"db": "pubmed", "term": " ".join(cfg["query"].split()),
                                             "retmax": cfg["retmax"], "sort": "relevance", "retmode": "json",
                                             "tool": cfg["tool"]}, timeout=30)
    r.raise_for_status()
    return r.json()["esearchresult"]["idlist"]


def parse_articles(xml_text: str) -> list[dict]:
    out = []
    for art in ET.fromstring(xml_text).iter("PubmedArticle"):
        parts = [(a.get("Label"), "".join(a.itertext()).strip()) for a in art.iter("AbstractText")]
        abstract = " ".join(f"{lab}: {t}" if lab else t for lab, t in parts if t)
        year = art.findtext(".//JournalIssue/PubDate/Year") or (art.findtext(".//JournalIssue/PubDate/MedlineDate") or "")[:4]
        out.append({"pmid": art.findtext(".//PMID"), "title": "".join(art.find(".//ArticleTitle").itertext()).strip(),
                    "journal": art.findtext(".//Journal/Title"), "year": int(year) if year.isdigit() else None,
                    "abstract": abstract})
    return [a for a in out if a["abstract"]]


def post(url: str, data: dict, tries: int = 4) -> str:
    for k in range(tries):  # NCBI returns occasional 502/429; back off and retry
        r = requests.post(url, data=data, timeout=60)
        if r.ok:
            return r.text
        time.sleep(2 ** k)
    r.raise_for_status()


def fetch(cfg: dict, pmids: list[str], batch: int = 50) -> list[dict]:
    arts = []
    for k in range(0, len(pmids), batch):
        time.sleep(cfg["delay_s"])  # polite: well under NCBI's 3 requests/s
        arts += parse_articles(post(cfg["efetch"], {"db": "pubmed", "id": ",".join(pmids[k:k + batch]),
                                                    "retmode": "xml", "tool": cfg["tool"]}))
    return arts
