import io
import re

from bs4 import BeautifulSoup
from pypdf import PdfReader

FIXES = {"�": "'", "’": "'", "‘": "'", "“": '"', "”": '"', "–": "-", "—": "-",
         "\xa0": " "}


def normalise(text: str) -> str:
    for bad, good in FIXES.items():
        text = text.replace(bad, good)
    lines = [re.sub(r"[ \t]+", " ", l).strip() for l in text.splitlines()]
    return "\n".join(l for l in lines if l)


def html_text(raw: bytes, start_prefix: str, stop_markers: list[str]) -> tuple[str, str]:
    # the site's pages: main content starts after the breadcrumb ("Home/...") and ends at the footer or side menu
    soup = BeautifulSoup(raw, "html.parser")
    title = normalise(soup.title.get_text(strip=True)) if soup.title else ""
    for t in soup(["script", "style", "noscript"]):
        t.decompose()
    for tr in soup.find_all("tr"):  # keep table rows on one line: "80% and above | A+ | 4.00"
        cells = [c.get_text(" ", strip=True) for c in tr.find_all(["td", "th"])]
        tr.replace_with(soup.new_string(" | ".join(c for c in cells if c) + "\n"))
    lines = normalise(soup.get_text("\n")).splitlines()
    start = next((i + 1 for i, l in enumerate(lines) if l.startswith(start_prefix)), 0)
    end = next((i for i, l in enumerate(lines[start:], start) if any(l.lower().startswith(m.lower())
                                                                    for m in stop_markers)), len(lines))
    return title, "\n".join(lines[start:end])


def pdf_text(raw: bytes) -> tuple[str, str]:
    pages = [p.extract_text() or "" for p in PdfReader(io.BytesIO(raw)).pages]
    text = normalise("\n".join(pages))
    title = next((l for l in text.splitlines() if "policy" in l.lower()), text.splitlines()[0] if text else "")
    return title, text
