"""Check which AIP chapters are in the PDF text vs which are being parsed."""
import json
import re
import requests
import tempfile
import pdfplumber
from pathlib import Path
from collections import defaultdict

from app.services.section_parser import AIP_SUBSECTION_PATTERN, _looks_like_toc_block

with open("data/regulations_manifest.json") as f:
    docs = json.load(f)

aip = next(d for d in docs if d["type"] == "AIP")
path = aip["path"].replace("\\", "/")
url = "https://pub-a32237578ade418f9375e48bb3f1982a.r2.dev/" + path

print("Downloading AIP...")
r = requests.get(url, timeout=180)
tmp = Path(tempfile.mktemp(suffix=".pdf"))
tmp.write_bytes(r.content)

all_text = ""
with pdfplumber.open(tmp) as pdf:
    for page in pdf.pages:
        all_text += (page.extract_text() or "") + "\n"
tmp.unlink()
print(f"Total chars: {len(all_text)}")

# 1. Find all AIP page markers and build a chapter → page count map
PAGE_PATTERN = re.compile(r"\b(GEN|ENR|AD)\s+(\d+(?:\.\d+)?)\s*-\s*\(?\d+\)?", re.IGNORECASE)
chapter_pages = defaultdict(set)
for m in PAGE_PATTERN.finditer(all_text):
    chapter_type = m.group(1).upper()
    chapter_num = m.group(2)
    chapter_key = f"AIP {chapter_type} {chapter_num}"
    chapter_pages[chapter_key].add(m.group(0))

print("\n=== AIP Chapters in PDF (by page marker count) ===")
for chapter in sorted(chapter_pages.keys()):
    count = len(chapter_pages[chapter])
    print(f"  {chapter:25s}  {count:3d} page markers")

# 2. Find all AIP_SUBSECTION_PATTERN matches and group by chapter
print("\n=== AIP Sections parsed per chapter ===")
matches = list(AIP_SUBSECTION_PATTERN.finditer(all_text))
chapter_sections = defaultdict(list)
for index, match in enumerate(matches):
    if "....." in match.group("heading"):
        continue
    start = match.start()
    end = matches[index + 1].start() if index + 1 < len(matches) else len(all_text)
    section_text = all_text[start:end].strip()
    if _looks_like_toc_block(section_text) or len(section_text) < 200:
        continue
    # Find nearest page marker before this match
    ctx = all_text[max(0, start - 2000):start]
    markers = re.findall(r"(GEN|ENR|AD)\s+(\d+(?:\.\d+)?)\s*-\s*\(?\d+\)?", ctx, re.IGNORECASE)
    if markers:
        ch_type, ch_num = markers[-1]
        chapter_key = f"AIP {ch_type.upper()} {ch_num}"
    else:
        chapter_key = "UNKNOWN"
    chapter_sections[chapter_key].append(match.group("label"))

for chapter in sorted(chapter_sections.keys()):
    count = len(chapter_sections[chapter])
    in_pdf = chapter in chapter_pages
    print(f"  {chapter:25s}  {count:4d} sections parsed  (in_pdf={in_pdf})")

# 3. Gap analysis: chapters in PDF with 0 sections parsed
print("\n=== GAPS: chapters in PDF but not parsed ===")
for chapter in sorted(chapter_pages.keys()):
    parsed_count = len(chapter_sections.get(chapter, []))
    page_count = len(chapter_pages[chapter])
    if parsed_count == 0:
        print(f"  {chapter:25s}  {page_count:3d} page markers  ZERO SECTIONS PARSED")
    elif parsed_count < page_count * 0.3:
        print(f"  {chapter:25s}  {page_count:3d} page markers  only {parsed_count} sections (low density)")

# 4. Check citation bug: AIP ENR 1.5 91.267
print("\n=== Citation bug check (CASR numbers as section IDs) ===")
casr_pattern = re.compile(r"\b91\.\d+\b")
for index, match in enumerate(matches):
    label = match.group("label")
    if re.match(r"^\d{2,3}\.\d+", label):  # looks like a CASR number
        start = match.start()
        ctx = all_text[max(0, start-500):start]
        markers = re.findall(r"(GEN|ENR|AD)\s+(\d+(?:\.\d+)?)\s*-\s*\(?\d+\)?", ctx, re.IGNORECASE)
        if markers:
            ch_type, ch_num = markers[-1]
            print(f"  label={label!r:15} in chapter AIP {ch_type.upper()} {ch_num}")
