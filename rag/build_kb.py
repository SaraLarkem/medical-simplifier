"""
build_kb.py
Project: Medical Information Simplifier
Purpose: Turn the MedlinePlus health-topics XML into a small local knowledge base
         (one JSON object per line) that the retriever can search offline.

Input:   the newest  data/kb/raw/mplus_topics_YYYY-MM-DD.xml
         Download "MedlinePlus Health Topic XML" from https://medlineplus.gov/xml.html
Output:  data/kb/medlineplus.jsonl       id, title, aliases, summary, url, source
         data/kb/medlineplus_meta.json   which XML it was built from, and counts

Only English topics are kept. Summaries are stored as plain text (HTML removed) and in
full; shortening happens at retrieval time.

Usage:
    python -m rag.build_kb
    python -m rag.build_kb --raw data/kb/raw --out data/kb/medlineplus.jsonl
"""

import argparse
import json
import re
import sys
import xml.etree.ElementTree as ET
from html.parser import HTMLParser
from pathlib import Path

RAW_PATTERN = re.compile(r"^mplus_topics_(\d{4}-\d{2}-\d{2})\.xml$")
SOURCE_NAME = "MedlinePlus"

# Tags after which a space is inserted, so words from separate paragraphs/list items
# don't run together when the HTML is flattened.
_BLOCK_TAGS = {"p", "br", "li", "ul", "ol", "h1", "h2", "h3", "h4", "div", "tr", "td"}


class _TextExtractor(HTMLParser):
    """Collects the visible text of an HTML fragment (entities are decoded by HTMLParser)."""

    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.parts = []

    def handle_starttag(self, tag, attrs):
        if tag in _BLOCK_TAGS:
            self.parts.append(" ")

    def handle_endtag(self, tag):
        if tag in _BLOCK_TAGS:
            self.parts.append(" ")

    def handle_data(self, data):
        self.parts.append(data)


def html_to_text(fragment):
    """Strip tags from an HTML fragment and collapse whitespace."""
    parser = _TextExtractor()
    parser.feed(fragment or "")
    parser.close()
    return re.sub(r"\s+", " ", "".join(parser.parts)).strip()


def find_latest_xml(raw_dir):
    """Return the mplus_topics_*.xml with the newest date in its filename."""
    candidates = []
    for path in Path(raw_dir).glob("mplus_topics_*.xml"):
        m = RAW_PATTERN.match(path.name)
        if m:
            candidates.append((m.group(1), path))
    if not candidates:
        sys.exit(
            f"No mplus_topics_YYYY-MM-DD.xml found in {raw_dir}.\n"
            "Download 'MedlinePlus Health Topic XML' from https://medlineplus.gov/xml.html "
            f"and save it in {raw_dir}/ without renaming it."
        )
    return max(candidates)[1]


def unique_aliases(title, names):
    """De-duplicate aliases case-insensitively and drop any that just repeat the title."""
    seen = {title.casefold()}
    out = []
    for name in names:
        name = re.sub(r"\s+", " ", name or "").strip()
        if name and name.casefold() not in seen:
            seen.add(name.casefold())
            out.append(name)
    return out


def parse_topics(xml_path):
    """Yield one record per English health topic, plus counts of what was skipped."""
    root = ET.parse(xml_path).getroot()
    stats = {"date_generated": root.get("date-generated"), "topics_in_file": len(root),
             "skipped_non_english": 0, "missing_summary": 0}
    records = []
    for topic in root.iter("health-topic"):
        if topic.get("language") != "English":
            stats["skipped_non_english"] += 1
            continue
        title = topic.get("title", "").strip()
        # "also-called" = synonyms shown on the page; "see-reference" = alternative names
        # that redirect to this topic (e.g. HbA1c -> A1C). Both are useful for matching.
        names = [e.text for e in topic.findall("also-called")]
        names += [e.text for e in topic.findall("see-reference")]
        summary = html_to_text(topic.findtext("full-summary"))
        if not summary:
            stats["missing_summary"] += 1
        records.append({
            "id": f"mlp-{topic.get('id')}",
            "title": title,
            "aliases": unique_aliases(title, names),
            "summary": summary,
            "url": topic.get("url", "").strip(),
            "source": SOURCE_NAME,
        })
    return records, stats


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--raw", default="data/kb/raw", help="folder containing mplus_topics_*.xml")
    ap.add_argument("--out", default="data/kb/medlineplus.jsonl")
    args = ap.parse_args()

    xml_path = find_latest_xml(args.raw)
    print(f"Reading {xml_path} ...")
    records, stats = parse_topics(xml_path)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    meta = {
        "source": SOURCE_NAME,
        "xml_file": xml_path.name,
        **stats,
        "topics_written": len(records),
        "aliases_total": sum(len(r["aliases"]) for r in records),
    }
    meta_path = out.with_name(out.stem + "_meta.json")
    meta_path.write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")

    print(f"Topics in file:        {stats['topics_in_file']}")
    print(f"Skipped (not English): {stats['skipped_non_english']}")
    print(f"Written:               {len(records)}  ({meta['aliases_total']} aliases)")
    print(f"Without a summary:     {stats['missing_summary']}")
    print(f"Saved: {out} and {meta_path}")


if __name__ == "__main__":
    main()
