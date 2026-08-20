"""Fetch a corpus of arXiv quantitative-finance abstracts.

The repo commits a manifest (document ids plus content checksums), not the
documents. That is a deliberate trade against convenience, for two reasons:

1. Reproducibility. recall@k is only comparable across runs if every run
   retrieves over the same documents. The manifest pins the exact snapshot;
   `python -m rag_eval_lab.loader --refetch` rebuilds it byte-for-byte, and
   checksums catch the case where arXiv serves a revised abstract.
2. Redistribution. Abstract text is the authors' copyright. Shipping ids and
   a loader is unambiguously fine; shipping 300 abstracts is a grey zone a
   portfolio repo does not need to sit in.

Nothing downstream knows about arXiv: any JSONL with doc_id, title, and
abstract fields works, which is what makes the corpus swappable.

arXiv API docs: https://info.arxiv.org/help/api/user-manual.html
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import time
import urllib.parse
import urllib.request
from pathlib import Path
from xml.etree import ElementTree

ARXIV_API = "http://export.arxiv.org/api/query"
ATOM = "{http://www.w3.org/2005/Atom}"
ARXIV_NS = "{http://arxiv.org/schemas/atom}"

# All nine q-fin subcategories, OR-ed explicitly because the API's wildcard
# behaviour is undocumented and this query is cheap either way.
QFIN_CATEGORIES = [
    "q-fin.CP", "q-fin.EC", "q-fin.GN", "q-fin.MF", "q-fin.PM",
    "q-fin.PR", "q-fin.RM", "q-fin.ST", "q-fin.TR",
]

PAGE_SIZE = 100
# arXiv asks for a 3 second gap between requests; ignoring that gets you
# silently served empty pages, which looks exactly like a bug in your code.
REQUEST_GAP_SECONDS = 3.0

DEFAULT_CORPUS = Path("data/corpus.jsonl")
DEFAULT_MANIFEST = Path("data/corpus_manifest.json")


def _normalise_whitespace(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def _abstract_checksum(abstract: str) -> str:
    return hashlib.sha256(abstract.encode("utf-8")).hexdigest()[:16]


def _parse_entry(entry: ElementTree.Element) -> dict:
    raw_id = entry.findtext(f"{ATOM}id") or ""
    # The Atom id is a URL like http://arxiv.org/abs/2408.01234v1; the tail is
    # the stable document id used everywhere downstream.
    doc_id = raw_id.rsplit("/", 1)[-1]
    categories = [
        c.get("term", "") for c in entry.findall(f"{ATOM}category") if c.get("term")
    ]
    primary = entry.find(f"{ARXIV_NS}primary_category")
    return {
        "doc_id": doc_id,
        "title": _normalise_whitespace(entry.findtext(f"{ATOM}title") or ""),
        "abstract": _normalise_whitespace(entry.findtext(f"{ATOM}summary") or ""),
        "categories": categories,
        "primary_category": (
            primary.get("term", "") if primary is not None
            else (categories[0] if categories else "")
        ),
        "published": entry.findtext(f"{ATOM}published") or "",
        "authors": [
            _normalise_whitespace(a.findtext(f"{ATOM}name") or "")
            for a in entry.findall(f"{ATOM}author")
        ],
        "url": raw_id,
    }


def _query_api(params: dict) -> list[dict]:
    encoded = urllib.parse.urlencode(params)
    with urllib.request.urlopen(f"{ARXIV_API}?{encoded}", timeout=30) as resp:
        root = ElementTree.fromstring(resp.read())
    return [_parse_entry(e) for e in root.findall(f"{ATOM}entry")]


def fetch_fresh(n: int = 300, categories: list[str] | None = None) -> list[dict]:
    """Fetch the n most recently submitted abstracts across the given categories."""
    categories = categories or QFIN_CATEGORIES
    query = " OR ".join(f"cat:{c}" for c in categories)
    docs: list[dict] = []
    seen: set[str] = set()
    start = 0
    while len(docs) < n:
        entries = _query_api(
            {
                "search_query": query,
                "start": start,
                "max_results": PAGE_SIZE,
                "sortBy": "submittedDate",
                "sortOrder": "descending",
            }
        )
        if not entries:
            break
        for doc in entries:
            # Cross-listed papers appear once per category; keep the first.
            if doc["doc_id"] in seen or not doc["abstract"]:
                continue
            seen.add(doc["doc_id"])
            docs.append(doc)
            if len(docs) >= n:
                break
        start += PAGE_SIZE
        time.sleep(REQUEST_GAP_SECONDS)
    return docs


def fetch_by_ids(doc_ids: list[str]) -> list[dict]:
    """Fetch specific documents, preserving the manifest's order."""
    by_id: dict[str, dict] = {}
    for i in range(0, len(doc_ids), PAGE_SIZE):
        batch = doc_ids[i : i + PAGE_SIZE]
        # Versioned ids (2408.01234v1) are accepted by id_list and pin the
        # exact revision, so pass them through untouched.
        entries = _query_api({"id_list": ",".join(batch), "max_results": PAGE_SIZE})
        for doc in entries:
            by_id[doc["doc_id"]] = doc
        if i + PAGE_SIZE < len(doc_ids):
            time.sleep(REQUEST_GAP_SECONDS)
    return [by_id[d] for d in doc_ids if d in by_id]


def write_corpus(docs: list[dict], path: Path = DEFAULT_CORPUS) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for doc in docs:
            f.write(json.dumps(doc, ensure_ascii=False) + "\n")


def write_manifest(docs: list[dict], path: Path = DEFAULT_MANIFEST) -> None:
    # One doc per line keeps the file small and its git diffs readable when
    # the snapshot is ever regenerated.
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write('{\n"source": "arXiv API, q-fin categories, sorted by submittedDate desc",\n')
        f.write(f'"n_docs": {len(docs)},\n"docs": {{\n')
        lines = [
            f'"{d["doc_id"]}": "{_abstract_checksum(d["abstract"])}"' for d in docs
        ]
        f.write(",\n".join(lines))
        f.write("\n}\n}\n")


def verify_against_manifest(docs: list[dict], path: Path = DEFAULT_MANIFEST) -> list[str]:
    """Return doc_ids whose text no longer matches the manifest checksum."""
    with open(path, encoding="utf-8") as f:
        manifest = json.load(f)
    expected = manifest["docs"]
    return [
        d["doc_id"]
        for d in docs
        if d["doc_id"] in expected
        and _abstract_checksum(d["abstract"]) != expected[d["doc_id"]]
    ]


def load_corpus(path: str | Path = DEFAULT_CORPUS) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--refetch",
        action="store_true",
        help="rebuild the pinned snapshot from the committed manifest",
    )
    parser.add_argument("--n", type=int, default=300, help="fresh fetch size")
    parser.add_argument("--out", default=str(DEFAULT_CORPUS))
    parser.add_argument("--manifest", default=str(DEFAULT_MANIFEST))
    args = parser.parse_args()

    if args.refetch:
        with open(args.manifest, encoding="utf-8") as f:
            manifest = json.load(f)
        ids = list(manifest["docs"])
        docs = fetch_by_ids(ids)
        drifted = verify_against_manifest(docs, Path(args.manifest))
        missing = len(ids) - len(docs)
        write_corpus(docs, Path(args.out))
        print(f"refetched {len(docs)}/{len(ids)} documents to {args.out}")
        if missing:
            print(f"warning: {missing} documents no longer retrievable")
        if drifted:
            print(f"warning: {len(drifted)} abstracts changed since the manifest: {drifted}")
    else:
        docs = fetch_fresh(n=args.n)
        write_corpus(docs, Path(args.out))
        write_manifest(docs, Path(args.manifest))
        print(f"wrote {len(docs)} documents to {args.out} and manifest to {args.manifest}")


if __name__ == "__main__":
    main()
