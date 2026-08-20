"""Retrieval evaluation: recall@k and MRR over the labelled question set.

Retrieval and generation are evaluated separately because "the answer was
wrong" has two different causes with two different fixes. If the right
document never reached the context window, the generator never had a chance
and no amount of prompting will help; if it was retrieved and the answer is
still wrong, retrieval is exonerated and the generation step owns the
failure. Measuring them together produces one number that cannot tell you
which lever to pull.

Metrics are computed at the document level. Retrievers rank chunks, but the
relevance labels live on documents (a human can say "this paper answers the
question" far more reliably than "this 220-word window answers it"), so
chunk rankings are collapsed to their best-ranked document first.

What these metrics do NOT catch, stated plainly:
- recall@k with self-authored labels measures agreement with the label
  author. An unlabelled but genuinely relevant document scores as a miss for
  the retriever when it is really a miss for the labeller.
- MRR only sees the first relevant hit; a system that finds one of three
  relevant documents looks identical to one that finds all three.
- Neither metric knows whether the retrieved text is sufficient to answer,
  only that the right document was ranked. Sufficiency shows up in the
  generation-side groundedness evaluation instead.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

DEFAULT_QUESTIONS = Path("eval/questions.jsonl")
RESULTS_DIR = Path("results")
KS = (1, 3, 5, 10)


def load_questions(path: str | Path = DEFAULT_QUESTIONS) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def doc_ranking(chunk_ranking: list[str]) -> list[str]:
    """Collapse a ranked list of chunk_ids to doc_ids, keeping best rank.

    chunk_ids are '<doc_id>#<i>'; a document appearing at chunk ranks 2 and 7
    is one document at rank 2, not two entries.
    """
    seen: set[str] = set()
    docs: list[str] = []
    for chunk_id in chunk_ranking:
        doc_id = chunk_id.rsplit("#", 1)[0]
        if doc_id not in seen:
            seen.add(doc_id)
            docs.append(doc_id)
    return docs


def recall_at_k(ranked_docs: list[str], relevant: list[str], k: int) -> float:
    """Fraction of relevant documents present in the top k."""
    if not relevant:
        raise ValueError("recall is undefined with no relevant documents")
    hits = len(set(ranked_docs[:k]) & set(relevant))
    return hits / len(relevant)


def reciprocal_rank(ranked_docs: list[str], relevant: list[str]) -> float:
    """1/rank of the first relevant document, 0 if absent from the ranking."""
    relevant_set = set(relevant)
    for rank, doc_id in enumerate(ranked_docs, start=1):
        if doc_id in relevant_set:
            return 1.0 / rank
    return 0.0


def evaluate_retriever(
    retriever, questions: list[dict], ks: tuple[int, ...] = KS, depth: int = 30
) -> dict:
    """Score one retriever over the answerable questions.

    depth is how many chunks are requested before collapsing to documents;
    it must exceed max(ks) with headroom because several chunks can collapse
    into one document.
    """
    per_question: list[dict] = []
    for q in questions:
        if not q["answerable"]:
            continue  # unanswerable questions belong to the generation eval
        ranked = doc_ranking([cid for cid, _ in retriever.search(q["question"], k=depth)])
        per_question.append(
            {
                "qid": q["qid"],
                "style": q["style"],
                "rr": reciprocal_rank(ranked, q["relevant_doc_ids"]),
                **{
                    f"recall@{k}": recall_at_k(ranked, q["relevant_doc_ids"], k)
                    for k in ks
                },
            }
        )

    def aggregate(rows: list[dict]) -> dict:
        n = len(rows)
        agg = {"n": n, "mrr": round(sum(r["rr"] for r in rows) / n, 3)}
        for k in ks:
            agg[f"recall@{k}"] = round(sum(r[f"recall@{k}"] for r in rows) / n, 3)
        return agg

    by_style: dict[str, list[dict]] = defaultdict(list)
    for row in per_question:
        by_style[row["style"]].append(row)

    return {
        "retriever": retriever.name,
        "overall": aggregate(per_question),
        "by_style": {style: aggregate(rows) for style, rows in by_style.items()},
        "per_question": per_question,
    }


def results_table(results: list[dict], ks: tuple[int, ...] = KS) -> str:
    """Render a GitHub-flavoured markdown table of the comparison."""
    header = "| retriever | " + " | ".join(f"recall@{k}" for k in ks) + " | MRR |"
    sep = "|---" * (len(ks) + 2) + "|"
    lines = [header, sep]
    for res in results:
        overall = res["overall"]
        cells = " | ".join(f"{overall[f'recall@{k}']:.3f}" for k in ks)
        lines.append(f"| {res['retriever']} | {cells} | {overall['mrr']:.3f} |")
    return "\n".join(lines)


def style_table(results: list[dict], k: int = 5) -> str:
    """Per-style breakdown: where each retriever wins and loses."""
    styles = sorted({s for res in results for s in res["by_style"]})
    header = "| retriever | " + " | ".join(f"{s} recall@{k}" for s in styles) + " |"
    sep = "|---" * (len(styles) + 1) + "|"
    lines = [header, sep]
    for res in results:
        cells = " | ".join(
            f"{res['by_style'][s][f'recall@{k}']:.3f}" if s in res["by_style"] else "-"
            for s in styles
        )
        lines.append(f"| {res['retriever']} | {cells} |")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--questions", default=str(DEFAULT_QUESTIONS))
    parser.add_argument("--out-dir", default=str(RESULTS_DIR))
    args = parser.parse_args()

    # Imports live here so the metric functions above stay importable (and
    # testable) without the corpus or the embedding stack present.
    from rag_eval_lab.chunking import build_chunks
    from rag_eval_lab.loader import load_corpus
    from rag_eval_lab.retrievers import BM25Retriever, DenseRetriever, HybridRetriever

    questions = load_questions(args.questions)
    chunks = build_chunks(load_corpus())
    bm25 = BM25Retriever(chunks)
    dense = DenseRetriever(chunks)
    hybrid = HybridRetriever(bm25, dense)

    results = [evaluate_retriever(r, questions) for r in (bm25, dense, hybrid)]

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "retrieval_metrics.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    table = results_table(results) + "\n\n" + style_table(results)
    (out_dir / "retrieval_metrics.md").write_text(table + "\n", encoding="utf-8")
    print(table)


if __name__ == "__main__":
    main()
