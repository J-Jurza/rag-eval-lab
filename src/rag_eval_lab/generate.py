"""Grounded answering and its evaluation: groundedness and refusal.

The generator is given the top retrieved chunks and told to answer only from
them, cite chunk numbers for every claim, and refuse with an exact sentence
when the context does not contain the answer. The exact refusal sentence is
not cosmetic: it makes refusal detection deterministic string matching
instead of another judgment call.

Generation is evaluated on two axes, separately from retrieval:

- Groundedness: an LLM judge splits each answer into atomic claims and checks
  each claim against the retrieved context only (not against world knowledge,
  not against the judge's opinion of truth). A claim can be grounded and
  still wrong if the source is wrong; groundedness is about whether the
  system stayed inside its evidence.
- Refusal: on the questions the corpus cannot answer, did the model refuse,
  or did it improvise? A RAG system that answers everything is not capable,
  it is unguarded.

Honesty about the judge, because it is the weakest link:
- The judge is the same model family as the generator, so systematic blind
  spots are shared. In production the judge gets validated against a sample
  of human judgments before its numbers are trusted; the notebook spot-checks
  a handful of claims by hand for exactly this reason.
- Claim splitting is itself a modelling decision the judge makes; two runs
  can split the same answer differently.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

REFUSAL_SENTENCE = "The provided context does not answer this question."

DEFAULT_MODEL = "gemini-2.5-flash"
GEN_DIR = Path("results/generation")
TOP_K = 5

ANSWER_PROMPT = """You answer questions using ONLY the numbered context passages below.

Rules:
- Every factual claim in your answer must come from the passages; cite the
  passage number like [2] after each claim.
- Do not use outside knowledge, even when you are confident.
- If the passages do not contain the information needed to answer, reply with
  exactly this sentence and nothing else: {refusal}

Context passages:
{context}

Question: {question}
Answer:"""

JUDGE_PROMPT = """You are auditing whether an answer is supported by its source context.
Split the answer into atomic factual claims. For each claim, decide whether
the context passages FULLY support it. Judge support strictly against the
context only; whether the claim is true in the real world is irrelevant.

Return JSON only, in this shape:
{{"claims": [{{"claim": "...", "supported": true}}]}}

If the answer is only a refusal or contains no factual claims, return
{{"claims": []}}.

Context passages:
{context}

Answer to audit:
{answer}"""


def _client():
    # Imported lazily: everything except the two API calls works without the
    # optional [gen] dependencies installed.
    from dotenv import load_dotenv
    from google import genai

    load_dotenv()
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise SystemExit(
            "GEMINI_API_KEY is not set. Copy .env.example to .env and add a key; "
            "retrieval and retrieval evaluation run without one."
        )
    return genai.Client(api_key=api_key)


def format_context(chunks: list[dict]) -> str:
    return "\n\n".join(
        f"[{i + 1}] (doc {c['doc_id']}) {c['text']}" for i, c in enumerate(chunks)
    )


def is_refusal(answer: str) -> bool:
    return REFUSAL_SENTENCE.lower() in answer.strip().lower()


def parse_judge_json(raw: str) -> list[dict]:
    """Parse the judge's claim list, tolerating markdown code fences.

    Anything unparseable raises: a silently empty claim list would score as
    perfectly grounded, which is the worst possible failure direction for an
    evaluation harness.
    """
    text = raw.strip()
    if text.startswith("```"):
        text = text.strip("`")
        if text.startswith("json"):
            text = text[4:]
    data = json.loads(text)
    claims = data["claims"]
    if not isinstance(claims, list):
        raise ValueError(f"claims is not a list: {claims!r}")
    for claim in claims:
        if not isinstance(claim.get("supported"), bool):
            raise ValueError(f"claim missing boolean 'supported': {claim!r}")
    return claims


def score_rows(rows: list[dict]) -> dict:
    """Aggregate judged answers into the generation-side metrics.

    Answerable questions report groundedness (supported claims / claims);
    unanswerable ones report refusal rate. An answerable question that was
    refused counts as a false refusal, reported separately: refusing
    everything would otherwise score as perfectly grounded.
    """
    answerable = [r for r in rows if r["answerable"]]
    unanswerable = [r for r in rows if not r["answerable"]]

    judged = [r for r in answerable if not r["refused"] and r["claims"]]
    total_claims = sum(len(r["claims"]) for r in judged)
    supported = sum(sum(c["supported"] for c in r["claims"]) for r in judged)

    return {
        "n_answerable": len(answerable),
        "n_unanswerable": len(unanswerable),
        "false_refusals": sum(r["refused"] for r in answerable),
        "refusal_rate_unanswerable": (
            round(sum(r["refused"] for r in unanswerable) / len(unanswerable), 3)
            if unanswerable
            else None
        ),
        "total_claims": total_claims,
        "supported_claims": supported,
        "groundedness": round(supported / total_claims, 3) if total_claims else None,
        "ungrounded_examples": [
            {"qid": r["qid"], "claim": c["claim"]}
            for r in judged
            for c in r["claims"]
            if not c["supported"]
        ],
    }


def _generate(client, model: str, prompt: str, retries: int = 3) -> str:
    for attempt in range(retries):
        try:
            response = client.models.generate_content(model=model, contents=prompt)
            return response.text or ""
        except Exception:
            if attempt == retries - 1:
                raise
            time.sleep(2 ** (attempt + 1))  # free-tier rate limits are real
    raise RuntimeError("unreachable")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=os.environ.get("GEMINI_MODEL", DEFAULT_MODEL))
    parser.add_argument("--out-dir", default=str(GEN_DIR))
    args = parser.parse_args()

    from rag_eval_lab.chunking import build_chunks
    from rag_eval_lab.evaluate import load_questions
    from rag_eval_lab.loader import load_corpus
    from rag_eval_lab.retrievers import BM25Retriever, DenseRetriever, HybridRetriever

    client = _client()
    questions = load_questions()
    chunks = build_chunks(load_corpus())
    by_id = {c["chunk_id"]: c for c in chunks}
    # The best retriever from the retrieval eval feeds the generator; the
    # comparison already happened upstream and this stage should not blur it.
    hybrid = HybridRetriever(BM25Retriever(chunks), DenseRetriever(chunks))

    rows = []
    for q in questions:
        top = [by_id[cid] for cid, _ in hybrid.search(q["question"], k=TOP_K)]
        context = format_context(top)
        answer = _generate(
            client,
            args.model,
            ANSWER_PROMPT.format(
                refusal=REFUSAL_SENTENCE, context=context, question=q["question"]
            ),
        )
        refused = is_refusal(answer)
        claims: list[dict] = []
        if not refused:
            raw = _generate(
                client, args.model, JUDGE_PROMPT.format(context=context, answer=answer)
            )
            claims = parse_judge_json(raw)
        rows.append(
            {
                "qid": q["qid"],
                "question": q["question"],
                "answerable": q["answerable"],
                "retrieved_docs": [c["doc_id"] for c in top],
                "answer": answer.strip(),
                "refused": refused,
                "claims": claims,
            }
        )
        print(f"{q['qid']}: {'refused' if refused else f'{len(claims)} claims judged'}")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "answers.jsonl", "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    summary = {"model": args.model, **score_rows(rows)}
    with open(out_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
