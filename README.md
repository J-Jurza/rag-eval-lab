# rag-eval-lab

A small retrieval-augmented generation (RAG) pipeline over arXiv quantitative-finance
abstracts, built so that the evaluation is the centrepiece rather than an afterthought.
Everything is hand-rolled where hand-rolling teaches something (BM25, rank fusion, the
metrics) and imported where it does not (the sentence encoder).

The organising idea: when a RAG system gives a wrong answer, two different failures
hide under one symptom. Either the right document never reached the model, which is a
retrieval failure, or it did and the model still went wrong, which is a generation
failure. The fixes have nothing in common, so the measurement keeps them apart:

- **Retrieval** is scored with recall@k and MRR against a labelled question set.
- **Generation** is scored on groundedness (is every claim supported by the retrieved
  context) and refusal behaviour (does it decline questions the corpus cannot answer).

## The measured comparison

Three retrievers over the same 402-chunk index, evaluated on 22 labelled questions:

| retriever | recall@1 | recall@3 | recall@5 | recall@10 | MRR |
|---|---|---|---|---|---|
| bm25 | 0.909 | 0.909 | 0.909 | 0.955 | 0.920 |
| dense | 0.818 | 1.000 | 1.000 | 1.000 | 0.902 |
| hybrid | 0.955 | 1.000 | 1.000 | 1.000 | 0.970 |

The aggregate hides the useful finding, which the per-style split exposes:

| retriever | keyword recall@5 | paraphrase recall@5 |
|---|---|---|
| bm25 | 1.000 | 0.800 |
| dense | 1.000 | 1.000 |
| hybrid | 1.000 | 1.000 |

BM25 is unbeatable when the question quotes exact identifiers ("SAML-D", "BANKNIFTY")
and goes blind when the question paraphrases ("digital coins" never matches
"cryptocurrencies"). Dense retrieval is the mirror image. The hybrid wins because the
two fail on different questions, and reciprocal rank fusion only needs one of them to
be right. [notebooks/walkthrough.ipynb](notebooks/walkthrough.ipynb) demonstrates both
failure directions on real questions, plus a chunk-boundary failure, and explains every
stage in prose.

Honesty about scale: 22 questions means one question moves a metric by 4.5 points, and
a 300-document corpus with single-document labels is an easy retrieval problem, which
is why the absolute numbers are high. The differences are directional. The harness is
the point: the same code re-runs unchanged on a corpus 100x larger, where these tables
start earning their keep.

Generation-side results (groundedness and refusal rates over the same question set)
are produced by `python -m rag_eval_lab.generate` into `results/generation/`.

## Running it

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[embed,dev]"

# rebuild the pinned corpus snapshot from the committed manifest
python -m rag_eval_lab.loader --refetch

pytest -q                          # pure-logic tests, no model, no API
python -m rag_eval_lab.evaluate    # the retrieval comparison above
```

No API key is needed for any of that. The generation layer is the only part that
calls an API (Gemini; a free-tier key from https://aistudio.google.com works):

```bash
pip install -e ".[gen]"
cp .env.example .env               # add GEMINI_API_KEY
python -m rag_eval_lab.generate
```

## Design decisions, and the trade-offs they are

- **A manifest is committed, not the corpus text.** `data/corpus_manifest.json` pins
  300 document ids with content checksums; the loader rebuilds the exact snapshot from
  the arXiv API. Reproducible metrics without redistributing authors' copyrighted
  abstracts. The trade: cloning does not give you the corpus offline, one command does.
- **No vector database.** 402 chunks embed to a 402x384 matrix (~600KB); exact cosine
  search is one matrix multiply. A vector DB earns its complexity when the matrix
  outgrows memory or needs concurrent updates. Neither is true here, and infrastructure
  without a measurable reason is resume decoration.
- **No orchestration framework.** Each pipeline stage is a short module with a
  docstring stating what it owns and what breaks it. A framework would save none of
  those lines and would hide the parts worth understanding. Frameworks earn their place
  when you need their integrations, not their abstractions.
- **BM25 is hand-rolled and pinned to hand-computed scores in tests.** Forty lines.
  The dense encoder is imported, because a trained model is not honestly reproducible
  in a weekend and pretending otherwise teaches nothing.
- **One chunk per abstract, title prepended.** Abstracts are already chunk-sized and
  self-contained; the sentence-aware splitter with overlap only engages past 220 words,
  a cap chosen against the embedding model's silent input truncation.
- **The question set is self-authored, and that bias is stated.** Questions were
  written against read abstracts, tagged keyword vs paraphrase, with five unanswerable
  questions for refusal testing. Self-authored labels measure agreement with the
  labeller; the real-world version (pooled labelling over real user questions) is
  described in the notebook.

## What the metrics do not catch

- recall@k against self-authored labels punishes the retriever for the labeller's
  misses: a genuinely relevant unlabelled document counts as a retrieval failure.
- MRR sees only the first relevant hit, so partial coverage of multi-document answers
  is invisible.
- Neither retrieval metric knows whether the retrieved text is sufficient to answer.
- Groundedness is not truth: an answer faithfully grounded in a wrong source scores
  perfectly. And the LLM judge shares a model family with the generator and is not
  validated against human judgments here; in production that calibration comes first.

## What this is NOT

This is a measurement harness around a deliberately small RAG system. For production:

- **Scale**: in-memory exact search dies somewhere past ~1M chunks; you would move to
  an ANN index (HNSW) or a vector store, and re-run this same evaluation to measure
  what the approximation costs you.
- **Freshness**: the corpus is a pinned snapshot by design. Production needs ingestion,
  re-embedding on content change, and eval questions that track corpus drift.
- **Latency and cost**: nothing here is batched, cached across requests, or streamed;
  the judge doubles API cost per answer and would be sampled, not run on everything.
- **Access control**: retrieval over documents with per-user permissions must filter at
  query time; there is none of that here.
- **Monitoring**: the eval runs on demand. Production runs it continuously on sampled
  live traffic, with human calibration of the judge and alerting on metric drift.
- **Judge validation**: the groundedness judge would be calibrated against a
  human-labelled sample before its numbers drive any decision.

## Repository map

```
src/rag_eval_lab/
  loader.py      corpus fetch + pinned manifest (reproducibility, redistribution)
  chunking.py    cleaning, title prepending, sentence-aware size cap
  retrievers.py  BM25 (hand-rolled), dense (MiniLM), hybrid (RRF)
  evaluate.py    recall@k, MRR, per-style breakdown; what they miss
  generate.py    grounded answering, refusal, LLM-judge groundedness
eval/questions.jsonl        27 labelled questions (22 answerable, 5 not)
data/corpus_manifest.json   the pinned snapshot: ids + checksums
results/                    committed metric tables
notebooks/walkthrough.ipynb the teaching walkthrough, executed in CI
tests/                      pure-logic tests: chunking, BM25, RRF, metrics, scoring
```

MIT licence. Corpus text remains the property of the respective paper authors and is
fetched from the arXiv API at build time, not stored in this repository.
