"""rag-eval-lab: a small RAG pipeline where the evaluation is the point.

The package is deliberately flat and framework-free. Each module maps to one
pipeline stage, in the order data flows through them:

    loader     -> fetch the corpus (arXiv q-fin abstracts) to data/corpus.jsonl
    chunking   -> clean text and enforce chunk size bounds
    retrievers -> BM25, dense, and hybrid retrieval over the chunks
    generate   -> grounded answering with an explicit refusal instruction
    evaluate   -> recall@k / MRR for retrieval; groundedness and refusal for generation
"""

__version__ = "0.1.0"
