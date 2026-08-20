"""Three retrievers over the same chunk index: BM25, dense, hybrid.

BM25 is hand-rolled (about forty lines) rather than imported, because on a
400-chunk corpus a library adds nothing except distance from the maths, and
the maths is the part worth being able to defend. Dense retrieval uses
sentence-transformers because hand-rolling a trained encoder is not a
weekend's honest work; the wrapper stays thin enough to swap models.

There is deliberately no vector database. 400 chunks embed to a matrix of
about 400 x 384 floats, roughly 600KB; exact cosine search over that is a
single matrix multiply. A vector DB earns its complexity somewhere past the
point where the matrix stops fitting in memory or index updates become
concurrent, and pretending otherwise here would be resume-driven
infrastructure.

All retrievers share one interface: search(query, k) returns the top-k
(chunk_id, score) pairs, best first. Scores are only comparable within one
retriever, which is exactly why the hybrid fuses ranks, not scores.
"""

from __future__ import annotations

import hashlib
import math
import re
from collections import Counter
from pathlib import Path

import numpy as np

_TOKEN = re.compile(r"[a-z0-9]+")

# Standard Robertson defaults. k1 controls term-frequency saturation, b how
# strongly scores are normalised by document length. Tuning them on 25
# labelled questions would be fitting noise, so they stay at the defaults.
BM25_K1 = 1.5
BM25_B = 0.75

EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
CACHE_DIR = Path("data/cache")

# Reciprocal rank fusion constant from Cormack et al. (2009). Large enough
# that a rank-1 result does not drown out everything below it.
RRF_K = 60


def tokenize(text: str) -> list[str]:
    return _TOKEN.findall(text.lower())


class BM25Retriever:
    """Okapi BM25 over the chunk texts.

    score(q, d) = sum over query terms of
        idf(t) * tf * (k1 + 1) / (tf + k1 * (1 - b + b * len_d / avg_len))
    with the Lucene-style idf ln(1 + (N - df + 0.5) / (df + 0.5)), which
    stays positive for terms that appear in most documents.
    """

    name = "bm25"

    def __init__(self, chunks: list[dict]):
        self.chunk_ids = [c["chunk_id"] for c in chunks]
        self.doc_tokens = [tokenize(c["text"]) for c in chunks]
        self.doc_lens = np.array([len(t) for t in self.doc_tokens], dtype=float)
        self.avg_len = float(self.doc_lens.mean()) if len(chunks) else 0.0
        self.tfs = [Counter(tokens) for tokens in self.doc_tokens]
        df: Counter = Counter()
        for tf in self.tfs:
            df.update(tf.keys())
        n = len(chunks)
        self.idf = {
            term: math.log(1 + (n - d + 0.5) / (d + 0.5)) for term, d in df.items()
        }

    def search(self, query: str, k: int = 10) -> list[tuple[str, float]]:
        scores = np.zeros(len(self.chunk_ids))
        for term in tokenize(query):
            idf = self.idf.get(term)
            if idf is None:
                continue  # unseen terms carry no evidence about this corpus
            for i, tf in enumerate(self.tfs):
                f = tf.get(term)
                if not f:
                    continue
                norm = 1 - BM25_B + BM25_B * self.doc_lens[i] / self.avg_len
                scores[i] += idf * f * (BM25_K1 + 1) / (f + BM25_K1 * norm)
        top = np.argsort(-scores)[:k]
        return [(self.chunk_ids[i], float(scores[i])) for i in top if scores[i] > 0]


class DenseRetriever:
    """Cosine similarity between MiniLM embeddings of query and chunks.

    Embeddings are L2-normalised at encode time so cosine similarity is a dot
    product. The chunk matrix is cached on disk keyed by a hash of the model
    name and chunk texts: re-embedding 400 chunks costs about a minute on CPU
    and zero information, so recomputing it every run would just make the
    feedback loop worse.
    """

    name = "dense"

    def __init__(self, chunks: list[dict], cache_dir: Path = CACHE_DIR):
        # Imported here, not at module top: BM25 and the eval harness must
        # work without the heavy optional dependency installed.
        from sentence_transformers import SentenceTransformer

        self.chunk_ids = [c["chunk_id"] for c in chunks]
        texts = [c["text"] for c in chunks]
        self.model = SentenceTransformer(EMBED_MODEL)

        key = hashlib.sha256(
            ("\x1e".join(texts) + EMBED_MODEL).encode("utf-8")
        ).hexdigest()[:16]
        cache_file = cache_dir / f"embeddings_{key}.npy"
        if cache_file.exists():
            self.matrix = np.load(cache_file)
        else:
            self.matrix = self.model.encode(
                texts, normalize_embeddings=True, show_progress_bar=False
            )
            cache_dir.mkdir(parents=True, exist_ok=True)
            np.save(cache_file, self.matrix)

    def search(self, query: str, k: int = 10) -> list[tuple[str, float]]:
        q = self.model.encode([query], normalize_embeddings=True)[0]
        scores = self.matrix @ q
        top = np.argsort(-scores)[:k]
        return [(self.chunk_ids[i], float(scores[i])) for i in top]


def rrf_fuse(
    rankings: list[list[str]], k: int = 10, rrf_k: int = RRF_K
) -> list[tuple[str, float]]:
    """Reciprocal rank fusion: score(d) = sum over rankings of 1/(rrf_k + rank).

    Fusing ranks instead of raw scores sidesteps the fact that BM25 scores
    and cosine similarities live on incompatible scales; calibrating those
    scales against each other is a research problem, and RRF gets most of the
    benefit without it.
    """
    scores: dict[str, float] = {}
    for ranking in rankings:
        for rank, chunk_id in enumerate(ranking):
            scores[chunk_id] = scores.get(chunk_id, 0.0) + 1.0 / (rrf_k + rank + 1)
    fused = sorted(scores.items(), key=lambda item: -item[1])
    return fused[:k]


class HybridRetriever:
    """RRF fusion of BM25 and dense rankings.

    Each base retriever contributes its top candidate_depth results; fusing
    from a deeper pool than the final k lets a document ranked (say) 12th by
    both retrievers surface into the fused top 10.
    """

    name = "hybrid"

    def __init__(self, bm25: BM25Retriever, dense: DenseRetriever, candidate_depth: int = 50):
        self.bm25 = bm25
        self.dense = dense
        self.candidate_depth = candidate_depth

    def search(self, query: str, k: int = 10) -> list[tuple[str, float]]:
        lexical = [cid for cid, _ in self.bm25.search(query, self.candidate_depth)]
        semantic = [cid for cid, _ in self.dense.search(query, self.candidate_depth)]
        return rrf_fuse([lexical, semantic], k=k)
