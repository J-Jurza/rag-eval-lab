"""Turn documents into retrieval units (chunks).

For this corpus the interesting decision is that there mostly is no decision:
an arXiv abstract is 150-250 words, which already sits inside the range that
works well for both BM25 and sentence embeddings. Splitting abstracts further
would destroy the one coherent argument each of them makes, and gluing
several together would blur relevance labels. So the default is one chunk
per document, with two deliberate interventions:

1. The title is prepended to the chunk text. Titles carry the highest-density
   signal in academic text and queries often paraphrase them.
2. Anything longer than max_words is split on sentence boundaries with
   overlap. On this corpus that path is rare; it exists because the pipeline
   claims to be corpus-agnostic, and a corpus of full documents would hit it
   immediately. The overlap trades index size for not stranding an answer
   across a boundary.

The failure mode this module owns: a bad chunk boundary makes the right
answer unretrievable no matter how good the retriever is. The notebook
demonstrates this on purpose.
"""

from __future__ import annotations

import re

# 220 words keeps a chunk inside roughly 300 tokens for typical embedding
# models, comfortably under the 512-token truncation limit of MiniLM-class
# encoders. Truncation is silent, which makes it the worst kind of bug: the
# index builds fine and recall just quietly degrades.
DEFAULT_MAX_WORDS = 220
DEFAULT_OVERLAP_WORDS = 40

# Good enough for abstracts; it will split "et al. (2020)" wrongly about once
# per few hundred documents. A production system over full papers should use
# a real sentence segmenter and, more importantly, measure the difference.
_SENTENCE_BOUNDARY = re.compile(r"(?<=[.!?])\s+(?=[A-Z(])")


def clean(text: str) -> str:
    """Collapse whitespace; leave content alone.

    Aggressive cleaning (stripping LaTeX, lowercasing, removing punctuation)
    belongs to individual retrievers, not the shared chunk text, because the
    generator needs to quote the original wording when it answers.
    """
    return re.sub(r"\s+", " ", text).strip()


def _split_sentences(text: str) -> list[str]:
    return [s for s in _SENTENCE_BOUNDARY.split(text) if s.strip()]


def _window_words(words: list[str], max_words: int, overlap_words: int) -> list[str]:
    step = max_words - overlap_words
    return [
        " ".join(words[i : i + max_words])
        for i in range(0, max(len(words) - overlap_words, 1), step)
    ]


def split_long_text(
    text: str,
    max_words: int = DEFAULT_MAX_WORDS,
    overlap_words: int = DEFAULT_OVERLAP_WORDS,
) -> list[str]:
    """Split text into pieces of at most max_words, preferring sentence ends.

    Sentences are packed greedily; a single sentence longer than max_words is
    hard-split on words so the cap is a real guarantee, not a hope.
    """
    words = text.split()
    if len(words) <= max_words:
        return [text]

    pieces: list[str] = []
    current: list[str] = []
    for sentence in _split_sentences(text):
        sentence_words = sentence.split()
        if len(sentence_words) > max_words:
            if current:
                pieces.append(" ".join(current))
                current = []
            pieces.extend(_window_words(sentence_words, max_words, overlap_words))
            continue
        if len(current) + len(sentence_words) > max_words:
            pieces.append(" ".join(current))
            # Carry the tail of the previous piece forward so a fact stated at
            # the boundary is fully present in at least one chunk.
            current = current[-overlap_words:] if overlap_words else []
        current.extend(sentence_words)
    if current:
        pieces.append(" ".join(current))
    return pieces


def build_chunks(
    docs: list[dict],
    max_words: int = DEFAULT_MAX_WORDS,
    overlap_words: int = DEFAULT_OVERLAP_WORDS,
) -> list[dict]:
    """Map documents to chunks: {chunk_id, doc_id, title, text}.

    chunk_id is doc_id plus a piece index, so retrieval hits trace back to a
    document, which is the unit the relevance labels are defined on.
    """
    chunks: list[dict] = []
    for doc in docs:
        title = clean(doc.get("title", ""))
        abstract = clean(doc.get("abstract", ""))
        if not abstract:
            continue
        body = f"{title}. {abstract}" if title else abstract
        for i, piece in enumerate(split_long_text(body, max_words, overlap_words)):
            chunks.append(
                {
                    "chunk_id": f"{doc['doc_id']}#{i}",
                    "doc_id": doc["doc_id"],
                    "title": title,
                    "text": piece,
                }
            )
    return chunks
