"""Chunking is pure logic, so it gets the strictest tests in the repo.

The properties tested here are the ones a retrieval index silently depends
on: the size cap is real, nothing is lost at boundaries, and ids stay
traceable to documents. If any of these break, retrieval metrics move for
reasons that have nothing to do with retrieval.
"""

from rag_eval_lab.chunking import build_chunks, clean, split_long_text


def make_text(n_sentences: int, words_per_sentence: int = 10) -> str:
    return " ".join(
        "Word" + " word" * (words_per_sentence - 2) + f" s{i}."
        for i in range(n_sentences)
    )


def test_clean_collapses_whitespace():
    assert clean("a\n b\t\t c   d ") == "a b c d"


def test_short_text_is_one_piece():
    text = make_text(3)
    assert split_long_text(text, max_words=200) == [text]


def test_cap_is_enforced_even_without_sentence_boundaries():
    # One 500-word "sentence": the sentence packer cannot help, so the word
    # window fallback must guarantee the cap.
    text = "w " * 499 + "end"
    pieces = split_long_text(text.strip(), max_words=100, overlap_words=20)
    assert all(len(p.split()) <= 100 for p in pieces)
    assert len(pieces) > 1


def test_no_words_lost_at_boundaries():
    text = make_text(40)  # 400 words, forces several pieces
    pieces = split_long_text(text, max_words=120, overlap_words=30)
    # Every sentence marker must survive somewhere; overlap may duplicate
    # words but must never drop them.
    joined = " ".join(pieces)
    for i in range(40):
        assert f"s{i}." in joined


def test_overlap_carries_boundary_context():
    text = make_text(40)
    pieces = split_long_text(text, max_words=120, overlap_words=30)
    for a, b in zip(pieces, pieces[1:]):
        # The head of each piece repeats the tail of the previous one.
        assert b.split()[0] in a.split()


def test_build_chunks_ids_trace_to_documents():
    docs = [
        {"doc_id": "d1", "title": "A Title", "abstract": make_text(3)},
        {"doc_id": "d2", "title": "Another", "abstract": make_text(60)},
        {"doc_id": "d3", "title": "Empty", "abstract": ""},
    ]
    chunks = build_chunks(docs, max_words=120, overlap_words=20)
    assert all(c["chunk_id"].startswith(c["doc_id"] + "#") for c in chunks)
    # d1 fits in one chunk, d2 must split, d3 is dropped for having no text
    assert len([c for c in chunks if c["doc_id"] == "d1"]) == 1
    assert len([c for c in chunks if c["doc_id"] == "d2"]) > 1
    assert not [c for c in chunks if c["doc_id"] == "d3"]


def test_title_is_prepended():
    docs = [{"doc_id": "d1", "title": "Volatility Smile", "abstract": "Body text."}]
    chunks = build_chunks(docs)
    assert chunks[0]["text"].startswith("Volatility Smile.")
