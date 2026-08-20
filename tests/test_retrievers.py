"""BM25 and RRF are pure functions of their inputs, so they are tested
against hand-computed expectations. The dense retriever needs a model
download, so its test is skipped where sentence-transformers is absent
(notably CI); what CI does guarantee is everything the comparison's
conclusions rest on arithmetically: the metrics, BM25, and the fusion.
"""

import math

import pytest

from rag_eval_lab.retrievers import BM25Retriever, rrf_fuse, tokenize


def chunk(cid: str, text: str) -> dict:
    return {"chunk_id": cid, "doc_id": cid, "title": "", "text": text}


TOY = [
    chunk("c1", "portfolio risk model"),
    chunk("c2", "portfolio portfolio portfolio optimisation"),
    chunk("c3", "market microstructure noise"),
    chunk("c4", "risk risk risk risk aversion"),
]


def test_tokenize_lowercases_and_strips_punctuation():
    assert tokenize("Value-at-Risk (VaR), 99%!") == ["value", "at", "risk", "var", "99"]


def test_rare_term_outranks_common_term():
    bm25 = BM25Retriever(TOY)
    # "microstructure" appears in one chunk, "portfolio" in two; a query for
    # the rare term must put its unique chunk first with a higher score than
    # any chunk scores for the common term alone.
    (top_id, top_score), *_ = bm25.search("microstructure", k=4)
    assert top_id == "c3"
    common_best = bm25.search("portfolio", k=1)[0][1]
    assert top_score > common_best


def test_term_frequency_saturates():
    bm25 = BM25Retriever(TOY)
    results = dict(bm25.search("risk", k=4))
    # c4 has tf=4 vs c1's tf=1, but the k1 saturation means its score must be
    # well under 4x c1's, once length normalisation is accounted for.
    assert results["c4"] > results["c1"]
    assert results["c4"] < 4 * results["c1"]


def test_bm25_score_matches_hand_computation():
    docs = [chunk("a", "x y"), chunk("b", "x x z")]
    bm25 = BM25Retriever(docs)
    # For term x: df=2, N=2 -> idf = ln(1 + 0.5/2.5); doc b: tf=2, len=3,
    # avg_len=2.5 -> norm = 1 - 0.75 + 0.75*3/2.5 = 1.15
    idf = math.log(1 + 0.5 / 2.5)
    expected = idf * 2 * 2.5 / (2 + 1.5 * 1.15)
    got = dict(bm25.search("x", k=2))["b"]
    assert got == pytest.approx(expected)


def test_unseen_query_terms_return_empty():
    bm25 = BM25Retriever(TOY)
    assert bm25.search("blockchain", k=4) == []


def test_rrf_rewards_agreement():
    ranking_a = ["c1", "c2", "c3"]
    ranking_b = ["c3", "c1", "c4"]
    fused = [cid for cid, _ in rrf_fuse([ranking_a, ranking_b], k=4)]
    # c1 (ranks 1 and 2) must beat c3 (ranks 3 and 1) and both must beat the
    # single-list entries c2 and c4.
    assert fused[0] == "c1"
    assert fused[1] == "c3"
    assert set(fused[2:]) == {"c2", "c4"}


def test_rrf_scores_are_sums_of_reciprocal_ranks():
    fused = dict(rrf_fuse([["c1"], ["c1"]], k=1, rrf_k=60))
    assert fused["c1"] == pytest.approx(2 / 61)


def test_dense_retriever_finds_paraphrase():
    pytest.importorskip("sentence_transformers")
    from rag_eval_lab.retrievers import DenseRetriever

    corpus = [
        chunk("housing", "Forecasting residential property prices with hedonic models"),
        chunk("crypto", "Volatility clustering in bitcoin returns"),
    ]
    dense = DenseRetriever(corpus)
    # No lexical overlap with the housing chunk: "home values" vs "residential
    # property prices". BM25 would return nothing useful here.
    top_id, _ = dense.search("predicting home values", k=1)[0]
    assert top_id == "housing"
