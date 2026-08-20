"""Metric arithmetic is the part of this repo whose correctness the whole
comparison rests on, so every function is pinned by hand-computed cases.
A subtle bug here (an off-by-one in rank, doc dedup losing order) would not
crash anything; it would just silently reorder the conclusions.
"""

import pytest

from rag_eval_lab.evaluate import (
    doc_ranking,
    evaluate_retriever,
    recall_at_k,
    reciprocal_rank,
    results_table,
)


def test_doc_ranking_collapses_chunks_keeping_best_rank():
    ranked_chunks = ["d2#1", "d1#0", "d2#0", "d3#0", "d1#2"]
    assert doc_ranking(ranked_chunks) == ["d2", "d1", "d3"]


def test_recall_at_k_hand_computed():
    ranked = ["a", "b", "c", "d"]
    assert recall_at_k(ranked, ["a"], 1) == 1.0
    assert recall_at_k(ranked, ["c"], 1) == 0.0
    assert recall_at_k(ranked, ["c"], 3) == 1.0
    assert recall_at_k(ranked, ["a", "d", "z"], 4) == pytest.approx(2 / 3)


def test_recall_requires_relevant_docs():
    with pytest.raises(ValueError):
        recall_at_k(["a"], [], 5)


def test_reciprocal_rank_hand_computed():
    assert reciprocal_rank(["a", "b", "c"], ["a"]) == 1.0
    assert reciprocal_rank(["a", "b", "c"], ["c"]) == pytest.approx(1 / 3)
    assert reciprocal_rank(["a", "b", "c"], ["z"]) == 0.0
    # first relevant hit wins even if a later relevant doc ranks worse
    assert reciprocal_rank(["a", "b", "c"], ["b", "c"]) == pytest.approx(1 / 2)


class FakeRetriever:
    """Returns a fixed chunk ranking regardless of the query."""

    name = "fake"

    def __init__(self, ranking):
        self.ranking = ranking

    def search(self, query, k=10):
        return [(cid, 1.0) for cid in self.ranking[:k]]


QUESTIONS = [
    {"qid": "q1", "style": "keyword", "answerable": True, "relevant_doc_ids": ["d1"], "question": "x"},
    {"qid": "q2", "style": "paraphrase", "answerable": True, "relevant_doc_ids": ["d9"], "question": "y"},
    {"qid": "q3", "style": "unanswerable", "answerable": False, "relevant_doc_ids": [], "question": "z"},
]


def test_evaluate_retriever_aggregates_and_skips_unanswerable():
    retriever = FakeRetriever(["d1#0", "d2#0", "d3#0"])
    result = evaluate_retriever(retriever, QUESTIONS, ks=(1, 3))
    # q1 hits at rank 1, q2 never hits; q3 must not be scored at all
    assert result["overall"]["n"] == 2
    assert result["overall"]["mrr"] == pytest.approx(0.5)
    assert result["overall"]["recall@1"] == pytest.approx(0.5)
    assert result["by_style"]["keyword"]["recall@1"] == 1.0
    assert result["by_style"]["paraphrase"]["recall@1"] == 0.0


def test_results_table_renders_all_retrievers():
    retriever = FakeRetriever(["d1#0"])
    result = evaluate_retriever(retriever, QUESTIONS, ks=(1, 3))
    table = results_table([result], ks=(1, 3))
    assert "| fake |" in table
    assert "recall@1" in table and "recall@3" in table
