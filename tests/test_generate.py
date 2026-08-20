"""The generation module's pure logic: refusal detection, judge output
parsing, and score aggregation. The API calls themselves are not tested
here; what CI protects is the arithmetic that turns judged claims into the
numbers the README reports.
"""

import pytest

from rag_eval_lab.generate import (
    REFUSAL_SENTENCE,
    format_context,
    is_refusal,
    parse_judge_json,
    score_rows,
)


def test_refusal_detection_is_exact_but_case_insensitive():
    assert is_refusal(REFUSAL_SENTENCE)
    assert is_refusal(REFUSAL_SENTENCE.upper())
    assert is_refusal(f"  {REFUSAL_SENTENCE}  ")
    assert not is_refusal("I do not know.")


def test_parse_judge_json_plain_and_fenced():
    plain = '{"claims": [{"claim": "a", "supported": true}]}'
    fenced = f"```json\n{plain}\n```"
    assert parse_judge_json(plain)[0]["supported"] is True
    assert parse_judge_json(fenced)[0]["claim"] == "a"


def test_parse_judge_json_refuses_to_guess():
    # An unparseable judge response must raise, never score as "no claims":
    # empty claims means perfectly grounded, the worst failure direction.
    with pytest.raises(Exception):
        parse_judge_json("The answer looks fine to me.")
    with pytest.raises(Exception):
        parse_judge_json('{"claims": [{"claim": "a", "supported": "yes"}]}')


def make_row(qid, answerable, refused, supported_flags):
    return {
        "qid": qid,
        "answerable": answerable,
        "refused": refused,
        "claims": [{"claim": f"c{i}", "supported": s} for i, s in enumerate(supported_flags)],
    }


def test_score_rows_hand_computed():
    rows = [
        make_row("q1", True, False, [True, True, False]),
        make_row("q2", True, False, [True]),
        make_row("q3", True, True, []),  # false refusal on an answerable question
        make_row("q4", False, True, []),  # correct refusal
        make_row("q5", False, False, [True]),  # improvised answer, should have refused
    ]
    summary = score_rows(rows)
    assert summary["groundedness"] == pytest.approx(3 / 4)
    assert summary["false_refusals"] == 1
    assert summary["refusal_rate_unanswerable"] == pytest.approx(0.5)
    assert summary["ungrounded_examples"] == [{"qid": "q1", "claim": "c2"}]


def test_format_context_numbers_from_one():
    chunks = [
        {"chunk_id": "a#0", "doc_id": "a", "text": "first"},
        {"chunk_id": "b#0", "doc_id": "b", "text": "second"},
    ]
    ctx = format_context(chunks)
    assert "[1] (doc a) first" in ctx
    assert "[2] (doc b) second" in ctx
