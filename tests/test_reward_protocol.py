"""The published reward protocol must replay exactly.

tests/fixtures/reward_protocol_golden.json is a frozen snapshot of the questions,
Jev request payloads, fixed-draft fixtures and calculate() outputs used for the
published run. Never regenerate it to make this test pass; fix the code instead.
"""

import json
from pathlib import Path

import httpx
import pytest

from fw_jev import reward

GOLDEN = json.loads((Path(__file__).parent / "fixtures/reward_protocol_golden.json").read_text())


def test_identity_constants():
    assert (reward.VERSION, reward.VARIANT, reward.MODEL, reward.ENDPOINT) == (
        GOLDEN["version"],
        GOLDEN["variant"],
        GOLDEN["model"],
        GOLDEN["endpoint"],
    )


@pytest.mark.parametrize("grounded", [False, True])
def test_questions(grounded):
    assert reward.questions(grounded=grounded) == GOLDEN["questions"]["grounded" if grounded else "ungrounded"]


@pytest.mark.parametrize("grounded", [False, True])
def test_jev_request_payload(monkeypatch, grounded):
    expected = GOLDEN["payloads"]["grounded" if grounded else "ungrounded"]
    state = json.loads(expected["state"])
    judge = reward.Jev("unit-test-key")
    sent = []

    def post(url, *, headers, json):
        sent.append({"url": url, "headers": headers, "json": json})
        answers = {
            "style": {"type": "choice", "probabilities": {"yes": 1.0, "no": 0.0, "uncertain": 0.0}},
            "quality": {"type": "score", "score": 3, "probabilities": {"0": 0, "1": 0, "2": 0, "3": 1}},
        }
        if grounded:
            answers["unsupported_claim"] = {
                "type": "choice",
                "probabilities": {"yes": 0.0, "no": 1.0, "uncertain": 0.0},
            }
        return httpx.Response(200, json={"model": reward.MODEL, "answers": answers})

    monkeypatch.setattr(judge.client, "post", post)
    try:
        judge.score(state["REQUEST"], state["DRAFT"], grounded=grounded)
    finally:
        judge.close()
    assert len(sent) == 1
    assert sent[0]["url"] == GOLDEN["endpoint"]
    assert sent[0]["headers"] == {"Authorization": "Bearer unit-test-key"}
    assert sent[0]["json"] == expected
    assert list(sent[0]["json"]) == list(expected)


def test_fixed_draft_fixtures_and_request_count():
    # The current preflight scores every canary and every grounding fixture, twice.
    assert [list(row) for row in reward.CANARIES] == GOLDEN["canaries"]
    assert [list(row) for row in reward.GROUNDING_FIXTURES] == GOLDEN["grounding_fixtures"]
    assert reward.PREFLIGHT_REQUESTS == GOLDEN["preflight_requests"] == 32


def test_every_calculate_case_replays_exactly():
    cases = GOLDEN["calculate_cases"]
    assert len(cases) == 576
    for index, case in enumerate(cases):
        result = reward.calculate(
            case["response"], case["draft"], truncated=case["truncated"], malformed=case["malformed"]
        )
        assert result == case["expected"], index
        assert result["version"] == GOLDEN["version"]
