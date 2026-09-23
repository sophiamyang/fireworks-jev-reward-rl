import json

import httpx

from fw_jev.config import load, prompt
from fw_jev.mock import response
from fw_jev.reward import Jev, calculate
from fw_jev.smoke import diagnose
from fw_jev.writing import PROTOCOL, judge_state, messages


def test_one_user_message_no_hidden_contract():
    text = "Write a post.\n\nSource: release notes."
    assert PROTOCOL == "user-only-v1"
    assert messages(text) == [{"role": "user", "content": text}]
    assert judge_state(text, "draft") == {"REQUEST": text, "DRAFT": "draft"}


def test_all_cases_use_same_writer_and_judge_content():
    _, train, val, _ = load("experiments/raw-base-v1/config.json")
    for row in train + val:
        request = prompt(dict(row, content_checks=["SECRET_REPORT_ONLY"]))
        assert messages(request)[0]["content"] == judge_state(request, "draft")["REQUEST"]
        assert "SECRET_REPORT_ONLY" not in request


def test_actual_jev_payload_has_only_user_request_and_draft():
    payloads = []

    def respond(request):
        payloads.append(json.loads(request.content))
        return httpx.Response(200, json=response(0.7))

    judge = Jev("test-key")
    judge.client.close()
    judge.client = httpx.Client(transport=httpx.MockTransport(respond))
    try:
        judge.score("Give me caption options", "Option 1: A\nOption 2: B")
    finally:
        judge.close()
    assert json.loads(payloads[0]["state"]) == judge_state(
        "Give me caption options", "Option 1: A\nOption 2: B"
    )


def test_menu_is_diagnostic_not_automatic_user_only_failure():
    group = []
    for i, value in enumerate((0.4, 0.6, 0.8, 0.9)):
        text = f"Option 1: Caption {i}\nOption 2: Another caption"
        group.append(
            {"case": {"id": "options"}, "sample": i, "draft": text, "score": calculate(response(value), text)}
        )
    d = diagnose(group, 0.1)
    assert d["style_signal_candidate"]
    assert all(x["nonanswer_hints"] == ["option_menu"] for x in d["samples"])
