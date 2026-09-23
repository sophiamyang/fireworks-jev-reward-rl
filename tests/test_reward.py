import json
from pathlib import Path

import httpx
import pytest

from fw_jev import calibration, report
from fw_jev import reward as jev
from fw_jev.config import load
from fw_jev.mock import response as mock_response
from fw_jev.rl_math import advantages
from fw_jev.runner import fingerprint

ROOT = Path(__file__).resolve().parents[1]


def response(style=0.8, quality=0.6, support=None):
    result = {
        "model": jev.MODEL,
        "grounding_applicable": support is not None,
        "answers": {
            "style": {"type": "choice", "probabilities": {"yes": style, "no": 1 - style, "uncertain": 0.0}},
            "quality": {
                "type": "score",
                "score": 3 * quality,
                "probabilities": {"0": 1 - quality, "1": 0.0, "2": 0.0, "3": quality},
            },
        },
    }
    if support is not None:
        result["answers"]["unsupported_claim"] = {
            "type": "choice",
            "probabilities": {"yes": 1 - support, "no": support, "uncertain": 0.0},
        }
    return result


def test_only_two_writing_questions_and_one_conditional_check():
    assert set(jev.questions()) == {"style", "quality"}
    assert set(jev.questions(grounded=True)) == {"style", "quality", "unsupported_claim"}
    assert jev.questions(grounded=True)["quality"]["criteria"] == jev.QUALITY_LEVELS
    for question in jev.questions(grounded=True).values():
        assert question["instructions"].startswith(jev.SAFETY)


def test_every_question_changes_reward_in_the_intended_direction():
    baseline = jev.calculate(response(support=0.8), "Useful draft.")
    assert baseline["reward"] == pytest.approx((0.8 + 0.6) / 2 * 0.8)
    for values in (dict(style=0.9, support=0.8), dict(quality=0.9, support=0.8), dict(support=0.9)):
        assert jev.calculate(response(**values), "Useful draft.")["reward"] > baseline["reward"]
    assert sum(baseline["effective_contributions"].values()) == pytest.approx(baseline["reward"])
    assert baseline["version"] == jev.VERSION


def test_uncertainty_uses_half_credit_not_dropped_mass():
    raw = response(support=0.8)
    raw["answers"]["style"]["probabilities"] = {"yes": 0.5, "no": 0.1, "uncertain": 0.4}
    raw["answers"]["unsupported_claim"]["probabilities"] = {"yes": 0.1, "no": 0.5, "uncertain": 0.4}
    assert jev.calculate(raw, "Useful draft.")["reward"] == pytest.approx((0.7 + 0.6) / 2 * 0.7)


@pytest.mark.parametrize(
    "draft,kwargs",
    [("", {}), ("word " * 40, {}), ("draft", {"truncated": True}), ("draft", {"malformed": True})],
)
def test_technical_zeros_preserved(draft, kwargs):
    assert jev.calculate(response(), draft, **kwargs)["reward"] == 0


def test_high_confidence_unsupported_claim_overrides_high_style():
    raw = response(style=0.95, quality=0.95, support=0.04)
    raw["answers"]["unsupported_claim"]["probabilities"] = {"yes": 0.95, "no": 0.04, "uncertain": 0.01}
    score = jev.calculate(raw, "We added a $50 credit.")
    assert score["raw_reward"] == pytest.approx(0.95)
    assert score["reward"] == 0 and score["grounding"]["blocked"]
    assert advantages([0.8, score["reward"], 0.75, 0.85])["advantages"][1] < 0


def test_uncertain_support_is_not_automatically_guilty():
    raw = response(support=0.1)
    raw["answers"]["unsupported_claim"]["probabilities"] = {"yes": 0.3, "no": 0.1, "uncertain": 0.6}
    assert jev.calculate(raw, "Thanks.")["reward"] > 0


def test_fiction_is_not_source_bound():
    score = jev.calculate(response(), "The lighthouse spoke.")
    assert not score["grounding"]["applicable"] and score["grounding_support_factor"] == 1
    assert score["reward"] > 0


def test_missing_support_answer_is_an_error_not_a_pass():
    raw = mock_response(0.9)
    raw["grounding_applicable"] = True
    with pytest.raises(ValueError):
        jev.components(raw)


@pytest.mark.parametrize(
    "mutation",
    [
        lambda r: r.update(model="other-model"),
        lambda r: r.update(grounding_applicable="false"),
        lambda r: r["answers"].pop("style"),
        lambda r: r["answers"].update(extra=r["answers"]["style"]),
        lambda r: r["answers"]["style"]["probabilities"].update(yes=float("nan")),
        lambda r: r["answers"]["quality"].update(score=0.0),
    ],
)
def test_invalid_responses_fail_closed(mutation):
    raw = response()
    mutation(raw)
    with pytest.raises(ValueError):
        jev.calculate(raw, "draft")


@pytest.mark.parametrize(
    "weights",
    [{"style": 0.5}, {"style": 0.5, "quality": 0.4}, {"style": 1.0, "quality": 0.0}, {"style": True, "quality": 0}],
)
def test_weights_must_be_the_two_positive_writing_weights(weights):
    with pytest.raises(ValueError):
        jev.validate_weights(weights)


def test_client_sends_only_the_current_questions():
    def handler(request):
        payload = json.loads(request.content)
        assert set(payload["questions"]) == {"style", "quality", "unsupported_claim"}
        assert payload["model"] == jev.MODEL
        return httpx.Response(200, json=response(support=0.8))

    judge = jev.Jev("unit-test-key")
    judge.client.close()
    judge.client = httpx.Client(transport=httpx.MockTransport(handler))
    try:
        raw = judge.score("request", "draft", grounded=True)
        assert raw["telemetry"]["question_count"] == 3
        assert jev.calculate(raw, "draft")["reward"] == pytest.approx(0.56)
    finally:
        judge.close()


@pytest.mark.parametrize("status,body", [(500, {}), (200, {"model": "other"})])
def test_client_errors_halt_instead_of_scoring(status, body):
    judge = jev.Jev("unit-test-key")
    judge.client.close()
    judge.client = httpx.Client(transport=httpx.MockTransport(lambda r: httpx.Response(status, json=body)))
    try:
        with pytest.raises(RuntimeError, match="no reward|HTTP"):
            judge.score("request", "draft")
    finally:
        judge.close()


def test_fixed_checks_use_the_style_component(tmp_path):
    c, _, _, _ = load(ROOT / "experiments/raw-base-v1/config.json")
    fixed = calibration.run(jev.make_judge(mock=True), tmp_path / "fixed", c["weights"])
    assert fixed["passed"] and len(fixed["checks"]) == 21
    assert "style_detects_awkwardness_repeat_0" in fixed["checks"]
    assert len(fixed["component_correlations"]) == 1
    frozen = json.loads((tmp_path / "fixed/frozen-check.json").read_text())
    assert frozen["version"] == jev.VERSION
    assert frozen["questions"] == jev.questions(grounded=True)
    records = json.loads((tmp_path / "fixed/records.json").read_text())
    assert len(records) == jev.PREFLIGHT_REQUESTS == 32
    assert set(fixed["protocol_sha256"]) <= set(fingerprint(ROOT))


def test_reports_refuse_to_mix_reward_versions():
    rows = [{"score": jev.calculate(response(), "draft")} for _ in range(2)]
    rows[1]["score"]["version"] = "different"
    with pytest.raises(ValueError, match="different reward protocols"):
        report.stats(rows)
