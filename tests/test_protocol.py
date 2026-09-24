"""Three-arm protocol, PK schedules, and CLI smoke test."""

from __future__ import annotations

import json

import numpy as np
import pytest

from machinic_psychopharmacology.cli import main
from machinic_psychopharmacology.protocol import (
    ProtocolConfig,
    evaluate_arm,
    interaction_report,
    run_protocol,
)
from machinic_psychopharmacology.schedule import (
    Pulse,
    inverted_u_gain,
    ir_work_blocks,
    oros_constant,
)
from machinic_psychopharmacology.vectors import mix_overlap, unit


def _setup(n: int = 48, d: int = 16, overlap: float = 0.5, seed: int = 4):
    g = np.random.default_rng(seed)
    v = unit(g.normal(size=d))
    extra = g.normal(size=d)
    v_f = mix_overlap(v, extra, overlap)
    focus = g.normal(0.7, 0.2, size=(n, 1))
    hallu = g.normal(0.5, 0.3, size=(n, 1))
    noise = g.normal(0.0, 0.1, size=(n, d))
    residuals = focus * v_f + hallu * v + noise
    return residuals, v_f, v


def test_ir_schedule_rebound() -> None:
    sched = ir_work_blocks(
        block_starts=[0.0],
        block_duration=10.0,
        peak=0.8,
        rebound=0.3,
        rebound_duration=4.0,
    )
    assert sched.value(5.0) == pytest.approx(0.8)
    assert sched.value(11.0) == pytest.approx(-0.3)
    assert sched.value(20.0) == pytest.approx(0.0)


def test_oros_is_flat() -> None:
    s = oros_constant(0.25)
    assert s.value(0.0) == s.value(99.0) == 0.25


def test_inverted_u_peaks_at_x0() -> None:
    assert inverted_u_gain(0.0, 1.0, x0=0.0, width=1.0) == pytest.approx(1.0)
    assert inverted_u_gain(2.0, 1.0, x0=0.0, width=1.0) < inverted_u_gain(
        0.5, 1.0, x0=0.0, width=1.0
    )


def test_pulse_validation() -> None:
    with pytest.raises(ValueError):
        Pulse(start=0, duration=0, peak=1)


def test_protocol_three_arms_run() -> None:
    residuals, v_f, v = _setup()
    cfg = ProtocolConfig(gamma=0.9, kappa=1.0, p_star=0.0)
    results = run_protocol(residuals, v_f, v, cfg)
    assert set(results) == {"stimulant", "raw_combo", "ortho_combo"}
    for m in results.values():
        assert m.n_tokens == residuals.shape[0]
        assert np.isfinite(m.efficacy)
        assert m.toxicity >= 0


def test_combo_arms_cap_toxicity() -> None:
    residuals, v_f, v = _setup(overlap=0.7)
    cfg = ProtocolConfig(gamma=1.2, kappa=1.0, p_star=0.0)
    results = run_protocol(residuals, v_f, v, cfg)
    raw = results["raw_combo"]
    ortho = results["ortho_combo"]
    stim = results["stimulant"]
    # Clamp to p*=0: hallucination probe is numerically zero on combo arms.
    assert raw.toxicity == pytest.approx(0.0, abs=1e-10)
    assert ortho.toxicity == pytest.approx(0.0, abs=1e-10)
    assert ortho.toxicity <= stim.toxicity + 1e-9
    assert raw.toxicity <= stim.toxicity + 1e-9
    assert np.isfinite(raw.efficacy) and np.isfinite(ortho.efficacy)


def test_interaction_report_tracks_overlap() -> None:
    residuals, v_f, v = _setup(overlap=0.6)
    report = interaction_report(v_f, v)
    assert report["cosine"] == pytest.approx(0.6, abs=0.02)
    assert report["orthogonalizable"] == 1.0


def test_evaluate_arm_times_must_match() -> None:
    residuals, v_f, v = _setup(n=8)
    with pytest.raises(ValueError):
        evaluate_arm(residuals, "stimulant", v_f, v, times=np.arange(3))


def test_cli_json_smoke(capsys: pytest.CaptureFixture[str]) -> None:
    rc = main(["--json", "--tokens", "32", "--dim", "12", "--seed", "1", "--overlap", "0.35"])
    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    assert "arms" in payload and "interaction" in payload
    assert set(payload["arms"]) == {"stimulant", "raw_combo", "ortho_combo"}


def test_cli_human_smoke(capsys: pytest.CaptureFixture[str]) -> None:
    rc = main(["--tokens", "16", "--dim", "8"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "ortho_combo" in out
    assert "efficacy" in out
