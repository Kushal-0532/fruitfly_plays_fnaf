import numpy as np
import pytest

import readout_policy as R
import sim_env
from actions import Action
from power import PowerModel
from scripts.audit_decisions import audit
from state import GameState


def state(**kw):
    s = GameState(t=0.0, hour=1, power_pct=80.0, usage=1, monitor_up=False, trusted=True)
    s.door_closed, s.light_on, s.hall = {"L": False, "R": False}, {"L": False, "R": False}, {"L": None, "R": None}
    for k, v in kw.items():
        setattr(s, k, v)
    return s


@pytest.fixture(scope="module")
def trace():
    return R.collect(range(8), nights=(2,))


def test_hysteresis_and_reasons():
    W = np.zeros((2, len(R.NAMES)))
    for i, side in enumerate("LR"):
        W[i, R.NAMES.index(f"hall_{side}")] = 8.0
        W[i, R.NAMES.index(f"closed_{side}")] = 4.0
    W[:, -1] = -4.0
    pol = R.ReadoutPolicy(PowerModel(), night=None, weights=W)
    lit = state(hall={"L": 0.9, "R": None}, light_on={"L": True, "R": False})
    assert pol.decide(lit, 9.0).action != Action.DOOR_L          # the light just came on: its ramp reads high, wait LOOK_MIN_S
    d = pol.decide(lit, 10.5)
    assert (d.action, d.reason) == (Action.DOOR_L, "readout_close_L")
    s = state(door_closed={"L": True, "R": False})
    assert pol.decide(s, 12.0).action != Action.DOOR_L          # p = sigmoid(0) = 0.5: between thresholds, door stays closed
    W[:, -1] = -9.0
    pol = R.ReadoutPolicy(PowerModel(), night=None, weights=W)
    d = pol.decide(s, 20.0)
    assert (d.action, d.reason) == (Action.DOOR_L, "readout_open_L")


def test_no_clicks_with_monitor_up_or_jam():
    W = np.zeros((2, len(R.NAMES)))
    W[:, -1] = 9.0                                            # wants both doors shut
    pol = R.ReadoutPolicy(PowerModel(), night=None, weights=W)
    assert pol.decide(state(monitor_up=True), 5.0).action not in (Action.DOOR_L, Action.DOOR_R)
    pol.jammed = {"L": True, "R": True}
    assert pol.decide(state(), 9.0).action not in (Action.DOOR_L, Action.DOOR_R)


def test_audit_accepts_readout_reasons():
    rows = [{"t": 0, "action": Action.DOOR_L, "reason": "readout_close_L"}, {"t": 1, "action": Action.DOOR_R, "reason": "readout_open_R"}]
    assert audit(rows)["violations"] == []
    assert audit([{"t": 0, "action": Action.DOOR_L, "reason": "readout_bogus"}])["violations"]


def test_imitation_agrees_90pct(trace):
    W, X, Y = R.fit_imitation(trace)
    assert R.agreement(W, X, Y)[0] >= 0.90


def test_saved_weights_run_full_nights():
    for night in (2, 3):
        r = sim_env.play(seed=1, night=night, policy="readout")
        assert r["audit"]["violations"] == [] and r["safe"] is None
        assert all(a in ("6am", "power_out", "foxy", "freddy", "bonnie_jam", "chica_jam") for a in [r["cause"]])


def test_trained_readout_reopens_without_evidence():
    """Live 2026-10-04 N3: one cove sighting shut L and the capped tclosed kept it shut all night (power out at 4 AM).
    With no fresh evidence a closed door must reopen on the saved weights within ~2 min."""
    from tests.fakes import make_state
    pol = R.ReadoutPolicy(PowerModel(), night=3)
    for k in range(1200):  # 120 s at 10 Hz, L shut, nothing seen
        t = k / 10
        x = pol.features(make_state(door_closed={"L": True, "R": False}, hall={"L": None, "R": None}, power_pct=60.0), t)
        if not pol.targets(x)["L"][0]:
            assert 20 < t < 120, t
            return
    raise AssertionError("L never reopened")


def test_closed_door_does_not_freeze_looks():
    """Live 2026-10-04 N3 round 2: with L shut the Supervisor's reopen_probe (swallowed by the readout) returned every step
    and no look or flip ran for 170 s. Looks/flips must keep coming while a door is held shut."""
    from tests.fakes import make_state
    pol = R.ReadoutPolicy(PowerModel(), night=3, weights=np.zeros((2, len(R.NAMES))) + np.eye(2, len(R.NAMES), 4) * 20)  # holds doors shut
    pol.close_t["L"] = 0.0
    acts = []
    for k in range(600):
        t = 10.0 + k / 10
        d = pol.decide(make_state(door_closed={"L": True, "R": False}, hall={"L": None, "R": None}, hour=2), t)
        if d.action != Action.NONE:
            acts.append(d.action)
            pol.last_action_t = t
    assert acts, "no look or flip in 60 s with L shut"


def test_fly_4b_verdict_closes_right_door():
    """Live 2026-10-05 N3 round 3: the fly saw 4B = 1.0 on 4 flips, the readout swallowed threat_close_R every time. The Supervisor's
    fly-verdict close must reach the door; other Supervisor door rules stay swallowed."""
    from tests.fakes import make_state
    pol = R.ReadoutPolicy(PowerModel(), night=3, weights=np.zeros((2, len(R.NAMES))))
    pol.r_out = True  # the fly saw 4B occupied on the last flip; the monitor is now down
    d = pol.decide(make_state(hall={"L": None, "R": None}, hour=2), 50.0)
    assert (d.action, d.reason) == (Action.DOOR_R, "threat_close_R")
