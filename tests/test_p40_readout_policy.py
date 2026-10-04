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
