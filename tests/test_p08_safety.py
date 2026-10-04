import json

import numpy as np

from actions import Action, Decision
from safety import FrameRing, Guards, safe_mode
from tests.fakes import make_state

NONE = Decision(Action.NONE)
DOOR = Decision(Action.DOOR_L, None, "threat_close_L")


def guards(tmp_path):
    return Guards(stop_path=tmp_path / "STOP")


def bad(**kw):
    return make_state(trusted=False, reasons=["missing:hour"], **kw)


def test_stop_file(tmp_path):
    g = guards(tmp_path)
    assert g.vet(make_state(), NONE, 0).action == Action.NONE
    (tmp_path / "STOP").write_text("")
    d = g.vet(make_state(), NONE, 1)
    assert (d.action, d.reason) == (Action.SAFE_MODE, "stop_file")


def test_untrusted_timer(tmp_path):
    g = guards(tmp_path)
    g.vet(bad(), NONE, 0)
    assert g.vet(bad(), NONE, 4.9).action == Action.NONE
    g.vet(make_state(), NONE, 5.0)          # trusted frame resets
    g.vet(bad(), NONE, 5.1)
    assert g.vet(bad(), NONE, 10.0).action == Action.NONE
    d = g.vet(bad(), NONE, 10.3)
    assert (d.action, d.reason) == (Action.SAFE_MODE, "untrusted_too_long")


def test_freeze_reasons(tmp_path):
    for r in ("frozen_clock", "capture_stall"):
        d = guards(tmp_path).vet(make_state(trusted=False, reasons=[r]), NONE, 0)
        assert (d.action, d.reason) == (Action.SAFE_MODE, r)


def run_door(g, state_fn, n):
    """Issue DOOR, keep vetting each 1.6 s with idle policy; return list of decisions."""
    out, t = [], 0.0
    d = g.vet(state_fn(t), DOOR, t)
    g.expect(d, t)
    out.append(d)
    for _ in range(n):
        t += 1.6
        d = g.vet(state_fn(t), NONE, t)
        out.append(d)
        if d.action != Action.NONE and d.action != Action.SAFE_MODE:
            g.expect(d, t)
    return out


def test_click_verify_retries(tmp_path):
    out = run_door(guards(tmp_path), lambda t: make_state(t=t), 4)
    assert [d.reason for d in out[1:]] == ["retry_1", "retry_2", "retry_3", "click_not_taken"]
    assert out[-1].action == Action.SAFE_MODE and out[1].action == Action.DOOR_L


def test_click_verify_recovers(tmp_path):
    g = guards(tmp_path)

    def st(t):  # door closes at the 2nd retry
        closed = t > 3.3
        return make_state(t=t, door_closed={"L": closed, "R": False})
    out = run_door(g, st, 4)
    assert [d.reason for d in out[1:3]] == ["retry_1", "retry_2"]
    assert all(d.action == Action.NONE for d in out[3:]) and g.pending is None
    d = g.vet(make_state(door_closed={"L": True, "R": False}), DOOR, 20)  # fresh action: counters reset
    g.expect(d, 20)
    assert g.vet(make_state(door_closed={"L": True, "R": False}), NONE, 21.6).reason == "retry_1"


def test_monitor_stuck(tmp_path):
    g = guards(tmp_path)
    up = make_state(monitor_up=True)
    assert g.vet(up, NONE, 0).action == Action.NONE
    d = g.vet(up, NONE, 6.0)
    assert (d.action, d.reason) == (Action.MONITOR, "monitor_second_gesture")
    assert g.vet(up, NONE, 6.5).action == Action.NONE
    d = g.vet(up, NONE, 7.6)
    assert (d.action, d.reason) == (Action.SAFE_MODE, "monitor_stuck")


def test_rate_limit(tmp_path):
    g = guards(tmp_path)
    ds = [g.vet(make_state(), Decision(Action.LIGHT_L), i * 1.4) for i in range(41)]
    assert all(d.action == Action.LIGHT_L for d in ds[:40])
    assert (ds[40].action, ds[40].reason) == (Action.NONE, "guard_rate")


def test_focus_refused(tmp_path):
    g = guards(tmp_path)
    for _ in range(2):
        g.note_refused()
    assert g.vet(make_state(), NONE, 0).action == Action.NONE
    g.note_refused()
    d = g.vet(make_state(), NONE, 1)
    assert (d.action, d.reason) == (Action.SAFE_MODE, "focus_refused")


def test_safe_mode_dump_and_latch(tmp_path, capsys):
    ring = FrameRing(30, 10)
    for i in range(400):
        ring.push(i / 10, np.zeros((4, 4, 3), np.uint8))
    p = safe_mode("because", ring, [{"t": 1}, {"t": 2}], tmp_path)
    assert (p / "reason.txt").read_text().strip() == "because"
    assert len(list((p / "frames").glob("*.png"))) == 300
    assert len(json.loads((p / "ring.json").read_text())) == 300
    assert len((p / "log_tail.jsonl").read_text().splitlines()) == 2
    assert "because" in capsys.readouterr().err
    g = guards(tmp_path)
    (tmp_path / "STOP").write_text("")
    g.vet(make_state(), NONE, 0)
    (tmp_path / "STOP").unlink()
    assert g.vet(make_state(), NONE, 1).action == Action.SAFE_MODE  # latched
    g.reset()
    assert g.vet(make_state(), NONE, 2).action == Action.NONE


def test_pending_door_or_light_dropped_when_monitor_pops_up():
    from tests.fakes import make_state
    g = Guards()
    st = make_state(light_on={"L": True, "R": False})
    g.vet(st, Decision(Action.LIGHT_L, None, "light_off"), 0.0)
    g.expect(Decision(Action.LIGHT_L, None, "light_off"), 0.0)
    up = make_state(monitor_up=True, light_on={"L": True, "R": False})
    d = g.vet(up, Decision(Action.NONE, None, "settle"), 3.0)  # past VERIFY_S: would have retried the click
    assert d.action == Action.NONE and g.pending is None
