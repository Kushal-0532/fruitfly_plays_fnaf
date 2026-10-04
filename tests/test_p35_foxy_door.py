import types

import config
import sim_env
from actions import Action
from policy import Supervisor
from power import PowerModel
from tests.fakes import make_state


def sup(**over):
    cfg = types.SimpleNamespace(**{**vars(config), "HOUR_SCALE": {}, "CHECK_PERIOD": {"L": 1e9, "R": 1e9}, **over})
    return Supervisor(PowerModel(), cfg, night=2), cfg


def look(s, t, cove, door_L=False, steps=3):
    """Feed `steps` frames of a 1C look (monitor up, cam 1C) and return the last decision."""
    first = d = None
    for k in range(steps):
        tt = t + 0.1 * k
        d = s.decide(make_state(t=tt, monitor_up=True, cam="1C", cove=cove, door_closed={"L": door_L, "R": False}), tt)
        first = first or (d if d.action != Action.NONE else None)
    return first or d, t + 0.1 * steps


def test_high_cove_lowers_monitor_then_closes_left_door():
    s, c = sup()
    s.decide(make_state(), 0)
    d, t = look(s, 5.0, 0.95)
    assert (d.action, d.reason) == (Action.MONITOR, "stall_down")
    t += c.SETTLE_S + 0.01
    d = s.decide(make_state(t=t), t)
    assert (d.action, d.reason) == (Action.DOOR_L, "foxy_close")


def test_later_low_cove_reopens_then_hall_look():
    s, c = sup(REOPEN_PROBE_S=None)
    s.decide(make_state(), 0)
    d, t = look(s, 5.0, 0.95)
    t += c.SETTLE_S + 0.01
    assert s.decide(make_state(t=t), t).reason == "foxy_close"
    # a flip begun after the close sees him home
    s.flip_t, s.last_flip = t + 1.0, t + 1.0
    t += 2.0 + c.SETTLE_S
    d, t = look(s, t, 0.1, door_L=True, steps=c.COVE_CLEAR_CONFIRM)
    assert (d.action, d.reason) == (Action.MONITOR, "stall_down")
    t += c.SETTLE_S + 0.01
    d = s.decide(make_state(t=t, door_closed={"L": True, "R": False}), t)
    assert (d.action, d.reason) == (Action.DOOR_L, "reopen_cove")
    t += c.SETTLE_S + 0.01
    assert s.decide(make_state(t=t), t).action == Action.LIGHT_L  # immediate hall look


def test_foxy_out_blocks_reopen_probe_on_left():
    s, c = sup(REOPEN_PROBE_S=5.0)
    s.decide(make_state(), 0)
    d, t = look(s, 5.0, 0.95)
    t += c.SETTLE_S + 0.01
    s.decide(make_state(t=t), t)
    shut = dict(door_closed={"L": True, "R": False}, light_on={"L": True, "R": False}, hall={"L": 0.97, "R": None})
    for tt in range(int(t) + 1, int(t) + 60):
        assert s.decide(make_state(t=float(tt), **shut), float(tt)).reason != "reopen_probe"


def test_single_noisy_frame_does_not_close():
    s, c = sup()
    s.decide(make_state(), 0)
    d, t = look(s, 5.0, 0.95, steps=1)
    assert d.reason != "stall_down" and not s.foxy_out


def test_sim_foxy_deaths_only_with_a_dead_left_door():
    """Night 2 sim: with the cove readout, Foxy kills (almost) only when Bonnie has jammed the left door. 6 AM stays out of
    reach because of power (phase 31 note), so this checks the Foxy handling only."""
    deaths, jam_free = 0, 0
    for seed in range(40):
        n = {}
        orig = sim_env.Night.die
        def die(self, cause, orig=orig, n=n):
            n["jam"] = self.jammed["bonnie"]; orig(self, cause)
        sim_env.Night.die = die
        try:
            r = sim_env.play(seed=seed)
        finally:
            sim_env.Night.die = orig
        deaths += r["cause"] == "foxy" and not n["jam"]
        assert r["audit"]["violations"] == []
    assert deaths <= 2, deaths


def test_a_few_low_frames_do_not_reopen():
    s, c = sup(REOPEN_PROBE_S=None)
    s.decide(make_state(), 0)
    d, t = look(s, 5.0, 0.95)
    t += c.SETTLE_S + 0.01
    s.decide(make_state(t=t), t)
    s.flip_t = t + 1.0
    t += 2.0 + c.SETTLE_S
    d, t = look(s, t, 0.1, door_L=True, steps=c.COVE_CLEAR_CONFIRM - 1)
    assert d.reason != "stall_down" and s.foxy_out
