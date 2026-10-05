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


def look(s, t, cove, door_L=False, steps=3, gone=None):
    """Feed `steps` frames of a 1C look (monitor up, cam 1C) and return the last decision. gone defaults to cove: an empty cove
    also scores high on the peek readout (held-out hit 1.0 at stage 4)."""
    gone = cove if gone is None else gone
    first = d = None
    pre = t - config.CAM_SETTLE_S - 0.01  # the camera has been on screen long enough for its fly scores to count
    s.decide(make_state(t=pre, monitor_up=True, cam="1C", door_closed={"L": door_L, "R": False}), pre)
    for k in range(steps):
        tt = t + 0.1 * k
        d = s.decide(make_state(t=tt, monitor_up=True, cam="1C", cove=cove, cove_gone=gone, door_closed={"L": door_L, "R": False}), tt)
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


def test_held_cove_repeats_do_not_confirm():
    """Live 2026-10-05 N3: one fresh high 1C frame, then the tracker's held copy, counted as 2 frames -> foxy_close on a static frame."""
    s, c = sup()
    s.decide(make_state(), 0)
    s.decide(make_state(t=4.0, monitor_up=True, cam="1C"), 4.0)
    for k in range(5):
        tt = 5.0 + 0.1 * k
        d = s.decide(make_state(t=tt, monitor_up=True, cam="1C", cove=0.95, cove_gone=0.95, fresh={"cove", "cove_gone"} if k == 0 else set()), tt)
        assert d.reason != "stall_down" and not s.foxy_out


def test_cove_ignored_right_after_camera_switch():
    s, c = sup()
    s.decide(make_state(), 0)
    for k in range(4):  # first frames of a look: transition static
        tt = 5.0 + 0.1 * k
        d = s.decide(make_state(t=tt, monitor_up=True, cam="1C", cove=0.95, cove_gone=0.95), tt)
        assert d.reason != "stall_down"


def test_peek_watches_instead_of_closing():
    """Live 2026-10-05 N3 round 2: closing on every peek (stage 2-3) ran the power out. A peek makes the flips watch the cove, faster."""
    s, c = sup()
    s.night = 3
    s.decide(make_state(), 0)
    d, t = look(s, 5.0, 0.95, gone=0.05, steps=4)
    assert d.reason != "stall_down" and not s.foxy_out and s._watching(t)
    s.last_flip, s.flip_n, s.clear4b_t = t, 1, t  # next flip: not a routine cove turn (flip_n odd), 4B seen empty just now
    assert s._flip_cam(make_state(t=t)) == c.STALL_CAM
    s.peek_t = None
    assert s._flip_cam(make_state(t=t)) == "4B"


def test_foxy_out_clears_when_someone_else_reopens():
    """Live 2026-10-05 N3 round 3: the readout reopened L, foxy_out stayed set, every flip went to 1C and Freddy walked in via 4B."""
    s, c = sup()
    s.decide(make_state(), 0)
    d, t = look(s, 5.0, 0.95)
    t += c.SETTLE_S + 0.01
    assert s.decide(make_state(t=t), t).reason == "foxy_close"
    t += 1.0
    s.decide(make_state(t=t, door_closed={"L": True, "R": False}), t)
    assert s.foxy_out
    t += 30.0
    s.decide(make_state(t=t, door_closed={"L": False, "R": False}), t)  # reopened by the readout policy
    assert not s.foxy_out
