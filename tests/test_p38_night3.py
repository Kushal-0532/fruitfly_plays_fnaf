import types

import config
from actions import Action
from policy import Supervisor
from power import PowerModel
from tests.fakes import make_state


def sup(night):
    cfg = types.SimpleNamespace(**{**vars(config), "HOUR_SCALE": {}, "CHECK_PERIOD": {"L": 1e9, "R": 1e9}})
    return Supervisor(PowerModel(), cfg, night=night)


def test_flip_camera_by_night_and_right_door_rule():
    s2, s3 = sup(2), sup(3)
    for s, n in ((s2, 2), (s3, 3)):
        s.flip_n = 1
    assert s2._flip_cam(make_state()) == "1C" and s3._flip_cam(make_state()) == "4B"
    s3.flip_n = config.COVE_LOOK_EVERY
    assert s3._flip_cam(make_state(door_closed={"L": False, "R": False})) == "4B"   # right door open: stay on 4B
    assert s3._flip_cam(make_state(door_closed={"L": False, "R": True})) == "1C"    # shut: Foxy gets his look
    s2.flip_n = config.COVE_LOOK_EVERY
    assert s2._flip_cam(make_state(door_closed={"L": False, "R": True})) == "1C"


def test_night3_flip_selects_4b():
    s = sup(3)
    s.decide(make_state(), 0)
    s.clear_t = {"L": 15.0, "R": 15.0}
    d = s.decide(make_state(t=20.0), 20.0)
    assert (d.action, d.reason) == (Action.MONITOR, "stall_flip")
    t = 20.0 + config.SETTLE_S + 0.01
    d = s.decide(make_state(t=t, monitor_up=True, cam="1A"), t)
    assert (d.action, d.arg, d.reason) == (Action.CAM, "4B", "stall_cam")


def look4b(s, t, score, steps=3):
    first = d = None
    pre = t - config.CAM_SETTLE_S - 0.01  # 4B on screen long enough for its fly scores to count
    s.decide(make_state(t=pre, monitor_up=True, cam="4B"), pre)
    for k in range(steps):
        tt = t + 0.1 * k
        d = s.decide(make_state(t=tt, monitor_up=True, cam="4B", cam4b=score), tt)
        first = first or (d if d.action != Action.NONE else None)
    return first or d, t + 0.1 * steps


def test_somebody_on_4b_lowers_monitor_then_closes_right_door():
    s = sup(3)
    s.decide(make_state(), 0)
    d, t = look4b(s, 5.0, 0.95)
    assert (d.action, d.reason) == (Action.MONITOR, "stall_down")
    t += config.SETTLE_S + 0.01
    d = s.decide(make_state(t=t), t)
    assert (d.action, d.reason) == (Action.DOOR_R, "threat_close_R")


def test_single_noisy_4b_frame_or_left_door_irrelevant():
    s = sup(3)
    s.decide(make_state(), 0)
    d, t = look4b(s, 5.0, 0.95, steps=1)
    assert d.reason != "stall_down"
    d, t = look4b(s, 6.0, 0.1)
    assert d.action == Action.NONE


def test_flip_holds_until_the_fly_judged_the_camera():
    """Live 2026-10-05 N3 round 2: 1 s holds gave the fly one settled verdict, an empty cove went unconfirmed and Foxy got in."""
    s = sup(3)
    s.decide(make_state(), 0)
    s.flip_cam, s.flip_t, s.last_flip = "4B", 10.0, 10.0
    up = lambda tt, v=None, fresh=set(): make_state(t=tt, monitor_up=True, cam="4B", cam4b=v, fresh=fresh)
    t = 10.0
    while t < 10.0 + config.STALL_HOLD_S + 0.3:   # no new verdicts: hold past STALL_HOLD_S
        assert s.decide(up(t), t).reason != "stall_down"
        t += 0.1
    for k in range(config.FLIP_VERDICTS):         # three fresh verdicts -> lower
        d = s.decide(up(t, 0.05, {"cam4b"}), t)
        t += 0.1
    assert d.reason == "stall_down"
