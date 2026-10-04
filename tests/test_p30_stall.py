import types

import config
import sim_env
from actions import Action
from policy import Supervisor
from power import PowerModel
from tests.fakes import make_state

LIT = dict(light_on={"L": False, "R": False})


def sup(night=2, **over):
    cfg = types.SimpleNamespace(**{**vars(config), **over})
    return Supervisor(PowerModel(), cfg, night=night), cfg


def seen_clear(s, t):
    s.decide(make_state(t=t, light_on={"L": True, "R": True}, hall={"L": 0.05, "R": 0.05}), t)


def test_night1_never_flips():
    s, c = sup(night=1, STALL_FROM_NIGHT=2)  # config now stalls from night 1 (phase 38 fix)
    for k in range(3000):
        t = k * 0.1
        seen_clear(s, t) if k % 50 == 0 else None
        assert s.decide(make_state(t=t), t).action != Action.MONITOR


def test_flip_sequence_then_hall_look():
    s, c = sup(HOUR_SCALE={}, STALL_PERIOD_S={h: 9 for h in range(6)}, CHECK_PERIOD={"L": 1e9, "R": 1e9})
    s.decide(make_state(), 0)
    seen_clear(s, 8.0)
    d = s.decide(make_state(t=9.0), 9.0)
    assert (d.action, d.reason) == (Action.MONITOR, "stall_flip")
    t = 9.0 + c.SETTLE_S + 0.01
    d = s.decide(make_state(t=t, monitor_up=True, cam="1A"), t)
    assert (d.action, d.arg, d.reason) == (Action.CAM, "1C", "stall_cam")
    t += c.SETTLE_S + 0.01
    assert s.decide(make_state(t=t, monitor_up=True, cam="1C"), t).action == Action.NONE
    t += c.STALL_HOLD_S
    d = s.decide(make_state(t=t, monitor_up=True, cam="1C"), t)
    assert (d.action, d.reason) == (Action.MONITOR, "stall_down")
    t += c.SETTLE_S + 0.01
    d = s.decide(make_state(t=t), t)
    assert (d.action, d.reason) == (Action.LIGHT_L, "hall_check") or d.action == Action.LIGHT_R


def test_stale_hall_checked_before_flip():
    s, c = sup(HOUR_SCALE={}, STALL_PERIOD_S={h: 9 for h in range(6)}, CHECK_PERIOD={"L": 1e9, "R": 1e9})
    s.decide(make_state(), 0)
    d = s.decide(make_state(t=9.0), 9.0)
    assert (d.action, d.reason) in ((Action.LIGHT_L, "hall_check"), (Action.LIGHT_R, "hall_check"))


def test_jammed_side_never_flips():
    s, c = sup(HOUR_SCALE={}, CHECK_PERIOD={"L": 8, "R": 1e9}, STALL_PERIOD_S={h: 9 for h in range(6)})
    s.decide(make_state(), 0)
    dark = lambda tt: make_state(t=tt, hall={"L": None, "R": None}, light_on={"L": False, "R": False})
    seen = []
    for k in range(80, 4000):
        t = k / 10
        seen.append((t, s.decide(dark(t), t)))
    jam_t = next(t for t, d in seen if d.reason == "jam_L")
    assert all(d.action != Action.MONITOR for t, d in seen if t > jam_t)


def test_sim_flips_cut_foxy_deaths_by_half():
    seeds = range(60)
    on = sum(sim_env.play(seed=s)["cause"] == "foxy" for s in seeds)
    off = sum(sim_env.play(seed=s, flips=False)["cause"] == "foxy" for s in seeds)
    assert on <= off / 2, (on, off)
