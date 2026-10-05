import types

import config
from actions import Action
from policy import REASONS, Supervisor
from power import PowerModel
from tests.fakes import make_state


def sup(**over):
    cfg = types.SimpleNamespace(**{**vars(config), **over})
    return Supervisor(PowerModel(), cfg), cfg


def calls(s, state_fn, t0, t1, dt=0.1):
    out, t = [], t0
    while t < t1 - 1e-9:
        out.append((round(t, 3), s.decide(state_fn(t), t)))
        t += dt
    return out


def acts(res):
    return [(t, d.action, d.reason) for t, d in res if d.action != Action.NONE]


def test_quiet_then_first_check():
    s, c = sup(HOUR_SCALE={})
    d = s.decide(make_state(), 0)
    assert (d.action, d.reason) == (Action.NONE, "idle")
    per = c.CHECK_PERIOD["L"]
    assert s.decide(make_state(), per - 0.1).action == Action.NONE
    d = s.decide(make_state(), per)
    assert (d.action, d.reason) == (Action.LIGHT_L, "hall_check")


def test_threat_close():
    s, _ = sup()
    s.decide(make_state(), 0)
    d = s.decide(make_state(hall={"L": 0.9, "R": 0.0}), 1)
    assert (d.action, d.reason) == (Action.DOOR_L, "threat_close_L")


def test_both_threats_higher_first():
    s, c = sup()
    st = lambda: make_state(hall={"L": 0.7, "R": 0.9})
    assert s.decide(st(), 1).action == Action.DOOR_R
    assert s.decide(make_state(door_closed={"L": False, "R": True}, hall={"L": 0.7, "R": 0.9}), 1.1).reason == "settle"
    d = s.decide(make_state(door_closed={"L": False, "R": True}, hall={"L": 0.7, "R": 0.9}), 1 + c.SETTLE_S)
    assert d.action == Action.DOOR_L


def closed_state(hall, light=True, t=0.0, **kw):
    return make_state(t=t, door_closed={"L": False, "R": True}, light_on={"L": False, "R": light},
                      hall={"L": 0.0, "R": hall}, **kw)


def test_hold_and_reopen():
    s, c = sup(CLOSED_LIGHT_S=1e9)  # the lit hold itself (by default the light goes off after CLOSED_LIGHT_S)
    s.decide(make_state(hall={"L": 0, "R": 0.9}), 0)  # closes R at t=0
    t = c.SETTLE_S
    # before MIN_HOLD_S: nothing
    while t < c.MIN_HOLD_S:
        assert s.decide(closed_state(0.0), t).action == Action.NONE
        t += 0.1
    t = c.MIN_HOLD_S
    # occupied -> stays closed
    for _ in range(20):
        d = s.decide(closed_state(0.9), t)
        assert d.action == Action.NONE and d.reason == "hold_occupied"
        t += 0.1
    # 4 empty then occupied resets streak
    for _ in range(c.EMPTY_FRAMES - 1):
        assert s.decide(closed_state(0.0), t).action == Action.NONE
        t += 0.1
    assert s.decide(closed_state(0.9), t).reason == "hold_occupied"
    t += 0.1
    seq = []
    for _ in range(c.EMPTY_FRAMES):
        seq.append(s.decide(closed_state(0.0), t))
        t += 0.1
    assert [d.action for d in seq[:-1]] == [Action.NONE] * (c.EMPTY_FRAMES - 1)
    assert (seq[-1].action, seq[-1].reason) == (Action.DOOR_R, "reopen_empty")


def test_conservation_doubles_period_and_blocks_speculative():
    cnt = {}
    for name, power, usage in [("normal", 99.0, 1), ("cons", 20.0, 2)]:
        s, _ = sup(HOUR_SCALE={})
        on_until, n, t = -1, 0, 0.0
        while t < 300:  # light comes on 0.3 s for 0.9 s after any look click (a dark look would jam)
            lit = t < on_until
            d = s.decide(make_state(t=t, power_pct=power, usage=usage, light_on={"L": lit, "R": lit}), t)
            if d.reason == "hall_check":
                n, on_until = n + 1, t + 0.9
            t += 0.1
        cnt[name] = n
    assert cnt["normal"] >= 10 and cnt["cons"] <= cnt["normal"] * 0.6, cnt
    for power, usage, expect in [(99.0, 1, Action.DOOR_L), (20.0, 2, Action.NONE)]:
        s, _ = sup()
        s.note_audio("L", 0.9)
        d = s.decide(make_state(power_pct=power, usage=usage, hall={"L": None, "R": 0.0}), 1)
        assert d.action == expect


def test_monitor_lower():
    s, c = sup()
    s.decide(make_state(monitor_up=True), 0)
    assert s.decide(make_state(monitor_up=True), c.MONITOR_MAX_UP_S - 0.1).action == Action.NONE
    d = s.decide(make_state(monitor_up=True), c.MONITOR_MAX_UP_S + 0.1)
    assert (d.action, d.reason) == (Action.MONITOR, "monitor_lower")


def test_untrusted_and_settle():
    s, c = sup()
    d = s.decide(make_state(trusted=False), 0)
    assert (d.action, d.reason) == (Action.NONE, "untrusted_wait")
    assert s.decide(make_state(hall={"L": 0.9, "R": 0}), 1).action == Action.DOOR_L
    assert s.decide(make_state(hall={"L": 0.9, "R": 0.9}), 1 + c.SETTLE_S - 0.01).reason == "settle"


def timeline(t):
    return make_state(t=t, hall={"L": 0.9 if 40 < t < 60 else 0.0, "R": 0.8 if 100 < t < 130 else 0.0},
                      foxy_stage=3 if 200 < t < 205 else 0, monitor_up=int(t) % 97 == 0 and t > 1)


def test_determinism_and_reasons():
    runs = []
    for _ in range(2):
        s, _ = sup()
        runs.append(calls(s, timeline, 0, 300))
    assert runs[0] == runs[1]
    assert all(d.reason in REASONS and d.reason for _, d in runs[0])
    assert acts(runs[0])


def test_light_left_on_by_a_check_is_switched_off():
    s, c = sup()
    st = make_state(light_on={"L": True, "R": False}, hall={"L": 0.0, "R": None})
    assert s.decide(st, 5.0).reason != "light_off"  # just came on: give the fly time to look
    d = s.decide(st, 5.0 + c.LOOK_MIN_S + 0.1)
    assert (d.action, d.reason) == (Action.LIGHT_L, "light_off")


def test_occupied_hall_beats_light_off():
    s, _ = sup()
    st = make_state(light_on={"L": True, "R": False}, hall={"L": 0.9, "R": None})
    assert s.decide(st, 5.0).reason == "threat_close_L"


def test_check_interval_shrinks_later_in_the_night():
    def first_check(hour):
        s, c = sup()
        t = 0.0
        while t < 120:
            d = s.decide(make_state(hour=hour), t)
            if d.reason == "hall_check":
                return t
            t += 0.1
        return None
    assert first_check(0) > first_check(2) > first_check(4)


def test_closed_door_reopens_blind_after_the_hold():
    s, c = sup(REOPEN_PROBE_S=12.0, CLOSED_LIGHT_S=1e9)
    closed = dict(door_closed={"L": True, "R": False}, light_on={"L": True, "R": False}, hall={"L": 0.97, "R": None})
    assert s.decide(make_state(**closed), 0.0).reason != "reopen_probe"
    d = s.decide(make_state(**closed), 12.5)
    assert (d.action, d.reason) == (Action.DOOR_L, "reopen_probe")



def dark(tt):
    return make_state(t=tt, hall={"L": None, "R": None}, light_on={"L": False, "R": False})


def test_dark_looks_jam_the_side_and_monitor_is_never_raised():
    s, c = sup(HOUR_SCALE={}, CHECK_PERIOD={"L": 8, "R": 1e9})
    s.decide(make_state(), 0)
    reasons, t = [], 8.0
    while t < 400:
        d = s.decide(dark(t), t)
        reasons.append((t, d.action, d.reason))
        t += 0.1
    jam_t = next(t for t, a, r in reasons if r == "jam_L")
    assert [r for _, _, r in reasons].count("hall_check") == c.JAM_CONFIRM  # two dark looks, then jam
    assert all(a not in (Action.MONITOR, Action.LIGHT_L) for tt, a, r in reasons if tt > jam_t)


def test_no_flip_while_a_look_is_unanswered():
    """Live 2026-10-05 N3 round 4: the R light stayed dark (Chica jammed it) and a stall flip raised the monitor 2 s later (F25)."""
    s, c = sup(HOUR_SCALE={})
    s.night = 3
    s.decide(make_state(), 0)
    look, t = None, 1.0
    while t < 120:
        d = s.decide(dark(t), t)
        if d.action != Action.NONE:
            s.last_action_t = t
        if d.reason == "hall_check":
            look = t
        assert not (d.action == Action.MONITOR and look is not None and t - look < c.LOOK_FAIL_S), (t, look)
        t += 0.1


def test_one_dark_look_or_a_hall_reading_is_not_a_jam():
    s, c = sup(HOUR_SCALE={}, CHECK_PERIOD={"L": 8, "R": 1e9})
    s.decide(make_state(), 0)
    assert s.decide(make_state(t=8.0), 8.0).reason == "hall_check"
    lit_by_hall = lambda tt: make_state(t=tt, hall={"L": 0.07, "R": None}, light_on={"L": False, "R": False})  # lamp misread, fly sees the lit hall
    for k in range(80, 140):
        assert not s.decide(lit_by_hall(k / 10), k / 10).reason.startswith("jam")



def test_no_light_click_on_a_closed_door():
    s, c = sup(REOPEN_PROBE_S=None)
    s.decide(make_state(hall={"L": 0, "R": 0.9}), 0)  # closes R
    for tt in range(1, 40):
        d = s.decide(closed_state(0.0, light=False, t=tt), tt)
        assert d.action not in (Action.LIGHT_R, Action.DOOR_R) or d.reason == "reopen_empty"
        assert d.action != Action.LIGHT_R


def test_reopen_probe_then_immediate_look():
    s, c = sup(REOPEN_PROBE_S=12.0, CLOSED_LIGHT_S=1e9)
    closed = dict(door_closed={"L": True, "R": False}, light_on={"L": True, "R": False}, hall={"L": 0.97, "R": None})
    s.decide(make_state(**closed), 0.0)
    assert s.decide(make_state(**closed), 12.5).reason == "reopen_probe"
    t = 12.5 + c.SETTLE_S + 0.1
    d = s.decide(make_state(t=t), t)
    assert (d.action, d.reason) == (Action.LIGHT_L, "hall_check")


def test_light_over_a_closed_door_goes_off():
    """Sim N3 2026-10-05: R shut on Chica with its light on, the light burned 12+ s and blocked every flip; Foxy ran unwatched."""
    s, c = sup(REOPEN_PROBE_S=None)
    s.decide(make_state(hall={"L": 0, "R": 0.9}), 0)  # closes R with the light on
    t, d = c.SETTLE_S, None
    while t < c.SETTLE_S + c.CLOSED_LIGHT_S + 1.0 and (d is None or d.reason != "light_off"):
        d = s.decide(closed_state(0.9, t=t), t)
        t += 0.1
    assert (d.action, d.reason) == (Action.LIGHT_R, "light_off") and t - c.SETTLE_S <= c.CLOSED_LIGHT_S + 0.2  # light first seen at SETTLE_S
