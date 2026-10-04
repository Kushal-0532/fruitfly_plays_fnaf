from state import GameState, Readings, StateTracker


def full(t, hour=1, power=90.0, **kw):
    d = dict(hour=hour, power_pct=power, usage=1, monitor_up=False,
             door_closed={"L": False, "R": False}, light_on={"L": False, "R": False},
             hall={"L": 0.0, "R": 0.0})
    d.update(kw)
    return Readings(t, **d)


def test_stale_expiry_boundary():
    tr = StateTracker()
    tr.update(full(0))
    assert tr.update(Readings(4.99, hour=1, power_pct=90.0)).usage == 1  # STALE_S usage = 5
    assert tr.update(Readings(5.01)).usage is None
    tr2 = StateTracker()
    tr2.update(full(0))
    assert tr2.update(Readings(0.49)).hall["L"] == 0.0   # hall stale 0.5
    assert tr2.update(Readings(0.51)).hall["L"] is None


def test_hour_regress():
    tr = StateTracker()
    tr.update(full(0, hour=3))
    s = tr.update(full(1, hour=2))
    assert s.hour == 3 and "hour_regress" in s.reasons


def test_hour_jump():
    tr = StateTracker()
    tr.update(full(0, hour=3))
    s = tr.update(full(1, hour=5))
    assert s.hour == 3 and "hour_jump" in s.reasons
    assert tr.update(full(2, hour=4)).hour == 4


def test_trusted_requires_fields():
    tr = StateTracker()
    assert tr.update(full(0)).trusted
    for k, name in [("hour", "hour"), ("power_pct", "power_pct"), ("monitor_up", "monitor_up")]:
        s = StateTracker().update(full(0, **{k: None}))
        assert not s.trusted and f"missing:{name}" in s.reasons
    s = StateTracker().update(full(0, door_closed={"L": False, "R": None}))
    assert not s.trusted and "missing:door_closed.R" in s.reasons


def test_freeze_watchdog():
    tr = StateTracker()
    for t in range(0, 20):
        s = tr.update(full(t))
    assert s.trusted
    s = tr.update(full(20))
    assert "frozen_clock" in s.reasons and not s.trusted
    s = tr.update(full(21, power=89.0))  # clock change resets
    assert s.trusted


def test_capture_stall():
    tr = StateTracker()
    tr.update(full(0))
    assert "capture_stall" not in tr.update(full(1.0), got_frame=False).reasons
    s = tr.update(full(2.0), got_frame=False)
    assert "capture_stall" in s.reasons and not s.trusted
    assert tr.update(full(2.1)).trusted


def test_snapshot_isolation():
    tr = StateTracker()
    s = tr.update(full(0))
    s.hall["L"] = 9.9
    s.door_closed["R"] = True
    s2 = tr.update(Readings(0.1))
    assert s2.hall["L"] == 0.0 and s2.door_closed["R"] is False
