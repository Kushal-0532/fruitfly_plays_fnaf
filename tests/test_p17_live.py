import json

import main
from readers import LiveReader
from state import Readings


class Stub:
    def __init__(self, **kw):
        self.kw, self.calls = kw, []

    def read(self, frame, t, **extra):
        self.calls.append(extra)
        return Readings(t, **self.kw)


def test_live_reader_merges_and_gates_hallway():
    h = Stub(hall={"L": 0.9, "R": None})
    r = LiveReader(Stub(hour=2, power_pct=80.0, usage=2), Stub(monitor_up=False, light_on={"L": True, "R": None}), h)
    out = r.read(None, 1.0)
    assert out.hour == 2 and out.hall == {"L": 0.9} and h.calls == [{"light_on": {"L": True}}]
    h2 = Stub(hall={"L": 0.9})
    up = LiveReader(Stub(hour=2), Stub(monitor_up=True), h2).read(None, 1.0)
    assert up.hall is None and h2.calls == []  # no hallway read while the monitor covers the office


def test_apply_fit_overrides(tmp_path, monkeypatch):
    import config
    p = tmp_path / "f.json"
    p.write_text(json.dumps({"overrides": {"MIN_HOLD_S": 9.0, "CHECK_PERIOD": {"L": 6, "R": 7}}}))
    monkeypatch.setattr(config, "MIN_HOLD_S", 5.0)
    monkeypatch.setattr(config, "CHECK_PERIOD", {"L": 8, "R": 8})
    main.apply_fit(p)
    assert config.MIN_HOLD_S == 9.0 and config.CHECK_PERIOD == {"L": 6, "R": 7}


def test_assume_open_makes_one_visible_door_enough():
    from state import StateTracker
    tr = StateTracker(assume_open=True)
    s = tr.update(Readings(0.0, hour=0, power_pct=99.0, monitor_up=False, door_closed={"L": False}, light_on={"L": False}))
    assert s.trusted and s.door_closed == {"L": False, "R": False}
    s = tr.update(Readings(10.0, hour=0, power_pct=98.0, monitor_up=False, door_closed={"L": True}))
    assert s.door_closed == {"L": True, "R": False}  # a later reading still overrides the assumption
