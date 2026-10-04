import pytest

import actuator
import config
from actions import Action, Decision
from actuator import Actuator, Refused
from geometry import Geometry, Uncalibrated, Window, center, ref_to_window
from tests.fakes import FakeClock, FakeRunner

RECT = (100, 100, 40, 20)
BUTTONS = {"door_L": RECT, "door_R": (1000, 100, 40, 20), "light_L": (50, 200, 20, 20),
           "monitor_bar": (400, 690, 400, 20), "off": (5000, 5000, 10, 10)}


def make(w=1920, h=1080, **kw):
    r, c = FakeRunner(), FakeClock()
    a = Actuator(77, Geometry(Window(77, 0, 0, w, h)), BUTTONS, run=r, sleep=c.sleep, clock=c.now, **kw)
    return a, r, c


def argvs(r):
    return [" ".join(x) for x in r.calls]


def test_door_click_argv():
    a, r, c = make()
    assert a.act(Decision(Action.DOOR_L))
    px, py = ref_to_window(a.geom, *center(RECT))
    assert argvs(r) == ["xdotool windowactivate --sync 77", f"xdotool mousemove --window 77 {px} {py}",
                        "xdotool mousedown 1", "xdotool mouseup 1"]
    assert c.t == pytest.approx(config.PAN_SETTLE + config.CLICK_HOLD)


def test_coords_1080p():
    a, r, _ = make()
    a.click("door_L")
    assert r.calls[1][-2:] == [str(v) for v in ref_to_window(a.geom, 120, 110)] == ["180", "165"]


def test_refuse_outside():
    a, r, _ = make()
    with pytest.raises(Refused) as e:
        a.click("off")
    assert e.value.reason and r.calls == []


def test_none_window():
    with pytest.raises(RuntimeError):
        Actuator(None, Geometry(Window(0, 0, 0, 1280, 720)), BUTTONS)


def test_rate_limit():
    a, r, c = make()
    assert a.act(Decision(Action.DOOR_L))
    n = len(r.calls)
    t0 = c.t
    assert a.act(Decision(Action.DOOR_R)) and len(r.calls) > n  # sent, after waiting out the gap
    assert c.t - t0 >= config.MIN_ACTION_GAP


def test_light_hold_mode(monkeypatch):
    monkeypatch.setattr(config, "LIGHT_MODE", "hold")
    a, r, _ = make()
    with a.light_hold("L"):
        assert "xdotool mouseup 1" not in argvs(r) and "xdotool mousedown 1" in argvs(r)
    assert argvs(r)[-1] == "xdotool mouseup 1"


def test_light_toggle_clicks_once():
    a, r, _ = make()
    a.act(Decision(Action.LIGHT_L))
    assert argvs(r).count("xdotool mousedown 1") == 1


def test_safe_mode_only_parks():
    a, r, _ = make()
    assert a.act(Decision(Action.SAFE_MODE))
    assert len(r.calls) == 1 and r.calls[0][:4] == ["xdotool", "mousemove", "--window", "77"]
    assert r.calls[0][4:] == ["960", "540"]


def test_none_sends_nothing():
    a, r, _ = make()
    assert not a.act(Decision(Action.NONE)) and r.calls == []


def test_probe_requires_calibration():
    with pytest.raises(Uncalibrated):
        actuator.probe("calibration.example.json")
