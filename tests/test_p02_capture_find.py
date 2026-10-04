import capture
from tests.fakes import FakeRunner

GEOM = "WINDOW=7\nX=10\nY=20\nWIDTH=1280\nHEIGHT=720\nSCREEN=0\n"


def test_first_candidate_wins(monkeypatch):
    r = FakeRunner(["", "42\n", GEOM])  # first candidate: no match; second: match; then geometry
    monkeypatch.setattr(capture, "_run", r)
    assert capture.find_window(["nope", "yes", "later"]) == (42, 10, 20, 1280, 720)
    assert ["xdotool", "search", "--name", "later"] not in r.calls


def test_fallback_full_display(monkeypatch):
    r = FakeRunner(["", "", "", "", "1920 1080\n"])
    monkeypatch.setattr(capture, "_run", r)
    assert capture.find_window(["a", "b", "c"]) == (None, 0, 0, 1920, 1080)


def test_list_windows_parse(monkeypatch):
    r = FakeRunner(["11\n12\n13\n", "Game\n", "\n", "Other win\n"])
    monkeypatch.setattr(capture, "_run", r)
    assert capture.list_windows() == [(11, "Game"), (13, "Other win")]
