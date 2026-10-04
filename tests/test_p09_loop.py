import json
import time

import config
from actions import Action
from sim import simulate
from tests.fakes import night_bonnie_left_visit, night_quiet

NIGHT_S = 6 * config.HOUR_S + 10


def test_quiet_night(tmp_path):
    summary, calls, rows = simulate(night_quiet(), NIGHT_S, out_dir=tmp_path)
    assert summary.status == "6am" and summary.hour_reached == 6
    assert not [c for c in calls if c[1].action in (Action.DOOR_L, Action.DOOR_R)]
    lines = (tmp_path / "run.jsonl").read_text().splitlines()
    assert "header" in json.loads(lines[0]) and len(rows) == len(lines) - 1
    assert summary.n_actions == len(calls)


def test_bonnie_visit_blind_reopen(tmp_path):
    summary, calls, rows = simulate(night_bonnie_left_visit(), 80, out_dir=tmp_path)
    doors = [(t, d) for t, d in calls if d.action == Action.DOOR_L]
    assert len(doors) >= 2 and doors[0][1].reason == "threat_close_L"
    assert doors[1][1].reason in ("reopen_probe", "reopen_empty") and doors[1][0] - doors[0][0] >= config.MIN_HOLD_S


def test_freeze_safe_mode(tmp_path):
    tl = [(100, {"hour": 1, "power_pct": 80.0})]
    summary, calls, rows = simulate(tl, 300, out_dir=tmp_path, power_mismatch=False)
    assert summary.status == "safe_mode" and summary.safe_reason == "frozen_clock"
    t_safe = rows[-1]["t"]
    assert 119 < t_safe < 122 and not [c for c in calls if c[0] > t_safe]
    assert list(tmp_path.glob("safe_*/reason.txt"))


def test_dry_run(tmp_path):
    summary, calls, rows = simulate(night_bonnie_left_visit(), 40, out_dir=tmp_path, dry_run=True)
    assert calls == [] and summary.n_actions == 0
    assert any(r["reason"] == "threat_close_L" for r in rows)


def test_capture_stall(tmp_path):
    summary, calls, rows = simulate([], 60, out_dir=tmp_path, stall_at=10)
    assert summary.status == "safe_mode" and summary.safe_reason == "capture_stall"


def test_sim_speed(tmp_path):
    t = time.perf_counter()
    simulate(night_quiet(), NIGHT_S, out_dir=tmp_path)
    assert time.perf_counter() - t < 5


def test_log_schema(tmp_path):
    _, _, rows = simulate(night_bonnie_left_visit(), 30, out_dir=tmp_path)
    keys = {"t", "hour", "power_pct", "usage", "monitor_up", "door_closed", "light_on", "cam", "hall",
            "foxy_stage", "cove", "cam4b", "trusted", "reasons", "action", "arg", "reason", "guard", "loop_ms"}
    assert all(set(r) == keys for r in rows)
