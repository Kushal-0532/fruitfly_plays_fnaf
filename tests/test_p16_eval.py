import json

import pytest

import evaluate as ev


def row(hour=1, power=90.0, mon=False, doors=(False, False), action="NONE", reason="idle", guard="", trusted=True):
    return {"hour": hour, "power_pct": power, "monitor_up": mon, "trusted": trusted,
            "door_closed": {"L": doors[0], "R": doors[1]}, "action": action, "reason": reason, "guard": guard}


def write(path, rows, night=2):
    path.write_text("\n".join(json.dumps(x) for x in [{"header": {"night": night}}] + rows) + "\n")
    return path


def test_wilson():
    lo, hi = ev.wilson(8, 10)
    assert lo == pytest.approx(0.490, abs=.001) and hi == pytest.approx(0.943, abs=.001)
    assert ev.wilson(0, 10)[0] == 0 and ev.wilson(10, 10)[1] == 1
    with pytest.raises(ValueError):
        ev.wilson(0, 0)


def test_death_causes(tmp_path):
    s = lambda rows: ev.summarize(ev.load_run(write(tmp_path / "r.jsonl", rows)))
    a = s([row(5, 30.0), row(6, 25.0)])
    assert a.reached_6am and a.death_cause == "6am" and a.power_at_hour5 == 30.0
    b = s([row(3, 50.0), row(3, 0.0)])
    assert b.death_cause == "power_out" and not b.reached_6am and b.last_hour == 3
    c = s([row(3, 50.0), row(3, 49.0, guard="SAFE_MODE:frozen_clock")])
    assert c.death_cause == "safe_mode" and c.safe_reason == "frozen_clock"
    assert s([row(2, 50.0)]).death_cause == "jumpscare_or_unknown"


def test_fractions_and_actions(tmp_path):
    rows = [row(mon=i < 3, doors=(i < 5, i < 1), action="DOOR_L" if i == 0 else "NONE",
                reason="threat_close_L" if i == 0 else "idle") for i in range(10)]
    st = ev.summarize(ev.load_run(write(tmp_path / "r.jsonl", rows)))
    assert st.frac_monitor_up == pytest.approx(0.3)
    assert st.frac_door_closed == pytest.approx(6 / 20)
    assert st.action_counts_by_reason == {"threat_close_L": 1}


def stats(wins, n):
    return [ev.RunStats(2, i < wins, 6, "6am", 10, 0, 0, {}, None) for i in range(n)]


def test_compare():
    assert ev.compare(stats(6, 10), stats(8, 10))["overlap"]
    assert not ev.compare(stats(1, 10), stats(9, 10))["overlap"]


def test_cli_golden(tmp_path, capsys):
    write(tmp_path / "run_a.jsonl", [row(6)])
    write(tmp_path / "run_b.jsonl", [row(6)])
    write(tmp_path / "run_c.jsonl", [row(3, 0.0)])
    write(tmp_path / "run_d.jsonl", [row(6)], night=1)  # other night ignored
    ev.main([str(tmp_path), "--night", "2", "--json", str(tmp_path / "o.json")])
    assert "Night 2: 2/3 reached 6 AM (66.7%, 95% CI 20.8%-93.9%)" in capsys.readouterr().out
    assert len(json.loads((tmp_path / "o.json").read_text())["runs"]) == 3
