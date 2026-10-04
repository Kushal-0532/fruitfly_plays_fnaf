import json

from scripts.audit_decisions import audit, main
from sim_env import play


def row(a, r):
    return {"t": 1.0, "action": a, "reason": r, "guard": ""}


def test_audit_counts_violations():
    res = audit([row("DOOR_L", "foxy_run"), row("DOOR_R", "threat_close_R"), row("LIGHT_L", "hall_check")])
    assert res["door_actions"] == 2 and len(res["violations"]) == 1
    assert audit([row("DOOR_L", "reopen_probe")])["violations"] == []


def test_cli_exit_codes(tmp_path):
    for name, reason, code in (("bad", "foxy_run", 1), ("ok", "threat_close_L", 0)):
        p = tmp_path / f"{name}.jsonl"
        p.write_text(json.dumps({"header": {}}) + "\n" + json.dumps(row("DOOR_L", reason)) + "\n")
        assert main([str(p)]) == code


def test_sim_nights_have_no_violations():
    assert all(play(seed=s)["audit"]["violations"] == [] for s in range(20))
