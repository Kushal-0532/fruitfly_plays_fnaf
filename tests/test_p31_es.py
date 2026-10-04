import json

import config
import es_tune
from es_tune import default_x, to_params


def test_defaults_roundtrip():
    p = to_params(default_x())
    assert abs(p.reopen_probe - config.REOPEN_PROBE_S) < 1e-6 and abs(p.stall_hold - config.STALL_HOLD_S) < 1e-6
    assert [round(v, 6) for v in p.stall_period] == [config.STALL_PERIOD_S[h] for h in range(6)]


def test_tiny_run_writes_proposal(tmp_path):
    out = tmp_path / "p.json"
    es_tune.main(["--gens", "1", "--lam", "4", "--seeds", "2", "--out", str(out)])
    d = json.loads(out.read_text())
    assert {"STALL_PERIOD_S", "STALL_HOLD_S", "REOPEN_PROBE_S"} <= set(d["overrides"]) and d["night"] == 2 and "bound_warnings" in d
