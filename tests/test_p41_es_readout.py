import random

import numpy as np

import es_readout as E
import sim_env


def test_noise_is_worse_but_valid():
    n = E.noisy(random.Random(1))
    assert n["hit"] <= sim_env.NOISE["hit"] and n["false_alarm"] >= sim_env.NOISE["false_alarm"]
    assert all(0 <= v <= 1 for v in n["cove_hit"].values())


def test_play_weights_override_and_reward():
    W = np.load(E.INIT_PATH)["W"]
    a = sim_env.play(seed=3, night=2, policy="readout", weights=W)
    b = sim_env.play(seed=3, night=2, policy="readout")
    assert a["t"] > 0 and 0 < E.reward(a) <= 1.0
    W0 = np.zeros_like(W) - 9.0                              # never closes a door: no stall livelock, the night still ends
    r = sim_env.play(seed=3, night=2, policy="readout", weights=W0)
    assert r["result"] != "timeout" and r["actions"] < 2000
    assert E.reward({"result": "6am", "t": 535}) == 1.0


def test_train_one_generation_and_table(tmp_path):
    out = tmp_path / "o.json"
    E.main(["--nights", "3", "--gens", "1", "--lam", "4", "--mu", "2", "--seeds", "2", "--heldout", "4", "--out", str(out)])
    import json
    d = json.loads(out.read_text())
    assert np.array(d["W"]).shape == (2, len(E.NAMES)) and len(d["heldout"]) == 3
    assert "hall_L" in E.table(np.zeros((2, len(E.NAMES))))
