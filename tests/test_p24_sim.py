import config
import sim_env
from es_tune import default_x, score, to_params


def test_policy_beats_no_doors_in_the_sim():
    seeds = range(30)  # night 4: Bonnie/Chica are the threat the doors answer (Night 2 is Foxy, no door logic yet)
    with_doors = sum(sim_env.play(seed=s, night=4, max_s=200)["result"] == "timeout" for s in seeds)
    without = sum(sim_env.play(seed=s, night=4, max_s=200, disable_doors=True)["result"] == "timeout" for s in seeds)
    assert with_doors > without + 5


def test_quiet_night_reaches_6am(monkeypatch):
    # with the FITTED rates (config_fit.json) the default look periods drain a quiet night to 0 before 6 AM: phase 31 tunes that
    monkeypatch.setattr(sim_env, "power_rates", lambda: dict(config.POWER_RATE_PCT_PER_S))
    r = sim_env.play(seed=0, quiet=True)
    assert r["result"] == "6am", r  # with nobody coming, the policy's checking must not drain the battery before 6 AM


def test_default_vector_roundtrips():
    p = to_params(default_x())
    assert abs(p.check_L - 8.0) < 1e-6 and abs(p.reopen_probe - 12.0) < 1e-6 and len(p.hour_scale) == 6


def test_score_is_deterministic():
    x = default_x()
    assert score((x, [1, 2, 3])) == score((x, [1, 2, 3]))
