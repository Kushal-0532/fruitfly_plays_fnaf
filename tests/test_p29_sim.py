import statistics
from collections import Counter

import config
import sim_env
from actions import Action
from sim_env import Night


class Always:
    """rng stub: randint(1, 20) always rolls `v`, choice takes the first, uniform the low end."""
    def __init__(self, v): self.v = v
    def randint(self, a, b): return self.v
    def choice(self, seq): return seq[0]
    def uniform(self, a, b): return a
    def random(self): return 0.5


def run(n, secs):
    for _ in range(int(secs / sim_env.DT)):
        if n.dead:
            break
        n.step()


def night(ai, nt=2, v=1, **ai_over):
    n = Night(0, night=nt)
    n.ai = {k: 0 for k in n.ai} | ai
    n.rng = Always(v)
    return n


def test_ai0_never_moves_ai20_every_tick():
    n = night({}, nt=3)
    run(n, 170)  # before the 2 AM step
    assert n.pos == {"bonnie": "1A", "chica": "1A", "freddy": "1A"} and n.foxy == 1
    n = night({"bonnie": 20}, v=20)
    run(n, config.TICK_S["bonnie"] * 2 + 0.1)
    assert n.pos["bonnie"] == "5"  # two ticks: 1A -> 1B -> first neighbour


def test_ai_steps_night2():
    n = Night(0, night=2)
    start = dict(n.ai)
    for hour, step in ((2, {"bonnie": 1}), (3, {"bonnie": 1, "chica": 1, "foxy": 1}), (4, {"bonnie": 1, "chica": 1, "foxy": 1})):
        n.t = hour * config.HOUR_S - 0.05
        n.step(0.1)
        for k, d in step.items():
            start[k] += d
        assert n.ai == start


def test_bonnie_jam_then_monitor_kills_but_not_if_never_raised():
    n = night({"bonnie": 20}, v=20)
    n.pos["bonnie"] = "door"
    run(n, config.TICK_S["bonnie"] + 0.2)
    assert n.jammed["bonnie"] and not n.dead
    n.apply(Action.DOOR_L, 0.2)
    assert not n.door["L"]  # jammed buttons do nothing
    n2 = night({"bonnie": 20}, v=20)
    n2.pos["bonnie"] = "door"
    run(n2, config.TICK_S["bonnie"] + 0.2)
    n.apply(Action.MONITOR, 0.2)
    assert n.dead == "bonnie_jam"
    n2.ai["bonnie"] = 0
    run(n2, 3 * 89)
    assert not n2.dead


def test_bonnie_at_door_closed_goes_back():
    n = night({"bonnie": 20}, v=20)
    n.pos["bonnie"], n.door["L"] = "door", True
    run(n, config.TICK_S["bonnie"] + 0.2)
    assert n.pos["bonnie"] == "1B" and not n.jammed["bonnie"]


def test_foxy_stalled_by_monitor_raised_every_6s():
    n = Night(1, night=2)
    n.ai["foxy"] = 3
    for k in range(int(6 * 89 / 6)):
        n.apply(Action.MONITOR, 1.0)
        n.apply(Action.MONITOR, 0.1)
        run(n, 4.9)
        assert n.dead is None or n.dead != "foxy"
    assert n.foxy <= 2  # the monitor is up at ~1 in 6 s but the post-lowering lock + ticks every 5 s: he never completes the run


def test_foxy_knock_drain():
    n = night({}, v=1)
    n.foxy, n.attack_t, n.door["L"] = 4, 0.0, True
    n.rates = {u: 0.0 for u in range(1, 5)}
    n.step(); assert round(n.power, 1) == 99.0
    n.foxy, n.attack_t = 4, 0.0
    n.step(); assert round(n.power, 1) == 93.0 and n.foxy == 1


def test_freddy_idle_night2_and_needs_monitor_night3():
    n = night({"freddy": 20}, nt=2, v=20)
    run(n, 300)
    assert n.pos["freddy"] == "1A"
    n = night({"freddy": 20}, nt=3, v=20)
    n.pos["freddy"] = "4B"
    run(n, 60)
    assert not n.freddy_in and not n.dead
    n.monitor = True
    run(n, 4)
    assert n.freddy_in


def test_calibration_night2_foxy_dominates_around_3am():
    rs = [sim_env.play(seed=s, flips=False) for s in range(200)]  # phase 29 baseline: no monitor raises
    c = Counter(r["cause"] for r in rs)
    med = statistics.median(r["death_t"] for r in rs if r["cause"] == "foxy")
    assert c.most_common(1)[0][0] == "foxy" and 240 <= med <= 420, (c, med)
