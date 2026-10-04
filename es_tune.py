"""Phase 24: (mu, lambda) evolution strategy over the policy parameters, reward = survive the sim night (partial credit for time).
Colab-friendly (stdlib + multiprocessing). Prints default vs tuned on held-out seeds and writes a PROPOSED overrides block:
nothing is applied until you run fit_config.py yourself.   python es_tune.py [--gens 25] [--out logs/es_proposal.json]"""
import argparse
import json
import math
import random
from collections import Counter
from multiprocessing import Pool
from pathlib import Path

import config
from sim_env import Params, play

NIGHT = 6 * config.HOUR_S
# name, default (from Params), low, high  -- searched in log space
SPACE = [("check_L", 3.0, 30.0), ("check_R", 3.0, 30.0)] + [(f"hs{h}", 0.3, 5.0) for h in range(6)] + \
        [("min_hold", 2.0, 15.0), ("reopen_probe", 6.0, 40.0), ("settle", 0.4, 2.0)] + \
        [(f"stall_h{h}", 3.0, 40.0) for h in range(6)] + [("stall_hold", 1.0, 3.0)]


def to_params(x):
    v = {n: math.exp(max(math.log(lo), min(math.log(hi), xi))) for (n, lo, hi), xi in zip(SPACE, x)}
    return Params(check_L=v["check_L"], check_R=v["check_R"], hour_scale=tuple(v[f"hs{h}"] for h in range(6)),
                  min_hold=v["min_hold"], reopen_probe=v["reopen_probe"], settle=v["settle"],
                  stall_period=tuple(v[f"stall_h{h}"] for h in range(6)), stall_hold=v["stall_hold"])


def default_x():
    p = Params()
    vals = [p.check_L, p.check_R, *p.hour_scale, p.min_hold, p.reopen_probe, p.settle, *p.stall_period, p.stall_hold]
    return [math.log(v) for v in vals]


def score(args):
    x, seeds, *rest = args
    p, tot, wins = to_params(x), 0.0, 0
    for s in seeds:
        r = play(p, seed=s, night=rest[0] if rest else 2)
        wins += r["result"] == "6am"
        tot += 1.0 if r["result"] == "6am" else 0.5 * r["t"] / NIGHT
    return tot / len(seeds), wins / len(seeds)


def causes(args):
    x, seeds, night = args
    p = to_params(x)
    return dict(Counter(play(p, seed=s, night=night)["cause"] for s in seeds))


def bound_warnings(x):
    out = []
    for (n, lo, hi), xi in zip(SPACE, x):
        f = (xi - math.log(lo)) / (math.log(hi) - math.log(lo))
        if f < 0.05 or f > 0.95:
            out.append(f"{n} ended at {'lower' if f < 0.5 else 'upper'} bound ({math.exp(min(max(xi, math.log(lo)), math.log(hi))):.2f}): widen the search range")
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--night", type=int, default=2)
    ap.add_argument("--gens", type=int, default=25)
    ap.add_argument("--lam", type=int, default=16)
    ap.add_argument("--mu", type=int, default=4)
    ap.add_argument("--seeds", type=int, default=24)
    ap.add_argument("--out", help="default logs/es_proposal_n<night>.json")
    a = ap.parse_args(argv)
    out = a.out or f"logs/es_proposal_n{a.night}.json"
    rng = random.Random(0)
    mean, sigma = default_x(), 0.35
    with Pool() as pool:
        for g in range(a.gens):
            seeds = [rng.randrange(10**6) for _ in range(a.seeds)]  # fresh seeds each generation: no memorising a night
            pop = [mean] + [[m + sigma * rng.gauss(0, 1) for m in mean] for _ in range(a.lam - 1)]
            res = pool.map(score, [(x, seeds, a.night) for x in pop])
            order = sorted(range(len(pop)), key=lambda i: -res[i][0])
            elite = [pop[i] for i in order[:a.mu]]
            mean = [sum(e[k] for e in elite) / a.mu for k in range(len(mean))]
            sigma = max(0.08, sigma * 0.93)
            print(f"gen {g:2d}  best {res[order[0]][0]:.3f} (wins {res[order[0]][1]:.2f})  mean-of-pop {sum(r[0] for r in res)/len(res):.3f}  sigma {sigma:.2f}", flush=True)
        held = [rng.randrange(10**6) for _ in range(300)]
        d, t = pool.map(score, [(default_x(), held, a.night), (mean, held, a.night)])
        cd, ct = pool.map(causes, [(default_x(), held, a.night), (mean, held, a.night)])
    print(f"held-out (300 nights, night {a.night}): default {d[1]:.2f} reach 6 AM, tuned {t[1]:.2f}")
    print("causes default:", cd, "\ncauses tuned:  ", ct)
    warn = bound_warnings(mean)
    for w in warn:
        print("WARNING:", w)
    p = to_params(mean)
    prop = {"CHECK_PERIOD": {"L": round(p.check_L, 2), "R": round(p.check_R, 2)}, "HOUR_SCALE": {str(h): round(v, 2) for h, v in enumerate(p.hour_scale)},
            "MIN_HOLD_S": round(p.min_hold, 2), "REOPEN_PROBE_S": round(p.reopen_probe, 2), "SETTLE_S": round(p.settle, 2),
            "STALL_PERIOD_S": {str(h): round(v, 2) for h, v in enumerate(p.stall_period)}, "STALL_HOLD_S": round(p.stall_hold, 2)}
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    Path(out).write_text(json.dumps({"night": a.night, "overrides": prop, "heldout_default": d[1], "heldout_tuned": t[1],
                                     "causes_default": cd, "causes_tuned": ct, "bound_warnings": warn}, indent=1))
    print(json.dumps(prop))


if __name__ == "__main__":
    main()
