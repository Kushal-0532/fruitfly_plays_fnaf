"""Phase 41: (mu, lambda) evolution strategy over the ReadoutPolicy weights W (2 sides x len(NAMES)), reward = survival fraction of the sim night
(+1 bonus for reaching 6 AM; the sim is harsher than the live game so wins are rare). flyvis stays frozen: the features are the fly scores.
Fresh train seeds every generation (shared by the population), noise on the fly features perturbed every generation, held-out seeds for the report.

  python es_readout.py --nights 2,3 --gens 15 --lam 8 --seeds 12 [--save]     # --save writes data/templates/readout_policy.npz
  python es_readout.py --eval --nights 2,3 --heldout 300              # imitation init vs trained file vs Supervisor on held-out seeds
"""
import argparse
import json
import random
import statistics
from collections import Counter
from multiprocessing import Pool
from pathlib import Path

import numpy as np

import config
import sim_env
from readout_policy import NAMES, WEIGHTS_PATH

NIGHT = 6 * config.HOUR_S
INIT_PATH = WEIGHTS_PATH.with_name("readout_policy_imitation.npz")
HELD_SEEDS = range(1000, 1300)      # never used for training (train seeds are drawn >= 2000)
VAL_SEEDS = range(1300, 1400)       # trained weights must beat the imitation init here or the init is kept


class Serial:
    """Stand-in for multiprocessing.Pool (--procs 1): one process, no extra heat."""
    def __enter__(self):
        return self

    def __exit__(self, *a):
        pass

    map = staticmethod(lambda f, xs: [f(x) for x in xs])


def noisy(rng):
    """Measured sim noise (sim_env.NOISE) made a bit worse at random: hit rates x0.7-1.0, false alarms x1-3, flicker x1-1.5."""
    n = dict(sim_env.NOISE)
    for k in ("hit", "cam4b_hit"):
        n[k] = n[k] * rng.uniform(0.7, 1.0)
    n["cove_hit"] = {k: v * rng.uniform(0.7, 1.0) for k, v in n["cove_hit"].items()}
    for k in ("false_alarm", "cove_false_alarm", "cam4b_false_alarm"):
        n[k] = min(0.3, n[k] * rng.uniform(1.0, 3.0))
    n["flicker"] = n["flicker"] * rng.uniform(1.0, 1.5)
    return n


def reward(r):
    return 1.0 if r["result"] == "6am" else r["t"] / NIGHT


def score(args):
    """-> mean reward of weights W on (night, seed) pairs, with the policy ('readout' or 'supervisor') and sim noise given."""
    w, jobs, noise, policy = args
    return sum(reward(sim_env.play(seed=s, night=n, noise=noise, policy=policy, weights=w)) for n, s in jobs) / len(jobs)


def evaluate(args):
    """-> per-night stats dict for held-out seeds (nominal noise)."""
    w, night, seeds, policy = args
    rs = [sim_env.play(seed=s, night=night, policy=policy, weights=w) for s in seeds]
    t = [r["t"] for r in rs]
    return {"mean_t": round(statistics.mean(t), 1), "median_t": round(statistics.median(t), 1), "wins": sum(r["result"] == "6am" for r in rs), "n": len(rs),
            "timeouts": sum(r["result"] == "timeout" for r in rs), "max_actions": max(r["actions"] for r in rs),
            "causes": dict(Counter(r["cause"] for r in rs).most_common())}


def table(W):
    out = [f"{'feature':<11}{'closes door L':>14}{'closes door R':>14}", "-" * 39]
    out += [f"{n:<11}{W[0, i]:>+14.2f}{W[1, i]:>+14.2f}" for i, n in enumerate(NAMES)]
    return "\n".join(out)


def train(nights, gens, lam, mu, nseeds, sigma0, rng, pool):
    W0 = np.load(INIT_PATH)["W"]
    mean, sigma, curve = W0.ravel().copy(), sigma0, []
    shape = W0.shape
    for g in range(gens):
        jobs = [(n, rng.randrange(2000, 10**6)) for n in nights for _ in range(nseeds)]
        noise = noisy(rng)
        pop = [mean] + [mean + sigma * np.array([rng.gauss(0, 1) for _ in mean]) for _ in range(lam - 1)]
        res = pool.map(score, [(p.reshape(shape), jobs, noise, "readout") for p in pop])
        order = sorted(range(lam), key=lambda i: -res[i])
        mean = np.mean([pop[i] for i in order[:mu]], axis=0)
        sigma = max(0.05, sigma * 0.95)
        curve.append((res[order[0]], res[0], float(np.mean(res))))
        print(f"gen {g:2d}  best {res[order[0]]:.3f}  parent {res[0]:.3f}  mean-of-pop {np.mean(res):.3f}  sigma {sigma:.2f}", flush=True)
    return W0, mean.reshape(shape), curve


def summary(name, st):
    return f"{name:<10} N{st['night']}: mean {st['mean_t']:6.1f} s  median {st['median_t']:6.1f} s  wins {st['wins']}/{st['n']}  timeouts {st['timeouts']}  max actions {st['max_actions']}  {st['causes']}"


def compare(W_by_policy, nights, n_held, pool):
    rows = []
    for night in nights:
        for name, (w, policy) in W_by_policy.items():
            rows.append((name, night, (w, night, list(HELD_SEEDS)[:n_held], policy)))
    res = pool.map(evaluate, [r[2] for r in rows])
    out = [{"name": r[0], "night": r[1], **st} for r, st in zip(rows, res)]
    for o in out:
        print(summary(o["name"], o))
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--nights", default="2,3")
    ap.add_argument("--gens", type=int, default=40)
    ap.add_argument("--lam", type=int, default=32)
    ap.add_argument("--mu", type=int, default=8)
    ap.add_argument("--seeds", type=int, default=24, help="train seeds per night per generation")
    ap.add_argument("--sigma", type=float, default=0.5)
    ap.add_argument("--procs", type=int, default=1, help="1 = serial (default); more = multiprocessing Pool")
    ap.add_argument("--heldout", type=int, default=300)
    ap.add_argument("--eval", action="store_true", help="only compare imitation init / saved weights / Supervisor")
    ap.add_argument("--save", action="store_true", help="write the trained W to data/templates/readout_policy.npz")
    ap.add_argument("--out", help="default logs/es_readout_n<nights>.json")
    a = ap.parse_args(argv)
    nights = [int(x) for x in a.nights.split(",")]
    out = a.out or f"logs/es_readout_n{''.join(map(str, nights))}.json"
    rng = random.Random(0)
    with (Pool(a.procs) if a.procs > 1 else Serial()) as pool:
        if a.eval:
            W0, W = np.load(INIT_PATH)["W"], np.load(WEIGHTS_PATH)["W"]
            curve = []
        else:
            W0, W, curve = train(nights, a.gens, a.lam, a.mu, a.seeds, a.sigma, rng, pool)
            vs = [(n, s) for n in nights for s in VAL_SEEDS]
            v0, v1 = pool.map(score, [(W0, vs, None, "readout"), (W, vs, None, "readout")])
            print(f"validation reward: imitation init {v0:.3f}, trained {v1:.3f}")
            if v1 < v0:
                print("WARNING: trained weights are worse than the imitation init on validation seeds; keeping init")
                W = W0
        cmp = compare({"supervisor": (None, "supervisor"), "imitation": (W0, "readout"), "trained": (W, "readout")}, nights, a.heldout, pool)
    print("\nweights (trained):\n" + table(W))
    if not a.eval:
        Path(out).parent.mkdir(parents=True, exist_ok=True)
        Path(out).write_text(json.dumps({"nights": nights, "W": W.tolist(), "names": NAMES, "curve_best_parent_mean": curve, "heldout": cmp}, indent=1))
        if a.save:
            np.savez(WEIGHTS_PATH, W=W, names=np.array(NAMES))
            print("wrote", WEIGHTS_PATH)


if __name__ == "__main__":
    main()
