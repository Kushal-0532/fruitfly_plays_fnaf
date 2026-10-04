"""Phase 42: Supervisor vs fly-readout policies in the sim, paired held-out seeds, bootstrap 95% CIs.
Policies: supervisor, imitation (readout init), trained (ES readout), shuffled (trained W, fly features permuted across channels = control).
No policy wins a sim night (power-out dominates), so the metric is survival time (s) and death causes, not win rate.
  nice -n 19 python compare_policies.py --nights 2 3 --seeds 60      # serial, single process (laptop overheats)"""
import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np

import readout_policy as RP
import sim_env

OUT = Path(__file__).parent / "logs/compare"
SEED0 = 1000                        # es_readout.HELD_SEEDS start: never used for training
PERM = [2, 3, 0, 1]                 # hall_L<-cove, hall_R<-cam4b, cove<-hall_L, cam4b<-hall_R (features 0-3 are the fly's scores)
POLICIES = ["supervisor", "imitation", "trained", "shuffled"]


def boot(x, stat=np.mean, n=2000, seed=0):
    """Percentile bootstrap 95% CI of stat over resampled seeds (same scheme as ab_report.boot_ci, resampling rows)."""
    x, rng = np.asarray(x, float), np.random.default_rng(seed)
    v = [stat(x[rng.integers(0, len(x), len(x))]) for _ in range(n)]
    return float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))


def run_policy(name, night, seeds, max_s=None):
    kw = {"supervisor": dict(policy="supervisor"), "imitation": dict(policy="readout", weights=np.load(RP.WEIGHTS_PATH.with_name("readout_policy_imitation.npz"))["W"]),
          "trained": dict(policy="readout"), "shuffled": dict(policy="readout")}[name]
    orig = RP.ReadoutPolicy.features
    if name == "shuffled":
        def shuf(self, s, t):
            x = orig(self, s, t)
            x[:4] = x[:4][PERM]
            return x
        RP.ReadoutPolicy.features = shuf
    try:
        return [sim_env.play(seed=s, night=night, max_s=max_s, **kw) for s in seeds]
    finally:
        RP.ReadoutPolicy.features = orig


def summarize(res, base):
    t = np.array([r["t"] for r in res])
    d = t - np.array([r["t"] for r in base])
    return {"n": len(res), "wins": sum(r["result"] == "6am" for r in res), "mean_t": float(t.mean()), "mean_ci": boot(t), "median_t": float(np.median(t)),
            "median_ci": boot(t, np.median), "paired_diff_vs_supervisor": float(d.mean()), "paired_diff_ci": boot(d),
            "causes": dict(Counter(r["cause"] for r in res).most_common())}


def plot(rep, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    nights = sorted(rep)
    fig, axs = plt.subplots(1, len(nights), figsize=(5 * len(nights), 4), sharey=True, squeeze=False)
    for ax, n in zip(axs[0], nights):
        for i, p in enumerate(POLICIES):
            r = rep[n][p]
            ax.bar(i, r["mean_t"], color="#888" if p == "shuffled" else "#3b6ea5" if p != "supervisor" else "#444",
                   yerr=[[r["mean_t"] - r["mean_ci"][0]], [r["mean_ci"][1] - r["mean_t"]]], capsize=4)
        ax.set_xticks(range(len(POLICIES)), POLICIES)
        ax.set_title(f"Night {n} (n={rep[n]['supervisor']['n']} paired seeds)")
        ax.axhline(535, ls=":", c="k", lw=.8)
    axs[0][0].set_ylabel("mean survival (s), bootstrap 95% CI")
    fig.tight_layout()
    fig.savefig(path, dpi=110)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--nights", type=int, nargs="+", default=[2, 3])
    ap.add_argument("--seeds", type=int, default=60)
    ap.add_argument("--max-s", type=float, default=None)
    ap.add_argument("--out", default=str(OUT))
    a = ap.parse_args(argv)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    seeds, rep, raw = range(SEED0, SEED0 + a.seeds), {}, {}
    for n in a.nights:
        runs = {p: run_policy(p, n, seeds, a.max_s) for p in POLICIES}
        raw[n] = {p: [r["t"] for r in rs] for p, rs in runs.items()}
        rep[n] = {p: summarize(rs, runs["supervisor"]) for p, rs in runs.items()}
    (out / "compare.json").write_text(json.dumps({"seeds": a.seeds, "seed0": SEED0, "report": rep, "raw_t": raw}, indent=1))
    lines = ["| night | policy | wins | mean s [95% CI] | median s [95% CI] | paired diff vs supervisor [95% CI] | deaths |", "|---|---|---|---|---|---|---|"]
    for n in a.nights:
        for p in POLICIES:
            r = rep[n][p]
            lines.append(f"| {n} | {p} | {r['wins']}/{r['n']} | {r['mean_t']:.0f} [{r['mean_ci'][0]:.0f}, {r['mean_ci'][1]:.0f}] | {r['median_t']:.0f} [{r['median_ci'][0]:.0f}, {r['median_ci'][1]:.0f}] | "
                         f"{r['paired_diff_vs_supervisor']:+.0f} [{r['paired_diff_ci'][0]:+.0f}, {r['paired_diff_ci'][1]:+.0f}] | {r['causes']} |")
    (out / "table.md").write_text("\n".join(lines) + "\n")
    plot(rep, out / "survival.png")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
