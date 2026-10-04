"""Phase 40: ReadoutPolicy. Door open/close = hysteresis(sigmoid(W . fly features)); same interface as policy.Supervisor.
Looks (hall light checks, stall flips, monitor lowering) stay the Supervisor's bookkeeping; every Supervisor door click is dropped
and replaced by the readout's. SPEC D12 extension: door reasons readout_close_L/R and readout_open_L/R (behind this policy only).

  python readout_policy.py --fit      # imitation init on Supervisor sim traces -> data/templates/readout_policy.npz
"""
import argparse
from pathlib import Path

import numpy as np

import config
from actions import Action, Decision
from policy import DOOR, Supervisor

WEIGHTS_PATH = Path(__file__).parent / "data/templates/readout_policy.npz"
READOUT_REASONS = frozenset(f"readout_{v}_{s}" for v in ("close", "open") for s in "LR")
CLOSE_T, OPEN_T = 0.6, 0.4        # hysteresis on the sigmoid: close above CLOSE_T, open below OPEN_T
NAMES = ["hall_L", "hall_R", "cove_conf", "cam4b_conf", "closed_L", "closed_R", "tclosed_L", "tclosed_R", "probe_L", "probe_R", "monitor", "power", "bias"]
CAM_SETTLE_S = 0.6   # s after a camera switch before its fly score counts (live: the first frames of a 1C look read 1.0)
TCLOSED_S = 12.0                  # time-closed feature scale (s); probe_X = door X has been shut >= TCLOSED_S (a step feature: AND-like rules stay linear)


class ReadoutPolicy(Supervisor):
    def __init__(self, power_model, cfg=None, night=None, weights=None):
        super().__init__(power_model, cfg, night)
        if weights is None:
            weights = np.load(WEIGHTS_PATH)["W"]
        self.W = np.asarray(weights, float)               # (2 sides L,R, len(NAMES))
        self.closed_t = {"L": None, "R": None}
        self.conf = {"cove": 0.0, "cam4b": 0.0}           # confirmed fly scores: a run of high frames on the camera, held until a run of low ones
        self.cam_since = (None, 0.0)
        self.run = {"cove": [0, 0], "cam4b": [0, 0]}      # [consecutive high frames, consecutive low frames]

    def _out(self, action, reason, t, arg=None):
        if action in (Action.DOOR_L, Action.DOOR_R):      # the Supervisor's own door rules are replaced by the readout
            return Decision(Action.NONE, None, "idle")
        return super()._out(action, reason, t, arg)

    def features(self, s, t):
        """The fly's scores (hall L/R, cove, 4B; None = not seen = 0), door state and time closed, power."""
        for side, key in (("L", "cove"), ("R", "cam4b")):
            if self.closed_t[side] is not None and not s.door_closed[side]:
                self.conf[key] = 0.0                      # the door that answers this camera was reopened: the sighting is consumed
            if s.door_closed[side]:
                self.closed_t[side] = t if self.closed_t[side] is None else self.closed_t[side]
            else:
                self.closed_t[side] = None
        if s.cam != self.cam_since[0]:
            self.cam_since = (s.cam, t)
        hall = {x: s.hall[x] if self.on_since[x] is not None and t - self.on_since[x] >= getattr(self.c, "LOOK_MIN_S", 0.0) else None for x in "LR"}
        cam_ok = t - self.cam_since[1] >= CAM_SETTLE_S
        for key, cam, thr, n_hi, n_lo in (("cove", "1C", self.c.COVE_THRESH, self.c.COVE_CONFIRM, self.c.COVE_CLEAR_CONFIRM),
                                          ("cam4b", "4B", self.c.CAM4B_THRESH, self.c.CAM4B_CONFIRM, self.c.COVE_CLEAR_CONFIRM)):
            v = getattr(s, key)
            if s.monitor_up and s.cam == cam and v is not None and cam_ok:  # the camera is on screen: one score per frame
                hi = v >= thr
                self.run[key] = [self.run[key][0] + 1 if hi else 0, 0 if hi else self.run[key][1] + 1]
                if self.run[key][0] >= n_hi:
                    self.conf[key] = 1.0
                elif self.run[key][1] >= n_lo:
                    self.conf[key] = 0.0
            if v is None:                                # sighting went stale (STALE_S): live run 2026-10-04 held the door shut all night on one cove frame
                self.conf[key] = 0.0
        closed = [float(bool(s.door_closed[x])) for x in "LR"]
        tcl = [0.0 if self.closed_t[x] is None else min(t - self.closed_t[x], 30.0) / TCLOSED_S for x in "LR"]
        v = lambda x: 0.0 if x is None else float(x)
        return np.array([v(hall["L"]), v(hall["R"]), self.conf["cove"], self.conf["cam4b"], *closed, *tcl, *[float(x >= 1.0) for x in tcl], float(bool(s.monitor_up)),
                         (s.power_pct or 0.0) / 100.0, 1.0])

    def targets(self, x):
        """-> {side: (want_closed, margin)} with hysteresis on the current door state in x."""
        p = 1.0 / (1.0 + np.exp(-(self.W @ x)))
        out = {}
        for i, side in enumerate("LR"):
            closed = x[4 + i] > 0.5
            out[side] = (p[i] > OPEN_T if closed else p[i] >= CLOSE_T, abs(p[i] - 0.5))
        return out

    def decide(self, s, t):
        d = super().decide(s, t)
        x = self.features(s, t)
        if d.action != Action.NONE or d.reason in ("untrusted_wait", "settle") or s.monitor_up:
            return d
        want = self.targets(x)
        cand = [(m, side) for side, (w, m) in want.items() if w != bool(s.door_closed[side]) and not self.jammed[side]]
        if not cand:
            return d
        side = max(cand)[1]
        verb = "close" if want[side][0] else "open"
        if verb == "close":
            self.close_t[side] = t
        return self._emit(DOOR[side], f"readout_{verb}_{side}", t)

    def _emit(self, action, reason, t):
        self.last_action_t = t
        return Decision(action, None, reason)


def fit_imitation(trace, l2=1e-4, flip_weight=10.0):
    """Weighted logistic regression (L-BFGS) of the Supervisor's door state after its decision on the fly features; transitions weigh more,
    steps where no click is possible (settle, monitor up) are skipped."""
    from scipy.optimize import minimize
    pol = ReadoutPolicy(None, weights=np.zeros((2, len(NAMES))))
    X, prev = [], 0.0
    for s, t, d, truth in trace:
        if t < prev:                      # a new night starts: fresh feature memory
            pol = ReadoutPolicy(None, weights=np.zeros((2, len(NAMES))))
        X.append(pol.features(s, t))
        prev = t
    X = np.array(X)
    Y = np.array([[truth[k] ^ (d.action == DOOR[k]) for k in "LR"] for s, t, d, truth in trace], float)

    ok = np.array([d.reason not in ("settle", "untrusted_wait") and not s.monitor_up for s, t, d, truth in trace])  # no click is possible on the other rows

    def fit(i):
        sw = np.where((X[:, 4 + i] > 0.5) != (Y[:, i] > 0.5), flip_weight, 1.0)
        sw = sw * ok / sw[ok].mean()

        def loss(w):
            p = 1 / (1 + np.exp(-X @ w))
            ll = -(sw * (Y[:, i] * np.log(p + 1e-9) + (1 - Y[:, i]) * np.log(1 - p + 1e-9))).mean() + l2 * w @ w
            return ll, X.T @ (sw * (p - Y[:, i])) / len(X) + 2 * l2 * w
        return minimize(loss, np.zeros(X.shape[1]), jac=True, method="L-BFGS-B", bounds=[(0, 0) if n == "power" else (None, None) for n in NAMES]).x  # power weight left 0 for ES
    return np.array([fit(0), fit(1)]), X, Y


def agreement(W, X, Y):
    """Fraction of steps where the readout's hysteresis target equals the Supervisor's door state (both sides), plus on transition steps."""
    pol = ReadoutPolicy(None, weights=W)
    ok = np.array([[pol.targets(x)[k][0] == (y[i] > 0.5) for i, k in enumerate("LR")] for x, y in zip(X, Y)])
    trans = np.array([[(x[4 + i] > 0.5) != (y[i] > 0.5) for i in range(2)] for x, y in zip(X, Y)])
    return float(ok.all(axis=1).mean()), float(ok[trans].mean()) if trans.any() else None


def collect(seeds, nights=(2, 3)):
    import sim_env
    trace = []
    for n in nights:
        for sd in seeds:
            sim_env.play(seed=sd, night=n, trace=trace)
    return trace


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--fit", action="store_true")
    ap.add_argument("--seeds", type=int, default=60)
    a = ap.parse_args(argv)
    W, X, Y = fit_imitation(collect(range(a.seeds)))
    print("train agreement (all steps, transition steps):", agreement(W, X, Y), "n =", len(X))
    Wh, Xh, Yh = fit_imitation(collect(range(1000, 1020)))
    print("held-out agreement:", agreement(W, Xh, Yh))
    WEIGHTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    np.savez(WEIGHTS_PATH, W=W, names=np.array(NAMES))
    print("wrote", WEIGHTS_PATH)


if __name__ == "__main__":
    main()
