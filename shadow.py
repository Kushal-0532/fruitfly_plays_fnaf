"""Phase 19: fly-brain hallway scorer, shadow only (never consulted by the Supervisor, decision D1)."""
import queue
import threading

import numpy as np

import config

N_BURST = 6
MOTION_TYPES = ("T4a", "T4b", "T5a", "T5b")


def hallway_diff(roi_gray, ref_gray):
    """|roi - ref| stretched to 0..255. Identical images -> all zeros."""
    d = np.abs(roi_gray.astype(np.float32) - ref_gray.astype(np.float32))
    m = d.max()
    return (d * (255.0 / m)).astype(np.uint8) if m > 0 else np.zeros(d.shape, np.uint8)


class FlyScorer:
    """Wraps perception.Perception; every burst starts from the network's resting state."""

    def __init__(self, perception=None):
        if perception is None:
            from perception import Perception
            perception = Perception()
        self.p = perception

    def _reset(self):
        self.p.state = self.p.net.steady_state(1.0, 1 / 50, 1)

    def score_burst(self, diffs):
        """-> {"L": float, "R": float}: summed motion-feature activity over the burst, per hex half."""
        self._reset()
        out = {"L": 0.0, "R": 0.0}
        for d in diffs:
            f = self.p.step(d)
            for t in MOTION_TYPES:
                for s in "LR":
                    out[s] += f[f"{t}_{s}"]
        return out


class Shadow:
    """Background worker: push(side, diff_frame) per lit hall frame; after N_BURST frames a burst is scored off-thread.
    latest() -> {"fly_L": ..., "fly_R": ...} (last finished burst per side, None before any)."""

    def __init__(self, scorer):
        self.scorer, self.buf, self.res = scorer, {"L": [], "R": []}, {"fly_L": None, "fly_R": None}
        self.q = queue.Queue(maxsize=4)
        threading.Thread(target=self._work, daemon=True).start()

    def push(self, side, diff):
        self.buf[side].append(diff)
        if len(self.buf[side]) >= N_BURST:
            burst, self.buf[side] = self.buf[side], []
            try:
                self.q.put_nowait((side, burst))
            except queue.Full:
                pass  # drop it: the shadow must never slow the loop

    def reset(self, side):
        self.buf[side] = []  # light went off: a burst must be consecutive lit frames

    def _work(self):
        while True:
            side, burst = self.q.get()
            r = self.scorer.score_burst(burst)
            self.res["fly_" + side] = r["L"] + r["R"]

    def latest(self):
        return dict(self.res)


class ShadowHook:
    """Plugs into main.run(shadow=...): feeds lit-hallway ROI diffs to a Shadow and returns the latest fly scores for the log row."""

    def __init__(self, geom, buttons, refs_path="data/templates/hall_ref.npz", scorer=None):
        from readers.hallway import LIT_MIN, _gray
        self._gray, self.lit_min = _gray, LIT_MIN
        self.geom, self.buttons = geom, buttons
        z = np.load(refs_path)
        self.refs = {s: z["ref" + s] for s in "LR"}
        self.shadow = Shadow(scorer or FlyScorer())

    def observe(self, frame, state, t):
        for s in "LR":
            if state.monitor_up is False and state.light_on.get(s) is True:
                g = self._gray(frame, self.geom, s, self.buttons)
                if g.shape == self.refs[s].shape and g.mean() >= self.lit_min:
                    self.shadow.push(s, hallway_diff(g, self.refs[s]))
                    continue
            self.shadow.reset(s)
        return self.shadow.latest()
