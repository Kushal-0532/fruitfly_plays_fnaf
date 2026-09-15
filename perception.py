"""Phase 03: RGB frames -> flyvis fly visual system -> small feature dict.

Stateful: network state persists across frames (one call per game frame).
Each frame: grayscale -> squash to 391x391 -> BoxEye hex render (721 hexals)
-> hold input for N_SUB Euler steps of DT -> read cell activity.
Features = baseline-subtracted rectified mean activity per cell type, split
left/right by hex column coordinate v (BoxEye maps x = kernel_size * v).
"""
import time

import numpy as np
import torch
import torchvision.transforms.functional as ttf

import flyvis  # noqa: F401  (sets default device, loads .env)
from flyvis import NetworkView
from flyvis.datasets.rendering.eye import BoxEye

MODEL = "flow/0000/000"
DT = 1 / 50
N_SUB = 5  # 5 * 1/50 s = 100 ms of fly-time per game frame (10 fps)
CELL_TYPES = ["T4a", "T4b", "T4c", "T4d", "T5a", "T5b", "T5c", "T5d", "L1", "L2"]


class Perception:
    def __init__(self, model=MODEL):
        self.net = NetworkView(model).init_network()
        self.net.eval()
        c = self.net.connectome
        self.eye = BoxEye(extent=15, kernel_size=13)
        self.size = self.eye.min_frame_size.tolist()
        v = np.asarray(c.nodes.v[:])
        self.masks = {}
        for t in CELL_TYPES:
            idx = np.asarray(c.nodes.layer_index[t][:])
            self.masks[t + "_L"] = idx[v[idx] < 0]
            self.masks[t + "_R"] = idx[v[idx] > 0]
        for prm in self.net.parameters():
            prm.requires_grad_(False)
        self.state = self.net.steady_state(1.0, DT, 1)
        self.baseline = self.state.nodes.activity.clone()
        self.last_ms = 0.0

    def to_hex(self, frame):
        """HxWx3 uint8 RGB (or HxW gray) -> (1,1,1,hexals) float in [0,1]."""
        g = frame.mean(axis=2) if frame.ndim == 3 else frame
        g = torch.as_tensor(np.ascontiguousarray(g), dtype=torch.float32)[None, None] / 255.0
        g = ttf.resize(g, self.size, antialias=True)
        return self.eye(g)

    @torch.no_grad()
    def step(self, frame):
        """One game frame in -> feature dict out. Keeps network state."""
        t = time.time()
        hx = self.to_hex(frame).repeat(1, N_SUB, 1, 1)
        self.net.stimulus.zero(1, N_SUB)
        self.net.stimulus.add_input(hx)
        self.state = self.net(self.net.stimulus(), DT, state=self.state, as_states=True)[-1]
        act = (self.state.nodes.activity - self.baseline).clamp(min=0)[0].numpy()
        feats = {k: float(act[m].mean()) for k, m in self.masks.items()}
        self.last_ms = (time.time() - t) * 1000
        return feats


def moving_bar(direction, n=20, size=200):
    """Synthetic frames: bright vertical bar sweeping left->right (+1) or right->left (-1)."""
    for i in range(n):
        f = np.full((size, size), 60, np.uint8)
        x = int(i / n * size) if direction > 0 else int((1 - i / n) * size)
        f[:, max(0, x - 8):x + 8] = 220
        yield f


if __name__ == "__main__":
    p = Perception()
    print("hexals", p.eye.hexals, "cells", p.baseline.shape[1], "size", p.size)
    tot = {}
    for d, name in ((1, "rightward"), (-1, "leftward")):
        acc = {}
        for f in moving_bar(d):
            for k, v in p.step(f).items():
                acc[k] = acc.get(k, 0) + v
        tot[name] = acc
        print(f"{name}: {p.last_ms:.0f} ms/step;",
              " ".join(f"{t}={acc[t+'_L']+acc[t+'_R']:.3f}" for t in ("T4a", "T4b", "T4c", "T4d", "T5a", "T5b")))
        for _ in range(10):
            p.step(np.full((200, 200), 60, np.uint8))  # settle
    # direction sanity: some T4/T5 subtype should prefer each direction
    r, l = tot["rightward"], tot["leftward"]
    pref = {t: (r[t + "_L"] + r[t + "_R"]) - (l[t + "_L"] + l[t + "_R"]) for t in ("T4a", "T4b", "T4c", "T4d")}
    print("rightward-minus-leftward per T4 subtype:", {k: round(v, 3) for k, v in pref.items()})
    best_r = max(pref, key=pref.get); best_l = min(pref, key=pref.get)
    assert pref[best_r] > 0 > pref[best_l], "no direction-selective response"
    print(f"OK: {best_r} prefers rightward, {best_l} prefers leftward; {p.last_ms:.0f} ms/step")
