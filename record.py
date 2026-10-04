"""Phase 10: record a play session to data/corpus/<name>/{frames/NNNNNN.png, meta.json}."""
import argparse
import json
import time
from pathlib import Path

import numpy as np

import config


def record_session(capture, out_dir, seconds, clock=time.monotonic, sleep=time.sleep, meta_extra=None):
    """Save frames at config.FPS for `seconds`. Returns the meta dict (also written to meta.json)."""
    from PIL import Image
    out = Path(out_dir)
    (out / "frames").mkdir(parents=True, exist_ok=True)
    t0, times, dt = clock(), [], 1 / config.FPS
    while True:
        ts = clock()
        if ts - t0 >= seconds:
            break
        Image.fromarray(np.ascontiguousarray(capture.grab())).save(out / "frames" / f"{len(times):06d}.png")
        times.append(round(ts - t0, 4))
        sleep(max(0.0, dt - (clock() - ts)))
    meta = {"t0": t0, "fps": config.FPS, "times": times, **(meta_extra or {})}
    (out / "meta.json").write_text(json.dumps(meta))
    return meta


class LiveRecorder:
    """Saves every new frame of a live run (corpus format) from a writer thread, so the run itself becomes labelable data
    (e.g. Chica/Foxy at the doors). Row t in logs/run_*.jsonl matches meta['times']."""

    def __init__(self, out_dir):
        import queue
        import threading
        self.out, self.times, self.q = Path(out_dir), [], queue.Queue(maxsize=200)
        (self.out / "frames").mkdir(parents=True, exist_ok=True)
        self.th = threading.Thread(target=self._run, daemon=True)
        self.th.start()

    def submit(self, frame, t):
        if not self.q.full():
            self.q.put((len(self.times), frame, t))
            self.times.append(round(t, 4))

    def _run(self):
        from PIL import Image
        while True:
            item = self.q.get()
            if item is None:
                return
            i, frame, _ = item
            Image.fromarray(np.ascontiguousarray(frame)).save(self.out / "frames" / f"{i:06d}.png")

    def close(self, meta_extra=None):
        self.q.put(None)
        self.th.join()
        (self.out / "meta.json").write_text(json.dumps({"t0": 0.0, "fps": config.FPS, "times": self.times, **(meta_extra or {})}))


def load_session(session_dir):
    """-> (meta, iterator of (t, frame ndarray)) in recording order."""
    from PIL import Image
    d = Path(session_dir)
    meta = json.loads((d / "meta.json").read_text())

    def frames():
        for i, t in enumerate(meta["times"]):
            yield t, np.asarray(Image.open(d / "frames" / f"{i:06d}.png").convert("RGB"))
    return meta, frames()


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--seconds", type=float, default=600)
    ap.add_argument("--night", type=int)
    a = ap.parse_args(argv)
    from capture import SCALE, Capture, find_window
    cap = Capture()
    wid, x, y, w, h = find_window()
    try:
        meta = record_session(cap, Path(config.CORPUS_DIR) / a.name, a.seconds, meta_extra={
            "window": {"wid": wid, "x": x, "y": y, "w": w, "h": h}, "capture_scale": SCALE, "night": a.night})
    finally:
        cap.close()
    print(f"recorded {len(meta['times'])} frames -> {config.CORPUS_DIR}/{a.name}")


if __name__ == "__main__":
    main()
