"""Phase 18: PipeWire game-audio onset detector. Plain DSP: per-side onsets raise hazard on that side.
Capture target / pw-record flags are unverified until the live probe (specs/LIVE-CHECKS.md L18)."""
import json
import queue
import subprocess
import threading
import wave
from dataclasses import dataclass

import numpy as np

import config


@dataclass
class Event:
    t: float
    side: str        # "L" | "R" | "B" (both/unknown)
    band: int
    strength: float  # energy / floor of the strongest band


def find_monitor_target(run=subprocess.run):
    """-> pw-record --target value for the default sink's monitor, or None (feature off)."""
    try:
        out = run(["pw-dump"], capture_output=True, text=True).stdout
        for obj in json.loads(out):
            for item in obj.get("metadata", []) or []:
                if item.get("key") == "default.audio.sink" and isinstance(item.get("value"), dict):
                    return item["value"].get("name")
    except Exception:
        pass
    try:  # fallback: default sink id from `wpctl status` ("*   47. Built-in Audio ...")
        for line in run(["wpctl", "status"], capture_output=True, text=True).stdout.splitlines():
            parts = line.replace("│", " ").split()
            if parts[:1] == ["*"] and len(parts) > 1 and parts[1].rstrip(".").isdigit():
                return parts[1].rstrip(".")
    except Exception:
        pass
    return None


def record_chunks(target, run=subprocess.Popen, rate=config.AUDIO_RATE, channels=2, chunk_s=0.1):
    """Yield int16 arrays (n, channels) from pw-record until its stdout ends."""
    cmd = ["pw-record", "--target", str(target), "-P", "stream.capture.sink=true", "--rate", str(rate),
           "--channels", str(channels), "--format", "s16", "-"]
    proc = run(cmd, stdout=subprocess.PIPE)
    size = int(rate * chunk_s) * channels * 2
    try:
        while True:
            buf = proc.stdout.read(size)
            if len(buf) < size:
                return
            yield np.frombuffer(buf, np.int16).reshape(-1, channels)
    finally:
        getattr(proc, "terminate", lambda: None)()


class OnsetDetector:
    def __init__(self, rate=config.AUDIO_RATE, bands=config.BANDS, cfg=config):
        self.rate, self.bands, self.c = rate, bands, cfg
        self.floor = None                          # (channels, bands)
        self.run_s = None                          # seconds above ratio, per channel
        self.fired = None
        self.last_event = -1e9

    def _energy(self, chunk):
        x = chunk.astype(np.float64) / 32768.0
        spec = np.abs(np.fft.rfft(x, axis=0)) ** 2 / len(x)
        freqs = np.fft.rfftfreq(len(x), 1 / self.rate)
        return np.array([[spec[(freqs >= lo) & (freqs < hi), ch].sum() for lo, hi in self.bands]
                         for ch in range(x.shape[1])]) + 1e-12

    def feed(self, chunk, t):
        c, dur = self.c, len(chunk) / self.rate
        e = self._energy(chunk)
        if self.floor is None:
            self.floor, self.run_s, self.fired = e.copy(), np.zeros(len(e)), np.zeros(len(e), bool)
            return []
        ratio = e / self.floor
        active = ratio > c.ONSET_RATIO                    # (ch, band)
        # floor: falls fast, rises slowly, never chases an onset (bounded rise while active)
        up = np.where(active, np.minimum(e, self.floor * c.ONSET_RATIO), e)
        self.floor += np.where(e < self.floor, 0.3, 0.02) * (up - self.floor)
        any_active = active.any(axis=1)
        self.run_s = np.where(any_active, self.run_s + dur, 0.0)
        self.fired &= any_active
        rising = (self.run_s >= c.MIN_ON_S - 1e-9) & ~self.fired
        if not rising.any():
            return []
        self.fired |= rising
        if t - self.last_event < c.REFRACTORY_S:
            return []
        strength = np.where(any_active[:, None], ratio, 0).max(axis=1)   # per channel
        band = int(np.argmax(np.where(any_active[:, None], ratio, 0).max(axis=0)))
        hi, lo = strength.max(), strength[strength > 0].min()
        if (strength > 0).sum() == 1:
            side = "LR"[int(np.argmax(strength))]
        elif hi / lo > c.PAN_RATIO:
            side = "LR"[int(np.argmax(strength))]
        else:
            side = "B"
        self.last_event = t
        return [Event(t, side, band, float(hi))]


def evaluate_events(events, truth, tol=1.0, duration_s=None):
    """events: [Event]; truth: [seconds]. One-to-one nearest matching within tol
    -> (precision, recall, false_per_min)."""
    free, hits = sorted(truth), 0
    for e in sorted(events, key=lambda e: e.t):
        near = [x for x in free if abs(x - e.t) <= tol]
        if near:
            free.remove(min(near, key=lambda x: abs(x - e.t)))
            hits += 1
    dur = duration_s or max([e.t for e in events] + list(truth) + [1.0])
    prec = hits / len(events) if events else 1.0
    rec = hits / len(truth) if truth else 1.0
    return prec, rec, (len(events) - hits) / (dur / 60)


def audio_enabled(path=config.POWER_FIT_PATH):
    try:
        return bool(json.load(open(path)).get("audio_enabled", config.AUDIO_ENABLED))
    except FileNotFoundError:
        return config.AUDIO_ENABLED


class AudioMonitor:
    """Background thread: pw-record -> OnsetDetector -> queue. poll() hands events to the main loop."""

    def __init__(self, target, enabled=None, chunks=record_chunks, detector=None):
        self.enabled = audio_enabled() if enabled is None else enabled
        self.q, self.logged = queue.Queue(), []
        self._chunks, self.det, self.target = chunks, detector or OnsetDetector(), target
        self.t = 0.0

    def start(self):
        threading.Thread(target=self._loop, daemon=True).start()

    def _loop(self):
        for chunk in self._chunks(self.target):
            for ev in self.det.feed(chunk, self.t):
                self.logged.append(ev)
                self.q.put(ev)
            self.t += len(chunk) / self.det.rate

    def apply_gate(self, precision, false_per_min):
        if precision < 0.7 or false_per_min > 1:
            self.enabled = False
            print("audio_logging_only: accuracy bar not met")
        return self.enabled

    def poll(self):
        out = []
        while not self.q.empty():
            out.append(self.q.get())
        return out if self.enabled else []


def forward(events, supervisor, t):
    """Raise hazard on the event's side (B: both sides)."""
    for ev in events:
        for side in ("LR" if ev.side == "B" else ev.side):
            supervisor.note_audio(side, t)


def probe(seconds=3.0):
    target = find_monitor_target()
    print("target:", target)
    if target is None:
        raise SystemExit("no monitor target found")
    chunks = []
    for ch in record_chunks(target):
        chunks.append(ch)
        if len(chunks) * len(ch) / config.AUDIO_RATE >= seconds:
            break
    data = np.concatenate(chunks)
    print("RMS per channel:", np.sqrt((data.astype(float) ** 2).mean(axis=0)))
    with wave.open("logs/audio_probe.wav", "wb") as w:
        w.setnchannels(2)
        w.setsampwidth(2)
        w.setframerate(config.AUDIO_RATE)
        w.writeframes(data.tobytes())
    print("wrote logs/audio_probe.wav")


if __name__ == "__main__":
    import sys
    if "--probe" not in sys.argv:
        sys.exit("usage: python audio.py --probe")
    probe()
