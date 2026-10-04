"""Phase 08: guards between policy and actuator. Fail loudly (SAFE_MODE) rather than play blind."""
import json
import sys
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path

import config
from actions import Action, Decision

SAFE = Action.SAFE_MODE
FREEZE_REASONS = ("frozen_clock", "capture_stall")


@dataclass
class _Pending:
    kind: Action
    arg: str | None
    field: tuple          # (state attr, side or None)
    before: object
    t_sent: float
    retries: int = 0


def _field_of(action):
    return {Action.DOOR_L: ("door_closed", "L"), Action.DOOR_R: ("door_closed", "R"),
            Action.LIGHT_L: ("light_on", "L"), Action.LIGHT_R: ("light_on", "R"),
            Action.MONITOR: ("monitor_up", None)}.get(action)


def _get(state, field):
    v = getattr(state, field[0])
    return v[field[1]] if field[1] else v


class Guards:
    def __init__(self, cfg=None, stop_path="STOP"):
        self.c, self.stop_path = cfg or config, Path(stop_path)
        self.reset()

    def reset(self):
        self.latched = None
        self.untrusted_since = None
        self.ever_trusted = False
        self.sent = deque()
        self.refused = 0
        self.pending = None
        self.up_since = None
        self.second_gesture = False
        self._state = None

    def note_refused(self):
        self.refused += 1

    def note_sent(self):
        self.refused = 0

    def _safe(self, reason):
        self.latched = self.latched or reason
        return Decision(SAFE, None, self.latched)

    def expect(self, d, t):
        f = _field_of(d.action)
        if f is None or d.reason == "monitor_second_gesture" or self._state is None:
            return
        if self.pending and self.pending.kind == d.action and d.reason.startswith("retry_"):
            self.pending.t_sent = t
        else:
            self.pending = _Pending(d.action, d.arg, f, _get(self._state, f), t)

    def confirm(self, state, t):
        p = self.pending
        if p and _get(state, p.field) is not None and _get(state, p.field) != p.before:
            self.pending = None

    def vet(self, state, d, t, mismatch=False):
        c = self.c
        self._state = state
        if self.latched:
            return Decision(SAFE, None, self.latched)
        if self.stop_path.exists():
            return self._safe("stop_file")
        if any(r in state.reasons for r in FREEZE_REASONS):
            return self._safe(next(r for r in FREEZE_REASONS if r in state.reasons))
        if state.trusted:
            self.untrusted_since, self.ever_trusted = None, True
        else:
            self.untrusted_since = t if self.untrusted_since is None else self.untrusted_since
            # the night-start title card shows no HUD for a few seconds: allow a longer first wait
            limit = c.UNTRUSTED_SAFE_S if self.ever_trusted else getattr(c, "STARTUP_GRACE_S", c.UNTRUSTED_SAFE_S)
            if t - self.untrusted_since > limit:
                return self._safe("untrusted_too_long")
        if mismatch:
            return self._safe("power_model_mismatch")
        if self.refused >= 3:
            return self._safe("focus_refused")

        self.confirm(state, t)
        if self.pending and state.monitor_up and self.pending.kind != Action.MONITOR and self.pending.kind != Action.CAM:
            # the camera view covers the office buttons (monitor popped up unexpectedly): a door/light click can't be verified or
            # retried now. Drop it and let the supervisor lower the monitor first.
            self.pending = None
        p = self.pending
        if p and p.kind in (Action.LIGHT_L, Action.LIGHT_R) and t - p.t_sent > c.VERIFY_S:
            # a light click is a toggle: retrying could undo a click that did land but was read late. The supervisor sees the
            # real state next and corrects it (light_off / hall_check), so just stop waiting.
            self.pending = p = None
        if p and t - p.t_sent > c.VERIFY_S:
            if p.retries >= c.MAX_RETRIES:
                return self._safe("click_not_taken")
            p.retries += 1
            d = Decision(p.kind, p.arg, f"retry_{p.retries}")
            return self._record(d, t)

        if state.monitor_up:
            self.up_since = t if self.up_since is None else self.up_since
            up = t - self.up_since
            if up >= c.MONITOR_STUCK_S + c.VERIFY_S and self.second_gesture:
                return self._safe("monitor_stuck")
            if up >= c.MONITOR_STUCK_S and not self.second_gesture:
                self.second_gesture = True
                return self._record(Decision(Action.MONITOR, None, "monitor_second_gesture"), t)
        else:
            self.up_since, self.second_gesture = None, False

        if d.action == SAFE:
            return self._safe(d.reason or "policy_safe_mode")
        return self._record(d, t)

    def _record(self, d, t):
        if d.action == Action.NONE:
            return d
        while self.sent and t - self.sent[0] >= 60:
            self.sent.popleft()
        if len(self.sent) >= self.c.MAX_ACTIONS_PER_MIN:
            return Decision(Action.NONE, None, "guard_rate")
        self.sent.append(t)
        return d


class FrameRing:
    def __init__(self, n_seconds=30, fps=10):
        self.buf = deque(maxlen=int(n_seconds * fps))

    def push(self, t, frame):
        self.buf.append((t, frame))

    def dump(self, outdir):
        from PIL import Image
        out = Path(outdir)
        (out / "frames").mkdir(parents=True, exist_ok=True)
        meta = []
        for i, (t, f) in enumerate(self.buf):
            name = f"frames/{i:04d}.png"
            Image.fromarray(f).save(out / name)
            meta.append({"file": name, "t": t})
        (out / "ring.json").write_text(json.dumps(meta))
        return out


def safe_mode(reason, ring, log_tail, outdir=None):
    base = Path(outdir or config.LOG_DIR)
    path = base / ("safe_" + time.strftime("%Y%m%d-%H%M%S"))
    n = 0
    while path.exists():
        n += 1
        path = base / f"safe_{time.strftime('%Y%m%d-%H%M%S')}_{n}"
    path.mkdir(parents=True)
    (path / "reason.txt").write_text(reason + "\n")
    ring.dump(path)
    (path / "log_tail.jsonl").write_text("".join(json.dumps(r) + "\n" for r in log_tail))
    print(f"SAFE MODE: {reason} (evidence in {path})\a", file=sys.stderr, flush=True)
    return path
