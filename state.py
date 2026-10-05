"""Phase 05: noisy per-frame Readings -> one GameState with staleness, trust and freeze detection.
Pure: time comes from Readings.t."""
import copy
from dataclasses import dataclass, field

import config

SIDES = ("L", "R")
DICT_FIELDS = ("door_closed", "light_on", "hall")
SCALARS = ("hour", "power_pct", "usage", "monitor_up", "cam", "foxy_stage", "cove", "cam4b", "cove_gone")


@dataclass
class Readings:
    t: float
    hour: int | None = None
    power_pct: float | None = None
    usage: int | None = None
    monitor_up: bool | None = None
    door_closed: dict | None = None
    light_on: dict | None = None
    cam: str | None = None
    hall: dict | None = None
    foxy_stage: int | None = None
    cam4b: float | None = None     # the fly's "somebody is on cam 4B" score (Chica / Freddy), held STALE_S["cam4b"] s
    cove: float | None = None      # the fly's "Foxy is out of the cove" score from a cam-1C look (held STALE_S["cove"] s)
    cove_gone: float | None = None  # the fly's "the cove is EMPTY: Foxy is running" score from a cam-1C look
    cam_feat: list | None = None   # the fly's activity over the camera view (monitor up only); never held across frames


@dataclass
class GameState:
    t: float
    hour: int | None = None
    power_pct: float | None = None
    usage: int | None = None
    monitor_up: bool | None = None
    door_closed: dict = field(default_factory=lambda: {"L": None, "R": None})
    light_on: dict = field(default_factory=lambda: {"L": None, "R": None})
    cam: str | None = None
    hall: dict = field(default_factory=lambda: {"L": None, "R": None})
    foxy_stage: int | None = None
    cove: float | None = None
    cam4b: float | None = None
    cove_gone: float | None = None
    cam_feat: list | None = None
    fresh: set | None = None       # scalars that came from a reading this step (not held); None = unknown, treat all as fresh
    trusted: bool = False
    reasons: list = field(default_factory=list)


class StateTracker:
    def __init__(self, cfg=config, assume_open=False):
        self.cfg = cfg
        self.assume_open = assume_open  # live: a night starts with doors open and lights off; only one side is on screen at a time
        self.val, self.seen = {}, {}  # key ("hour" or "hall.L") -> last good value / time
        self.last_hour = None
        self.change_t = None          # last time hour or power changed
        self.raw = {"hour": None, "power_pct": None}
        self.frame_t = None           # t of the last update that had a frame

    def _hold(self, key, new, t, reasons):
        stale = self.cfg.STALE_S[key.split(".")[0]]
        if new is not None:
            self.val[key], self.seen[key] = new, t
        elif key in self.seen and t - self.seen[key] > stale:
            self.val[key] = None
        return self.val.get(key)

    def update(self, r: Readings, got_frame=True):
        t, reasons = r.t, []
        if self.assume_open and not self.seen:
            for k in ("door_closed", "light_on"):
                for side in SIDES:
                    self.val[f"{k}.{side}"], self.seen[f"{k}.{side}"] = False, t
        hour = r.hour
        if hour is not None and self.last_hour is not None:
            if hour < self.last_hour:
                reasons.append("hour_regress")
                hour = None
            elif hour > self.last_hour + 1:
                reasons.append("hour_jump")
                hour = None
        if hour is not None:
            self.last_hour = hour

        for k in ("hour", "power_pct"):  # freeze watchdog watches raw changes
            v = hour if k == "hour" else r.power_pct
            if v is not None and v != self.raw[k]:
                self.raw[k], self.change_t = v, t
        if self.change_t is None:
            self.change_t = t

        s = GameState(t=t)
        vals = {"hour": hour}
        for k in SCALARS[1:]:
            vals[k] = getattr(r, k)
        for k in SCALARS:
            setattr(s, k, self._hold(k, vals[k], t, reasons))
        for k in DICT_FIELDS:
            d = getattr(r, k) or {}
            setattr(s, k, {side: self._hold(f"{k}.{side}", d.get(side), t, reasons) for side in SIDES})

        for name in self.cfg.REQUIRED_FIELDS:
            base, _, side = name.partition(".")
            v = getattr(s, base)[side] if side else getattr(s, base)
            if v is None:
                reasons.append(f"missing:{name}")

        if got_frame:
            self.frame_t = t
            if t - self.change_t >= self.cfg.FREEZE_S:
                reasons.append("frozen_clock")
        else:
            if self.frame_t is None:
                self.frame_t = t
            if t - self.frame_t >= self.cfg.CAPTURE_STALL_S:
                reasons.append("capture_stall")

        s.cam_feat = r.cam_feat if s.monitor_up else None
        s.fresh = {k for k in SCALARS if vals[k] is not None}
        s.reasons = reasons
        s.trusted = not any(x.startswith(("missing:", "frozen", "capture")) for x in reasons)
        return copy.deepcopy(s)  # snapshot: caller mutation can't reach tracker (tracker keeps no refs anyway)
