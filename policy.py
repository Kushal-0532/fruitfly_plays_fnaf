"""Phase 04: heuristic policy. Feature dict + time -> one discrete action.

Actions are single key presses (UCN keys toggle), so the policy tracks what
it believes the door/monitor state is.

Rules, in priority order:
  1. settle: ignore everything for SETTLE s after any action (door and monitor
     animations are themselves huge motion spikes)
  2. monitor up -> lower it after CAM_HOLD s (frame shows the camera feed,
     not the office, so motion there says nothing about the doors)
  3. door closed too long -> reopen after DOOR_HOLD s (power drain)
  4. motion on a side rises MOTION_THRESH above its running baseline -> close
     that side's door
  5. every CAM_PERIOD s -> raise monitor (stalls Foxy)
"""
from dataclasses import dataclass, field

NONE, CAM_TOGGLE, LEFT_DOOR, RIGHT_DOOR = "NONE", "CAM_TOGGLE", "LEFT_DOOR", "RIGHT_DOOR"

# Calibration knobs. Tune from logs/run_*.csv in phase 06.
MOTION_TYPES = ("T4a", "T4b", "T5a", "T5b")  # horizontal motion detectors
MOTION_THRESH = 0.05  # live test: quiet ~0.03-0.05 per type, burst ~0.13
BASELINE_ALPHA = 0.05  # EMA rate for the running motion baseline
RIGHT_DOOR_ENABLED = False  # user-confirmed: Freddy and Foxy both come LEFT
SETTLE = 1.5
CAM_PERIOD = 12.0
CAM_HOLD = 2.0
DOOR_HOLD = 6.0


@dataclass
class State:
    left_closed: bool = False
    right_closed: bool = False
    monitor_up: bool = False
    t_action: float = -1e9
    t_door: dict = field(default_factory=lambda: {"L": -1e9, "R": -1e9})
    t_monitor: float = 0.0
    baseline: dict = field(default_factory=dict)


def motion(feats, side):
    return sum(feats[f"{t}_{side}"] for t in MOTION_TYPES)


def decide(feats, t, s):
    """-> action string. Mutates s."""
    m = {side: motion(feats, side) for side in "LR"}
    excess = {side: m[side] - s.baseline.get(side, m[side]) for side in "LR"}
    for side in "LR":
        # ponytail: baseline only updates while not triggered, so a lingering
        # animatronic doesn't get absorbed. Upgrade: per-side robust stats.
        if excess[side] < MOTION_THRESH:
            s.baseline[side] = (1 - BASELINE_ALPHA) * s.baseline.get(side, m[side]) + BASELINE_ALPHA * m[side]

    def act(a):
        s.t_action = t
        if a == CAM_TOGGLE:
            s.monitor_up = not s.monitor_up
            s.t_monitor = t
        elif a == LEFT_DOOR:
            s.left_closed = not s.left_closed
            s.t_door["L"] = t
        elif a == RIGHT_DOOR:
            s.right_closed = not s.right_closed
            s.t_door["R"] = t
        return a

    if t - s.t_action < SETTLE:
        return NONE
    if s.monitor_up:
        return act(CAM_TOGGLE) if t - s.t_monitor >= CAM_HOLD else NONE
    if s.left_closed and t - s.t_door["L"] >= DOOR_HOLD:
        return act(LEFT_DOOR)
    if s.right_closed and t - s.t_door["R"] >= DOOR_HOLD:
        return act(RIGHT_DOOR)
    side = max("LR" if RIGHT_DOOR_ENABLED else "L", key=excess.get)
    if excess[side] >= MOTION_THRESH:
        closed = s.left_closed if side == "L" else s.right_closed
        if not closed:
            return act(LEFT_DOOR if side == "L" else RIGHT_DOOR)
    if t - s.t_monitor >= CAM_PERIOD and not (s.left_closed or s.right_closed):
        return act(CAM_TOGGLE)
    return NONE


def _feats(l=0.03, r=0.03):
    return {f"{t}_{side}": (l if side == "L" else r) for t in MOTION_TYPES for side in "LR"}


if __name__ == "__main__":
    s, dt, trace = State(), 0.1, []

    def run(t0, t1, f):
        t = t0
        while t < t1 - 1e-9:
            a = decide(f, round(t, 2), s)
            if a != NONE:
                trace.append((round(t, 1), a))
            t += dt

    run(0, 5, _feats())  # quiet: build baseline, no action
    assert trace == [], trace
    run(5, 6, _feats(l=0.10))  # left spike
    assert trace[-1][1] == LEFT_DOOR and s.left_closed, trace
    run(6, 11.5, _feats())  # quiet: door reopens after DOOR_HOLD
    assert trace[-1] == (11.0, LEFT_DOOR) and not s.left_closed, trace
    s.t_monitor = 11.5  # push cam schedule out of the way for the right-door check
    RIGHT_DOOR_ENABLED = True  # exercise right-door path too
    run(13, 14, _feats(r=0.10))  # right spike
    assert trace[-1][1] == RIGHT_DOOR and s.right_closed, trace
    run(14, 20, _feats())  # right reopens
    assert trace[-1][1] == RIGHT_DOOR and not s.right_closed, trace
    n = len(trace)
    run(20, 30, _feats())  # quiet long enough -> monitor up (t=23.5) then down
    cams = [a for _, a in trace[n:]]
    assert cams == [CAM_TOGGLE, CAM_TOGGLE] and not s.monitor_up, trace[n:]
    n = len(trace)
    s2 = State(monitor_up=True, t_monitor=29.5, baseline=dict(s.baseline))
    assert decide(_feats(l=0.5), 30.0, s2) == NONE  # monitor up: no door action
    for tt, a in trace:
        print(f"{tt:5.1f}s {a}")
    print("OK")
