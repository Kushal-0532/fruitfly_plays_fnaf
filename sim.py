"""Offline harness: runs main.run on a simulated world with fake clock/capture/actuator."""
import json
import tempfile
from pathlib import Path

import numpy as np

import config
from actions import Action
from main import run
from policy import Supervisor
from power import PowerModel
from readers.scripted import ScriptedReader
from safety import Guards
from state import StateTracker
from tests.fakes import FakeClock


class World:
    """Tiny game model so scripted readers react to the bot's actions (doors, lights, monitor, power)."""

    def __init__(self):
        self.door = {"L": False, "R": False}
        self.light = {"L": False, "R": False}  # toggle mode, like the real game
        self.monitor, self.cam, self.power, self.t = False, None, 100.0, 0.0

    def usage(self, t):
        u = 1 + sum(self.door.values()) + sum(self.light.values()) + self.monitor
        return min(u, 4)

    def snapshot(self, t):
        self.power -= config.POWER_RATE_PCT_PER_S[self.usage(t)] * (t - self.t)
        self.t = t
        return dict(hour=min(int(t // config.HOUR_S), 6), power_pct=self.power, usage=self.usage(t),
                    monitor_up=self.monitor, door_closed=dict(self.door), cam=self.cam, foxy_stage=0,
                    light_on=dict(self.light), hall={"L": 0.0, "R": 0.0})

    def apply(self, d, t):
        a = d.action
        if a in (Action.DOOR_L, Action.DOOR_R):
            self.door[a.name[-1]] ^= True
        elif a in (Action.LIGHT_L, Action.LIGHT_R):
            self.light[a.name[-1]] ^= True
        elif a == Action.MONITOR:
            self.monitor ^= True
        elif a == Action.CAM:
            self.cam = d.arg


class FakeCapture:
    def __init__(self, now, stall_at=None):
        self.now, self.stall_at, self.frame = now, stall_at, np.zeros((4, 4, 3), np.uint8)

    def grab(self):
        return None if self.stall_at is not None and self.now() >= self.stall_at else self.frame


class FakeActuator:
    def __init__(self, now, world):
        self.now, self.world, self.calls = now, world, []

    def act(self, d):
        self.calls.append((self.now(), d))
        self.world.apply(d, self.now())
        return True


def simulate(timeline, duration, night=1, stall_at=None, out_dir=None, cfg=None, **run_kw):
    """-> (RunSummary, [(t, Decision)], [log rows])."""
    out = Path(out_dir or tempfile.mkdtemp())
    clock, world = FakeClock(), World()
    act = FakeActuator(clock.now, world)
    sup = Supervisor(PowerModel(), cfg)
    summary = run(FakeCapture(clock.now, stall_at), ScriptedReader(timeline, world), act, StateTracker(),
                  sup, Guards(cfg, stop_path=out / "STOP"), out / "run.jsonl", night, max_seconds=duration,
                  clock=clock.now, sleep=clock.sleep, safe_dir=out, **run_kw)
    rows = [json.loads(l) for l in open(out / "run.jsonl")][1:]
    return summary, act.calls, rows
