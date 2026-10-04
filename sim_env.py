"""Phase 29: FNaF 1 night with the documented AI (config.AI_START/AI_STEP/TICK_S, facts F21-F29), no game and no fly network.
Bonnie/Chica walk to their door and jam it if it is open (the next monitor raise kills), Foxy runs after stage 3 unless the monitor
stalls him, Freddy (night >= 3) enters from 4B. The fly's danger signal uses hit/false-alarm/flicker rates measured on the corpus.

  python sim_env.py --night 2 --seeds 200     # cause histogram + median death time for the default policy
"""
import argparse
import json
import random
import statistics
import types
from collections import Counter, deque
from dataclasses import dataclass

import config
from scripts.audit_decisions import audit
from actions import Action
from policy import Supervisor
from power import PowerModel
from state import Readings, StateTracker

DT = 0.1
ACT_S = 1.1                      # a click takes about this long (PAN_SETTLE + hold + settle of the actuator)
CAM_SIGNAL = 0.4                 # U: fly camera-feature shift when an enemy is on the viewed camera (noise sd is 0.02)
SIDE = {"bonnie": "L", "chica": "R"}
# measured with scripts/fly_danger_corpus.py on n2_a
NOISE = {"hit": 0.98, "false_alarm": 0.002, "flicker": 0.12,
         "cove_hit": {2: 0.2, 3: 0.52, 4: 1.0}, "cove_false_alarm": 0.02,
         "cam4b_hit": 0.95, "cam4b_false_alarm": 0.02}  # phase 39 fit: held-out hit 1.00 (16 Chica frames, 5 looks), 1.9% false alarms; rounded down  # cove: phase 34 held-out per-frame rates at the 2% threshold  # corpus n2_a: 86% of occupied lit frames >= 0.5 (14% flicker zeros), 0 false alarms on 61 empty lit frames


def power_rates():
    """Fitted drain rates (config_fit.json "power_rates") when present, else config.POWER_RATE_PCT_PER_S."""
    try:
        return {int(k): v for k, v in json.load(open(config.POWER_FIT_PATH))["power_rates"].items()}
    except (FileNotFoundError, KeyError):
        return dict(config.POWER_RATE_PCT_PER_S)


@dataclass
class Params:
    """Everything the ES may change; maps onto config names."""
    check_L: float = 8.0
    check_R: float = 8.0
    hour_scale: tuple = (3.0, 2.5, 1.5, 1.0, 0.75, 0.75)
    min_hold: float = 5.0
    reopen_probe: float = 12.0
    settle: float = 0.8
    stall_period: tuple = tuple(config.STALL_PERIOD_S[h] for h in range(6))
    stall_hold: float = config.STALL_HOLD_S
    attention: bool = True

    def cfg(self):
        d = {**vars(config), "CHECK_PERIOD": {"L": self.check_L, "R": self.check_R},
             "HOUR_SCALE": dict(enumerate(self.hour_scale)), "MIN_HOLD_S": self.min_hold,
             "REOPEN_PROBE_S": self.reopen_probe, "SETTLE_S": self.settle,
             "STALL_PERIOD_S": dict(enumerate(self.stall_period)), "STALL_HOLD_S": self.stall_hold, "ATTENTION": self.attention}
        return types.SimpleNamespace(**d)


class Night:
    def __init__(self, seed, noise=None, night=2, quiet=False, rates=None):
        self.rng = random.Random(seed)
        self.noise, self.night, self.rates = noise or NOISE, night, rates or power_rates()
        self.ai = dict(config.AI_START[night])
        if self.ai["freddy"] < 0:
            self.ai["freddy"] = self.rng.choice([1, 2])
        self.quiet = quiet
        if quiet:
            self.ai = dict.fromkeys(self.ai, 0)
        self.t, self.power = 0.0, 100.0
        self.door = {"L": False, "R": False}
        self.light = {"L": False, "R": False}
        self.monitor = False
        self.cam = "1A"
        self.view = None                       # side the pointer last went to (what is on screen), None = centre / monitor
        self.dead = self.death_t = None
        self.pos = {"bonnie": "1A", "chica": "1A", "freddy": "1A"}
        self.jammed = {"bonnie": False, "chica": False}
        self.next_tick = dict(config.TICK_S)   # first movement opportunity one period in
        self.foxy = 1                          # 1-3 in the cove, 4 = running
        self.lock_until, self.attack_t, self.knocks = 0.0, None, 0
        self.freddy_in = False
        self.hist = {"L": deque(maxlen=config.HALL_SMOOTH_FRAMES), "R": deque(maxlen=config.HALL_SMOOTH_FRAMES)}

    def usage(self):
        return min(1 + sum(self.door.values()) + sum(self.light.values()) + self.monitor, 4)

    def hour(self):
        return min(int(self.t // config.HOUR_S), 6)

    def roll(self, who):
        return self.rng.randint(1, 20) <= self.ai[who]

    def die(self, cause):
        self.dead, self.death_t = cause, self.t

    # --- world dynamics
    def step(self, dt=DT):
        h0 = self.hour()
        self.t += dt
        if self.hour() != h0 and not self.quiet:
            for who, d in config.AI_STEP.get(self.hour(), {}).items():
                self.ai[who] += d
        self.power -= (self.rates[self.usage()] + config.PASSIVE_DRAIN.get(self.night, 0.0)) * dt
        if self.power <= 0:
            return self.die("power_out")
        for who in ("freddy", "bonnie", "chica", "foxy"):
            while self.t >= self.next_tick[who] and not self.dead:
                self.next_tick[who] += config.TICK_S[who]
                getattr(self, f"tick_{who}" if who in ("freddy", "foxy") else "tick_walker")(who)
        if self.dead:
            return
        if self.foxy == 4:
            if self.monitor and self.cam == "2A":
                self.attack_t = min(self.attack_t, self.t)  # F27: viewing 2A sends him at once
            if self.t >= self.attack_t:
                if self.door["L"]:
                    self.knocks += 1
                    self.power -= 5 * self.knocks - 4
                    self.foxy, self.attack_t = 1, None
                else:
                    return self.die("foxy")
        if self.freddy_in and not self.monitor and self.rng.random() < 0.25 * dt:
            self.die("freddy")

    def tick_walker(self, who):
        if self.jammed[who] or not self.roll(who):
            return
        if self.pos[who] == "door":
            if self.door[SIDE[who]]:
                self.pos[who] = self.rng.choice(config.SIM_DOOR_RETURN[who])  # F23/F24: blocked, it leaves
            else:
                self.jammed[who] = True                                         # F24: open door -> jam
        else:
            self.pos[who] = self.rng.choice(config.SIM_PATHS[who][self.pos[who]])

    def tick_foxy(self, _):
        if self.foxy == 4 or self.monitor or self.t < self.lock_until or not self.roll("foxy"):
            return  # F26: fails while the monitor is up, and for a random time after
        self.foxy += 1
        if self.foxy == 4:
            self.attack_t = self.t + config.FOXY_RUN_S

    def tick_freddy(self, _):
        if self.night < 3 or self.freddy_in or not self.roll("freddy"):
            return
        i = config.FREDDY_PATH.index(self.pos["freddy"])
        if i < len(config.FREDDY_PATH) - 1:
            if not self.monitor:
                self.pos["freddy"] = config.FREDDY_PATH[i + 1]
        elif self.door["R"]:
            self.pos["freddy"] = "4A"
        elif self.monitor and self.cam != "4B":
            self.freddy_in = True

    def apply(self, action, dur=ACT_S, arg=None):
        """Perform an action, advancing the world by its duration (clicks block, like the live actuator)."""
        side = action.name[-1] if action.name[-1] in "LR" else None
        jam = side and self.jammed[next(k for k, v in SIDE.items() if v == side)]
        if action in (Action.DOOR_L, Action.DOOR_R):
            if not jam:
                self.door[side] ^= True
            self.view = side
        elif action in (Action.LIGHT_L, Action.LIGHT_R):
            if not jam:
                self.light[side] ^= True
            self.view = side
        elif action == Action.CAM:
            self.cam = arg
        elif action == Action.MONITOR:
            self.monitor ^= True
            self.view = None
            if self.monitor and any(self.jammed.values()):
                self.die(next(k for k, v in self.jammed.items() if v) + "_jam")  # F25
            elif not self.monitor:
                self.lock_until = self.t + self.rng.uniform(*config.FOXY_LOCK_S)
        for _ in range(int(round(dur / DT))):
            if self.dead:
                break
            self.step()

    # --- observation (what the live readers would report)
    def at_door(self, s):
        return any(self.pos[w] == "door" for w in SIDE if SIDE[w] == s)

    def danger(self, s):
        if not self.light[s] or self.door[s] or self.view != s or self.monitor:
            self.hist[s].clear()
            return None
        if self.rng.random() < self.noise["flicker"]:
            v = 0.0
        elif self.at_door(s):
            v = 0.97 if self.rng.random() < self.noise["hit"] else 0.1
        else:
            v = 0.97 if self.rng.random() < self.noise["false_alarm"] else 0.02
        self.hist[s].append(v)
        return max(self.hist[s])  # same smoothing as fly_brain.FlyHallway

    def cove_obs(self):
        """The fly's cove score while cam 1C is on screen: per-frame hit rate by Foxy stage, 2% false alarms at stage 1."""
        if not (self.monitor and self.cam == "1C"):
            return None
        p = self.noise["cove_hit"].get(self.foxy, self.noise["cove_false_alarm"])
        return 0.95 if self.rng.random() < p else 0.05

    def cam4b_obs(self):
        """The fly's 4B score: somebody (Chica / Freddy) standing on 4B."""
        if not (self.monitor and self.cam == "4B"):
            return None
        p = self.noise["cam4b_hit"] if any(self.pos[w] == "4B" for w in ("chica", "freddy")) else self.noise["cam4b_false_alarm"]
        return 0.95 if self.rng.random() < p else 0.05

    def cam_feat(self):
        shown = any(p == self.cam for w, p in self.pos.items() if w != "freddy")
        return [self.rng.gauss(0.5 * (i + 1) + (CAM_SIGNAL if shown and i < 3 else 0.0), 0.02) for i in range(6)]

    def readings(self):
        vis = (not self.monitor)
        dc = {s: (self.door[s] if vis and self.view == s else None) for s in "LR"}
        lo = {s: (self.light[s] if vis and self.view == s else None) for s in "LR"}
        return Readings(self.t, hour=self.hour(), power_pct=float(round(self.power)), usage=self.usage(), monitor_up=self.monitor,
                        door_closed=dc, light_on=lo, cam=self.cam if self.monitor else None, cam_feat=self.cam_feat() if self.monitor else None,
                        hall={s: self.danger(s) for s in "LR"}, cove=self.cove_obs(), cam4b=self.cam4b_obs())


def play(params=None, seed=0, noise=None, max_s=None, disable_doors=False, quiet=False, night=2, flips=True):
    """-> dict(result, cause, death_t, t, hour, power, actions, audit). result/cause is '6am' or a death cause."""
    params = params or Params()
    n = Night(seed, noise, night, quiet)
    tracker, sup = StateTracker(assume_open=True), Supervisor(PowerModel(), params.cfg(), night=night if flips else None)
    n_act, limit, rows = 0, (max_s or (6 * config.HOUR_S + 1)), []
    while not n.dead and n.t < limit:
        r = n.readings()
        s = tracker.update(r, got_frame=True)
        sup.pm.observe(n.t, s.power_pct, s.usage)
        d = sup.decide(s, n.t)
        rows.append({"t": n.t, "action": d.action, "reason": d.reason})
        if d.action in (Action.DOOR_L, Action.DOOR_R) and disable_doors:
            d = type(d)(Action.NONE, None, "doors_disabled")
        if d.action not in (Action.NONE, Action.SAFE_MODE):
            n.apply(d.action, ACT_S if d.action != Action.CAM else 0.5, d.arg)
            sup.last_action_t = n.t
            n_act += 1
        else:
            n.step()
    res = n.dead or ("6am" if n.hour() >= 6 else "timeout")
    return {"result": res, "cause": res, "death_t": n.death_t, "t": round(n.t, 1), "hour": n.hour(), "power": round(n.power, 1),
            "actions": n_act, "audit": audit(rows)}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--night", type=int, default=2)
    ap.add_argument("--seeds", type=int, default=200)
    a = ap.parse_args(argv)
    rs = [play(seed=s, night=a.night) for s in range(a.seeds)]
    print(f"night {a.night}, default policy, {a.seeds} seeds:", dict(Counter(r["cause"] for r in rs).most_common()))
    dt = [r["death_t"] for r in rs if r["death_t"]]
    print("median death t:", round(statistics.median(dt), 1) if dt else None)


if __name__ == "__main__":
    main()
