"""Phase 07: deterministic supervisor. GameState -> exactly one Decision with a stable reason code.
All thresholds come from config (U until fitted). Pure given (state, t) plus own small memory."""
import config
from attention import Attention
from actions import Action, Decision

# SPEC D12: every door click carries one of these (a fly verdict); scripts/audit_decisions.py checks logs against it
FLY_DOOR_REASONS = frozenset({"threat_close_L", "threat_close_R", "foxy_close", "freddy_close",
                              "reopen_empty", "reopen_cove", "reopen_probe"})
REASONS = frozenset({
    "untrusted_wait", "settle", "monitor_lower", "light_off", "stall_flip", "stall_cam", "stall_down", "hold_occupied",
    "hall_check", "jam_L", "jam_R", "idle"}) | FLY_DOOR_REASONS

DOOR = {"L": Action.DOOR_L, "R": Action.DOOR_R}
LIGHT = {"L": Action.LIGHT_L, "R": Action.LIGHT_R}


class Supervisor:
    def __init__(self, power_model, cfg=None, night=None):
        self.pm, self.c, self.night = power_model, cfg or config, night  # night None = no stall flips
        self.t0 = None
        self.last_action_t = None
        self.close_t = {}                      # side -> when we saw/made the door closed
        self.streak = {"L": 0, "R": 0}
        self.last_check = {"L": None, "R": None}
        self.jammed = {"L": False, "R": False}  # F24: a dead light button on an open door = jammed; supervisor memory only
        self.audio_t = {}
        self.up_since = None
        self.on_since = {"L": None, "R": None}  # when each side's light was first seen on
        self.alt = "L"
        self.attn = Attention(self.c)
        self.look_t = {}                       # side -> when a hall_check light click was issued and not yet seen to light
        self.look_fails = {"L": 0, "R": 0}     # consecutive hall checks whose light never came on
        self.clear_t = {"L": None, "R": None}  # last time the fly saw this hallway lit and empty
        self.foxy_out, self.foxy_close_t, self.cove_clear, self.cove_n, self.clear_n = False, None, False, 0, 0
        self.foxy_shut_seen = False            # the left door was seen shut after foxy_close
        self.peek_n, self.peek_t = 0, None     # the fly saw Foxy out of his curtain (stage 2-3): watch the cove more (PEEK_WATCH_S)
        self.r_out, self.cam4b_n, self.clear4b_t = False, 0, None  # phase 39: the fly saw somebody (Chica / Freddy) on cam 4B  # phase 35: the fly saw Foxy out of his cove
        self.last_flip = None                  # stall flips (F26): when the monitor was last raised for one
        self.flip_cam, self.flip_n = None, 0   # camera of the flip in progress; flips started
        self.flip_t = self.cam_t = None        # raise time of the flip in progress, and when its camera first showed
        self.cam_since = (None, 0.0)           # camera on screen and since when (fly camera scores count after CAM_SETTLE_S)
        self.flip_verdicts = 0                 # new settled fly verdicts on the flip camera so far

    def note_audio(self, side, t):
        self.audio_t[side] = t

    def _hazard(self, side, t):
        a = self.audio_t.get(side)
        return a is not None and t - a <= self.c.AUDIO_HAZARD_S

    def cam_score(self, s, key, cam, t):
        """-> the fly's score `key` if it is a NEW verdict on camera `cam`, settled CAM_SETTLE_S after the switch; else None.
        The tracker holds cove/cam4b for STALE_S, so s.cove alone repeats an old look (live 2026-10-05 N3: a 1C score from 13 s
        earlier opened every look and counted as confirming frames)."""
        if s.cam != self.cam_since[0]:
            self.cam_since = (s.cam, t)
        v = getattr(s, key)
        if not s.monitor_up or s.cam != cam or v is None or (s.fresh is not None and key not in s.fresh):
            return None
        return v if t - self.cam_since[1] >= self.c.CAM_SETTLE_S else None

    def _out(self, action, reason, t, arg=None):
        assert action not in (Action.DOOR_L, Action.DOOR_R) or reason in FLY_DOOR_REASONS, reason
        if action != Action.NONE:
            self.last_action_t = t
        return Decision(action, arg, reason)

    def decide(self, s, t):
        c = self.c
        if self.t0 is None:
            self.t0 = t
            self.last_check = {"L": t, "R": t}
            self.attn.last = {k: t for k in ("L", "R")}
            self.last_flip = t
        if not s.trusted:
            return Decision(Action.NONE, None, "untrusted_wait")
        if self.last_action_t is not None and t - self.last_action_t < c.SETTLE_S:
            return Decision(Action.NONE, None, "settle")

        if s.monitor_up:
            self.up_since = t if self.up_since is None else self.up_since
        else:
            self.up_since = None
        for side in "LR":
            self.on_since[side] = (self.on_since[side] if self.on_since[side] is not None else t) if s.light_on[side] is True else None
            if s.door_closed[side]:
                self.close_t.setdefault(side, t)
            else:
                self.close_t.pop(side, None)

        for side in "LR":
            if s.hall[side] is not None and s.light_on[side] is True and s.door_closed[side] is False:
                self.attn.saw_hall(side, s.hall[side], t)
                if s.hall[side] < c.HALL_THRESH:
                    self.clear_t[side] = t
            if s.light_on[side] is True or s.hall[side] is not None:  # a hall reading exists only while the light is really on (the lamp reader flickers)
                self.look_t.pop(side, None)
                self.look_fails[side] = 0
        conserve = self.pm.needs_conservation(s.power_pct, s.usage, s.hour)
        mult = c.CONSERVE_MULT if conserve else 1.0
        power_ok = s.power_pct > c.MIN_CHECK_POWER
        door_open_L = s.door_closed["L"] is False

        # 4. monitor lowering
        if s.monitor_up and t - self.up_since > c.MONITOR_MAX_UP_S:
            return self._out(Action.MONITOR, "monitor_lower", t)

        # 5. threats
        threats = []
        for side in "LR":
            if s.door_closed[side] is not False:
                continue
            h = s.hall[side]
            if h is not None and h >= c.HALL_THRESH:
                threats.append((h, side))
            elif h is None and not conserve and self._hazard(side, t):
                threats.append((c.HALL_THRESH, side))
        if threats:
            return self._close(max(threats)[1], f"threat_close_{max(threats)[1]}", t)

        # 5a. Foxy: two fly readouts on a cam-1C look. "Out of his curtain" (cove, stage 2-3) only makes the flips watch him
        # (peek_t): a watched Foxy cannot move (F26), and closing on peeks ran the power out (live 2026-10-05 N3 round 2: L shut 38%).
        # "The cove is empty" (cove_gone, stage 4: he is running) closes the left door. Doors cannot be clicked with the monitor up,
        # so: lower the monitor at once, then close (foxy_close); reopen only after a LATER look sees him back in the cove
        cove, gone = self.cam_score(s, "cove", "1C", t), self.cam_score(s, "cove_gone", "1C", t)
        if not (s.monitor_up and s.cam == "1C"):
            self.cove_n = self.clear_n = self.peek_n = 0
        else:  # steps without a new fly verdict keep the counts
            if gone is not None:
                hi = gone >= c.COVE_GONE_THRESH
                self.cove_n = self.cove_n + 1 if hi else 0
                self.clear_n = 0 if hi else self.clear_n + 1  # one low frame proves nothing: he must be seen home on several
            if cove is not None:
                self.peek_n = self.peek_n + 1 if cove >= c.COVE_THRESH else 0
                if self.peek_n >= c.COVE_CONFIRM:
                    self.peek_t = t
        if s.monitor_up and not self.foxy_out and self.cove_n >= c.COVE_CONFIRM:
            self.foxy_out, self.flip_t, self.cam_t = True, None, None
            if s.door_closed["L"]:
                self.foxy_close_t = t  # already shut (threat close): just keep it shut
            return self._out(Action.MONITOR, "stall_down", t)
        if self.foxy_out and self.foxy_close_t is None and not s.monitor_up:
            if s.door_closed["L"] is False:
                self.foxy_close_t = t
                return self._close("L", "foxy_close", t)
            self.foxy_close_t = t
        if (self.foxy_out and self.foxy_close_t is not None and s.monitor_up and self.flip_t is not None
                and self.flip_t > self.foxy_close_t and self.clear_n >= c.COVE_CLEAR_CONFIRM):
            self.cove_clear, self.flip_t, self.cam_t = True, None, None
            return self._out(Action.MONITOR, "stall_down", t)
        # the left door shut on him and was reopened by someone else (ReadoutPolicy owns reopens): he is no longer loose. Live 2026-10-05
        # N3 round 3: foxy_out stayed set after the readout reopened, every flip went to 1C, 4B unwatched 330 s, Freddy got in at 5 AM
        if self.foxy_out and self.foxy_close_t is not None and s.door_closed["L"]:
            self.foxy_shut_seen = True
        if self.foxy_out and self.foxy_shut_seen and s.door_closed["L"] is False:
            self.foxy_out, self.cove_clear, self.foxy_close_t, self.foxy_shut_seen = False, False, None, False
        if self.foxy_out and self.cove_clear and not s.monitor_up and s.door_closed["L"]:
            self.foxy_out, self.cove_clear, self.foxy_close_t, self.foxy_shut_seen = False, False, None, False
            self.last_check["L"] = t - c.CHECK_PERIOD["L"] * 10
            self.attn.last["L"] = self.last_check["L"]
            return self._out(DOOR["L"], "reopen_cove", t)

        # 5a'. cam 4B (phase 39): somebody on 4B is on the way to the right door. Same dance as Foxy: lower the monitor, then close R
        # (reason threat_close_R: a fly verdict). Reopen is the normal reopen_probe / hall-look path.
        v4b = self.cam_score(s, "cam4b", "4B", t)
        if not (s.monitor_up and s.cam == "4B"):
            self.cam4b_n = 0
        elif v4b is not None:
            self.cam4b_n = self.cam4b_n + 1 if v4b >= c.CAM4B_THRESH else 0
        if v4b is not None and v4b < c.CAM4B_THRESH:
            self.clear4b_t = t  # last time the fly saw 4B empty: leaving it for the cove is safe for a while
        if s.monitor_up and not self.r_out and self.cam4b_n >= c.CAM4B_CONFIRM and s.door_closed["R"] is False:
            self.r_out, self.flip_t, self.cam_t = True, None, None
            return self._out(Action.MONITOR, "stall_down", t)
        if self.r_out and not s.monitor_up:
            self.r_out = False
            if s.door_closed["R"] is False:
                return self._close("R", "threat_close_R", t)

        # 5b. a hall check left the light on (toggle mode) and nobody was there: switch it off again (drain, and the next check
        # would otherwise toggle it off unseen)
        if not s.monitor_up:
            for side in "LR":
                # keep the light on long enough for the fly to respond to what the light reveals before judging 'empty'
                if (s.door_closed[side] is False and s.light_on[side] is True
                        and t - self.on_since[side] >= getattr(c, "LOOK_MIN_S", 0.0)):
                    return self._out(LIGHT[side], "light_off", t)
                # a door closed while its light was on: the look is over. A burning light drains and blocks every flip (_attend waits
                # for lights off), so Foxy ran unwatched (sim N3 2026-10-05: R shut on Chica with the light on, no flip for 36 s)
                if (s.door_closed[side] and s.light_on[side] is True and self.on_since[side] is not None
                        and t - self.on_since[side] >= c.CLOSED_LIGHT_S):
                    return self._out(LIGHT[side], "light_off", t)

        # 6. conditional hold of closed doors
        note = None
        probe = getattr(c, "REOPEN_PROBE_S", None)
        for side in "LR":
            if not s.door_closed[side] or t - self.close_t[side] < c.MIN_HOLD_S or s.monitor_up:
                continue
            if probe is not None and t - self.close_t[side] >= probe and not (side == "L" and self.foxy_out):
                # the fly cannot see through a closed door (its patch shows the slab), so reopen on a timer and let the next
                # look decide; an animatronic still there is seen again and the door closes
                self.last_check[side] = t - c.CHECK_PERIOD[side] * 10  # look right away
                self.attn.last[side] = self.last_check[side]
                return self._out(DOOR[side], "reopen_probe", t)
            h = s.hall[side]
            if s.light_on[side] is not True:
                self.streak[side] = 0  # no light click on a closed door: the fly sees only the slab
            elif h is None:
                self.streak[side] = 0
            elif h >= c.HALL_THRESH:
                self.streak[side] = 0
                note = note or "hold_occupied"
            else:
                self.streak[side] += 1
                if self.streak[side] >= c.EMPTY_FRAMES:
                    self.streak[side] = 0
                    return self._out(DOOR[side], "reopen_empty", t)

        # 7b. a hall-check light that never came on with the door open: the buttons are dead (F24 jam). Never click that side
        # again and never raise the monitor (F25: the next raise is the jumpscare)
        if not s.monitor_up:
            for side in "LR":
                t0 = self.look_t.get(side)
                if t0 is None or s.door_closed[side] is not False or t - t0 < c.LOOK_FAIL_S:
                    continue
                self.look_t.pop(side)
                self.look_fails[side] += 1
                if self.look_fails[side] < c.JAM_CONFIRM:  # one dark look can be a misread (live run 224453: false jam_R killed every flip)
                    self.attn.last[side] = t - c.CHECK_PERIOD[side] * 10  # look again right away
                    continue
                self.jammed[side] = True
                return Decision(Action.NONE, None, f"jam_{side}")

        # 8. attention: the fly's evidence picks what to look at next (a hallway light or a camera)
        if c.ATTENTION:
            d = self._attend(s, t, power_ok, conserve, mult)
            if d is not None:
                return d
        elif power_ok and not s.monitor_up:
            # fixed look timers: the interval shrinks as the night gets more dangerous (HOUR_SCALE), grows when power is
            # short (mult), and the side that has gone longest unchecked (or has a recent audio cue) goes first
            scale = getattr(c, "HOUR_SCALE", {}).get(s.hour, 1.0)
            due = [x for x in "LR" if s.door_closed[x] is False
                   and t - self.last_check[x] >= c.CHECK_PERIOD[x] * mult * scale]
            if due:
                side = max(due, key=lambda x: (self._hazard(x, t), t - self.last_check[x]))
                self.last_check[side] = t
                return self._out(LIGHT[side], "hall_check", t)

        return Decision(Action.NONE, None, note or "idle")

    def _flip_cam(self, s):
        """Night 2: the cove. Night 3+: cam 4B (Freddy only enters from 4B while it is NOT on screen, F29), except every
        COVE_LOOK_EVERY-th flip looks at the cove, and only while the right door is already shut (leaving 4B with it open is how he gets in)."""
        c = self.c
        cam = c.STALL_CAM_BY_NIGHT.get(self.night, c.STALL_CAM)
        if self.foxy_out:  # Foxy loose: every flip is his
            return c.STALL_CAM
        # Foxy peeking (watching): every flip may look at the cove, but only with 4B covered (sim N3: cove-only flips let Freddy in, F29)
        if self.night >= 3 and (self._watching(self.last_flip) or self.flip_n % c.COVE_LOOK_EVERY == 0) and (
                s.door_closed["R"] or (self.clear4b_t is not None and self.last_flip - self.clear4b_t <= c.CAM4B_CLEAR_S)):
            return c.STALL_CAM
        return cam

    def _watching(self, t):
        return self.peek_t is not None and t - self.peek_t <= self.c.PEEK_WATCH_S

    def _flip(self, s, t):
        """The monitor is up for a stall flip: pick 1C, hold, lower. A flip is a look (bookkeeping), never a door decision."""
        c = self.c
        cam = self.flip_cam or c.STALL_CAM
        if s.cam != cam:
            self.cam_t = None
            return self._out(Action.CAM, "stall_cam", t, cam)
        if self.cam_t is None:
            self.cam_t, self.flip_verdicts = t, 0
        key = {"1C": "cove_gone", "4B": "cam4b"}.get(cam)
        if key and self.cam_score(s, key, cam, t) is not None:
            self.flip_verdicts += 1
        # hold until the fly has judged the camera FLIP_VERDICTS times (live 2026-10-05: half the 1 s looks gave it one verdict after
        # CAM_SETTLE_S, an empty cove went unconfirmed and Foxy got in), never past FLIP_MAX_S
        if t - self.cam_t < c.STALL_HOLD_S or (key and self.flip_verdicts < c.FLIP_VERDICTS and t - self.cam_t < c.FLIP_MAX_S):
            return None
        self.flip_t = self.cam_t = None
        for x in "LR":  # the doors were unwatched for the whole flip: check both halls right away
            self.attn.last[x] = t - c.CHECK_PERIOD[x] * 10
        return self._out(Action.MONITOR, "stall_down", t)

    def _attend(self, s, t, power_ok, conserve, mult):
        c, a = self.c, self.attn
        if s.monitor_up:
            return self._flip(s, t) if self.flip_t is not None else None  # raised for something else: step 4 lowers it
        if self.flip_t is not None and t - self.flip_t > c.FLIP_MAX_S + 3.0:  # a flip with a camera switch takes ~2 s before its hold
            self.flip_t = self.cam_t = None                               # the raise never happened / the monitor is already down
        if not power_ok:
            return None
        if any(s.light_on[x] is True for x in "LR"):
            return None  # a light is still on: switch it off first (live 2026-10-04: a second check started 0.6 s after the first, both lights read on, the light_off clicks looped all night)
        scale = getattr(c, "HOUR_SCALE", {}).get(s.hour, 1.0)
        jam = any(self.jammed.values())
        # a look still pending: wait for the light (or LOOK_FAIL_S -> jam), and never flip meanwhile (live 2026-10-05 N3 round 4: the R
        # light stayed dark = Chica jammed it, the bot raised the monitor 2 s later, F25 jumpscare)
        halls = [x for x in "LR" if s.door_closed[x] is False and not self.jammed[x] and x not in self.look_t]
        if (self.night is not None and self.night >= c.STALL_FROM_NIGHT and not jam and self.flip_t is None and not self.look_t
                and not any(s.light_on[x] is True for x in "LR")
                and t - self.last_flip >= c.STALL_PERIOD_S.get(s.hour, 6) * mult * (c.FOXY_OUT_FLIP_MULT if self.foxy_out or self._watching(t) else 1.0)):
            safe_s = {x: c.CAM_SAFE_BY_NIGHT.get(self.night, {}).get(x, c.CAM_SAFE_S) for x in "LR"}  # night 3+: Chica can arrive and enter within one flip
            stale = [x for x in halls if self.clear_t[x] is None or t - self.clear_t[x] > safe_s[x]]
            if not stale:  # never go blind to the doors on a guess
                self.last_flip = self.flip_t = t
                self.flip_n += 1
                self.flip_cam = self._flip_cam(s)
                return self._out(Action.MONITOR, "stall_flip", t)
            k = min(stale, key=lambda x: self.clear_t[x] or -1.0)  # look at the stalest hall first, then flip
            a.last[k] = t
            self.look_t[k] = t
            return self._out(LIGHT[k], "hall_check", t)
        k = a.choose(t, halls, scale, mult)
        if k is None:
            return None
        k = k[0]
        a.last[k] = t
        self.look_t[k] = t
        return self._out(LIGHT[k], "hall_check", t)

    def _close(self, side, reason, t):
        self.close_t[side] = t
        self.streak[side] = 0
        return self._out(DOOR[side], reason, t)
