"""All tunables. Unverified game facts (specs/docs/fnaf1-facts.md) get added here as phases need them."""

# xdotool --name candidates for the FNaF 1 window (Wine/Bottles); first match wins
WINDOW_TITLE_CANDIDATES = ["Five Nights at Freddy", "FiveNightsatFreddys", "FNAF"]

REF_W, REF_H = 1280, 720  # reference window size all coordinates are expressed in
FPS = 10                  # control loop rate
CALIBRATION_PATH = "calibration.json"
CORPUS_DIR = "data/corpus"
LOG_DIR = "logs"

# Actuator (all U until the Phase 04 live probe, see specs/docs/fnaf1-facts.md F4/F5)
LIGHT_MODE = "toggle"      # "toggle" | "hold"
LIGHT_HOLD = 1.0           # s, one light() call in hold mode
PAN_SETTLE = 1.0           # s after mousemove before pressing (office view pans with pointer)
CLICK_HOLD = 0.06          # s mousedown -> mouseup
MONITOR_DWELL = 0.5        # s hovering the bottom bar to flip the monitor
MIN_ACTION_GAP = 0.25      # s between actions
PARK_REF = (640, 360)      # reference px, no button hovered

# State tracker (Phase 05)
# door_closed never expires: only the bot changes doors and only one side is on screen at a time (live: the left door expired mid-hold and tripped safe mode)
STALE_S = {"hour": 5, "power_pct": 5, "usage": 5, "monitor_up": 1, "door_closed": 1e9, "light_on": 120,
           "cam": 2, "hall": 0.5, "foxy_stage": 8, "cove": 30, "cam4b": 15}  # s a None reading keeps the last good value
FREEZE_S = 20          # s hour+power unchanged while frames arrive -> frozen_clock
CAPTURE_STALL_S = 2    # s without frames -> capture_stall
REQUIRED_FIELDS = ("hour", "power_pct", "monitor_up", "door_closed.L", "door_closed.R")

# Power model (Phase 06)
POWER_RATE_PCT_PER_S = {1: 0.10, 2: 0.20, 3: 0.30, 4: 0.40}  # U: pessimistic placeholders, fitted in Phase 17
HOUR_S = 89                                                   # U F3: seconds per in-game hour
POWER_FIT_PATH = "config_fit.json"

# Supervisor (Phase 07). All U: Phase 14 fits HALL_THRESH, Phase 17 fits holds/periods, Phase 15 the Foxy values.
SETTLE_S = 0.8             # s of NONE after any action
HALL_THRESH = 0.5          # hallway occupancy score that counts as "occupied"
REOPEN_PROBE_S = 12.0      # U: s a closed door is held before reopening to probe (None = only reopen when a lit look sees the hall empty)
LOOK_MIN_S = 1.0           # U: s a light stays on before the bot may call the hall empty and switch it off (the fly needs a few frames)
HALL_SMOOTH_FRAMES = 5     # fly danger: max over this many frames while the light is on (the game flickers the hallway light)
MIN_HOLD_S = 5.0           # s a closed door stays closed before reopen is considered
EMPTY_FRAMES = 5           # consecutive lit+empty frames needed to reopen
MONITOR_MAX_UP_S = 4.0     # s before the supervisor lowers the monitor
CHECK_PERIOD = {"L": 8, "R": 8}  # s between hall light checks at HOUR_SCALE 1.0
HOUR_SCALE = {0: 3.0, 1: 2.5, 2: 1.5, 3: 1.0, 4: 0.75, 5: 0.75}  # U: look rarely early (nothing has moved yet), often late
CONSERVE_MULT = 2.0
ATTENTION = True           # fly-evidence-driven looking (attention.py); False = the old fixed hall-check timers
STALL_FROM_NIGHT = 1       # U: stall flips (F26: Foxy fails every move while the monitor is up) from this night. 1: run c_n1_live was killed by Foxy at 5 AM on Night 1 (F22 says his AI is 0 there)
STALL_PERIOD_S = {0: 10, 1: 8, 2: 7, 3: 6, 4: 6, 5: 6}  # U: s between monitor raises, by hour (phase 31 tunes)
STALL_HOLD_S = 1.0         # s the monitor stays up after cam 1C shows (the fly's response lands in the log)
STALL_CAM = "1C"
LOOK_FAIL_S = 2.0          # s after a hall-check click with the light still not on = blocked look (retry once, then close that door)
CAM_SAFE_S = 10.0          # s: a camera look needs both open halls seen empty this recently
ATTN_HEAT_GAIN = 3.0       # drive multiplier per unit of heat
ATTN_HEAT_TAU_S = 25.0
ATTN_HEAT_MAX = 2.0
AUDIO_HAZARD_S = 5.0       # s a note_audio() cue keeps a side hazardous
MIN_CHECK_POWER = 3.0      # % below which speculative checks stop

# Safety guards (Phase 08)
UNTRUSTED_SAFE_S = 5       # s continuously untrusted -> SAFE_MODE
STARTUP_GRACE_S = 25       # s allowed before the first trusted state (night-start title card has no HUD)
MAX_ACTIONS_PER_MIN = 40
VERIFY_S = 1.5             # s for an action's state change to show up
MAX_RETRIES = 3
MONITOR_STUCK_S = 6        # s monitor up -> second lowering gesture, then SAFE_MODE

# Audio cue detector (Phase 18). All U until the live check L18.
AUDIO_ENABLED = False       # flipped via config_fit.json key "audio_enabled" only if the accuracy bar is met
AUDIO_RATE = 16000
BANDS = ((100, 800), (800, 3000), (3000, 7000))  # Hz
ONSET_RATIO = 4.0           # band energy / noise floor
MIN_ON_S = 0.1              # s above ratio before an onset fires
REFRACTORY_S = 0.5          # s after an event during which no new one fires
PAN_RATIO = 2.0             # louder/quieter channel ratio needed to call a side, else "B"

# Faithful sim (phase 29). D = documented AI, specs/docs/fnaf1-facts.md F21-F29
AI_START = {1: dict(bonnie=0, chica=0, foxy=0, freddy=0), 2: dict(bonnie=3, chica=1, foxy=1, freddy=0),
            3: dict(bonnie=0, chica=5, foxy=2, freddy=-1), 4: dict(bonnie=2, chica=4, foxy=6, freddy=1),
            5: dict(bonnie=5, chica=7, foxy=5, freddy=3), 6: dict(bonnie=10, chica=12, foxy=6, freddy=4)}  # D F22; freddy -1 = Night 3's random 1-or-2
AI_STEP = {2: dict(bonnie=1), 3: dict(bonnie=1, chica=1, foxy=1), 4: dict(bonnie=1, chica=1, foxy=1)}  # D F22: added at the hour key (2 AM, 3 AM, 4 AM)
TICK_S = dict(freddy=3.02, bonnie=4.97, chica=4.98, foxy=5.01)  # D F21
SIM_PATHS = {  # U: uniform choice among neighbours; "door" = the office door of that animatronic (F23)
    "bonnie": {"1A": ["1B"], "1B": ["5", "2A"], "5": ["1B", "2A"], "2A": ["3", "2B"], "3": ["2A", "door"], "2B": ["3", "door"]},
    "chica": {"1A": ["1B"], "1B": ["7", "6"], "7": ["6", "4A"], "6": ["7", "4A"], "4A": ["4B", "1B"], "4B": ["4A", "door"]}}
SIM_DOOR_RETURN = {"bonnie": ["1B"], "chica": ["1B", "4A"]}  # F23: where a blocked animatronic goes
FREDDY_PATH = ["1A", "1B", "7", "6", "4A", "4B"]  # D F29
FOXY_LOCK_S = (0.83, 17.48)  # D F26: random lock after the monitor goes down
FOXY_RUN_S = 25.0            # D F27
PASSIVE_DRAIN = {n: 0.0 for n in range(1, 7)}  # U: %/s extra drain per night. TODO fit from live logs (phase 37)
COLLECT_PERIOD_S = 25.0     # U: --collect-cove: s between stall flips, so Foxy progresses past stage 1 (phase 33)
COLLECT_HOLD_S = 2.0
COVE_GAIN = 4.0              # U: |cam picture - stage-1 reference| amplification for the fly's cam-1C input (phase 34)
COVE_THRESH = 0.86          # U: cove score that counts as "Foxy is out" (fit_cove writes the fitted threshold into data/templates/fly_cove.npz; main loads it)
COVE_CONFIRM = 2            # U: consecutive frames at/above the threshold before the supervisor acts
FOXY_OUT_FLIP_MULT = 0.5    # U: stall flips come twice as often while Foxy is loose
JAM_CONFIRM = 2             # consecutive dark hall-check looks before a side counts as jammed (F24)
COVE_CLEAR_CONFIRM = 5       # U: consecutive low-cove frames on a later 1C look before reopening the left door (run 225159: 1-frame reopen dithered 10x)
STALL_CAM_BY_NIGHT = {2: "1C", 3: "4B", 4: "4B", 5: "4B", 6: "4B"}  # phase 38
COVE_LOOK_EVERY = 3         # U: night >= 3: every 3rd flip looks at the cove (only with the right door shut) so Foxy stays under the fly's eye
CAM_SAFE_BY_NIGHT = {n: {"R": 5.0} for n in (3, 4, 5, 6)}  # U: per side, s a hall must have been seen clear before a flip (an all-sides 5 s window never fits: sim livelock). Night 3 rounds 1-2 (runs 231259, 231825): Chica killed within 1 s of lowering the monitor after a 4B flip
CAM4B_THRESH = 0.86         # U: cam4b score that counts as "somebody is on 4B" (fit_cove(cam="4B") writes the fitted one into data/templates/fly_cam4b.npz)
CAM4B_CONFIRM = 2           # U: consecutive frames before the supervisor acts
CAM4B_CLEAR_S = 15.0         # U: a cove flip on night 3+ is allowed if the fly saw 4B empty this recently (or the right door is shut)
