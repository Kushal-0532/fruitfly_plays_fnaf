"""Phase 09: the control loop. Only wires; every part is injected so sim.py can run it offline."""
import argparse
import hashlib
import json
import subprocess
import time
import types
from collections import deque
from dataclasses import dataclass
from pathlib import Path

import numpy as np

import config
from actions import Action, Decision
from audio import forward as forward_audio
from safety import FrameRing, safe_mode
from state import Readings

ROW_KEYS = ("hour", "power_pct", "usage", "monitor_up", "door_closed", "light_on", "cam", "hall",
            "foxy_stage", "cove", "cam4b", "trusted", "reasons")


@dataclass
class RunSummary:
    status: str            # "6am" | "timeout" | "safe_mode"
    hour_reached: int | None
    n_actions: int
    safe_reason: str | None = None


def _config_hash():
    d = {k: repr(v) for k, v in sorted(vars(config).items()) if k.isupper()}
    return hashlib.md5(json.dumps(d).encode()).hexdigest()[:10]


def _git_sha():
    try:
        r = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, timeout=2)
        return r.stdout.strip() or None
    except Exception:
        return None


def _grab(capture, last_n):
    """-> (frame or None, got_new_frame, n). Real Capture re-serves its latest frame, so use its counter."""
    try:
        frame = capture.grab()
    except Exception:
        return None, False, last_n
    n = getattr(capture, "n", None)
    return frame, frame is not None and (n is None or n != last_n), n


def run(capture, reader, actuator, tracker, supervisor, guards, log_path, night, dry_run=False,
        max_seconds=None, clock=time.monotonic, sleep=time.sleep, power_mismatch=True, safe_dir=None,
        audio=None, shadow=None, viz=None, recorder=None):
    from actuator import Refused
    Path(log_path).parent.mkdir(parents=True, exist_ok=True)
    log = open(log_path, "w")
    log.write(json.dumps({"header": {"night": night, "ts": time.strftime("%Y%m%d-%H%M%S"),
                                     "config_hash": _config_hash(), "dry_run": dry_run,
                                     "git_sha_or_none": _git_sha()}}) + "\n")
    ring, tail = FrameRing(30, config.FPS), deque(maxlen=300)
    t0, n_actions, hour, last_n = clock(), 0, None, None
    status, safe_reason = "timeout", None
    dt = 1 / config.FPS
    try:
        while True:
            ts, w0 = clock(), time.perf_counter()
            t = ts - t0
            frame, got, last_n = _grab(capture, last_n)
            if frame is not None:
                ring.push(t, frame)
                if recorder is not None and got:
                    recorder.submit(frame, t)
            readings = reader.read(frame, t) if got else Readings(t)
            state = tracker.update(readings, got_frame=got)
            pm = supervisor.pm
            pm.observe(t, state.power_pct, state.usage)
            mism = power_mismatch and pm.mismatch(t, state.power_pct, state.usage)
            if audio is not None:
                forward_audio(audio.poll(), supervisor, t)
            dec = supervisor.decide(state, t)
            # dry run: nothing is sent, so the guards only judge perception (freeze/untrusted/power), never action effects
            vetted = guards.vet(state, Decision(Action.NONE, None, dec.reason) if dry_run else dec, t, mismatch=mism)

            if vetted.action not in (Action.SAFE_MODE, Action.NONE) and not dry_run:
                try:
                    if actuator.act(vetted):
                        supervisor.last_action_t = clock() - t0  # settle counts from the END of the (blocking) action, so frames get read
                        n_actions += 1
                        guards.note_sent()
                        guards.expect(vetted, clock() - t0)  # verify from when the click ended, not when it started
                except Refused:
                    guards.note_refused()
            if state.hour is not None:
                hour = max(hour or 0, state.hour)

            row = {"t": round(t, 3), **{k: getattr(state, k) for k in ROW_KEYS},
                   "action": dec.action.name, "arg": dec.arg, "reason": dec.reason,
                   "guard": "" if (vetted.action, vetted.reason) == (dec.action, dec.reason)
                   else f"{vetted.action.name}:{vetted.reason}",
                   "loop_ms": round((time.perf_counter() - w0) * 1000, 2)}
            if viz is not None:
                viz.update(state, row)
            if shadow is not None and got and frame is not None:
                row.update(shadow.observe(frame, state, t))  # logging only, never read back by the supervisor
            log.write(json.dumps(row) + "\n")
            log.flush()
            tail.append(row)

            if vetted.action == Action.SAFE_MODE:
                safe_reason, status = vetted.reason, "safe_mode"
                safe_mode(vetted.reason, ring, list(tail), safe_dir)
                break
            if state.hour == 6:
                status = "6am"
                break
            if max_seconds is not None and t >= max_seconds:
                break
            sleep(max(0.0, dt - (clock() - ts)))
    finally:
        log.close()
    return RunSummary(status, hour, n_actions, safe_reason)


def apply_fit(path=config.POWER_FIT_PATH):
    """Overrides written by fit_config.py ("overrides" in config_fit.json) replace config values for this process."""
    try:
        for k, v in json.load(open(path)).get("overrides", {}).items():
            setattr(config, k, {int(a) if a.lstrip("-").isdigit() else a: b for a, b in v.items()} if isinstance(v, dict) else v)
    except FileNotFoundError:
        pass
    try:
        config.CAM4B_THRESH = float(np.load("data/templates/fly_cam4b.npz")["threshold"])
    except FileNotFoundError:
        pass


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--night", type=int, required=True)
    ap.add_argument("--dry-run", action="store_true", help="read and log only; never sends input")
    ap.add_argument("--max-seconds", type=float)
    ap.add_argument("--perception", choices=("fly", "pixel"), default="fly",
                    help="who judges the hallways: the fly's neurons (default) or the pixel-difference detector")
    ap.add_argument("--viz", nargs="?", const=8765, type=int, metavar="PORT", help="live fly-brain page on http://localhost:PORT")
    ap.add_argument("--shadow", action="store_true", help="also score hallway bursts with the fly model (logged, never used)")
    ap.add_argument("--record", metavar="NAME", help="also save every frame to data/corpus/NAME (for labeling Chica/Foxy later)")
    ap.add_argument("--collect-cove", action="store_true", help="phase 33: flip to cam 1C rarely so Foxy leaves the cove (recording data)")
    ap.add_argument("--policy", choices=("supervisor", "readout"), default="supervisor",
                    help="who takes the door decisions: the supervisor rules (default) or the trained fly readout (phase 43)")
    a = ap.parse_args(argv)
    from actuator import Actuator
    from capture import SCALE, Capture, find_window
    from geometry import Geometry, Window, load_buttons
    from policy import Supervisor
    from record import LiveRecorder
    from power import PowerModel, load_rates
    from readers import live_reader
    from safety import Guards
    from state import StateTracker

    apply_fit()
    try:
        config.COVE_THRESH = float(np.load("data/templates/fly_cove.npz")["threshold"])
    except FileNotFoundError:
        pass
    if a.collect_cove:
        config.STALL_PERIOD_S = {h: config.COLLECT_PERIOD_S for h in range(6)}
        config.STALL_HOLD_S = config.COLLECT_HOLD_S
    buttons = load_buttons(config.CALIBRATION_PATH)
    cap = Capture()
    cap.grab(timeout=20)  # block until the first frame: the screencast can take a few seconds to start, which read as a stall
    geom = Geometry(Window(*find_window()), SCALE)
    try:
        pm = load_rates()
    except (FileNotFoundError, KeyError):
        pm = PowerModel()
    # dry run: the human drives, so a monitor held up longer than the bot would is not a fault
    dry_cfg = types.SimpleNamespace(**{k: getattr(config, k) for k in dir(config) if k.isupper()})
    dry_cfg.MONITOR_STUCK_S = 1e9
    brain = viz = None
    if a.perception == "fly" or a.viz:
        from fly_brain import FlyBrain, FlyHallway, Tee
        brain = FlyBrain(geom, buttons, view=bool(a.viz))
        if a.perception == "fly" and brain.readout is None:
            raise SystemExit("no fly readout: run `python fly_brain.py` first")
    if a.viz:
        from live_viz import LiveViz
        viz = LiveViz(brain, a.viz)
        print(f"live fly brain: http://localhost:{a.viz}", flush=True)
    from readers.hallway import HallwayDetector
    hallway = FlyHallway(brain) if a.perception == "fly" else (Tee(brain, HallwayDetector(geom, buttons)) if brain else None)
    shadow = None
    if a.shadow:
        from shadow import ShadowHook
        shadow = ShadowHook(geom, buttons)
    ts = time.strftime("%Y%m%d-%H%M%S")
    out = Path(config.LOG_DIR)
    rec = None
    try:
        rec = LiveRecorder(Path(config.CORPUS_DIR) / a.record) if a.record else None
        summary = run(cap, live_reader(geom, buttons, hallway), Actuator(cap.wid, geom, buttons), StateTracker(assume_open=True), (Supervisor if a.policy == "supervisor" else __import__("readout_policy").ReadoutPolicy)(pm, night=a.night),
                      Guards(dry_cfg if a.dry_run else None, stop_path=out / "STOP"), out / f"run_{ts}.jsonl", a.night, dry_run=a.dry_run,
                      max_seconds=a.max_seconds, safe_dir=out / f"safe_{ts}", shadow=shadow, viz=viz, recorder=rec)
    finally:
        if rec is not None:
            from capture import find_window as _fw
            wid, x, y, w, h = _fw()
            rec.close({"window": {"wid": wid, "x": x, "y": y, "w": w, "h": h}, "capture_scale": SCALE, "night": a.night})
        cap.close()
    print(summary)


if __name__ == "__main__":
    main()
