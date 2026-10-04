"""Phase 01 env check: python version, packages, tools, session type, weights. Exit 0 iff hard checks pass."""
import importlib
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
ok = True


def report(name, passed, detail="", hard=True):
    global ok
    ok &= passed or not hard
    print(f"{'OK' if passed else ('FAIL' if hard else 'WARN')}  {name} {detail}")


report("python 3.11", sys.version_info[:2] == (3, 11), sys.version.split()[0])
for mod in ["flyvis", "torch", "PIL", "dbus_next", "numpy", "scipy", "sklearn"]:
    try:
        m = importlib.import_module(mod)
        report(f"import {mod}", True, getattr(m, "__version__", ""))
    except Exception as e:
        report(f"import {mod}", False, repr(e))
for name, cmd in [("xdotool", ["xdotool", "--version"]), ("pw-record", ["pw-record", "--version"]),
                  ("python3-gi", ["/usr/bin/python3", "-c", "import gi"])]:
    try:
        report(name, subprocess.run(cmd, capture_output=True).returncode == 0)
    except FileNotFoundError:
        report(name, False, "missing")
sess = os.environ.get("XDG_SESSION_TYPE", "?")
report("session", sess != "wayland", sess, hard=False)
report("flyvis weights", (ROOT / "data/results/flow/0000/000").exists())
sys.exit(0 if ok else 1)
