"""Phase 01 smoke test: imports, versions, session type, xdotool."""
import os
import shutil
import subprocess

import torch
import flyvis

print("torch", torch.__version__, "cuda", torch.cuda.is_available())
print("flyvis", flyvis.__version__, flyvis.__file__)
sess = os.environ.get("XDG_SESSION_TYPE", "?")
print("session", sess, "" if sess == "x11" else "<-- WARN: wayland: capture via portal, xdotool via Xwayland")
xd = shutil.which("xdotool")
print("xdotool", subprocess.run([xd, "--version"], capture_output=True, text=True).stdout.strip() if xd else "MISSING: sudo apt install xdotool")
