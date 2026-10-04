"""Grab frames of the FNaF 1 window on GNOME Wayland.

xdg-desktop-portal ScreenCast (dbus-next) -> PipeWire node -> gst_pipe.py
(system python, GStreamer) -> raw RGB frames on a pipe -> numpy.
mss does not work here (XGetImage fails on Xwayland root).
"""
import asyncio
import ctypes
import signal
import os
import secrets
import subprocess
import sys
import threading
import time
from pathlib import Path

import numpy as np

import config
from dbus_next import Message, Variant
from dbus_next.aio import MessageBus

SCALE = 2  # ponytail: capture at 1/2 res; perception downsamples far more anyway
FPS = config.FPS
TOKEN_FILE = Path(__file__).with_name(".portal_token")
PORTAL = "org.freedesktop.portal.Desktop"
PORTAL_PATH = "/org/freedesktop/portal/desktop"


_run = subprocess.run  # indirection so tests can inject a FakeRunner


def _out(*argv):
    return _run(list(argv), capture_output=True, text=True).stdout


def find_window(names=None):
    """-> (wid, x, y, w, h) of the first window matching any candidate title.
    Falls back to the full display (wid None) if none match."""
    for name in names or config.WINDOW_TITLE_CANDIDATES:
        wids = _out("xdotool", "search", "--name", name).split()
        if wids:
            g = dict(l.split("=") for l in _out("xdotool", "getwindowgeometry", "--shell", wids[0]).split())
            return int(wids[0]), int(g["X"]), int(g["Y"]), int(g["WIDTH"]), int(g["HEIGHT"])
    # ponytail: xdotool --name regex misses the Wine window; fall back to substring match on the listing
    for wid, title in list_windows():
        if any(n.lower() in title.lower() for n in names or config.WINDOW_TITLE_CANDIDATES):
            g = dict(l.split("=") for l in _out("xdotool", "getwindowgeometry", "--shell", str(wid)).split())
            return wid, int(g["X"]), int(g["Y"]), int(g["WIDTH"]), int(g["HEIGHT"])
    w, h = display_size()
    return None, 0, 0, w, h


def list_windows():
    """-> [(wid, title)] for every X window with a non-empty name."""
    out = []
    for wid in _out("xdotool", "search", "--name", "").split():
        title = _out("xdotool", "getwindowname", wid).strip()
        if title:
            out.append((int(wid), title))
    return out


def display_size():
    return tuple(map(int, _out("xdotool", "getdisplaygeometry").split()))


async def _portal_session():
    """Full ScreenCast handshake. -> (pipewire_fd, node_id)."""
    bus = await MessageBus(negotiate_unix_fd=True).connect()
    sender = bus.unique_name[1:].replace(".", "_")
    responses = {}

    def handler(msg):
        if msg.member == "Response" and msg.interface == "org.freedesktop.portal.Request":
            responses[msg.path] = msg.body  # (code, results)

    bus.add_message_handler(handler)

    async def request(member, sig, body, opts):
        token = "t" + secrets.token_hex(4)
        opts = {**opts, "handle_token": Variant("s", token)}
        path = f"{PORTAL_PATH}/request/{sender}/{token}"
        reply = await bus.call(Message(destination=PORTAL, path=PORTAL_PATH,
                                       interface="org.freedesktop.portal.ScreenCast",
                                       member=member, signature=sig, body=body + [opts]))
        assert reply.error_name is None, reply.body
        while path not in responses:
            await asyncio.sleep(0.05)
        code, results = responses.pop(path)
        assert code == 0, f"{member} cancelled/failed (code {code})"
        return {k: v.value for k, v in results.items()}

    r = await request("CreateSession", "a{sv}", [], {"session_handle_token": Variant("s", "ucn")})
    session = r["session_handle"]
    opts = {"types": Variant("u", 1), "cursor_mode": Variant("u", 1), "persist_mode": Variant("u", 2)}
    if TOKEN_FILE.exists():
        opts["restore_token"] = Variant("s", TOKEN_FILE.read_text().strip())
    await request("SelectSources", "oa{sv}", [session], opts)
    r = await request("Start", "osa{sv}", [session, ""], {})
    if "restore_token" in r:
        TOKEN_FILE.write_text(r["restore_token"])
    node = r["streams"][0][0]
    reply = await bus.call(Message(destination=PORTAL, path=PORTAL_PATH,
                                   interface="org.freedesktop.portal.ScreenCast",
                                   member="OpenPipeWireRemote", signature="oa{sv}", body=[session, {}]))
    fd = reply.unix_fds[0]
    return fd, node, bus  # keep bus alive or session dies


class Capture:
    def __init__(self, fps=FPS, scale=SCALE, overlay=True):
        self.wid, x, y, w, h = find_window()
        self.overlay = None
        if self.wid is not None and overlay:
            # A lone fullscreen Proton window gets unredirected (sent straight to
            # the display), which starves the screencast: stream negotiates, zero
            # frames. Any window overlapping it forces compositing again. 4x4 px,
            # override-redirect so it never takes focus. Clearing
            # _NET_WM_BYPASS_COMPOSITOR on the game did NOT help (tested).
            self.overlay = subprocess.Popen(
                ["xmessage", "-xrm", "*overrideRedirect: True",
                 "-geometry", f"4x4+{x + w - 4}+{y + h - 4}", "-buttons", "", " "],
                stderr=subprocess.DEVNULL,
                preexec_fn=lambda: ctypes.CDLL("libc.so.6").prctl(1, signal.SIGTERM))
        self.region = (y // scale, (y + h) // scale, x // scale, (x + w) // scale)  # y0,y1,x0,x1
        dw, dh = display_size()
        self.w, self.h = dw // scale, dh // scale
        fd, node, self._bus = asyncio.get_event_loop().run_until_complete(_portal_session())
        self.proc = subprocess.Popen(
            ["/usr/bin/python3", str(Path(__file__).with_name("gst_pipe.py")),
             str(fd), str(node), str(self.w), str(self.h), str(fps)],
            stdout=subprocess.PIPE, pass_fds=(fd,),
            preexec_fn=lambda: ctypes.CDLL("libc.so.6").prctl(1, signal.SIGTERM))  # PR_SET_PDEATHSIG: die with parent
        self.latest, self.n = None, 0
        threading.Thread(target=self._reader, daemon=True).start()

    def _reader(self):
        size = self.w * self.h * 3
        while True:
            buf = self.proc.stdout.read(size)
            if len(buf) < size:
                break
            self.latest = np.frombuffer(buf, np.uint8).reshape(self.h, self.w, 3)
            self.n += 1

    def grab(self, timeout=10):
        """Latest frame, cropped to game window. HxWx3 uint8 RGB."""
        t = time.time()
        while self.latest is None:
            if time.time() - t > timeout or self.proc.poll() is not None:
                raise RuntimeError(
                    "no frames from gst_pipe. If the helper is still alive, the screen is not updating: "
                    "a frozen/hung fullscreen game produces no new frames. Otherwise see stderr.")
            time.sleep(0.01)
        y0, y1, x0, x1 = self.region
        return self.latest[y0:y1, x0:x1]

    def stream(self, fps=FPS):
        dt = 1 / fps
        while True:
            t = time.time()
            yield self.grab()
            time.sleep(max(0, dt - (time.time() - t)))

    def close(self):
        self.proc.terminate()
        if self.overlay:
            self.overlay.terminate()


if __name__ == "__main__":
    from PIL import Image
    for wid, title in list_windows():
        print(wid, title)
    cap = Capture()
    print("window", cap.wid, "region", cap.region, "stream", cap.w, cap.h)
    f = cap.grab()
    Path("logs").mkdir(exist_ok=True)
    Image.fromarray(f).save("logs/frame.png")
    print("saved logs/frame.png", f.shape, "mean", round(float(f.mean()), 1))
    t0, n0 = time.time(), cap.n
    time.sleep(3)
    print("source fps %.1f" % ((cap.n - n0) / (time.time() - t0)))
    cap.close()
    sys.exit(0)
