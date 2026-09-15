#!/usr/bin/python3
"""Runs under SYSTEM python3 (has gi). Streams raw RGB frames from a portal
PipeWire node to stdout. Spawned by capture.py; not for direct use.
usage: gst_pipe.py FD NODE WIDTH HEIGHT FPS
"""
import sys

import gi

gi.require_version("Gst", "1.0")
from gi.repository import GLib, Gst  # noqa: E402

fd, node, w, h, fps = map(int, sys.argv[1:6])
Gst.init(None)
pipe = Gst.parse_launch(
    f"pipewiresrc fd={fd} path={node} do-timestamp=true ! videorate drop-only=true "
    f"! video/x-raw,framerate={fps}/1 ! videoconvert ! videoscale "
    f"! video/x-raw,format=RGB,width={w},height={h},pixel-aspect-ratio=1/1 "
    f"! fdsink fd=1 sync=false"
)
loop = GLib.MainLoop()


def on_msg(_bus, msg):
    if msg.type in (Gst.MessageType.ERROR, Gst.MessageType.EOS):
        print("gst_pipe:", msg.type, msg.parse_error() if msg.type == Gst.MessageType.ERROR else "", file=sys.stderr)
        loop.quit()


bus = pipe.get_bus()
bus.add_signal_watch()
bus.connect("message", on_msg)
pipe.set_state(Gst.State.PLAYING)
try:
    loop.run()
finally:
    pipe.set_state(Gst.State.NULL)
