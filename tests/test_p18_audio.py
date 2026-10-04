import io
import json
import subprocess

import numpy as np

import audio
from audio import Event, OnsetDetector, evaluate_events, find_monitor_target, record_chunks

RATE, CH = 16000, 1600


def signal(seconds, bursts, seed=0, noise=30):
    """bursts: (t0, dur, amp_L, amp_R) of a 400 Hz tone over low white noise."""
    rnd = np.random.RandomState(seed)
    n = int(seconds * RATE)
    x = rnd.randn(n, 2) * noise
    tt = np.arange(n) / RATE
    for t0, dur, aL, aR in bursts:
        m = (tt >= t0) & (tt < t0 + dur)
        tone = np.sin(2 * np.pi * 400 * tt[m])
        x[m, 0] += aL * tone
        x[m, 1] += aR * tone
    return np.clip(x, -32768, 32767).astype(np.int16)


def run(x):
    det, ev = OnsetDetector(), []
    for i in range(0, len(x) - CH + 1, CH):
        ev += det.feed(x[i:i + CH], i / RATE)
    return ev


def test_left_right_both():
    for amps, side in [((8000, 0), "L"), ((0, 8000), "R"), ((8000, 8000), "B")]:
        ev = run(signal(4, [(1.95, 0.3, *amps)]))
        assert len(ev) == 1 and ev[0].side == side and abs(ev[0].t - 1.95) < 0.2, (side, ev)


def test_noise_no_events():
    assert len(run(signal(60, [], noise=500))) < 1


def test_refractory():
    assert len(run(signal(4, [(1.0, 0.3, 8000, 0), (1.4, 0.3, 8000, 0)]))) == 1


def test_evaluate_events():
    ev = [Event(1.0, "L", 0, 5), Event(10.0, "R", 0, 5), Event(30.0, "L", 0, 5)]
    p, r, fpm = evaluate_events(ev, [1.5, 10.2, 20.0], duration_s=60)
    assert (p, r) == (2 / 3, 2 / 3) and fpm == 1.0


PIPEWIRE_DUMP = json.dumps([{"type": "PipeWire:Interface:Node", "info": {}},
                            {"type": "PipeWire:Interface:Metadata",
                             "metadata": [{"key": "default.audio.sink", "value": {"name": "alsa_output.pci.analog"}}]}])
WPCTL = "Audio\n ├─ Sinks:\n │      46. HDMI [vol: 1.0]\n │  *   47. Built-in Audio Analog Stereo [vol: 0.40]\n"


def fake_run(out):
    return lambda argv, **kw: subprocess.CompletedProcess(argv, 0, stdout=out, stderr="")


def test_find_monitor_target():
    assert find_monitor_target(fake_run(PIPEWIRE_DUMP)) == "alsa_output.pci.analog"
    def wp_only(argv, **kw):
        return subprocess.CompletedProcess(argv, 0, stdout=WPCTL if argv[0] == "wpctl" else "[]", stderr="")
    assert find_monitor_target(wp_only) == "47"
    assert find_monitor_target(fake_run("")) is None


def test_record_chunks_shape():
    data = signal(0.5, []).tobytes()

    class P:
        stdout = io.BytesIO(data)
        def terminate(self): pass
    seen = []
    chunks = list(record_chunks("t", run=lambda cmd, **kw: seen.append(cmd) or P()))
    assert len(chunks) == 5 and all(c.shape == (1600, 2) and c.dtype == np.int16 for c in chunks)
    assert "pw-record" in seen[0][0]


def test_monitor_gate_and_forward():
    from actions import Action
    mon = audio.AudioMonitor("t", enabled=False, chunks=lambda t: iter([]))
    mon.q.put(Event(1, "L", 0, 5))
    assert mon.poll() == []
    assert mon.apply_gate(0.5, 0) is False
    got = []
    sup = type("S", (), {"note_audio": lambda self, s, t: got.append(s)})()
    audio.forward([Event(1, "B", 0, 5), Event(2, "L", 0, 5)], sup, 3.0)
    assert got == ["L", "R", "L"]
