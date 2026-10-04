"""Phase 21: a self-contained replay page with (a) the fly visual system's activity on a recorded clip, and (b) the bot's
state/decision timeline from a run log. No external resources: open the html file straight from disk.

  python export_replay.py --session data/corpus/n2_a --frames 3600:3720 --run logs/run_20261002-141947.jsonl --out replay/bonnie.html

The brain maps are the flyvis network's own cell activity (perception.Perception, same model as the live bot's shadow mode),
baseline-subtracted, one hexagon per cell on the 721-cell hex lattice, per cell type.
"""
import argparse
import base64
import io
import json
from pathlib import Path

import numpy as np

# retina -> lamina -> medulla -> motion detectors: a row per stage
from fly_brain import JPEG_W, VIEW_TYPES as CELL_TYPES


def extract_brain(session, f0, f1, types=CELL_TYPES, warmup=10):
    """-> dict with hex coordinates and per-frame int8 activity per cell type (scaled by the clip's 99th percentile)."""
    from PIL import Image
    from perception import Perception

    d = Path(session)
    meta = json.loads((d / "meta.json").read_text())
    p = Perception()
    c = p.net.connectome
    u, v = np.asarray(c.nodes.u[:]), np.asarray(c.nodes.v[:])
    idx = {t: np.asarray(c.nodes.layer_index[t][:]) for t in types}
    grid = np.stack([u[idx[types[0]]], v[idx[types[0]]]], 1)
    for t in types:  # every cell type sits on the same lattice, in the same order
        assert (np.stack([u[idx[t]], v[idx[t]]], 1) == grid).all(), t

    def load(i):
        return Image.open(d / "frames" / f"{i:06d}.png").convert("RGB")

    for i in range(max(0, f0 - warmup), f0):
        p.step(np.asarray(load(i)))  # let the network leave its resting state before recording
    raw, jpegs = [], []
    for i in range(f0, f1):
        im = load(i)
        p.step(np.asarray(im))
        act = (p.state.nodes.activity - p.baseline)[0].numpy()
        raw.append({t: act[idx[t]] for t in types})
        buf = io.BytesIO()
        im.resize((JPEG_W, int(JPEG_W * im.height / im.width))).save(buf, "JPEG", quality=70)
        jpegs.append(base64.b64encode(buf.getvalue()).decode())
    # show each cell's change relative to its own average over the clip: the steady brightness offset of a dark room would
    # otherwise paint the first stages one flat colour
    dev = {t: np.stack([r[t] for r in raw]) for t in types}
    dev = {t: a - a.mean(axis=0) for t, a in dev.items()}
    scale = {t: float(np.percentile(np.abs(a), 99)) or 1.0 for t, a in dev.items()}
    q = {t: np.clip(np.round(dev[t] / scale[t] * 127), -127, 127).astype(int).tolist() for t in types}
    return {"types": types, "grid": grid.tolist(), "act": q, "jpegs": jpegs, "t0": meta["times"][f0],
            "times": meta["times"][f0:f1], "frames": [f0, f1]}


def frame_info(session, f0, f1):
    """Per-frame readings from the same readers the bot uses: hour, power, hall scores (pixel baseline), labels."""
    import csv

    import config
    from geometry import load_buttons
    from PIL import Image
    from readers import hallway as H
    from readers.buttons import ButtonReader
    from readers.digits import DigitReader, _geom

    d = Path(session)
    meta = json.loads((d / "meta.json").read_text())
    geom, buttons = _geom(meta), load_buttons(config.CALIBRATION_PATH)
    digits, btn, hall = DigitReader(geom, buttons), ButtonReader(geom, buttons), H.HallwayDetector(geom, buttons)
    labels = {}
    if (d / "labels.csv").exists():
        labels = {int(r["frame"]): r for r in csv.DictReader(open(d / "labels.csv"))}
    out = []
    for i in range(f0, f1):
        f = np.asarray(Image.open(d / "frames" / f"{i:06d}.png").convert("RGB"))
        r = btn.read(f, 0)
        row = {"hour": digits.read_hour(f), "power": digits.read_power(f), "monitor": r.monitor_up,
               "hallL": hall.score(f, "L", (r.light_on or {}).get("L")), "hallR": hall.score(f, "R", (r.light_on or {}).get("R"))}
        if i in labels:
            row["label"] = {k: labels[i][k] for k in ("hall_L", "hall_R") if labels[i].get(k)}
        out.append(row)
    return out


def load_run(path):
    rows = [json.loads(l) for l in open(path)]
    rows = [r for r in rows if "header" not in r]
    keys = ("t", "hour", "power_pct", "usage", "monitor_up", "door_closed", "light_on", "cam", "hall", "action", "reason", "guard")
    # keep every 5th quiet row, every row with an action
    return [{k: r.get(k) for k in keys} for i, r in enumerate(rows) if i % 5 == 0 or r.get("action") != "NONE" or r.get("guard")]


PAGE = r"""<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Fly brain plays FNaF</title>
<style>
:root{--bg:#0e1116;--panel:#161b22;--ink:#e6edf3;--mute:#8b98a5;--line:#2a313c;--acc:#58a6ff;--warn:#f0883e;--good:#3fb950}
@media (prefers-color-scheme: light){:root{--bg:#f6f8fa;--panel:#fff;--ink:#1f2328;--mute:#59636e;--line:#d1d9e0;--acc:#0969da;--warn:#bc4c00;--good:#1a7f37}}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);font:14px/1.45 system-ui,sans-serif}
main{max-width:1180px;margin:0 auto;padding:16px}h1{font-size:20px;margin:0 0 2px}.sub{color:var(--mute);margin:0 0 14px}
.panel{background:var(--panel);border:1px solid var(--line);border-radius:10px;padding:12px;margin-bottom:14px}
.row{display:flex;gap:14px;flex-wrap:wrap}.game{flex:0 0 340px}.game img{width:100%;border-radius:6px;display:block;background:#000}
.maps{flex:1 1 560px;display:grid;grid-template-columns:repeat(4,1fr);gap:8px}.maps figure{margin:0}
.maps canvas{width:100%;aspect-ratio:1;background:#05070a;border-radius:6px;display:block}
figcaption{font-size:12px;color:var(--mute);text-align:center}figcaption b{color:var(--ink)}
.bar{display:flex;align-items:center;gap:10px;margin-top:10px;flex-wrap:wrap}input[type=range]{flex:1;min-width:160px}
button{background:var(--panel);color:var(--ink);border:1px solid var(--line);border-radius:6px;padding:4px 12px;cursor:pointer}
.stats{display:flex;gap:16px;flex-wrap:wrap;margin-top:8px;font-variant-numeric:tabular-nums}.stats span{color:var(--mute)}.stats b{color:var(--ink)}
canvas.tl{width:100%;height:230px;display:block}.legend{font-size:12px;color:var(--mute);margin-top:6px}
#tip{position:fixed;pointer-events:none;background:var(--panel);border:1px solid var(--line);padding:4px 8px;border-radius:6px;font-size:12px;display:none}
footer{color:var(--mute);font-size:12px;margin-top:6px}
</style></head><body><main>
<h1>Fly brain plays Five Nights at Freddy's</h1>
<p class="sub">Left: the game. Right: the activity of a real connectome-constrained fly visual system (flyvis, Lappalainen et al. 2024, <i>Nature</i>) watching it. One hexagon per cell; red = above that cell's average over the clip, blue = below, so the maps show what is changing.</p>
<section class="panel" id="brain"><div class="row"><div class="game"><img id="shot" alt="game frame"><div class="stats" id="stats"></div></div>
<div class="maps" id="maps"></div></div>
<div class="bar"><button id="play">Play</button><input id="seek" type="range" min="0" max="0" value="0"><span id="clock"></span></div>
<div class="legend" id="note"></div></section>
<section class="panel" id="botpanel"><b>The bot's night</b><div class="legend">Power, camera/door/light state and every decision. Hover for the reason code.</div>
<canvas class="tl" id="tl"></canvas></section>
<footer>Fly model: flyvis (TuragaLab). Replay generated offline by export_replay.py; no network resources.</footer>
<div id="tip"></div></main>
<script>
const D=__DATA__;
const $=id=>document.getElementById(id);
function setup(){
 const B=D.brain; if(!B){$('brain').style.display='none';} else {
 const maps=$('maps'); B.cv={};
 B.types.forEach(t=>{const f=document.createElement('figure');const c=document.createElement('canvas');c.width=c.height=240;f.appendChild(c);
  const cap=document.createElement('figcaption');cap.innerHTML='<b>'+t+'</b> '+(D.names[t]||'');f.appendChild(cap);maps.appendChild(f);B.cv[t]=c;});
 const g=B.grid,xs=g.map(([u,v])=>v+0.5*u),ys=g.map(([u,v])=>u*0.8660254);
 B.x0=Math.min(...xs);B.x1=Math.max(...xs);B.y0=Math.min(...ys);B.y1=Math.max(...ys);B.xs=xs;B.ys=ys;
 $('seek').max=B.jpegs.length-1; $('note').textContent=D.note||'';
 draw(0);}
 if(!D.run||!D.run.length){$('botpanel').style.display='none';} else drawTL();
}
function col(v){ // -127..127 -> diverging blue/dark/red
 const a=Math.min(1,Math.abs(v)/127);const k=Math.pow(a,0.6);
 return v>=0?`rgb(${Math.round(20+235*k)},${Math.round(24+70*k)},${Math.round(30)})`:`rgb(${Math.round(20)},${Math.round(40+90*k)},${Math.round(40+215*k)})`;}
function draw(i){
 const B=D.brain;$('shot').src='data:image/jpeg;base64,'+B.jpegs[i];
 const W=240,pad=8,sx=(W-2*pad)/(B.x1-B.x0),sy=(W-2*pad)/(B.y1-B.y0),s=Math.min(sx,sy),r=s*0.55;
 B.types.forEach(t=>{const c=B.cv[t].getContext('2d');c.fillStyle='#05070a';c.fillRect(0,0,W,W);const a=B.act[t][i];
  for(let k=0;k<a.length;k++){const x=pad+(B.xs[k]-B.x0)*s+(W-2*pad-(B.x1-B.x0)*s)/2,y=pad+(B.ys[k]-B.y0)*s+(W-2*pad-(B.y1-B.y0)*s)/2;
   c.fillStyle=col(a[k]);c.beginPath();for(let j=0;j<6;j++){const an=Math.PI/3*j+Math.PI/6;c.lineTo(x+r*Math.cos(an),y+r*Math.sin(an));}c.closePath();c.fill();}});
 const f=D.info&&D.info[i],tt=(B.times[i]-B.t0).toFixed(1);
 $('clock').textContent='frame '+(B.frames[0]+i)+'  ·  '+tt+' s';
 if(f){const lab=f.label?Object.entries(f.label).map(([k,v])=>k+'='+v).join(' '):'';
  $('stats').innerHTML=['hour',f.hour==null?'?':f.hour,'power',f.power==null?'?':f.power+'%','monitor',f.monitor?'up':'down',
   'hall L',f.hallL==null?'dark':f.hallL.toFixed(2),'hall R',f.hallR==null?'dark':f.hallR.toFixed(2)].reduce((h,v,k,a)=>k%2?h:h+'<div><span>'+v+'</span> <b>'+a[k+1]+'</b></div>','')+(lab?'<div><span>label</span> <b>'+lab+'</b></div>':'');}
 $('seek').value=i;
}
let timer=null;
$('play').onclick=()=>{ if(timer){clearInterval(timer);timer=null;$('play').textContent='Play';return;}
 $('play').textContent='Pause';timer=setInterval(()=>{let i=(+$('seek').value+1)%D.brain.jpegs.length;draw(i);},100);};
$('seek').oninput=e=>draw(+e.target.value);
function drawTL(){
 const R=D.run,cv=$('tl'),dpr=window.devicePixelRatio||1,W=cv.clientWidth,H=cv.clientHeight;cv.width=W*dpr;cv.height=H*dpr;
 const c=cv.getContext('2d');c.scale(dpr,dpr);const cs=getComputedStyle(document.documentElement);const ink=cs.getPropertyValue('--ink'),mute=cs.getPropertyValue('--mute'),line=cs.getPropertyValue('--line'),acc=cs.getPropertyValue('--acc'),warn=cs.getPropertyValue('--warn'),good=cs.getPropertyValue('--good');
 const T=R[R.length-1].t,L=44,Rm=10,top=8,pw=H*0.5,x=t=>L+(W-L-Rm)*t/T,y=p=>top+pw*(1-p/100);
 c.font='11px system-ui';c.fillStyle=mute;c.strokeStyle=line;
 [0,50,100].forEach(p=>{c.beginPath();c.moveTo(L,y(p));c.lineTo(W-Rm,y(p));c.stroke();c.fillText(p+'%',6,y(p)+4);});
 c.strokeStyle=acc;c.lineWidth=2;c.beginPath();let st=false;R.forEach(r=>{if(r.power_pct==null)return;st?c.lineTo(x(r.t),y(r.power_pct)):c.moveTo(x(r.t),y(r.power_pct));st=true;});c.stroke();c.lineWidth=1;
 // hour ticks
 let lh=null,lx=-99;R.forEach(r=>{if(r.hour!=null&&r.hour!==lh){lh=r.hour;c.strokeStyle=line;c.beginPath();c.moveTo(x(r.t),top);c.lineTo(x(r.t),top+pw);c.stroke();if(x(r.t)-lx>48){c.fillStyle=mute;c.fillText((r.hour||12)+' AM',x(r.t)+3,top+11);lx=x(r.t);}}});
 // state bands
 const bands=[['monitor',r=>r.monitor_up,acc],['door L',r=>r.door_closed&&r.door_closed.L,warn],['door R',r=>r.door_closed&&r.door_closed.R,warn],['light L',r=>r.light_on&&r.light_on.L,good],['light R',r=>r.light_on&&r.light_on.R,good]];
 const by=top+pw+14,bh=(H-by-26)/bands.length;
 bands.forEach(([n,f,k],j)=>{c.fillStyle=mute;c.fillText(n,2,by+j*bh+bh*0.75);c.fillStyle=k;R.forEach((r,i)=>{if(f(r)){const x0=x(r.t),x1=i+1<R.length?x(R[i+1].t):x0+2;c.fillRect(x0,by+j*bh+1,Math.max(1.5,x1-x0),bh-2);}});});
 // decisions
 D.marks=[];R.forEach(r=>{if(r.action&&r.action!=='NONE'){const xx=x(r.t),yy=H-12;c.fillStyle=r.reason&&r.reason.startsWith('threat')?warn:ink;c.fillRect(xx-1,yy-6,2,12);D.marks.push([xx,r]);}
  if(r.hall){['L','R'].forEach(s=>{if(r.hall[s]!=null&&r.hall[s]>0.5){c.fillStyle=warn;c.beginPath();c.arc(x(r.t),top+pw+6,3,0,7);c.fill();}});}});
 cv.onmousemove=e=>{const b=cv.getBoundingClientRect(),mx=e.clientX-b.left;let best=null,bd=9;D.marks.forEach(([xx,r])=>{if(Math.abs(xx-mx)<bd){bd=Math.abs(xx-mx);best=r;}});const tip=$('tip');
  if(best){tip.style.display='block';tip.style.left=e.clientX+12+'px';tip.style.top=e.clientY+12+'px';tip.textContent=best.t.toFixed(1)+' s  '+best.action+(best.arg?' '+best.arg:'')+'  ('+best.reason+')'+(best.guard?'  guard: '+best.guard:'');}else tip.style.display='none';};
 cv.onmouseleave=()=>$('tip').style.display='none';
}
setup();window.addEventListener('resize',()=>{if(D.run&&D.run.length)drawTL();});
</script></body></html>"""

NAMES = {"R1": "photoreceptor", "L1": "lamina, on/off contrast", "L2": "lamina, on/off contrast", "Mi1": "medulla, ON pathway",
         "T4a": "ON motion detector", "T4b": "ON motion detector", "T4c": "ON motion detector", "T4d": "ON motion detector",
         "T5a": "OFF motion detector", "T5b": "OFF motion detector", "T5c": "OFF motion detector", "T5d": "OFF motion detector"}


def build(session=None, frames=None, run=None, note=""):
    data = {"names": NAMES, "note": note}
    if session:
        f0, f1 = frames
        data["brain"] = extract_brain(session, f0, f1)
        data["info"] = frame_info(session, f0, f1)
    if run:
        data["run"] = load_run(run)
    return PAGE.replace("__DATA__", json.dumps(data, separators=(",", ":")))


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--session", help="corpus session dir, e.g. data/corpus/n2_a")
    ap.add_argument("--frames", default="3600:3720", help="start:stop frame indices of the clip")
    ap.add_argument("--run", help="bot run log (jsonl) for the timeline panel")
    ap.add_argument("--out", default="replay/replay.html")
    ap.add_argument("--note", default="")
    a = ap.parse_args(argv)
    if not (a.session or a.run):
        ap.error("need --session and/or --run")
    f0, f1 = map(int, a.frames.split(":"))
    html = build(a.session, (f0, f1), a.run, a.note)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(html)
    print(f"wrote {a.out} ({len(html) / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
