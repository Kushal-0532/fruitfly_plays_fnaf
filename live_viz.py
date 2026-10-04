"""Live view of the fly brain while the bot plays: python main.py --night 1 --viz  ->  http://localhost:8765
Stdlib HTTP server in a thread; the page polls /state a few times a second. Read-only: it never touches the game."""
import json
import threading
from collections import deque
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import config
from export_replay import NAMES

PAGE = r"""<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Fly brain, live</title>
<style>
:root{--bg:#0e1116;--panel:#161b22;--ink:#e6edf3;--mute:#8b98a5;--line:#2a313c;--acc:#58a6ff;--warn:#f0883e;--good:#3fb950;--bad:#f85149}
@media (prefers-color-scheme: light){:root{--bg:#f6f8fa;--panel:#fff;--ink:#1f2328;--mute:#59636e;--line:#d1d9e0;--acc:#0969da;--warn:#bc4c00;--good:#1a7f37;--bad:#cf222e}}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);font:14px/1.45 system-ui,sans-serif}
main{max-width:1240px;margin:0 auto;padding:14px}h1{font-size:19px;margin:0}.sub{color:var(--mute);margin:2px 0 12px}
.panel{background:var(--panel);border:1px solid var(--line);border-radius:10px;padding:12px}
.grid{display:grid;grid-template-columns:350px 1fr;gap:12px}@media(max-width:900px){.grid{grid-template-columns:1fr}}
.game img{width:100%;border-radius:6px;display:block;background:#000}
.maps{display:grid;grid-template-columns:repeat(4,1fr);gap:8px}.maps figure{margin:0}.maps canvas{width:100%;aspect-ratio:1;background:#05070a;border-radius:6px;display:block}
figcaption{font-size:11px;color:var(--mute);text-align:center}figcaption b{color:var(--ink)}
.stats{display:grid;grid-template-columns:1fr 1fr;gap:2px 12px;margin-top:8px;font-variant-numeric:tabular-nums}.stats span{color:var(--mute)}
.danger{display:grid;grid-template-columns:1fr 1fr;gap:12px;margin-top:12px}
.meter{height:14px;border-radius:7px;background:var(--line);overflow:hidden}.meter i{display:block;height:100%;width:0;background:var(--good);transition:width .12s}
.neurons{display:flex;gap:6px;margin-top:8px}.neurons div{flex:1;text-align:center;font-size:11px;color:var(--mute)}
.neurons i{display:block;height:46px;position:relative;background:var(--line);border-radius:4px;overflow:hidden}.neurons i b{position:absolute;left:0;right:0;bottom:50%;background:var(--warn)}
canvas.spark{width:100%;height:70px;display:block}
#log{height:150px;overflow:auto;font:12px ui-monospace,monospace;color:var(--mute);margin-top:8px}#log div b{color:var(--ink)}#log .threat b{color:var(--bad)}
h2{font-size:13px;margin:0 0 4px}.tag{font-size:11px;color:var(--mute)}
</style></head><body><main>
<h1>Fly brain plays Five Nights at Freddy's, live</h1>
<p class="sub">A connectome-constrained fly visual system (flyvis) watches the game. Hexagons = cells (red above that cell's own average, blue below). The outlined patches are the parts of its retina that look at each hallway.</p>
<div class="grid"><div class="panel game"><img id="shot" alt=""><div class="stats" id="stats"></div><div id="log"></div></div>
<div class="panel"><div class="maps" id="maps"></div>
<div class="danger" id="danger"></div></div></div></main>
<script>
const $=id=>document.getElementById(id),NAMES=__NAMES__;let built=false,hist={L:[],R:[]};const RT=['R1','L1','L2','Mi1'];
function col(v){const a=Math.min(1,Math.abs(v)/127),k=Math.pow(a,0.6);return v>=0?`rgb(${20+235*k|0},${24+70*k|0},30)`:`rgb(20,${40+90*k|0},${40+215*k|0})`;}
function build(s){const m=$('maps');Object.keys(s.maps).forEach(t=>{const f=document.createElement('figure'),c=document.createElement('canvas');c.width=c.height=220;c.id='m_'+t;f.appendChild(c);
 const g=document.createElement('figcaption');g.innerHTML='<b>'+t+'</b> '+(NAMES[t]||'');f.appendChild(g);m.appendChild(f);});
 const d=$('danger');['L','R'].forEach(x=>{const e=document.createElement('div');e.innerHTML='<h2>'+(x=='L'?'Left':'Right')+' hallway <span class="tag" id="t'+x+'"></span></h2><div class="meter"><i id="mt'+x+'"></i></div>'+
  '<div class="neurons" id="n'+x+'">'+RT.map(t=>'<div>'+t+'<i><b></b></i></div>').join('')+'</div><canvas class="spark" id="sp'+x+'"></canvas>';d.appendChild(e);});built=true;}
function hexes(s){const g=s.grid,xs=g.map(([u,v])=>v+0.5*u),ys=g.map(([u,v])=>u*0.8660254);const x0=Math.min(...xs),x1=Math.max(...xs),y0=Math.min(...ys),y1=Math.max(...ys);
 const W=220,p=8,sc=Math.min((W-2*p)/(x1-x0),(W-2*p)/(y1-y0)),r=sc*0.55,ox=(W-(x1-x0)*sc)/2,oy=(W-(y1-y0)*sc)/2;return {xs,ys,x0,y0,sc,r,ox,oy,W};}
function draw(s){const H=hexes(s),inR={L:new Set(s.regions.L),R:new Set(s.regions.R)};
 Object.entries(s.maps).forEach(([t,a])=>{const c=$('m_'+t).getContext('2d');c.fillStyle='#05070a';c.fillRect(0,0,H.W,H.W);
  for(let k=0;k<a.length;k++){const x=H.ox+(H.xs[k]-H.x0)*H.sc,y=H.oy+(H.ys[k]-H.y0)*H.sc;c.fillStyle=col(a[k]);c.beginPath();for(let j=0;j<6;j++){const an=Math.PI/3*j+Math.PI/6;c.lineTo(x+H.r*Math.cos(an),y+H.r*Math.sin(an));}c.closePath();c.fill();
   if(inR.L.has(k)||inR.R.has(k)){c.strokeStyle=inR.L.has(k)?'#58a6ff':'#f0883e';c.lineWidth=1.2;c.stroke();}}});}
function spark(id,arr,thr){const c=$(id),dpr=devicePixelRatio||1,W=c.clientWidth,Hh=c.clientHeight;c.width=W*dpr;c.height=Hh*dpr;const x=c.getContext('2d');x.scale(dpr,dpr);
 const cs=getComputedStyle(document.documentElement);x.strokeStyle=cs.getPropertyValue('--line');x.beginPath();x.moveTo(0,Hh*(1-thr));x.lineTo(W,Hh*(1-thr));x.stroke();
 x.strokeStyle=cs.getPropertyValue('--warn');x.lineWidth=1.6;x.beginPath();arr.forEach((v,i)=>{const xx=W*i/Math.max(1,arr.length-1),yy=Hh*(1-(v==null?0:v));i?x.lineTo(xx,yy):x.moveTo(xx,yy);});x.stroke();}
function update(s){
 if(!s.maps){return;} if(!built)build(s); $('shot').src='data:image/jpeg;base64,'+s.jpeg; draw(s);
 ['L','R'].forEach(x=>{const d=s.danger[x];hist[x].push(d);if(hist[x].length>150)hist[x].shift();
  $('mt'+x).style.width=(d==null?0:d*100)+'%';$('mt'+x).style.background=d!=null&&d>s.thresh?'var(--bad)':'var(--good)';
  $('t'+x).textContent=d==null?'(light off: fly sees a dark hall, no verdict)':'danger '+d.toFixed(2);
  [...$('n'+x).children].forEach((e,k)=>{const v=s.contrib[x][k],h=Math.min(1,Math.abs(v)/3)*50,b=e.querySelector('b');b.style.height=h+'%';b.style[v>=0?'bottom':'top']='50%';b.style[v>=0?'top':'bottom']='auto';});
  spark('sp'+x,hist[x],s.thresh);});
 const b=s.bot||{};$('stats').innerHTML=[['hour',b.hour],['power',b.power_pct==null?null:b.power_pct+'%'],['monitor',b.monitor_up?'up':'down'],['cove',s.cove==null?null:s.cove.toFixed(2)],['usage',b.usage],
  ['door L',b.door_closed&&(b.door_closed.L?'CLOSED':'open')],['door R',b.door_closed&&(b.door_closed.R?'CLOSED':'open')],['light L',b.light_on&&(b.light_on.L?'on':'off')],['light R',b.light_on&&(b.light_on.R?'on':'off')],['fly step',s.ms+' ms'],['t',b.t==null?null:b.t.toFixed(0)+' s']]
  .map(([k,v])=>'<div><span>'+k+'</span> '+(v==null?'?':v)+'</div>').join('');
 $('log').innerHTML=(s.log||[]).slice().reverse().map(r=>'<div class="'+(r.reason.startsWith('threat')?'threat':'')+'">'+r.t.toFixed(1)+' s <b>'+r.action+(r.arg?' '+r.arg:'')+'</b> '+r.reason+'</div>').join('');}
async function tick(){try{const r=await fetch('/state');update(await r.json());}catch(e){}setTimeout(tick,150);}tick();
</script></body></html>"""


class LiveViz:
    def __init__(self, brain, port=8765, thresh=None):
        self.brain, self.port, self.thresh = brain, port, thresh if thresh is not None else config.HALL_THRESH
        self.bot, self.log, self.lock = {}, deque(maxlen=40), threading.Lock()
        viz = self

        class H(BaseHTTPRequestHandler):
            def log_message(self, *a):
                pass

            def do_GET(self):
                if self.path.startswith("/state"):
                    body = json.dumps(viz.state()).encode()
                    ctype = "application/json"
                else:
                    body = PAGE.replace("__NAMES__", json.dumps(NAMES)).encode()
                    ctype = "text/html; charset=utf-8"
                self.send_response(200)
                self.send_header("Content-Type", ctype)
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

        self.server = ThreadingHTTPServer(("127.0.0.1", port), H)
        threading.Thread(target=self.server.serve_forever, daemon=True).start()

    def update(self, state, row):
        """Called by main.run every loop with the tracked state and the log row."""
        with self.lock:
            self.bot = {"t": row["t"], "hour": state.hour, "power_pct": state.power_pct, "usage": state.usage, "monitor_up": state.monitor_up,
                        "door_closed": state.door_closed, "light_on": state.light_on}
            if row["action"] != "NONE":
                self.log.append({"t": row["t"], "action": row["action"], "arg": row["arg"], "reason": row["reason"]})

    def state(self):
        snap = self.brain.snapshot() or {}
        with self.lock:
            return {**snap, "bot": self.bot, "log": list(self.log), "thresh": self.thresh}

    def close(self):
        self.server.shutdown()
