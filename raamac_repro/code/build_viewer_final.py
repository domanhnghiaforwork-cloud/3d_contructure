"""Viewer 3D MỘT cloud hoàn thiện duy nhất (week{N}_final.ply) + tham chiếu để đối chiếu.

Không có toggle trước/sau căn chỉnh — chỉ một bản final (đã căn chỉnh, mét thật, đa shot).
Cloud tham chiếu chuyên nghiệp bật/tắt bằng checkbox 'Chồng tham chiếu' để người xem tự đánh giá.
Tự chứa, WebGL thuần, không CDN. Mở file:// là chạy.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
PKG = HERE.parent
sys.path.insert(0, str(HERE))
from build_viewer_6clouds import load_reduce, quantize  # noqa: E402

REF_PLY = PKG / "data" / "raw" / "week26.ply"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--week", type=int, required=True)
    ap.add_argument("--max-points", type=int, default=150000)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()
    import build_viewer_6clouds as bv
    bv.MAX_POINTS = args.max_points
    out = HERE.parent / "artifacts" / f"week{args.week}"
    final = out / f"week{args.week}_final.ply"
    if not final.exists():
        raise SystemExit(f"Thiếu {final} — chạy merge_shots.py trước")
    OUT = args.out or (PKG / "viewer" / f"viewer_final_week{args.week}.html")

    fxyz, frgb, fn, fd = load_reduce(final)
    rxyz, rrgb, rn, rd = load_reduce(REF_PLY)
    allpts = np.concatenate([fxyz, rxyz])
    c = (allpts.max(0) + allpts.min(0)) / 2
    e = float((allpts.max(0) - allpts.min(0)).max())
    fp, fc, fgz = quantize(fxyz, frgb, c, e)
    rp, rc, rgz = quantize(rxyz, rrgb, c, e)
    merge = json.loads((out / "merge_summary.json").read_text()) if (out / "merge_summary.json").exists() else {}
    clouds = [
        dict(name=f"1 · week{args.week} FINAL (repro đa shot)",
             cfg=f"{merge.get('n_shots_merged','?')} shot video hợp nhất · đã căn chỉnh, mét thật",
             note="Bản hoàn thiện DUY NHẤT: dựng riêng từng shot video rồi hợp nhất cloud đã căn chỉnh "
                  "(đa góc). Bật 'Chồng tham chiếu' để tự đối chiếu với cloud chuyên nghiệp.",
             orig=fn, shown=len(fxyz), pos=fp, col=fc, gz=fgz),
        dict(name="2 · week26.ply chuyên nghiệp (tham chiếu)",
             cfg="Scan chuyên nghiệp RAAMAC — để ĐỐI CHIẾU, không phải bản repro",
             note="Cloud chuyên nghiệp phủ kín nhà đa góc. Dùng làm mốc so sánh trực quan.",
             orig=rn, shown=len(rxyz), pos=rp, col=rc, gz=rgz),
    ]
    html = HTML.replace("__DATA__", json.dumps({"clouds": clouds}, ensure_ascii=False, separators=(",", ":"))).replace("__WEEK__", str(args.week))
    OUT.write_text(html, encoding="utf-8")
    print(f"final {fn:,}->{len(fxyz):,} | ref {rn:,}->{len(rxyz):,}")
    print(f"viewer {OUT} ({OUT.stat().st_size/1e6:.2f} MB)")


HTML = r"""<!DOCTYPE html>
<html lang="vi"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Cloud hoàn thiện week__WEEK__</title>
<style>
html,body{margin:0;height:100%;overflow:hidden;background:#0e1116;color:#e6e6e6;font:14px/1.45 system-ui,Segoe UI,Roboto,sans-serif}
#c{position:absolute;inset:0;width:100%;height:100%;display:block;touch-action:none;cursor:grab}
#c:active{cursor:grabbing}
#bar{position:absolute;top:10px;left:10px;right:10px;display:flex;flex-wrap:wrap;gap:6px;align-items:center}
#bar button{background:#1c2330;color:#cfd8e3;border:1px solid #33405a;border-radius:8px;padding:7px 11px;cursor:pointer;font-size:13px}
#bar button:hover{background:#26304a}#bar button.on{background:#2f6feb;border-color:#2f6feb;color:#fff}
#bar .sep{width:1px;height:22px;background:#33405a;margin:0 3px}
#info{position:absolute;left:10px;bottom:10px;max-width:470px;background:rgba(14,17,22,.82);border:1px solid #2a3346;border-radius:10px;padding:10px 13px}
#info h2{margin:0 0 4px;font-size:15px;color:#7fb3ff}#info .m{color:#9aa7b8;font-size:12.5px}#info .note{margin-top:5px;font-size:13px}
#hint{position:absolute;right:10px;bottom:10px;background:rgba(14,17,22,.82);border:1px solid #2a3346;border-radius:10px;padding:8px 12px;font-size:12px;color:#9aa7b8}
label{font-size:12.5px;color:#9aa7b8;display:flex;gap:5px;align-items:center;user-select:none}
</style></head><body>
<canvas id="c"></canvas><div id="bar"></div><div id="info"></div>
<div id="hint">Kéo: xoay · Kéo phải/Shift+kéo: pan · Lăn: zoom<br>←/→ hoặc 1–2: chuyển cloud · O: chồng tham chiếu · R: đặt lại góc</div>
<script>
const DATA=__DATA__;const REF=DATA.clouds.length-1;
const canvas=document.getElementById('c');const gl=canvas.getContext('webgl',{antialias:true});
if(!gl){document.getElementById('info').innerHTML='<h2>Trình duyệt không hỗ trợ WebGL</h2>';throw new Error('no webgl');}
function sh(t,s){const x=gl.createShader(t);gl.shaderSource(x,s);gl.compileShader(x);if(!gl.getShaderParameter(x,gl.COMPILE_STATUS))throw new Error(gl.getShaderInfoLog(x));return x;}
function program(v,f){const p=gl.createProgram();gl.attachShader(p,sh(gl.VERTEX_SHADER,v));gl.attachShader(p,sh(gl.FRAGMENT_SHADER,f));gl.linkProgram(p);return p;}
const prog=program(`attribute vec3 aPos;attribute vec3 aCol;uniform mat4 uMVP;uniform float uSize;varying vec3 vCol;void main(){gl_Position=uMVP*vec4(aPos,1.0);gl_PointSize=uSize;vCol=aCol;}`,`precision mediump float;varying vec3 vCol;uniform float uAlpha;void main(){gl_FragColor=vec4(vCol,uAlpha);}`);
const loc={pos:gl.getAttribLocation(prog,'aPos'),col:gl.getAttribLocation(prog,'aCol'),mvp:gl.getUniformLocation(prog,'uMVP'),size:gl.getUniformLocation(prog,'uSize'),alpha:gl.getUniformLocation(prog,'uAlpha')};
const lprog=program(`attribute vec3 aPos;uniform mat4 uMVP;void main(){gl_Position=uMVP*vec4(aPos,1.0);}`,`precision mediump float;uniform vec4 uColor;void main(){gl_FragColor=uColor;}`);
const lloc={pos:gl.getAttribLocation(lprog,'aPos'),mvp:gl.getUniformLocation(lprog,'uMVP'),color:gl.getUniformLocation(lprog,'uColor')};
gl.enable(gl.DEPTH_TEST);gl.clearColor(0.055,0.067,0.086,1);
function b64bytes(s){const b=atob(s);const u=new Uint8Array(b.length);for(let i=0;i<b.length;i++)u[i]=b.charCodeAt(i);return u;}
const cache=[];
function gpu(i){if(cache[i])return cache[i];const m=DATA.clouds[i];const u=new Uint16Array(b64bytes(m.pos).buffer);const p=new Float32Array(u.length);for(let j=0;j<u.length;j++)p[j]=u[j]/32767.5-1;const pb=gl.createBuffer();gl.bindBuffer(gl.ARRAY_BUFFER,pb);gl.bufferData(gl.ARRAY_BUFFER,p,gl.STATIC_DRAW);const cb=gl.createBuffer();gl.bindBuffer(gl.ARRAY_BUFFER,cb);gl.bufferData(gl.ARRAY_BUFFER,b64bytes(m.col),gl.STATIC_DRAW);return cache[i]={pb,cb,n:m.shown};}
let helpB=null,helpN=0,helpGz=null;
function helpers(gz){if(helpB&&helpGz===gz)return;helpGz=gz;const v=[];for(let x=-1;x<=1.001;x+=0.2)v.push(x,-1,gz,x,1,gz);for(let y=-1;y<=1.001;y+=0.2)v.push(-1,y,gz,1,y,gz);helpN=v.length/3;v.push(-1,-1,gz,-0.4,-1,gz,-1,-1,gz,-1,-0.4,gz,-1,-1,gz,-1,-1,gz+0.6);helpB=gl.createBuffer();gl.bindBuffer(gl.ARRAY_BUFFER,helpB);gl.bufferData(gl.ARRAY_BUFFER,new Float32Array(v),gl.STATIC_DRAW);}
const sub=(a,b)=>[a[0]-b[0],a[1]-b[1],a[2]-b[2]];const dot=(a,b)=>a[0]*b[0]+a[1]*b[1]+a[2]*b[2];
const cross=(a,b)=>[a[1]*b[2]-a[2]*b[1],a[2]*b[0]-a[0]*b[2],a[0]*b[1]-a[1]*b[0]];
const norm=a=>{const l=Math.hypot(a[0],a[1],a[2]);return[a[0]/l,a[1]/l,a[2]/l];};
function mul(a,b){const o=new Float32Array(16);for(let c=0;c<4;c++)for(let r=0;r<4;r++){let s=0;for(let k=0;k<4;k++)s+=a[k*4+r]*b[c*4+k];o[c*4+r]=s;}return o;}
function persp(f,a,n,fa){const t=1/Math.tan(f/2);return new Float32Array([t/a,0,0,0,0,t,0,0,0,0,(fa+n)/(n-fa),-1,0,0,2*fa*n/(n-fa),0]);}
function lookAt(e,c,u){const z=norm(sub(e,c)),x=norm(cross(u,z)),y=cross(z,x);return new Float32Array([x[0],y[0],z[0],0,x[1],y[1],z[1],0,x[2],y[2],z[2],0,-dot(x,e),-dot(y,e),-dot(z,e),1]);}
let cur=0,yaw=0.7,pitch=0.35,dist=3.1,target=[0,0,0],dragging=false,dragMode='rotate',pinch=0;
const HOME={yaw:0.7,pitch:0.35,dist:3.1};const ptrs=new Map();
canvas.addEventListener('contextmenu',e=>e.preventDefault());
canvas.addEventListener('pointerdown',e=>{try{canvas.setPointerCapture(e.pointerId);}catch(_){}ptrs.set(e.pointerId,{x:e.clientX,y:e.clientY});dragging=true;pinch=0;dragMode=(e.button===2||e.shiftKey)?'pan':'rotate';});
canvas.addEventListener('pointermove',e=>{const p=ptrs.get(e.pointerId);if(!p)return;const dx=e.clientX-p.x,dy=e.clientY-p.y;p.x=e.clientX;p.y=e.clientY;if(ptrs.size===2){const[a,b]=[...ptrs.values()];const d=Math.hypot(a.x-b.x,a.y-b.y);if(pinch>0)dist=Math.max(0.4,Math.min(30,dist*pinch/d));pinch=d;return;}if(dragMode==='pan'){const f=[-Math.cos(pitch)*Math.cos(yaw),-Math.cos(pitch)*Math.sin(yaw),-Math.sin(pitch)];const r=norm(cross(f,[0,0,1])),u=cross(r,f),s=dist*0.0016;target=[target[0]-r[0]*dx*s+u[0]*dy*s,target[1]-r[1]*dx*s+u[1]*dy*s,target[2]-r[2]*dx*s+u[2]*dy*s];}else{yaw-=dx*0.005;pitch=Math.max(-1.5,Math.min(1.5,pitch+dy*0.005));}});
const up=e=>{ptrs.delete(e.pointerId);if(!ptrs.size)dragging=false;pinch=0;};
canvas.addEventListener('pointerup',up);canvas.addEventListener('pointercancel',up);
canvas.addEventListener('wheel',e=>{e.preventDefault();dist=Math.max(0.4,Math.min(30,dist*Math.exp(e.deltaY*0.0012)));},{passive:false});
const bar=document.getElementById('bar');
const btns=DATA.clouds.map((m,i)=>{const b=document.createElement('button');b.textContent=m.name;b.onclick=()=>show(i);bar.append(b);return b;});
bar.append(Object.assign(document.createElement('span'),{className:'sep'}));
const gridL=document.createElement('label');gridL.innerHTML='<input type="checkbox" id="grid" checked> Lưới & trục';bar.append(gridL);
const ovrL=document.createElement('label');ovrL.innerHTML='<input type="checkbox" id="ovr" checked> Chồng tham chiếu';bar.append(ovrL);
const spin=document.createElement('label');spin.innerHTML='<input type="checkbox" id="auto" checked> Tự xoay';bar.append(spin);
const reset=document.createElement('button');reset.textContent='Đặt lại góc';reset.onclick=()=>{yaw=HOME.yaw;pitch=HOME.pitch;dist=HOME.dist;target=[0,0,0];};bar.append(reset);
function show(i){cur=i;btns.forEach((b,j)=>b.classList.toggle('on',j===i));const m=DATA.clouds[i];document.getElementById('ovr').disabled=(i===REF);document.getElementById('info').innerHTML=`<h2>${m.name}</h2><div class="m">${m.cfg}</div><div class="m">Điểm gốc: ${m.orig.toLocaleString('vi-VN')} · Hiển thị: ${m.shown.toLocaleString('vi-VN')}</div><div class="note">${m.note}</div>`;helpers(m.gz);gpu(i);}
addEventListener('keydown',e=>{if(e.key==='ArrowRight')show((cur+1)%DATA.clouds.length);else if(e.key==='ArrowLeft')show((cur+DATA.clouds.length-1)%DATA.clouds.length);else if(e.key>='1'&&e.key<='2')show(+e.key-1);else if(e.key==='o'||e.key==='O'){const o=document.getElementById('ovr');o.checked=!o.checked;}else if(e.key==='r'||e.key==='R'){yaw=HOME.yaw;pitch=HOME.pitch;dist=HOME.dist;target=[0,0,0];}});
function resize(){const d=devicePixelRatio||1;canvas.width=innerWidth*d;canvas.height=innerHeight*d;gl.viewport(0,0,canvas.width,canvas.height);}addEventListener('resize',resize);resize();
function drawPoints(g,mvp,a){gl.useProgram(prog);gl.uniformMatrix4fv(loc.mvp,false,mvp);gl.uniform1f(loc.size,2.0*(devicePixelRatio||1));gl.uniform1f(loc.alpha,a);gl.bindBuffer(gl.ARRAY_BUFFER,g.pb);gl.enableVertexAttribArray(loc.pos);gl.vertexAttribPointer(loc.pos,3,gl.FLOAT,false,0,0);gl.bindBuffer(gl.ARRAY_BUFFER,g.cb);gl.enableVertexAttribArray(loc.col);gl.vertexAttribPointer(loc.col,3,gl.UNSIGNED_BYTE,true,0,0);gl.drawArrays(gl.POINTS,0,g.n);}
function frame(){requestAnimationFrame(frame);if(document.getElementById('auto').checked&&!dragging)yaw+=0.0022;const off=[dist*Math.cos(pitch)*Math.cos(yaw),dist*Math.cos(pitch)*Math.sin(yaw),dist*Math.sin(pitch)];const eye=[target[0]+off[0],target[1]+off[1],target[2]+off[2]];const mvp=mul(persp(0.9,canvas.width/canvas.height,0.05,100),lookAt(eye,target,[0,0,1]));gl.clear(gl.COLOR_BUFFER_BIT|gl.DEPTH_BUFFER_BIT);if(document.getElementById('grid').checked&&helpB){gl.useProgram(lprog);gl.uniformMatrix4fv(lloc.mvp,false,mvp);gl.bindBuffer(gl.ARRAY_BUFFER,helpB);gl.enableVertexAttribArray(lloc.pos);gl.vertexAttribPointer(lloc.pos,3,gl.FLOAT,false,0,0);gl.uniform4f(lloc.color,0.13,0.24,0.42,0.55);gl.drawArrays(gl.LINES,0,helpN);gl.uniform4f(lloc.color,0.88,0.33,0.33,1);gl.drawArrays(gl.LINES,helpN,2);gl.uniform4f(lloc.color,0.33,0.75,0.48,1);gl.drawArrays(gl.LINES,helpN+2,2);gl.uniform4f(lloc.color,0.36,0.56,0.98,1);gl.drawArrays(gl.LINES,helpN+4,2);}
const overlay=document.getElementById('ovr').checked&&cur!==REF;if(overlay){gl.enable(gl.BLEND);gl.blendFunc(gl.SRC_ALPHA,gl.ONE_MINUS_SRC_ALPHA);gl.depthMask(false);drawPoints(gpu(REF),mvp,0.35);gl.depthMask(true);gl.disable(gl.BLEND);}drawPoints(gpu(cur),mvp,1);}
show(0);frame();
</script></body></html>"""

if __name__ == "__main__":
    main()