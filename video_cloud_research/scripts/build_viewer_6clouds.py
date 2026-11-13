"""Đóng gói 6 cloud video→3D (THÔ + SAU CĂN CHỈNH) + cloud tham chiếu week26.ply
vào MỘT file HTML tự chứa. Không CDN/URL ngoài: WebGL thuần, nhúng base64, mở file:// là chạy.

- Thô: fused.ply gốc, chuẩn hóa riêng từng cloud (khung/scale tùy ý của model).
- Sau căn chỉnh: recon_aligned.ply khung ref; riêng MASt3R 25f chỉ có khung IFC
  → đổi về ref bằng nghịch đảo G3 đóng băng. 6 bản căn chỉnh + cloud tham chiếu
  dùng CHUNG một hệ chuẩn hóa (union bbox) để so sánh đúng vị trí/scale.
- Tiện ích giống viewer_3d.html: lưới + trục XYZ (checkbox), kéo phải/Shift+kéo = pan,
  checkbox chồng cloud tham chiếu lên bản căn chỉnh để đối chiếu.
Mỗi biến thể: lọc ngoại lai xa (percentile 0.5–99.5), voxel-downsample ≤150k điểm.
"""
import base64
import itertools
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
ART = ROOT / "artifacts"
REF_PLY = ROOT.parent / "data/schependomlaan/raw/week26.ply"
G3 = ROOT.parent / "artifacts/week26_registration_v3/candidate_transform.json"
OUT = ROOT / "viewer_6clouds.html"
MAX_POINTS = 150_000
GRID = 256

CLOUDS = [
    dict(dir="colmap_run", name="1 · COLMAP SfM+MVS",
         cfg="113 frame @2fps · SIFT CPU · sequential matcher · PatchMatch CUDA",
         note="Biến dạng: 2 khối khác scale — không căn chỉnh được.",
         al_path="alignment/recon_aligned.ply", al_frame="ref"),
    dict(dir="dust3r_run", name="2 · DUSt3R swin-2",
         cfg="384px · 113 frame · conf>1.5 · niter 300 · 289 s · loss 0.011",
         note="Méo nội tại, trôi scale (non-metric) — cổng căn chỉnh trượt.",
         al_path="dust3r_run/alignment_ref/recon_aligned.ply", al_frame="ref"),
    dict(dir="dust3r_logwin", name="3 · DUSt3R logwin n1000",
         cfg="384px · 113 frame · conf>1.5 · niter 1000 · 765 s · loss 0.014",
         note="Phủ rộng nhất nhưng méo phi cứng (bbox xiên ~58 m). Test 61,9%.",
         al_path="dust3r_logwin/alignment_ref/recon_aligned.ply", al_frame="ref"),
    dict(dir="dust3r_hi", name="4 · DUSt3R logwin n1500",
         cfg="384px · 113 frame · conf>1.5 · niter 1500 · 1057 s · loss 0.0139",
         note="Thêm vòng alignment KHÔNG giảm méo. Test 61,9%.",
         al_path="dust3r_hi/alignment_ref/recon_aligned.ply", al_frame="ref"),
    dict(dir="mast3r_run", name="5 · MASt3R 25f@384",
         cfg="25 frame @384px · stride 9",
         note="Sập thành cụm nhỏ (coverage_bim 0,5%) — không dùng được.",
         al_path="mast3r_run/alignment_rigid/recon_aligned.ply", al_frame="ifc→ref (nghịch đảo G3)"),
    dict(dir="mast3r_full", name="6 · MASt3R 57f@256",
         cfg="57 frame @256px · logwin · conf 1.0 · niter 1000 · 315 s · loss 0.060",
         note="TỐT NHẤT từ video: metric, cứng, đúng scale thật, phủ ít. Test 66,7%.",
         al_path="mast3r_full/alignment_ref/recon_aligned.ply", al_frame="ref"),
]

TYPE_MAP = {"float": "<f4", "double": "<f8", "uchar": "u1", "char": "i1",
            "ushort": "<u2", "short": "<i2", "uint": "<u4", "int": "<i4"}


def read_ply(path):
    with open(path, "rb") as f:
        assert f.readline().strip() == b"ply"
        fmt, n, props, in_vertex = None, 0, [], False
        while True:
            line = f.readline().decode("ascii").strip()
            if line == "end_header":
                break
            parts = line.split()
            if parts[0] == "format":
                fmt = parts[1]
            elif parts[0] == "element":
                in_vertex = parts[1] == "vertex"
                if in_vertex:
                    n = int(parts[2])
            elif parts[0] == "property" and in_vertex:
                assert parts[1] != "list"
                props.append((parts[2], TYPE_MAP[parts[1]]))
        assert fmt in ("binary_little_endian", "ascii"), fmt
        if fmt == "ascii":
            rows = np.loadtxt(itertools.islice(f, n), dtype=np.float64, ndmin=2)
            out = np.empty(n, dtype=np.dtype(props))
            for i, (name, _) in enumerate(props):
                out[name] = rows[:, i]
            return out
        return np.fromfile(f, dtype=np.dtype(props), count=n)


def load_reduce(path):
    data = read_ply(path)
    xyz = np.stack([data["x"], data["y"], data["z"]], axis=1).astype(np.float64)
    pre = "" if "red" in data.dtype.names else "diffuse_"
    rgb = np.stack([data[pre + "red"], data[pre + "green"], data[pre + "blue"]], axis=1).astype(np.uint8)
    n_orig = len(xyz)
    lo, hi = np.percentile(xyz, 0.5, axis=0), np.percentile(xyz, 99.5, axis=0)
    keep = np.all((xyz >= lo) & (xyz <= hi), axis=1)
    xyz, rgb = xyz[keep], rgb[keep]
    extent = float((xyz.max(axis=0) - xyz.min(axis=0)).max())
    keys = np.floor((xyz - xyz.min(axis=0)) / (extent / GRID)).astype(np.int64)
    kv = np.ascontiguousarray(keys).view(np.dtype((np.void, keys.dtype.itemsize * 3)))
    _, first = np.unique(kv, return_index=True)
    if len(first) > MAX_POINTS:
        first = np.random.default_rng(7).choice(first, MAX_POINTS, replace=False)
    return xyz[first], rgb[first], n_orig, n_orig - int(keep.sum())


def quantize(xyz, rgb, center, extent):
    norm = (xyz - center) / extent * 2
    q = np.clip(np.round((norm + 1) * 0.5 * 65535), 0, 65535).astype("<u2")
    gz = float((xyz[:, 2].min() - center[2]) / extent * 2)
    return (base64.b64encode(q.tobytes()).decode(),
            base64.b64encode(rgb.tobytes()).decode(), gz)


def main():
    clouds, al_xyzs, al_rgbs = [], [], []
    for c in CLOUDS:
        raw_path = ART / c["dir"] / "fused.ply"
        xyz, rgb, n_orig, dropped = load_reduce(raw_path)
        center = (xyz.max(axis=0) + xyz.min(axis=0)) / 2
        extent = float((xyz.max(axis=0) - xyz.min(axis=0)).max())
        pos, col, gz = quantize(xyz, rgb, center, extent)
        entry = dict(c, orig=n_orig, shown=len(xyz), dropped=dropped, gz=gz,
                     sizeMB=round(raw_path.stat().st_size / 1e6, 1), pos=pos, col=col)
        al_path = ART / c["al_path"]
        axyz, argb, an_orig, an_dropped = load_reduce(al_path)
        if c["al_frame"].startswith("ifc"):
            t = np.asarray(json.loads(G3.read_text()), dtype=np.float64)
            axyz = (axyz - t[:3, 3]) @ t[:3, :3]
        mj = json.loads((al_path.parent / "metrics.json").read_text())
        entry["al"] = dict(path=c["al_path"], frame=c["al_frame"],
                           scale=round(float(mj.get("scale", 0)), 3),
                           cov=round(mj["coverage_ref_at_tau"]["0.2"], 3)
                           if mj.get("coverage_ref_at_tau", {}).get("0.2") is not None else None,
                           orig=an_orig, shown=len(axyz), dropped=an_dropped)
        al_xyzs.append(axyz)
        al_rgbs.append(argb)
        clouds.append(entry)
        print(f"{c['dir']}: thô {n_orig:,}→{len(xyz):,} | căn chỉnh {an_orig:,}→{len(axyz):,}", flush=True)

    rxyz, rrgb, rn_orig, rn_dropped = load_reduce(REF_PLY)
    rcenter = (rxyz.max(axis=0) + rxyz.min(axis=0)) / 2
    rextent = float((rxyz.max(axis=0) - rxyz.min(axis=0)).max())
    rpos, rcol, rgz = quantize(rxyz, rrgb, rcenter, rextent)
    ref = dict(dir="ref", name="7 · week26.ply tham chiếu",
               cfg="Scan CHUYÊN NGHIỆP của bộ dữ liệu (RAAMAC dựng từ video drone bằng SfM)",
               note="Đối chiếu: RAAMAC 95,24% test với cloud này. Không phải cloud dựng lại của nhánh video.",
               orig=rn_orig, shown=len(rxyz), dropped=rn_dropped, gz=rgz,
               sizeMB=round(REF_PLY.stat().st_size / 1e6, 1), pos=rpos, col=rcol,
               al=dict(path="data/schependomlaan/raw/week26.ply", frame="ref gốc",
                       scale=1.0, cov=None, orig=rn_orig, shown=len(rxyz), dropped=rn_dropped))
    al_xyzs.append(rxyz)
    al_rgbs.append(rrgb)
    clouds.append(ref)
    print(f"week26.ply: gốc {rn_orig:,}→{len(rxyz):,}", flush=True)

    allal = np.concatenate(al_xyzs)
    a_center = (allal.max(axis=0) + allal.min(axis=0)) / 2
    a_extent = float((allal.max(axis=0) - allal.min(axis=0)).max())
    for entry, axyz, argb in zip(clouds, al_xyzs, al_rgbs):
        pos, col, gz = quantize(axyz, argb, a_center, a_extent)
        entry["al"].update(pos=pos, col=col, gz=gz)

    html = HTML.replace("__DATA__", json.dumps({"clouds": clouds}, ensure_ascii=False, separators=(",", ":")))
    OUT.write_text(html, encoding="utf-8")
    print(f"Đã ghi {OUT} ({OUT.stat().st_size/1e6:.1f} MB)")


HTML = r"""<!DOCTYPE html>
<html lang="vi">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>6 cloud 3D từ video + tham chiếu week26.ply</title>
<style>
  html,body{margin:0;height:100%;overflow:hidden;background:#0e1116;color:#e6e6e6;
    font:14px/1.45 system-ui,Segoe UI,Roboto,sans-serif}
  #c{position:absolute;inset:0;width:100%;height:100%;display:block;touch-action:none;cursor:grab}
  #c:active{cursor:grabbing}
  #bar{position:absolute;top:10px;left:10px;right:10px;display:flex;flex-wrap:wrap;gap:6px;align-items:center}
  #bar button{background:#1c2330;color:#cfd8e3;border:1px solid #33405a;border-radius:8px;
    padding:7px 11px;cursor:pointer;font-size:13px}
  #bar button:hover{background:#26304a}
  #bar button.on{background:#2f6feb;border-color:#2f6feb;color:#fff}
  #bar button.mode.on{background:#2ea06f;border-color:#2ea06f}
  #bar .nav{font-weight:700}
  #bar .sep{width:1px;height:22px;background:#33405a;margin:0 3px}
  #info{position:absolute;left:10px;bottom:10px;max-width:470px;background:rgba(14,17,22,.82);
    border:1px solid #2a3346;border-radius:10px;padding:10px 13px}
  #info h2{margin:0 0 4px;font-size:15px;color:#7fb3ff}
  #info .m{color:#9aa7b8;font-size:12.5px}
  #info .note{margin-top:5px;font-size:13px}
  #hint{position:absolute;right:10px;bottom:10px;background:rgba(14,17,22,.82);
    border:1px solid #2a3346;border-radius:10px;padding:8px 12px;font-size:12px;color:#9aa7b8}
  label{font-size:12.5px;color:#9aa7b8;display:flex;gap:5px;align-items:center;user-select:none}
  label.dis{opacity:.4}
</style>
</head>
<body>
<canvas id="c"></canvas>
<div id="bar"></div>
<div id="info"></div>
<div id="hint">Kéo: xoay · Kéo phải / Shift+kéo: pan · Lăn / chụm: thu phóng<br>
←/→ hoặc 1–7: chuyển cloud · T: thô ↔ căn chỉnh · R: đặt lại góc<br>
Trục: <span style="color:#e05555">X đỏ</span> · <span style="color:#55c07a">Y lục</span> · <span style="color:#5b8ff9">Z lam</span></div>
<script>
const DATA = __DATA__;
const REF = DATA.clouds.length - 1;
const canvas = document.getElementById('c');
const gl = canvas.getContext('webgl', {antialias: true});
if (!gl) { document.getElementById('info').innerHTML = '<h2>Trình duyệt không hỗ trợ WebGL</h2>'; throw new Error('no webgl'); }

function sh(type, src){ const s = gl.createShader(type); gl.shaderSource(s, src); gl.compileShader(s);
  if (!gl.getShaderParameter(s, gl.COMPILE_STATUS)) throw new Error(gl.getShaderInfoLog(s)); return s; }
function program(vs, fs){ const p = gl.createProgram();
  gl.attachShader(p, sh(gl.VERTEX_SHADER, vs)); gl.attachShader(p, sh(gl.FRAGMENT_SHADER, fs));
  gl.linkProgram(p); return p; }
const prog = program(
  `attribute vec3 aPos; attribute vec3 aCol; uniform mat4 uMVP; uniform float uSize;
   varying vec3 vCol; void main(){ gl_Position = uMVP * vec4(aPos,1.0); gl_PointSize = uSize; vCol = aCol; }`,
  `precision mediump float; varying vec3 vCol; uniform float uAlpha;
   void main(){ gl_FragColor = vec4(vCol, uAlpha); }`);
const loc = { pos: gl.getAttribLocation(prog,'aPos'), col: gl.getAttribLocation(prog,'aCol'),
  mvp: gl.getUniformLocation(prog,'uMVP'), size: gl.getUniformLocation(prog,'uSize'),
  alpha: gl.getUniformLocation(prog,'uAlpha') };
const lprog = program(
  `attribute vec3 aPos; uniform mat4 uMVP; void main(){ gl_Position = uMVP * vec4(aPos,1.0); }`,
  `precision mediump float; uniform vec4 uColor; void main(){ gl_FragColor = uColor; }`);
const lloc = { pos: gl.getAttribLocation(lprog,'aPos'), mvp: gl.getUniformLocation(lprog,'uMVP'),
  color: gl.getUniformLocation(lprog,'uColor') };
gl.enable(gl.DEPTH_TEST); gl.clearColor(0.055, 0.067, 0.086, 1);

function b64bytes(s){ const bin = atob(s); const b = new Uint8Array(bin.length);
  for (let i = 0; i < bin.length; i++) b[i] = bin.charCodeAt(i); return b; }
const cache = [];
function gpu(i, mode){
  const slot = cache[i] || (cache[i] = {});
  if (slot[mode]) return slot[mode];
  const m = mode === 'raw' ? DATA.clouds[i] : DATA.clouds[i].al;
  const u16 = new Uint16Array(b64bytes(m.pos).buffer);
  const pos = new Float32Array(u16.length);
  for (let j = 0; j < u16.length; j++) pos[j] = u16[j] / 32767.5 - 1;
  const pb = gl.createBuffer(); gl.bindBuffer(gl.ARRAY_BUFFER, pb); gl.bufferData(gl.ARRAY_BUFFER, pos, gl.STATIC_DRAW);
  const cb = gl.createBuffer(); gl.bindBuffer(gl.ARRAY_BUFFER, cb); gl.bufferData(gl.ARRAY_BUFFER, b64bytes(m.col), gl.STATIC_DRAW);
  return slot[mode] = { pb, cb, n: m.shown };
}

/* lưới + trục: dựng lại theo mặt phẳng đáy (gz) của biến thể đang xem */
let helpB = null, helpGridN = 0, helpGz = null;
function helpers(gz){
  if (helpB && helpGz === gz) return;
  helpGz = gz;
  const v = [];
  for (let x = -1; x <= 1.001; x += 0.2) v.push(x, -1, gz, x, 1, gz);
  for (let y = -1; y <= 1.001; y += 0.2) v.push(-1, y, gz, 1, y, gz);
  helpGridN = v.length / 3;
  v.push(-1,-1,gz, -0.4,-1,gz,  -1,-1,gz, -1,-0.4,gz,  -1,-1,gz, -1,-1,gz+0.6);
  helpB = gl.createBuffer(); gl.bindBuffer(gl.ARRAY_BUFFER, helpB);
  gl.bufferData(gl.ARRAY_BUFFER, new Float32Array(v), gl.STATIC_DRAW);
}

const sub=(a,b)=>[a[0]-b[0],a[1]-b[1],a[2]-b[2]];
const dot=(a,b)=>a[0]*b[0]+a[1]*b[1]+a[2]*b[2];
const cross=(a,b)=>[a[1]*b[2]-a[2]*b[1],a[2]*b[0]-a[0]*b[2],a[0]*b[1]-a[1]*b[0]];
const norm=a=>{const l=Math.hypot(a[0],a[1],a[2]);return [a[0]/l,a[1]/l,a[2]/l];};
function mul(a,b){const o=new Float32Array(16);
  for(let c=0;c<4;c++)for(let r=0;r<4;r++){let s=0;for(let k=0;k<4;k++)s+=a[k*4+r]*b[c*4+k];o[c*4+r]=s;}return o;}
function persp(fov,asp,n,f){const t=1/Math.tan(fov/2);
  return new Float32Array([t/asp,0,0,0, 0,t,0,0, 0,0,(f+n)/(n-f),-1, 0,0,2*f*n/(n-f),0]);}
function lookAt(e,c,u){const z=norm(sub(e,c)),x=norm(cross(u,z)),y=cross(z,x);
  return new Float32Array([x[0],y[0],z[0],0, x[1],y[1],z[1],0, x[2],y[2],z[2],0, -dot(x,e),-dot(y,e),-dot(z,e),1]);}

let cur = 0, mode = 'raw', yaw = 0.7, pitch = 0.35, dist = 3.1, target = [0,0,0];
let dragging = false, dragMode = 'rotate', pinch = 0;
const HOME = { yaw: 0.7, pitch: 0.35, dist: 3.1 };
const ptrs = new Map();
canvas.addEventListener('contextmenu', e => e.preventDefault());
canvas.addEventListener('pointerdown', e => {
  try { canvas.setPointerCapture(e.pointerId); } catch (_) {}
  ptrs.set(e.pointerId, {x:e.clientX, y:e.clientY});
  dragging = true; pinch = 0;
  dragMode = (e.button === 2 || e.shiftKey) ? 'pan' : 'rotate';
});
canvas.addEventListener('pointermove', e => {
  const p = ptrs.get(e.pointerId); if (!p) return;
  const dx = e.clientX - p.x, dy = e.clientY - p.y; p.x = e.clientX; p.y = e.clientY;
  if (ptrs.size === 2) { const [a,b] = [...ptrs.values()]; const d = Math.hypot(a.x-b.x, a.y-b.y);
    if (pinch > 0) dist = Math.max(0.4, Math.min(30, dist * pinch / d)); pinch = d; return; }
  if (dragMode === 'pan') {
    const f = [-Math.cos(pitch)*Math.cos(yaw), -Math.cos(pitch)*Math.sin(yaw), -Math.sin(pitch)];
    const r = norm(cross(f, [0,0,1])), u = cross(r, f), s = dist * 0.0016;
    target = [target[0]-r[0]*dx*s+u[0]*dy*s, target[1]-r[1]*dx*s+u[1]*dy*s, target[2]-r[2]*dx*s+u[2]*dy*s];
  } else { yaw -= dx * 0.005; pitch = Math.max(-1.5, Math.min(1.5, pitch + dy * 0.005)); }
});
const up = e => { ptrs.delete(e.pointerId); if (!ptrs.size) dragging = false; pinch = 0; };
canvas.addEventListener('pointerup', up); canvas.addEventListener('pointercancel', up);
canvas.addEventListener('wheel', e => { e.preventDefault();
  dist = Math.max(0.4, Math.min(30, dist * Math.exp(e.deltaY * 0.0012))); }, {passive:false});

const bar = document.getElementById('bar');
const prev = document.createElement('button'); prev.textContent = '←'; prev.className = 'nav';
const next = document.createElement('button'); next.textContent = '→'; next.className = 'nav';
prev.onclick = () => show((cur + DATA.clouds.length - 1) % DATA.clouds.length);
next.onclick = () => show((cur + 1) % DATA.clouds.length);
bar.append(prev, next);
const btns = DATA.clouds.map((m, i) => { const b = document.createElement('button');
  b.textContent = m.name; b.onclick = () => show(i); bar.append(b); return b; });
bar.append(Object.assign(document.createElement('span'), {className: 'sep'}));
const modeBtns = {};
for (const [key, label] of [['raw', 'Thô'], ['al', 'Sau căn chỉnh']]) {
  const b = document.createElement('button'); b.textContent = label; b.className = 'mode';
  b.onclick = () => setMode(key); bar.append(b); modeBtns[key] = b;
}
bar.append(Object.assign(document.createElement('span'), {className: 'sep'}));
const gridL = document.createElement('label');
gridL.innerHTML = '<input type="checkbox" id="grid" checked> Lưới & trục';
bar.append(gridL);
const ovrL = document.createElement('label');
ovrL.innerHTML = '<input type="checkbox" id="ovr"> Chồng tham chiếu';
ovrL.title = 'Chỉ dùng ở chế độ Sau căn chỉnh: chồng week26.ply mờ lên cloud đang xem';
bar.append(ovrL);
const spin = document.createElement('label');
spin.innerHTML = '<input type="checkbox" id="auto" checked> Tự xoay';
bar.append(spin);
const reset = document.createElement('button'); reset.textContent = 'Đặt lại góc';
reset.onclick = () => { yaw = HOME.yaw; pitch = HOME.pitch; dist = HOME.dist; target = [0,0,0]; };
bar.append(reset);

function setMode(m){ mode = m;
  modeBtns.raw.classList.toggle('on', m === 'raw');
  modeBtns.al.classList.toggle('on', m === 'al');
  show(cur); }

function show(i){
  cur = i;
  btns.forEach((b, j) => b.classList.toggle('on', j === i));
  const m = DATA.clouds[i], a = m.al;
  const v = mode === 'raw' ? m : a;
  const ovr = document.getElementById('ovr');
  ovr.disabled = (mode !== 'al' || i === REF);
  ovrL.classList.toggle('dis', ovr.disabled);
  const head = mode === 'raw'
    ? `<div class="m">Chế độ: THÔ — khung/scale tùy ý của model · ${i === REF ? 'data/schependomlaan/raw/week26.ply' : 'artifacts/' + m.dir + '/fused.ply'}</div>`
    : `<div class="m">Chế độ: SAU CĂN CHỈNH — ${i === REF ? a.path : 'artifacts/' + a.path}</div>` +
      `<div class="m">Khung: ${a.frame} · scale ≈ ${a.scale} · coverage_ref@0.2: ${a.cov == null ? '—' : a.cov}</div>` +
      `<div class="m">Các bản căn chỉnh + tham chiếu dùng chung hệ chuẩn hóa (khung ref) để so sánh đúng vị trí/scale.</div>`;
  document.getElementById('info').innerHTML =
    `<h2>${m.name}</h2><div class="m">${m.cfg}</div>` + head +
    `<div class="m">Điểm gốc: ${v.orig.toLocaleString('vi-VN')} · Hiển thị: ${v.shown.toLocaleString('vi-VN')}` +
    (mode === 'raw' ? ` · File: ${m.sizeMB} MB` : '') + `</div>` +
    `<div class="note">${m.note}</div>` +
    `<div class="m">Nguồn video: week26.mp4 — drone 1 quỹ đạo, 55 s, 1920×1080 (Schependomlaan).</div>`;
  helpers(v.gz);
  gpu(i, mode);
}
addEventListener('keydown', e => {
  if (e.key === 'ArrowRight') show((cur + 1) % DATA.clouds.length);
  else if (e.key === 'ArrowLeft') show((cur + DATA.clouds.length - 1) % DATA.clouds.length);
  else if (e.key >= '1' && e.key <= '7') show(+e.key - 1);
  else if (e.key === 't' || e.key === 'T') setMode(mode === 'raw' ? 'al' : 'raw');
  else if (e.key === 'r' || e.key === 'R') { yaw = HOME.yaw; pitch = HOME.pitch; dist = HOME.dist; target = [0,0,0]; }
});

function resize(){ const dpr = devicePixelRatio || 1;
  canvas.width = innerWidth * dpr; canvas.height = innerHeight * dpr;
  gl.viewport(0, 0, canvas.width, canvas.height); }
addEventListener('resize', resize); resize();

function drawPoints(g, mvp, alpha){
  gl.useProgram(prog);
  gl.uniformMatrix4fv(loc.mvp, false, mvp);
  gl.uniform1f(loc.size, 2.0 * (devicePixelRatio || 1));
  gl.uniform1f(loc.alpha, alpha);
  gl.bindBuffer(gl.ARRAY_BUFFER, g.pb); gl.enableVertexAttribArray(loc.pos);
  gl.vertexAttribPointer(loc.pos, 3, gl.FLOAT, false, 0, 0);
  gl.bindBuffer(gl.ARRAY_BUFFER, g.cb); gl.enableVertexAttribArray(loc.col);
  gl.vertexAttribPointer(loc.col, 3, gl.UNSIGNED_BYTE, true, 0, 0);
  gl.drawArrays(gl.POINTS, 0, g.n);
}
function frame(){
  requestAnimationFrame(frame);
  if (document.getElementById('auto').checked && !dragging) yaw += 0.0022;
  const off = [dist*Math.cos(pitch)*Math.cos(yaw), dist*Math.cos(pitch)*Math.sin(yaw), dist*Math.sin(pitch)];
  const eye = [target[0]+off[0], target[1]+off[1], target[2]+off[2]];
  const mvp = mul(persp(0.9, canvas.width/canvas.height, 0.05, 100), lookAt(eye, target, [0,0,1]));
  gl.clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT);
  if (document.getElementById('grid').checked && helpB) {
    gl.useProgram(lprog);
    gl.uniformMatrix4fv(lloc.mvp, false, mvp);
    gl.bindBuffer(gl.ARRAY_BUFFER, helpB); gl.enableVertexAttribArray(lloc.pos);
    gl.vertexAttribPointer(lloc.pos, 3, gl.FLOAT, false, 0, 0);
    gl.uniform4f(lloc.color, 0.13, 0.24, 0.42, 0.55);
    gl.drawArrays(gl.LINES, 0, helpGridN);
    gl.uniform4f(lloc.color, 0.88, 0.33, 0.33, 1); gl.drawArrays(gl.LINES, helpGridN, 2);
    gl.uniform4f(lloc.color, 0.33, 0.75, 0.48, 1); gl.drawArrays(gl.LINES, helpGridN+2, 2);
    gl.uniform4f(lloc.color, 0.36, 0.56, 0.98, 1); gl.drawArrays(gl.LINES, helpGridN+4, 2);
  }
  const overlay = document.getElementById('ovr').checked && mode === 'al' && cur !== REF;
  if (overlay) {
    gl.enable(gl.BLEND); gl.blendFunc(gl.SRC_ALPHA, gl.ONE_MINUS_SRC_ALPHA); gl.depthMask(false);
    drawPoints(gpu(REF, 'al'), mvp, 0.4);
    gl.depthMask(true); gl.disable(gl.BLEND);
  }
  drawPoints(gpu(cur, mode), mvp, 1);
}
const hm = location.hash.match(/^#([1-7])(c?)(r?)$/);
if (hm && hm[2]) setMode('al');
if (hm && hm[3]) document.getElementById('ovr').checked = true;
show(hm ? +hm[1] - 1 : 0); frame();
</script>
</body>
</html>
"""


if __name__ == "__main__":
    main()