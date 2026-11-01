#!/usr/bin/env python3
"""Offline browser UI for labeling canonical badminton-court landmarks."""
from __future__ import annotations

import argparse
import json
import sys
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote, urlparse


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from pipeline.court_keypoints import LANDMARK_BY_NAME, schema_payload, validate_annotation  # noqa: E402


ANNOTATOR_HTML = r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Court landmark annotator</title>
<style>
:root{color-scheme:dark;--bg:#0b1118;--panel:#151e29;--line:#2b3a4d;--accent:#48d7a2;--warn:#ffd166}*{box-sizing:border-box}
html,body{width:100%;height:100%;overflow:hidden}body{margin:0;background:var(--bg);color:#f5f7fa;font-family:Inter,system-ui,sans-serif;display:grid;grid-template-rows:auto minmax(0,1fr)}header{padding:14px 18px;border-bottom:1px solid var(--line);display:flex;gap:12px;align-items:center;flex-wrap:wrap}
button,select{background:#202c3b;color:#fff;border:1px solid #394b61;border-radius:8px;padding:9px 12px}button:hover{border-color:var(--accent)}button.primary{background:#176b51}
button,select{max-width:100%}.layout{display:grid;grid-template-columns:minmax(0,1fr) 340px;min-width:0;min-height:0;height:auto;overflow:hidden}
.stage{display:flex;align-items:center;justify-content:center;min-width:0;min-height:0;overflow:hidden;padding:12px;overscroll-behavior:none}canvas{display:block;width:auto;height:auto;max-width:100%;max-height:100%;border:1px solid var(--line);cursor:crosshair}.sidebar{min-height:0;height:100%;overflow-y:auto;overscroll-behavior:contain;padding:14px;border-left:1px solid var(--line);background:var(--panel)}
.landmark{display:flex;gap:8px;align-items:center;width:100%;text-align:left;margin:5px 0}.landmark.active{border-color:var(--warn);box-shadow:0 0 0 1px var(--warn)}.dot{width:10px;height:10px;border-radius:50%;background:#58687b}.done .dot{background:var(--accent)}.skipped .dot{background:#8993a0}.meta{color:#9fb0c3;font-size:.88rem}.help{padding:12px 14px;margin:10px 0 14px;border:1px solid #526579;border-radius:10px;background:#101923;line-height:1.4}.help strong{color:var(--warn)}.shortcuts{display:grid;grid-template-columns:auto 1fr;gap:6px 10px;margin-top:12px;padding-top:12px;border-top:1px solid #334457;font-size:.86rem;color:#b7c7d8}.shortcuts div:nth-child(odd){white-space:nowrap}kbd{display:inline-block;min-width:23px;padding:1px 5px;border:1px solid #526579;border-bottom-width:2px;border-radius:5px;background:#202c3b;color:#fff;text-align:center;font:inherit;font-size:.78rem}#status{color:var(--warn)}#progress{margin:8px 0 12px;color:#b7c7d8;font-size:.9rem}
@media(max-width:850px){html,body{height:auto;min-height:100%;overflow:auto}body{display:block}.layout{grid-template-columns:1fr;height:auto;overflow:visible}.stage{height:65vh}.sidebar{height:auto;overflow:visible;border-left:0;border-top:1px solid var(--line)}}
</style></head><body>
<header><button id="prev">← Previous</button><select id="imageSelect"></select><button id="next">Next →</button><button id="skip">Not visible / off-screen</button><button id="clear">Clear selected</button><button class="primary" id="save">Save annotation</button><span id="status"></span></header>
<div class="layout"><div class="stage"><canvas id="canvas"></canvas></div><aside class="sidebar"><h2>Court landmarks</h2><div class="help"><strong>Never guess a missing point.</strong><br>If the selected intersection is clearly visible, click its exact center. If it is outside the image, occluded, or ambiguous, choose <em>Not visible / off-screen</em>.<br><br><strong>Far</strong> = across the net. <strong>Near</strong> = camera side.<div class="shortcuts"><div><kbd>J</kbd> / <kbd>↓</kbd></div><div>Next landmark</div><div><kbd>K</kbd> / <kbd>↑</kbd></div><div>Previous landmark</div><div><kbd>V</kbd></div><div>Mark not visible</div><div><kbd>C</kbd></div><div>Clear selected</div><div><kbd>S</kbd></div><div>Save annotation</div><div><kbd>[</kbd> / <kbd>]</kbd></div><div>Previous / next image</div></div></div><div id="progress"></div><div id="landmarks"></div></aside></div>
<script>
const state={items:[],schema:[],index:0,selected:0,points:{},image:null};const canvas=document.querySelector('#canvas'),ctx=canvas.getContext('2d');
async function init(){const data=await (await fetch('/api/items')).json();state.items=data.items;state.schema=data.schema;renderList();const hash=new URLSearchParams(location.search).get('image');const found=state.items.findIndex(x=>x.image===hash);state.index=found>=0?found:0;await load();}
function emptyPoints(){return Object.fromEntries(state.schema.map(x=>[x.name,undefined]));}
async function load(){const item=state.items[state.index];if(!item)return;document.querySelector('#imageSelect').innerHTML=state.items.map((x,i)=>`<option value="${i}" ${i===state.index?'selected':''}>${x.image}${x.annotated?' ✓':''}</option>`).join('');const existing=await (await fetch('/api/annotation/'+encodeURIComponent(item.image))).json();state.points=emptyPoints();if(existing.points)Object.assign(state.points,existing.points);state.image=new Image();state.image.onload=()=>{canvas.width=state.image.naturalWidth;canvas.height=state.image.naturalHeight;draw();};state.image.src='/images/'+encodeURIComponent(item.image);setStatus(`${state.index+1}/${state.items.length} · ${item.source_video}`);renderList();}
function draw(){ctx.drawImage(state.image,0,0);state.schema.forEach((lm,i)=>{const p=state.points[lm.name];if(Array.isArray(p)){ctx.beginPath();ctx.arc(p[0],p[1],i===state.selected?14:10,0,Math.PI*2);ctx.fillStyle=i===state.selected?'#ffd166':'#48d7a2';ctx.fill();ctx.strokeStyle='#07120e';ctx.lineWidth=3;ctx.stroke();ctx.font='28px system-ui';ctx.fillStyle='#fff';ctx.fillText(String(i+1),p[0]+14,p[1]-14);}});}
function renderList(){const root=document.querySelector('#landmarks');root.innerHTML='';let visible=0,invisible=0;state.schema.forEach((lm,i)=>{const p=state.points[lm.name];if(Array.isArray(p))visible++;else if(p===null)invisible++;const b=document.createElement('button');b.className='landmark '+(i===state.selected?'active ':'')+(Array.isArray(p)?'done':p===null?'skipped':'');b.innerHTML=`<span class="dot"></span><span>${i+1}. ${lm.label}</span>`;b.onclick=()=>{state.selected=i;renderList();draw();};root.appendChild(b);});const unresolved=state.schema.length-visible-invisible;document.querySelector('#progress').textContent=`Visible: ${visible} · Not visible: ${invisible} · Unresolved: ${unresolved}`;}
canvas.onclick=e=>{const r=canvas.getBoundingClientRect(),sx=canvas.width/r.width,sy=canvas.height/r.height,lm=state.schema[state.selected];state.points[lm.name]=[(e.clientX-r.left)*sx,(e.clientY-r.top)*sy];if(state.selected<state.schema.length-1)state.selected++;renderList();draw();scrollSelectedLandmark();};
function scrollSelectedLandmark(){const sidebar=document.querySelector('.sidebar'),el=document.querySelectorAll('.landmark')[state.selected];if(!sidebar||!el)return;const top=el.offsetTop-(sidebar.clientHeight-el.clientHeight)/2;sidebar.scrollTop=Math.max(0,top);}
function selectLandmark(delta){state.selected=Math.max(0,Math.min(state.schema.length-1,state.selected+delta));renderList();draw();scrollSelectedLandmark();}
function markInvisible(){state.points[state.schema[state.selected].name]=null;if(state.selected<state.schema.length-1)state.selected++;renderList();draw();scrollSelectedLandmark();}
function clearSelected(){state.points[state.schema[state.selected].name]=undefined;renderList();draw();}
document.querySelector('#skip').onclick=markInvisible;document.querySelector('#clear').onclick=clearSelected;
async function save(){const item=state.items[state.index],missing=state.schema.filter(lm=>state.points[lm.name]===undefined);if(missing.length){setStatus(`Resolve ${missing.length} unmarked landmarks first`);return;}const payload={...item,image_width:canvas.width,image_height:canvas.height,points:state.points,complete:true};const response=await fetch('/api/annotation/'+encodeURIComponent(item.image),{method:'POST',headers:{'content-type':'application/json'},body:JSON.stringify(payload)});const result=await response.json();if(!response.ok){setStatus(result.error||'Save failed');return;}state.items[state.index].annotated=true;setStatus('Saved');renderList();}
async function moveImage(delta){state.index=Math.max(0,Math.min(state.items.length-1,state.index+delta));state.selected=0;await load();}
document.querySelector('#save').onclick=save;document.querySelector('#prev').onclick=()=>moveImage(-1);document.querySelector('#next').onclick=()=>moveImage(1);document.querySelector('#imageSelect').onchange=async e=>{state.index=Number(e.target.value);state.selected=0;await load();};
document.addEventListener('keydown',e=>{if(e.metaKey||e.ctrlKey||e.altKey||e.target.matches('select,input,textarea'))return;const key=e.key.toLowerCase();if(key==='j'||key==='arrowdown'){e.preventDefault();selectLandmark(1);}else if(key==='k'||key==='arrowup'){e.preventDefault();selectLandmark(-1);}else if(key==='v'){e.preventDefault();markInvisible();}else if(key==='c'){e.preventDefault();clearSelected();}else if(key==='s'){e.preventDefault();save();}else if(key==='['){e.preventDefault();moveImage(-1);}else if(key===']'){e.preventDefault();moveImage(1);}});
function setStatus(x){document.querySelector('#status').textContent=x;}init();
</script></body></html>"""


class AnnotationHandler(BaseHTTPRequestHandler):
    dataset_dir: Path

    def send_json(self, payload: object, status: HTTPStatus = HTTPStatus.OK) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        if parsed.path == "/":
            body = ANNOTATOR_HTML.encode("utf-8")
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        if parsed.path == "/api/items":
            manifest = json.loads((self.dataset_dir / "manifest.json").read_text())
            review_path = self.dataset_dir / "review.json"
            review = json.loads(review_path.read_text()).get("items", {}) if review_path.exists() else {}
            annotated = {path.stem for path in (self.dataset_dir / "annotations").glob("*.json")}
            items = [
                {**item, "annotated": Path(item["image"]).stem in annotated}
                for item in manifest["items"]
                if review.get(item["image"], {}).get("status") != "rejected"
            ]
            self.send_json({"items": items, "schema": schema_payload()})
            return
        if parsed.path.startswith("/api/annotation/"):
            filename = Path(unquote(parsed.path.removeprefix("/api/annotation/"))).name
            annotation = self.dataset_dir / "annotations" / f"{Path(filename).stem}.json"
            self.send_json(json.loads(annotation.read_text()) if annotation.exists() else {})
            return
        if parsed.path.startswith("/images/"):
            filename = Path(unquote(parsed.path.removeprefix("/images/"))).name
            image_path = self.dataset_dir / "images" / filename
            if not image_path.is_file():
                self.send_error(HTTPStatus.NOT_FOUND)
                return
            body = image_path.read_bytes()
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", "image/jpeg")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        self.send_error(HTTPStatus.NOT_FOUND)

    def do_POST(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        if not parsed.path.startswith("/api/annotation/"):
            self.send_error(HTTPStatus.NOT_FOUND)
            return
        filename = Path(unquote(parsed.path.removeprefix("/api/annotation/"))).name
        manifest = json.loads((self.dataset_dir / "manifest.json").read_text())
        allowed = {item["image"] for item in manifest["items"]}
        if filename not in allowed:
            self.send_json({"error": "image is not in the manifest"}, HTTPStatus.BAD_REQUEST)
            return
        length = int(self.headers.get("Content-Length", "0"))
        if length <= 0 or length > 1_000_000:
            self.send_json({"error": "invalid payload length"}, HTTPStatus.BAD_REQUEST)
            return
        try:
            payload = json.loads(self.rfile.read(length))
        except json.JSONDecodeError:
            self.send_json({"error": "invalid JSON"}, HTTPStatus.BAD_REQUEST)
            return
        errors = validate_annotation(payload)
        if errors:
            self.send_json({"error": "; ".join(errors)}, HTTPStatus.BAD_REQUEST)
            return
        annotation_path = self.dataset_dir / "annotations" / f"{Path(filename).stem}.json"
        annotation_path.write_text(json.dumps(payload, indent=2))
        self.send_json({"saved": str(annotation_path)})

    def log_message(self, format: str, *args: object) -> None:
        print(f"[annotator] {format % args}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Serve the offline court-landmark annotator")
    parser.add_argument("--dataset", type=Path, default=Path("data/court_dataset"))
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args(argv)
    manifest = args.dataset / "manifest.json"
    if not manifest.is_file():
        parser.error(f"manifest not found: {manifest}; run prepare_court_dataset.py first")
    AnnotationHandler.dataset_dir = args.dataset.resolve()
    server = ThreadingHTTPServer((args.host, args.port), AnnotationHandler)
    print(f"Court annotator: http://{args.host}:{args.port}")
    print(f"Dataset: {AnnotationHandler.dataset_dir}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
