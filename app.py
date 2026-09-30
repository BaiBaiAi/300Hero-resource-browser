from __future__ import annotations

import argparse
import io
import json
import mimetypes
import os
import sys
import threading
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, quote, unquote, urlparse

from PIL import Image

import jmp_core as jc
from audio_backend import inspect_bank, decode_track
from dat_table_backend import parse_dat_table
from catalog import Catalog, kind, owner
from resource_inspector import inspect_resource
from model_backend import parse_raw_model, browser_model, model_frame
from live2d_backend import Live2DBrowserPreview
from settings_store import choose_game_dir, has_jmp_files, load_settings, save_settings


FROZEN = bool(getattr(sys, "frozen", False))
BUNDLE_ROOT = Path(getattr(sys, "_MEIPASS", Path(__file__).resolve().parent))
ROOT = Path(sys.executable).resolve().parent if FROZEN else Path(__file__).resolve().parent
WEB = BUNDLE_ROOT / "web"
SETTINGS = ROOT / "config" / "settings.json"
DEFAULT_GAME_DIR = r"D:\JumpGame\300Hero"
PORT = 9961
IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".gif", ".bmp", ".dds", ".tga", ".webp"}
AUDIO_EXTS = {".wav", ".ogg", ".mp3", ".flac", ".m4a"}


def json_bytes(value):
    return json.dumps(value, ensure_ascii=False).encode("utf-8")


class State:
    def __init__(self, game_dir):
        self.lock = threading.RLock()
        self.game_dir = game_dir
        self.packs = []
        self.catalog = None
        self.by_id = {}
        self.error = ""
        self.model_cache = {}
        self.live2d_preview = None

    def load(self, game_dir=None):
        with self.lock:
            if game_dir:
                self.game_dir = os.path.abspath(game_dir)
            if not has_jmp_files(self.game_dir):
                raise FileNotFoundError("所选目录中没有 Data*.jmp")
            self.packs = jc.load_packs(self.game_dir)
            valid = [pack for pack in self.packs if pack.parse_ok]
            if not valid:
                raise ValueError("没有可读取的 JMP 资源包")
            self.catalog = Catalog(valid)
            self.by_id = {f"{Path(pack.file).name}:{entry.index}": (pack, entry)
                          for pack, entry in self.catalog.pairs}
            self.model_cache.clear()
            save_settings(SETTINGS, {"gameDir": self.game_dir})
            self.error = ""
            return self.summary()

    def summary(self):
        return {
            "gameDir": self.game_dir,
            "packCount": len(self.packs),
            "validPackCount": sum(pack.parse_ok for pack in self.packs),
            "resourceCount": len(self.by_id),
            "packs": [{"name": pack.base_name(), "ok": pack.parse_ok,
                       "count": len(pack.entries), "warning": pack.warning,
                       "error": pack.error} for pack in self.packs],
        }

    def pair(self, rid):
        pair = self.by_id.get(rid)
        if not pair:
            raise FileNotFoundError("资源不存在或资源包已经重新加载")
        return pair

    @staticmethod
    def row(pack, entry, reason=""):
        ext = Path(entry.path).suffix.lower()
        return {"id": f"{Path(pack.file).name}:{entry.index}", "pack": pack.base_name(),
                "index": entry.index, "path": entry.path, "name": Path(entry.path).name,
                "ext": ext or "(无)", "category": kind(entry.path), "hero": owner(entry.path),
                "csize": entry.csize, "rsize": entry.rsize, "md5": entry.md5, "reason": reason}

    def search(self, query, category, pack_name, hero, page, page_size):
        pairs = self.catalog.search(query, category,
            {pack.file for pack in self.packs if pack.base_name() == pack_name} if pack_name else None)
        if hero:
            hero = str(int(hero))
            pairs = [(pack, entry) for pack, entry in pairs if owner(entry.path) == hero]
        total = len(pairs)
        start = max(0, (page - 1) * page_size)
        return {"total": total, "page": page, "pageSize": page_size,
                "items": [self.row(*pair) for pair in pairs[start:start + page_size]]}

    def related(self, hero):
        hero = str(int(hero))
        pairs = [(pack, entry) for pack, entry in self.catalog.pairs if owner(entry.path) == hero]
        rows = [self.row(pack, entry, "角色编号 " + hero) for pack, entry in pairs]
        counts = {}
        for row in rows:
            counts[row["category"]] = counts.get(row["category"], 0) + 1
        return {"hero": hero, "total": len(rows), "counts": counts, "items": rows}


def make_handler(state):
    class Handler(BaseHTTPRequestHandler):
        server_version = "300HeroResourceBrowser/1.0"

        def log_message(self, *_):
            return

        def send_data(self, status, data, content_type):
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(data)

        def send_json(self, value, status=200):
            self.send_data(status, json_bytes(value), "application/json; charset=utf-8")

        def fail(self, exc, status=500):
            self.send_json({"error": str(exc)}, status)

        def do_GET(self):
            parsed = urlparse(self.path)
            path, query = unquote(parsed.path), parse_qs(parsed.query)
            try:
                if path == "/api/status":
                    self.send_json(state.summary()); return
                if path == "/api/select-game-dir":
                    selected = choose_game_dir(state.game_dir)
                    if not selected: self.fail(ValueError("没有选择目录"), 400); return
                    self.send_json(state.load(selected)); return
                if path == "/api/reload":
                    self.send_json(state.load(query.get("gameDir", [state.game_dir])[0])); return
                if path == "/api/search":
                    self.send_json(state.search(query.get("q", [""])[0], query.get("category", ["全部"])[0],
                        query.get("pack", [""])[0], query.get("hero", [""])[0],
                        max(1, int(query.get("page", [1])[0])), min(500, max(20, int(query.get("pageSize", [100])[0]))))); return
                if path == "/api/related":
                    self.send_json(state.related(query.get("hero", [""])[0])); return
                if path == "/api/live2d":
                    rid = query.get("id", [""])[0]
                    _pack, entry = state.pair(rid)
                    if state.live2d_preview:
                        try: state.live2d_preview.close()
                        except Exception: pass
                    state.live2d_preview = Live2DBrowserPreview(
                        [p for p in state.packs if p.parse_ok], entry.path, WEB / "live2d_viewer")
                    self.send_json({"view":"live2d-web", "name":state.live2d_preview.manifest["name"],
                        "motions":state.live2d_preview.manifest["motions"],
                        "url":"/live2d-runtime/index.html"}); return
                if path == "/api/audio-track":
                    pack, entry = state.pair(query.get("id", [""])[0])
                    raw = jc.read_entry(pack, entry)
                    info = inspect_bank(raw)
                    index = int(query.get("track", [0])[0])
                    if index < 0 or index >= len(info["tracks"]): raise ValueError("音轨编号超出范围")
                    self.send_data(200, decode_track(raw, info["tracks"][index]), "audio/wav"); return
                if path == "/api/model-frame":
                    rid = query.get("id", [""])[0]
                    model = state.model_cache.get(rid)
                    if model is None:
                        pack, entry = state.pair(rid); model = parse_raw_model(jc.read_entry(pack, entry)); state.model_cache[rid] = model
                    self.send_json({"meshes": model_frame(model, int(query.get("action", [0])[0]), float(query.get("time", [0])[0]))}); return
                if path in ("/api/preview", "/api/raw"):
                    pack, entry = state.pair(query.get("id", [""])[0])
                    raw = jc.read_entry(pack, entry)
                    ext = Path(entry.path).suffix.lower()
                    if path == "/api/raw":
                        if ext in IMAGE_EXTS:
                            image = Image.open(io.BytesIO(raw)).convert("RGBA")
                            out = io.BytesIO(); image.save(out, "PNG")
                            self.send_data(200, out.getvalue(), "image/png"); return
                        if ext in AUDIO_EXTS:
                            self.send_data(200, raw, mimetypes.guess_type(entry.path)[0] or "application/octet-stream"); return
                        self.fail(ValueError("该类型没有媒体预览流"), 400); return
                    result = State.row(pack, entry)
                    result["size"] = len(raw)
                    result["view"] = "image" if ext in IMAGE_EXTS else "audio" if ext in AUDIO_EXTS else "report"
                    if result["view"] == "image":
                        try:
                            Image.open(io.BytesIO(raw)).verify()
                        except Exception as exc:
                            result["view"] = "report"
                            result["report"] = inspect_resource(raw, ext) + "\n\n图片解码提示：" + str(exc)
                    if ext == ".dat":
                        try:
                            table = parse_dat_table(raw)
                            result.update(view="table", columns=list(table.columns), rows=[list(row) for row in table.rows[:1000]], rowCount=len(table.rows))
                        except Exception as exc:
                            result["report"] = inspect_resource(raw, ext) + "\n\nDAT 表格解析失败：" + str(exc)
                    elif ext in {".bank", ".fsb"}:
                        try:
                            audio = inspect_bank(raw)
                            result.update(view="bank", tracks=[{"name":t["name"], "codec":t["codec"],
                                "rate":t["rate"], "channels":t["channels"], "duration":t["duration"]}
                                for t in audio["tracks"][:1000]])
                            result["report"] = "FMOD 音频资源\n音频块：%d\n音轨：%d\n\n%s" % (audio["fsb5"], audio["track_count"], "\n".join(
                                "%d. %s · %s · %dHz · %.2fs" % (i+1, t["name"], t["codec"], t["rate"], t["duration"]) for i,t in enumerate(audio["tracks"][:500])))
                        except Exception as exc: result["report"] = inspect_resource(raw, ext) + "\n\n音频解析提示：" + str(exc)
                    elif ext in {".x", ".x3d", ".eg3d"}:
                        try:
                            rid = f"{Path(pack.file).name}:{entry.index}"
                            parsed_model = parse_raw_model(raw); state.model_cache[rid] = parsed_model
                            model = browser_model(parsed_model)
                            model_dir = entry.path.replace("\\", "/").rsplit("/", 1)[0]
                            texture_ids = {}
                            for texture in model.get("textures", []):
                                match = state.catalog.resolve_texture(texture, model_dir)
                                if match:
                                    tp,te=match; texture_ids[texture]=f"{Path(tp.file).name}:{te.index}"
                            model["textureIds"] = texture_ids
                            result.update(view="model", model=model, modelFrameApi="/api/model-frame?id=" + quote(rid))
                        except Exception as exc:
                            result["report"] = inspect_resource(raw, ext) + "\n\n模型预览解析失败：" + str(exc)
                    elif (ext in {".moc", ".mtn"} or entry.path.lower().endswith(".model.json")
                          or (ext == ".json" and "\\live2d\\" in entry.path.lower())):
                        result.update(view="live2d", live2dApi="/api/live2d?id=" + quote(f"{Path(pack.file).name}:{entry.index}"))
                    elif result["view"] == "report" and "report" not in result:
                        result["report"] = inspect_resource(raw, ext)
                    self.send_json(result); return
                if path.startswith("/live2d-runtime/"):
                    if not state.live2d_preview: self.fail(FileNotFoundError("没有正在预览的 Live2D"), 404); return
                    root = state.live2d_preview.root.resolve()
                    target = (root / path[len("/live2d-runtime/"):]).resolve()
                    if root not in target.parents or not target.is_file(): self.fail(FileNotFoundError("Live2D 文件不存在"), 404); return
                    self.send_data(200, target.read_bytes(), mimetypes.guess_type(target.name)[0] or "application/octet-stream"); return
                rel = "index.html" if path in ("", "/") else path.lstrip("/")
                target = (WEB / rel).resolve()
                if WEB.resolve() not in target.parents and target != WEB.resolve():
                    self.fail(PermissionError("禁止访问"), 403); return
                if not target.is_file(): self.fail(FileNotFoundError("文件不存在"), 404); return
                self.send_data(200, target.read_bytes(), mimetypes.guess_type(target.name)[0] or "application/octet-stream")
            except FileNotFoundError as exc: self.fail(exc, 404)
            except (ValueError, jc.JmpError) as exc: self.fail(exc, 400)
            except Exception as exc: self.fail(exc)

    return Handler


def main():
    parser = argparse.ArgumentParser(description="300英雄资源浏览器")
    parser.add_argument("--game-dir")
    parser.add_argument("--no-browser", action="store_true")
    args = parser.parse_args()
    requested = args.game_dir or load_settings(SETTINGS).get("gameDir") or DEFAULT_GAME_DIR
    if not has_jmp_files(requested): requested = choose_game_dir(requested) or requested
    state = State(os.path.abspath(requested))
    state.load()
    server = ThreadingHTTPServer(("127.0.0.1", PORT), make_handler(state))
    url = f"http://127.0.0.1:{PORT}/"
    print("300英雄资源浏览器已启动：" + url)
    if not args.no_browser: threading.Timer(0.5, lambda: webbrowser.open(url)).start()
    try: server.serve_forever()
    except KeyboardInterrupt: pass
    finally: server.server_close()


if __name__ == "__main__":
    main()
