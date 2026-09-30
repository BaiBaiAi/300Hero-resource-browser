"""Extract a Cubism 2 model family from JMP and serve the browser preview."""
from __future__ import annotations

import json
import re
import shutil
import tempfile
from pathlib import Path, PureWindowsPath

import jmp_core as jc


class Live2DError(ValueError):
    pass


def _norm(path: str) -> str:
    return path.replace('/', '\\').lower()


def _clean_json(raw: bytes):
    text = raw.decode('utf-8-sig', 'replace')
    # Some shipped legacy model settings are JavaScript object literals rather
    # than strict JSON (for example `sound:` without quotes).
    text = re.sub(r'([,{]\s*)([A-Za-z_$][\w$]*)"?\s*:', r'\1"\2":', text)
    text = re.sub(r',\s*([}\]])', r'\1', text)
    return json.loads(text)


def _catalog(packs):
    return {_norm(entry.path): (pack, entry) for pack in packs for entry in pack.entries}


def _find_model_json(catalog, selected_path):
    selected = PureWindowsPath(selected_path)
    ext = selected.suffix.lower()
    base = selected.parent.parent if ext == '.mtn' else selected.parent
    prefix = _norm(str(base)) + '\\'
    direct_json = [(path, pair) for path, pair in catalog.items()
                   if PureWindowsPath(path).parent == PureWindowsPath(_norm(str(base)))
                   and path.endswith('.json')]
    matches = []
    for path, pair in direct_json:
        try:
            settings = _clean_json(jc.read_entry(*pair))
        except (OSError, ValueError, json.JSONDecodeError):
            continue
        # 300英雄的一部分 Cubism 2 配置名为 512.json 等，并不以
        # .model.json 结尾。用内容判断，避免把同目录的普通 JSON 当模型配置。
        if isinstance(settings, dict) and settings.get('model') and settings.get('textures'):
            matches.append((path, pair, settings))
    if not matches:
        raise Live2DError('同一 Live2D 目录中没有找到有效的模型配置 JSON')
    if ext == '.moc':
        stem = selected.stem.lower()
        preferred = [item for item in matches
                     if PureWindowsPath(str(item[2].get('model', ''))).stem.lower() == stem]
        if preferred:
            matches = preferred
    matches.sort(key=lambda item: (not item[0].endswith('.model.json'), item[0]))
    return matches[0]


def extract_preview(packs, selected_path: str, destination) -> dict:
    """Extract one model.json family to a safe temporary web directory."""
    catalog = _catalog(packs)
    model_path, (model_pack, model_entry), settings = _find_model_json(catalog, selected_path)
    model_dir = str(PureWindowsPath(model_path).parent)
    required = [settings.get('model')]
    required.extend(settings.get('textures') or [])
    if settings.get('physics'):
        required.append(settings['physics'])
    motions = []
    for group, definitions in (settings.get('motions') or {}).items():
        for index, definition in enumerate(definitions or []):
            filename = definition.get('file')
            if filename:
                required.append(filename)
                motions.append({'label': '%s · %s' % (group, PureWindowsPath(filename).name),
                                'url': './assets/' + filename.replace('\\', '/'), 'group': group,
                                'index': index})
    # Some shipped head models omit the motions object entirely. Discover all
    # sibling .mtn files and synthesize a Cubism 2 motion group for the viewer.
    known_motion_paths = {_norm(model_dir + '\\' + item['url'][9:].replace('/', '\\'))
                          for item in motions}
    discovered = []
    model_prefix = _norm(model_dir) + '\\'
    selected_norm = _norm(selected_path)
    for path in catalog:
        if path.startswith(model_prefix) and path.endswith('.mtn') and path not in known_motion_paths:
            relative = path[len(model_prefix):]
            discovered.append((path != selected_norm, relative, path))
    discovered.sort()
    if discovered:
        clean_motions = dict(settings.get('motions') or {})
        group = list(clean_motions.get('auto') or [])
        for _not_selected, relative, _path in discovered:
            group.append({'file': relative.replace('\\', '/')})
            required.append(relative)
            motions.append({'label': 'auto · ' + PureWindowsPath(relative).name,
                            'url': './assets/' + relative.replace('\\', '/'),
                            'group': 'auto', 'index': len(group) - 1})
        clean_motions['auto'] = group
        settings = dict(settings)
        settings['motions'] = clean_motions
    missing = []
    root = Path(destination) / 'assets'
    root.mkdir(parents=True, exist_ok=True)
    for relative in dict.fromkeys(item for item in required if item):
        source_path = _norm(model_dir + '\\' + relative)
        pair = catalog.get(source_path)
        if not pair:
            missing.append(relative)
            continue
        target = root.joinpath(*PureWindowsPath(relative).parts)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(jc.read_entry(*pair))
    if missing:
        raise Live2DError('模型依赖缺失：' + '、'.join(missing))
    # Rewrite invalid legacy trailing commas and keep only browser-safe relative paths.
    clean_settings = dict(settings)
    (root / 'model.json').write_text(json.dumps(clean_settings, ensure_ascii=False, indent=2), encoding='utf-8')
    manifest = {'name': settings.get('name') or PureWindowsPath(model_path).stem,
                'source': model_path, 'textureCount': len(settings.get('textures') or []),
                'motions': motions}
    (root / 'manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding='utf-8')
    return manifest


class Live2DBrowserPreview:
    """Extract a model family beside the bundled community web viewer."""

    def __init__(self, packs, selected_path: str, viewer_dir):
        self.temp = tempfile.TemporaryDirectory(prefix='rs_live2d_browser_')
        self.root = Path(self.temp.name)
        shutil.copytree(viewer_dir, self.root, dirs_exist_ok=True)
        self.manifest = extract_preview(packs, selected_path, self.root)

    def close(self):
        self.temp.cleanup()
