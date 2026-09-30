from __future__ import annotations

import collections
import posixpath
import re


TEXTURES = {".dds", ".tga", ".png", ".bmp", ".jpg", ".jpeg", ".webp"}
MODELS = {".x", ".model"}


def key(path):
    value = path.replace("\\", "/").lower()
    return value[3:] if value.startswith("../") else value


def kind(path):
    value = key(path)
    ext = posixpath.splitext(value)[1]
    if "/live2d/" in value or ext in {".moc", ".mtn"} or value.endswith(".model.json"):
        return "Live2D"
    if ext in TEXTURES:
        return "贴图"
    if "/audio/" in value or ext in {".bank", ".wav", ".ogg", ".mp3", ".fsb"}:
        return "语音/音效"
    if "/effect/" in value or "/magic/" in value:
        return "技能特效"
    if ext in MODELS:
        return "模型"
    return "配置/其他"


def owner(path):
    value = key(path)
    patterns = [
        r"/character/(?:roleaction|role)/(\d+)(?=[_/.])",
        r"/audio/hero/(\d+)(?=[_/.])",
        r"/audio/pick/pick(\d+)(?=[_.])",
        r"/(?:effect|magic)/skill/(\d+)(?=[_/])",
    ]
    for pattern in patterns:
        match = re.search(pattern, value)
        if match:
            return match.group(1)
    return None


class Catalog:
    def __init__(self, packs):
        self.pairs = [(pack, entry) for pack in packs if pack.parse_ok for entry in pack.entries]
        self.paths = collections.defaultdict(list)
        self.texture_names = collections.defaultdict(list)
        self.texture_stems = collections.defaultdict(list)
        for pair in self.pairs:
            normalized = key(pair[1].path)
            self.paths[normalized].append(pair)
            base = posixpath.basename(normalized)
            if posixpath.splitext(base)[1] in TEXTURES:
                self.texture_names[base].append(pair)
                self.texture_stems[posixpath.splitext(base)[0]].append(pair)

    def resolve_texture(self, texture, model_dir=""):
        """Resolve a material texture without rescanning the full JMP catalog."""
        normalized = key(texture)
        base = posixpath.basename(normalized)
        stem = posixpath.splitext(base)[0]
        candidates = self.texture_names.get(base) or self.texture_stems.get(stem) or ()
        if model_dir:
            directory = key(model_dir).rstrip("/") + "/"
            for pair in candidates:
                if key(pair[1].path).startswith(directory):
                    return pair
        return candidates[0] if candidates else None

    def search(self, query="", category="全部", packs=None):
        terms = query.lower().replace("\\", "/").split()
        return [(pack, entry) for pack, entry in self.pairs
                if (packs is None or pack.file in packs)
                and (category == "全部" or kind(entry.path) == category)
                and all(term in key(entry.path) for term in terms)]
