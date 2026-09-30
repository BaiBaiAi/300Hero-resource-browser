"""Content-based readers for the remaining JMP resource formats."""
from __future__ import annotations

import math
import re
import struct
from collections import Counter


TEXT_EXTENSIONS = {
    '.ini', '.txt', '.lua', '.xml', '.json', '.csv', '.log', '.htm', '.html',
    '.mtn', '.fx', '.psh', '.ps', '.vs', '.string', '.tab',
}


def decode_text(raw: bytes) -> tuple[str, str] | None:
    """Return decoded text and encoding only when the byte stream is plausibly text."""
    if raw.startswith((b'\xff\xfe', b'\xfe\xff')):
        encoding = 'utf-16' if raw.startswith(b'\xff\xfe') else 'utf-16-be'
        return raw.decode(encoding, 'replace').lstrip('\ufeff'), encoding
    if raw.startswith(b'\xef\xbb\xbf'):
        return raw.decode('utf-8-sig', 'replace'), 'utf-8-sig'
    for encoding in ('utf-8', 'gb18030'):
        try:
            text = raw.decode(encoding)
        except UnicodeDecodeError:
            continue
        if not text:
            return text, encoding
        good = sum(ch.isprintable() or ch in '\r\n\t' for ch in text)
        if good / len(text) >= .92 and '\x00' not in text:
            return text, encoding
    return None


def _ascii_strings(raw: bytes, minimum=4, limit=400):
    pattern = rb'[\x20-\x7e]{%d,}' % minimum
    return [(m.start(), m.group().decode('ascii')) for m in re.finditer(pattern, raw)][:limit]


def _utf16_strings(raw: bytes, minimum=4, limit=100):
    pattern = rb'(?:(?:[\x20-\x7e]|[\x80-\xff])\x00){%d,}' % minimum
    out = []
    for match in re.finditer(pattern, raw):
        value = match.group().decode('utf-16le', 'replace')
        if sum(ch.isprintable() for ch in value) / len(value) >= .9:
            out.append((match.start(), value))
            if len(out) >= limit:
                break
    return out


def _short(value: str, size=180):
    value = value.replace('\r', '\\r').replace('\n', '\\n').replace('\t', '\\t')
    return value if len(value) <= size else value[:size] + '…'


def inspect_live2d_motion(raw: bytes) -> str:
    decoded = decode_text(raw)
    if not decoded:
        return inspect_binary(raw, '.mtn')
    text, encoding = decoded
    settings, tracks = {}, []
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith('#'):
            continue
        if '=' not in line:
            continue
        name, value = line.split('=', 1)
        if name.startswith('$'):
            settings[name[1:]] = value
        else:
            values = value.split(',') if value else []
            tracks.append((name, len(values), values[:3], values[-3:]))
    fps = settings.get('fps', '?')
    frames = max((item[1] for item in tracks), default=0)
    try:
        duration = frames / float(fps)
        duration_text = '%.2f 秒' % duration
    except (TypeError, ValueError, ZeroDivisionError):
        duration_text = '未知'
    lines = ['🎭 Live2D 动作数据', f'编码: {encoding}', f'帧率: {fps}',
             f'淡入/淡出: {settings.get("fadein", "?")} / {settings.get("fadeout", "?")} ms',
             f'参数轨道: {len(tracks)}    最大帧数: {frames}    估算时长: {duration_text}', '',
             '参数轨道（名称 / 帧数 / 首尾采样）:']
    lines.extend('  · %s / %d / %s … %s' % (name, count, ', '.join(first), ', '.join(last))
                 for name, count, first, last in tracks[:200])
    if len(tracks) > 200:
        lines.append('  …其余 %d 条轨道省略' % (len(tracks) - 200))
    lines.extend(['', '原始文本:', '-' * 80, text[:200_000]])
    if len(text) > 200_000:
        lines.append('\n（文本过长，预览仅显示前 200000 个字符）')
    return '\n'.join(lines)


def inspect_live2d_model(raw: bytes) -> str:
    strings = _ascii_strings(raw, 4, 2000)
    params = []
    textures = []
    for _, value in strings:
        if value.startswith(('Param', 'PARAM_')) and value not in params:
            params.append(value)
        if re.search(r'\.(?:png|dds|tga|jpg)$', value, re.I) and value not in textures:
            textures.append(value)
    version = raw[3] if raw[:3] == b'moc' and len(raw) > 3 else '?'
    lines = ['🎭 Live2D Cubism 2 模型（MOC）', f'格式版本字节: {version}',
             f'文件大小: {len(raw)} 字节', f'可识别参数: {len(params)}',
             f'纹理引用: {len(textures)}', '', '参数名:']
    lines.extend('  · ' + value for value in params[:300])
    if textures:
        lines.extend(['', '纹理引用:'] + ['  · ' + value for value in textures[:100]])
    lines.append('\n说明: MOC 是 Live2D 编译模型；当前显示公开字符串和结构摘要。')
    return '\n'.join(lines)


def inspect_legacy_ui(raw: bytes) -> str:
    strings = _ascii_strings(raw, 4, 4000)
    version = raw.split(b'\0', 1)[0].decode('ascii', 'replace') if raw else '?'
    controls, textures, other = [], [], []
    for offset, value in strings:
        item = (offset, value)
        if value.startswith('ID_'):
            controls.append(item)
        elif re.search(r'\.(?:dds|png|tga|bmp|jpg)$', value, re.I):
            textures.append(item)
        elif value != version and len(value) >= 4:
            other.append(item)
    lines = ['🧩 旧版 UI 二进制布局', f'版本标识: {version}', f'文件大小: {len(raw)} 字节',
             f'控件标识: {len(controls)}    贴图引用: {len(textures)}    其他字符串: {len(other)}', '',
             '控件:']
    lines.extend('  0x%08X  %s' % item for item in controls[:500])
    lines.extend(['', '贴图:'])
    lines.extend('  0x%08X  %s' % item for item in textures[:500])
    if other:
        lines.extend(['', '其他可读字符串:'])
        lines.extend('  0x%08X  %s' % (off, _short(value)) for off, value in other[:300])
    return '\n'.join(lines)


def inspect_xmap(raw: bytes) -> str:
    if len(raw) < 12:
        return inspect_binary(raw, '.xmap')
    version, reserved, count = struct.unpack_from('<fII', raw)
    pos, records = 12, []
    try:
        for _ in range(min(count, 1_000_000)):
            end = raw.index(b'\n', pos); model = raw[pos:end].decode('gb18030', 'replace'); pos = end + 1
            end = raw.index(b'\n', pos); group = raw[pos:end].decode('gb18030', 'replace'); pos = end + 1
            if pos + 40 > len(raw):
                raise ValueError
            flag = struct.unpack_from('<I', raw, pos)[0]; pos += 4
            values = struct.unpack_from('<9f', raw, pos); pos += 36
            records.append((model, group, flag, values))
    except (ValueError, struct.error):
        return inspect_binary(raw, '.xmap', note='XMAP 记录在 0x%X 处不完整' % pos)
    lines = ['🗺 XMAP 场景对象表', f'版本值: {version:g}    保留值: {reserved}',
             f'声明对象数: {count}    成功读取: {len(records)}    剩余: {len(raw)-pos} 字节', '',
             '序号 | 模型 | 分组 | 标志 | 位置 XYZ | 旋转 XYZ | 缩放 XYZ']
    for index, (model, group, flag, values) in enumerate(records[:1000], 1):
        nums = ' | '.join(','.join('%.4g' % value for value in values[i:i+3]) for i in (0, 3, 6))
        lines.append('%d | %s | %s | %d | %s' % (index, model, group, flag, nums))
    if len(records) > 1000:
        lines.append('…其余 %d 条记录省略' % (len(records) - 1000))
    return '\n'.join(lines)


def inspect_map(raw: bytes) -> str:
    entries = []
    pos = 0
    while pos + 8 <= min(len(raw), 4096):
        tag = raw[pos:pos+4]
        if not all(32 <= byte < 127 for byte in tag):
            break
        value = struct.unpack_from('<I', raw, pos + 4)[0]
        entries.append((pos, tag.decode('ascii'), value))
        # inf0 is a sized metadata chunk; following entries form a tag/offset directory.
        pos += 8 + value if pos == 0 and tag == b'inf0' and value <= 1024 else 8
    strings = _ascii_strings(raw, 4, 300)
    lines = ['🗺 MAP 场景容器', f'文件大小: {len(raw)} 字节',
             f'头部段目录: {len(entries)} 项', '', '段标记 / 数值（大小或绝对偏移）:']
    lines.extend('  0x%08X  %-4s  0x%08X (%d)' % (off, tag, value, value)
                 for off, tag, value in entries)
    lines.extend(['', '文件内可读标识（前 300 项）:'])
    lines.extend('  0x%08X  %s' % (off, _short(value)) for off, value in strings)
    lines.append('\n说明: MAP 含地形、纹理和关卡二进制段；当前展示段目录与内部标识。')
    return '\n'.join(lines)


def inspect_ttf(raw: bytes) -> str:
    if len(raw) < 12:
        return inspect_binary(raw, '.ttf')
    scaler, count = raw[:4], struct.unpack_from('>H', raw, 4)[0]
    tables = []
    for index in range(min(count, 4096)):
        pos = 12 + index * 16
        if pos + 16 > len(raw):
            break
        tag, checksum, offset, length = struct.unpack_from('>4sIII', raw, pos)
        tables.append((tag.decode('latin1'), checksum, offset, length,
                       offset + length <= len(raw)))
    names = []
    name_entry = next((item for item in tables if item[0] == 'name' and item[4]), None)
    if name_entry:
        base, length = name_entry[2], name_entry[3]
        try:
            _, ncount, storage = struct.unpack_from('>HHH', raw, base)
            for i in range(min(ncount, 1000)):
                platform, encoding, language, name_id, size, offset = struct.unpack_from('>HHHHHH', raw, base+6+i*12)
                start = base + storage + offset; data = raw[start:start+size]
                text = data.decode('utf-16-be' if platform in (0, 3) else 'latin1', 'replace')
                if text and text not in [item[-1] for item in names]:
                    names.append((name_id, language, text))
        except (struct.error, UnicodeError):
            pass
    lines = ['🔤 TrueType/OpenType 字体', '缩放器: ' + scaler.hex(' '),
             f'文件大小: {len(raw)} 字节    表数量: {count}    有效目录项: {len(tables)}', '', '名称记录:']
    lines.extend('  nameID=%d lang=0x%04X  %s' % item for item in names[:100])
    lines.extend(['', '字体表:'])
    lines.extend('  %-4s  offset=0x%08X  size=%d  %s' % (tag, off, size, '有效' if valid else '越界')
                 for tag, _, off, size, valid in tables)
    return '\n'.join(lines)


def inspect_binary(raw: bytes, ext='', note='') -> str:
    scan_limit = 16 * 1024 * 1024
    scanned = raw if len(raw) <= scan_limit else raw[:8*1024*1024] + raw[-8*1024*1024:]
    counts = Counter(scanned)
    entropy = -sum((count/len(scanned)) * math.log2(count/len(scanned)) for count in counts.values()) if scanned else 0
    ascii_values = _ascii_strings(scanned, 4, 400)
    utf16_values = _utf16_strings(scanned, 4, 100)
    words = [struct.unpack_from('<I', raw, pos)[0] for pos in range(0, min(len(raw)-3, 64), 4)]
    lines = ['📦 二进制资源结构摘要', f'扩展名: {ext or "无"}    大小: {len(raw)} 字节',
             '文件头: ' + raw[:32].hex(' '), f'字节熵: {entropy:.3f} bit/byte',
             '前部小端 uint32: ' + ', '.join(str(value) for value in words)]
    if len(raw) > scan_limit:
        lines.append('字符串/熵扫描范围: 文件头尾各 8 MB（大文件限流）')
    if note:
        lines.append('解析提示: ' + note)
    lines.extend(['', f'ASCII 字符串 ({len(ascii_values)} 项，最多显示 400):'])
    lines.extend('  0x%08X  %s' % (off, _short(value)) for off, value in ascii_values)
    if utf16_values:
        lines.extend(['', f'UTF-16LE 字符串 ({len(utf16_values)} 项，最多显示 100):'])
        lines.extend('  0x%08X  %s' % (off, _short(value)) for off, value in utf16_values)
    preview = raw[:4096]
    lines.extend(['', '十六进制（前 4KB）:'])
    lines.extend('%08X  %-47s  %s' % (i, ' '.join('%02X' % b for b in preview[i:i+16]),
                 ''.join(chr(b) if 32 <= b < 127 else '.' for b in preview[i:i+16]))
                 for i in range(0, len(preview), 16))
    return '\n'.join(lines)


def inspect_resource(raw: bytes, ext: str) -> str:
    """Return a readable format-specific report for a remaining resource type."""
    ext = ext.lower()
    if ext == '.mtn' or raw.startswith(b'# Live2D Animator Motion Data'):
        return inspect_live2d_motion(raw)
    if ext == '.moc' or raw[:3] == b'moc':
        return inspect_live2d_model(raw)
    if ext == '.ui' and raw.startswith(b'20110629a\0'):
        return inspect_legacy_ui(raw)
    if ext == '.xmap':
        return inspect_xmap(raw)
    if ext == '.map' and raw[:4] == b'inf0':
        return inspect_map(raw)
    if ext == '.ttf' or raw[:4] in (b'\x00\x01\x00\x00', b'OTTO'):
        return inspect_ttf(raw)
    decoded = decode_text(raw)
    if decoded:
        text, encoding = decoded
        return '📄 文本资源\n编码: %s    字符数: %d\n%s\n%s' % (encoding, len(text), '-'*80,
                                                                       text[:500_000])
    return inspect_binary(raw, ext)
