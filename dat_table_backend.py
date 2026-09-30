"""Read protobuf-style game ``.dat`` files as generic tables.

The game does not ship the matching ``.proto`` schemas.  This module therefore
keeps field numbers as column names and never guesses business meanings.
"""
from __future__ import annotations

import math
import struct
from dataclasses import dataclass


class DatTableError(ValueError):
    pass


@dataclass(frozen=True)
class DatTable:
    columns: tuple[str, ...]
    rows: tuple[tuple[object, ...], ...]
    container_field: int | None


def _varint(data: bytes, pos: int) -> tuple[int, int]:
    value = 0
    for shift in range(0, 70, 7):
        if pos >= len(data):
            raise DatTableError("varint 被截断")
        byte = data[pos]
        pos += 1
        value |= (byte & 0x7f) << shift
        if byte < 0x80:
            return value, pos
    raise DatTableError("varint 超过 10 字节")


def _wire_fields(data: bytes, *, max_fields: int = 1_000_000):
    fields = []
    pos = 0
    while pos < len(data):
        key, pos = _varint(data, pos)
        number, wire = key >> 3, key & 7
        if not number or wire not in (0, 1, 2, 5):
            raise DatTableError("无效 protobuf 字段")
        if wire == 0:
            value, pos = _varint(data, pos)
        elif wire == 1:
            if pos + 8 > len(data):
                raise DatTableError("fixed64 字段被截断")
            value = data[pos:pos + 8]
            pos += 8
        elif wire == 5:
            if pos + 4 > len(data):
                raise DatTableError("fixed32 字段被截断")
            value = data[pos:pos + 4]
            pos += 4
        else:
            size, pos = _varint(data, pos)
            if size > len(data) - pos:
                raise DatTableError("长度字段超出文件末尾")
            value = data[pos:pos + size]
            pos += size
        fields.append((number, wire, value))
        if len(fields) > max_fields:
            raise DatTableError("字段数量异常")
    return fields


def _text(data: bytes) -> str | None:
    if not data:
        return ""
    for encoding in ("utf-8", "gb18030"):
        try:
            value = data.decode(encoding)
        except UnicodeDecodeError:
            continue
        printable = sum(ch.isprintable() or ch in "\r\n\t" for ch in value)
        if printable / max(1, len(value)) >= .9 and "\x00" not in value:
            return value
    return None


def _scalar(wire: int, value):
    if wire == 0:
        return value
    if wire == 1:
        unsigned = int.from_bytes(value, "little")
        floating = struct.unpack("<d", value)[0]
        return floating if math.isfinite(floating) and abs(floating) >= 1e-100 else unsigned
    if wire == 5:
        unsigned = int.from_bytes(value, "little")
        floating = struct.unpack("<f", value)[0]
        return floating if math.isfinite(floating) and abs(floating) >= 1e-30 else unsigned
    text = _text(value)
    return text if text is not None else "0x" + value.hex()


def _record(data: bytes, prefix: str = "", depth: int = 0) -> dict[str, object]:
    if depth > 5:
        return {prefix.rstrip("."): "0x" + data.hex()}
    values: dict[str, list[object]] = {}
    for number, wire, value in _wire_fields(data):
        name = f"{prefix}字段{number}"
        nested = None
        if wire == 2 and value:
            try:
                nested_fields = _wire_fields(value, max_fields=20_000)
                if nested_fields:
                    nested = _record(value, name + ".", depth + 1)
            except DatTableError:
                pass
        if nested is not None:
            for key, item in nested.items():
                values.setdefault(key, []).append(item)
        else:
            values.setdefault(name, []).append(_scalar(wire, value))
    return {key: items[0] if len(items) == 1 else " | ".join(map(str, items))
            for key, items in values.items()}


def parse_dat_table(raw: bytes) -> DatTable:
    """Parse a schema-less protobuf payload into rows and stable columns."""
    if not raw:
        raise DatTableError("文件为空")
    top = _wire_fields(raw)
    candidates: dict[int, list[bytes]] = {}
    for number, wire, value in top:
        if wire != 2 or not value:
            continue
        try:
            if _wire_fields(value, max_fields=20_000):
                candidates.setdefault(number, []).append(value)
        except DatTableError:
            pass
    if candidates:
        container, messages = max(candidates.items(), key=lambda item: (len(item[1]), -item[0]))
        # A single embedded message is still a useful one-row table.  If other
        # top-level values exist, retain those by treating the whole file as a row.
        if len(messages) == 1 and len(top) > 1:
            records = [_record(raw)]
            container = None
        else:
            records = [_record(message) for message in messages]
    else:
        records = [_record(raw)]
        container = None
    if not records or not any(records):
        raise DatTableError("未识别到可显示的数据字段")
    columns = []
    for record in records:
        for key in record:
            if key not in columns:
                columns.append(key)
    rows = tuple(tuple(record.get(column, "") for column in columns) for record in records)
    return DatTable(tuple(columns), rows, container)


def format_preview(table: DatTable, *, limit: int = 200) -> str:
    columns = ("行号",) + table.columns
    body = [(index,) + row for index, row in enumerate(table.rows[:limit], 1)]
    widths = []
    for col, name in enumerate(columns):
        values = [str(name)] + [str(row[col]) for row in body]
        widths.append(min(36, max(len(value) for value in values)))
    def line(row):
        return " | ".join(str(value)[:widths[i]].ljust(widths[i]) for i, value in enumerate(row))
    out = ["📊 二进制数据表（Protocol Buffers，无字段定义）",
           f"记录数: {len(table.rows)}    数据列: {len(table.columns)}",
           "列名按二进制字段编号显示；在获得对应 .proto 定义前不猜测字段业务含义。", "",
           line(columns), "-+-".join("-" * width for width in widths)]
    out.extend(line(row) for row in body)
    if len(table.rows) > limit:
        out.append(f"\n预览仅显示前 {limit} 条。")
    return "\n".join(out)
