from __future__ import annotations
import math, struct, zlib
import numpy as np
from jumpx_backend import clean_tracks
from eg3d_backend import parse_model as parse_eg3d_model

JUMPX_INDEX_HEAD = b'WUYAXI@SINA.CN\x00'
JUMPX_OFF_OBF = 1000000000

def _cs(buf, off, maxlen=96):
    """从 buf[off:] 读取 null 结尾字符串。"""
    if not (0 <= off < len(buf)):
        return ''
    end = buf.find(b'\x00', off, off + maxlen)
    if end == -1:
        end = min(off + maxlen, len(buf))
    return buf[off:end].decode('gbk', 'replace')


def parse_jumpx_model(raw: bytes) -> dict:
    """完整解析 JUMPX (.x)：返回几何体顶点/索引/材质，供 3D 渲染。

    格式依据社区 JumpXToolchain (github.com/silent-reader-cn)：
      头 80B → version u32 → 表长 u32 → 表项 [name4][vsz4][value] ×N
      → indexSize/modelSize/indexSizeCompressed/modelSizeCompressed 各 u32
      → zlib(index 流: 'WUYAXI@SINA.CN\\0' + 结构体数组)
      → zlib(model 流: 顶点 float3 / 法线 float3 / uv float2 / 索引 u16×3)
      偏移字段存储时 +1000000000 混淆。
    """
    if len(raw) < 104 or raw[:5] != b'JUMPX':
        raise ValueError('非 JUMPX 模型或文件头不完整')
    off = 80
    version = struct.unpack_from('<I', raw, off)[0]; off += 4
    table_len = struct.unpack_from('<I', raw, off)[0]; off += 4
    fields = {}
    table_end = off + table_len
    if table_end + 16 > len(raw):
        raise ValueError('JUMPX 区块表超出文件边界')
    while off + 12 <= table_end:
        name = raw[off:off + 4].decode('latin1', 'replace')
        vsz = struct.unpack_from('<I', raw, off + 4)[0]
        if vsz < 1 or off + 8 + vsz > table_end:
            raise ValueError('JUMPX 无效区块表项')
        val = raw[off + 8:off + 8 + vsz]
        off += 8 + vsz
        fields[name] = struct.unpack('<I', val)[0] if vsz == 4 else val
    if off != table_end:
        raise ValueError('JUMPX 区块表长度错误')
    index_size, model_size, index_c, model_c = struct.unpack_from('<IIII', raw, off)
    off += 16
    if off + index_c + model_c > len(raw):
        raise ValueError('JUMPX 压缩数据被截断')
    index_buf = zlib.decompress(raw[off:off + index_c])
    model_buf = zlib.decompress(raw[off + index_c: off + index_c + model_c])
    if len(index_buf) != index_size or len(model_buf) != model_size:
        raise ValueError('JUMPX 解压长度与声明不一致')
    out = {'format': 'jumpX', 'version': version, 'fields': fields,
           'geometries': [], 'textures': [], 'materials': [],
           'bones': [], 'actions': [], 'particles': []}
    # 粒子特效记录（已逆向, 2026-09-12 实证 034.x/033_skin6.x 共 7 条）：
    #   aprt = index 流偏移；每条记录定长 1068B：
    #   +0 u32 model流参数偏移(40B/条) · +4 标志 · +8 特效名[80B]
    #   +88 u32 挂点骨骼索引 · +92 相对骨骼偏移 3×f32 · 其余为发射参数
    nprt = fields.get('nprt', 0)
    aprt = fields.get('aprt', 0)
    if nprt and isinstance(aprt, int) and 0 <= aprt < len(index_buf):
        for _pi in range(nprt):
            _b = aprt + _pi * 1068
            if _b + 104 > len(index_buf):
                break
            _u0, _fl = struct.unpack_from('<II', index_buf, _b)
            _nm = index_buf[_b + 8:_b + 88].split(b'\x00')[0].decode(
                'gbk', 'replace')
            _bi, _ox, _oy, _oz = struct.unpack_from('<I3f', index_buf, _b + 88)
            out['particles'].append({'name': _nm, 'flags': _fl, 'moff': _u0,
                                     'bone': _bi, 'off': (_ox, _oy, _oz)})
    # 纹理名
    ntex, atex = fields.get('ntex', 0), fields.get('atex', 0)
    for i in range(ntex):
        b = atex + i * 8
        if b + 8 > len(index_buf):
            break
        name_off = struct.unpack_from('<I', index_buf, b + 4)[0]
        nm = _cs(index_buf, name_off)
        out['textures'].append(nm)
    # 材质（记录 48B：+0 名字偏移 +8 混合标志(0x14000/0x10000) +12 纹理索引 +.. 参数；
    # 尾部 u32 为 model 流参数块偏移+1e9）
    nmtl, amtl = fields.get('nmtl', 0), fields.get('amtl', 0)
    for i in range(nmtl):
        b = amtl + i * 48
        if b + 48 > len(index_buf):
            break
        words = struct.unpack_from('<12I', index_buf, b)
        nm = _cs(index_buf, words[0])
        out['materials'].append({'name': nm, 'tex': words[3]})
    # 几何体（v8: 记录 124B，偏移 u32 混淆+1e9：
    #  +0 headOff +4 0x40 +8 nameOff +12 objId +16 matId +20 flags[2]
    #  +28 vcount +32 fcount +36 pos(f3) +40 normal(f3) +44 extra(f3)
    #  +52 uv(f2) +68 joints +76 INDEX(u16×3×fcount) +60 invisible
    #  +64 mainBone +68 bonesOff +72 box[6] +96 unknown1 +100..123 扩展）
    ngeo, ageo = fields.get('ngeo', 0), fields.get('ageo', 0)
    for i in range(ngeo):
        b = ageo + i * 124
        if b + 124 > len(index_buf):
            break
        geo_flags = struct.unpack_from('<I', index_buf, b + 4)[0]
        # 0x40/0x80 are mesh feature flags, not required vertex layout tags.
        # Static map meshes also use zero with the same 124-byte descriptor.
        geo_layout = geo_flags & ~0xC0
        if geo_layout not in (0, 3):
            raise ValueError('未支持的 JUMPX 网格布局: 0x%X' % geo_flags)
        name_off = struct.unpack_from('<I', index_buf, b + 8)[0]
        mat_id = struct.unpack_from('<I', index_buf, b + 16)[0]
        vcount = struct.unpack_from('<I', index_buf, b + 28)[0]
        fcount = struct.unpack_from('<I', index_buf, b + 32)[0]
        voff = struct.unpack_from('<I', index_buf, b + 36)[0]
        noff = struct.unpack_from('<I', index_buf, b + 44)[0]
        uoff = struct.unpack_from('<I', index_buf, b + 52)[0]
        ioff = struct.unpack_from('<I', index_buf, b + 76)[0]
        box = struct.unpack_from('<6f', index_buf, b + 96)
        jon = struct.unpack_from('<Q', index_buf, b + 68)[0]   # 逐顶点关节表(model流)
        jon = jon - JUMPX_OFF_OBF if jon >= JUMPX_OFF_OBF else jon
        bsoff = struct.unpack_from('<I', index_buf, b + 92)[0]  # 蒙皮权重块(model流)
        bsoff = (bsoff - JUMPX_OFF_OBF if bsoff >= JUMPX_OFF_OBF else bsoff) if bsoff else None
        voff -= JUMPX_OFF_OBF if voff >= JUMPX_OFF_OBF else 0
        noff -= JUMPX_OFF_OBF if noff >= JUMPX_OFF_OBF else 0
        uoff -= JUMPX_OFF_OBF if uoff >= JUMPX_OFF_OBF else 0
        ioff -= JUMPX_OFF_OBF if ioff >= JUMPX_OFF_OBF else 0
        g = {'name': _cs(index_buf, name_off), 'materialId': mat_id,
             'vertexCount': vcount, 'faceCount': fcount, 'box': box,
             'positions': None, 'indices': None, 'normals': None, 'uvs': None,
             'jointOff': jon, 'weightOff': bsoff, 'recordOffset': b}
        g['flags'] = geo_flags
        try:
            if geo_layout == 3 and voff == 0 and vcount:
                import numpy as np
                from jumpx_backend import unpack_position
                po = struct.unpack_from('<I', index_buf, b + 40)[0] - JUMPX_OFF_OBF
                no = struct.unpack_from('<I', index_buf, b + 48)[0] - JUMPX_OFF_OBF
                high = np.asarray(struct.unpack_from('<3f', index_buf, b + 96))
                low = np.asarray(struct.unpack_from('<3f', index_buf, b + 108))
                if po < 0 or no < 0:
                    raise ValueError('JUMPX 压缩网格地址无效')
                words = np.frombuffer(model_buf, '<u4', vcount, po)
                g['positions'] = tuple(unpack_position(words, low, high).ravel())
                normals = np.frombuffer(model_buf, 'i1', vcount * 3, no).reshape(-1,3).astype(float)
                normals /= np.maximum(np.linalg.norm(normals,axis=1,keepdims=True),1e-12)
                g['normals'] = tuple(normals.ravel())
            elif voff >= 0 and vcount:
                g['positions'] = struct.unpack_from('<%df' % (vcount * 3), model_buf, voff)
            if geo_layout != 3 and noff > 0 and vcount:
                g['normals'] = struct.unpack_from('<%df' % (vcount * 3), model_buf, noff)
            if uoff >= 0 and vcount:
                g['uvs'] = struct.unpack_from('<%df' % (vcount * 2), model_buf, uoff)
            if ioff >= 0 and fcount:
                g['indices'] = struct.unpack_from('<%dH' % (fcount * 3), model_buf, ioff)
        except struct.error as ex:
            g['error'] = str(ex)
        out['geometries'].append(g)

    # ---- 骨骼（index 流 172B 记录，结构与 jump2fbx XBone 一致）
    nbon, abon = fields.get('nbon', 0), fields.get('abon', 0)
    if nbon and abon:
        abon = abon - JUMPX_OFF_OBF if abon >= JUMPX_OFF_OBF else abon

        def _dobf(v):
            return v - JUMPX_OFF_OBF if v >= JUMPX_OFF_OBF else v
        for i in range(nbon):
            b = abon + i * 172
            if b + 172 > len(index_buf):
                break
            name_off, parent, fcount = struct.unpack_from('<3I', index_buf, b + 8)
            matrix = struct.unpack_from('<16f', index_buf, b + 24)
            vis_n, vis_o, pos_n, pos_o, _z3, rot_n = struct.unpack_from('<6I', index_buf, b + 132)
            rot_o, _z4, sc_n, sc_o = struct.unpack_from('<4I', index_buf, b + 156)
            out['bones'].append({
                'name': _cs(index_buf, name_off), 'recordOffset': b,
                'parent': -1 if parent == 0xFFFFFFFF else parent,
                'frameCount': fcount, 'matrix': matrix,
                'posCount': pos_n, 'posOff': _dobf(pos_o),
                'rotCount': rot_n, 'rotOff': _dobf(rot_o),
                'scaleCount': sc_n, 'scaleOff': _dobf(sc_o)})
        # ---- 动作表（index 流 90B 记录：name[80]+start/end/other u16+ffs）
        nact, aact = fields.get('nact', 0), fields.get('aact', 0)
        if nact and aact:
            aact = aact - JUMPX_OFF_OBF if aact >= JUMPX_OFF_OBF else aact
            for i in range(nact):
                b = aact + i * 90
                if b + 90 > len(index_buf):
                    break
                rec = index_buf[b:b + 90]
                nm = rec[:80].split(b'\x00')[0].decode('gbk', 'replace')
                start, end, other = struct.unpack_from('<3H', rec, 80)
                out['actions'].append({'name': nm, 'start': start,
                                       'end': end, 'other': other})
    # ---- 蒙皮权重（model 流，每顶点 24B：count u8 + bones[4]u8 + pad3 + weight[4]f32）
    jfmt = struct.Struct('<B4B3x4f')
    for g in out['geometries']:
        jo = g.get('weightOff', -1)
        g['joints'] = None
        n = g['vertexCount']
        if jo is not None and jo >= 0 and n and jo + n * 24 <= len(model_buf):
            try:
                g['joints'] = list(jfmt.iter_unpack(model_buf[jo:jo + n * 24]))
            except struct.error:
                g['joints'] = None
    # ---- 动画轨道（model 流：pos=帧×3f32，rot=帧×4f32 四元数 x,y,z,w）
    for b in out['bones']:
        b['posT'] = b['rotT'] = None
        if b['posCount'] and b['posOff'] >= 0 and \
                b['posOff'] + b['posCount'] * 12 <= len(model_buf):
            b['posT'] = bytes(model_buf[b['posOff']:
                                        b['posOff'] + b['posCount'] * 12])
        if b['rotCount'] and b['rotOff'] >= 0 and \
                b['rotOff'] + b['rotCount'] * 16 <= len(model_buf):
            b['rotT'] = bytes(model_buf[b['rotOff']:
                                        b['rotOff'] + b['rotCount'] * 16])
    # Scale and visibility are separate optional tracks, never inferred from
    # frameCount alone (which can be nonzero while an address is null).
    for bone in out['bones']:
        base = bone['recordOffset']
        bone['scaleT'] = bone['visibleT'] = None
        count, pointer = struct.unpack_from('<II', index_buf, base + 164)
        if count and pointer >= JUMPX_OFF_OBF:
            offset = pointer - JUMPX_OFF_OBF
            if offset + count * 12 > len(model_buf):
                raise ValueError('JUMPX 缩放轨道越界')
            bone['scaleT'] = model_buf[offset:offset + count * 12]
        count, pointer = struct.unpack_from('<II', index_buf, base + 132)
        if count and pointer >= JUMPX_OFF_OBF:
            offset = pointer - JUMPX_OFF_OBF
            if offset + count * 4 > len(model_buf):
                raise ValueError('JUMPX 显隐轨道越界')
            bone['visibleT'] = model_buf[offset:offset + count * 4]
    timeline = max((bone['frameCount'] for bone in out['bones']), default=1)
    for g in out['geometries']:
        g['visibleT'] = None
        pointer = struct.unpack_from('<I', index_buf, g['recordOffset'] + 60)[0]
        if pointer >= JUMPX_OFF_OBF:
            offset = pointer - JUMPX_OFF_OBF
            if offset + timeline * 4 <= len(model_buf):
                g['visibleT'] = model_buf[offset:offset + timeline * 4]
        if g.get('indices') and max(g['indices']) >= g['vertexCount']:
            raise ValueError('JUMPX 网格三角形索引越界')
    from jumpx_backend import clean_tracks
    out['warnings'] = clean_tracks(out['bones'])
    if not out['actions'] and timeline > 1:
        out['actions'] = [{'name': '完整动画', 'start': 0, 'end': timeline - 1}]
    valid_actions = []
    for action in out['actions']:
        if not 0 <= action['start'] <= action['end'] < timeline:
            out['warnings'].append('动作 %s 的帧范围 %d–%d 超出有效轨道，已跳过' %
                                   (action['name'], action['start'], action['end']))
        else:
            valid_actions.append(action)
    out['actions'] = valid_actions
    out['model_buf'] = model_buf
    return out



def parse_raw_model(raw: bytes):
    return parse_eg3d_model(raw) if raw[:4] == b'EG3D' else parse_jumpx_model(raw)


def browser_model(model):
    meshes=[]; total_v=total_t=0
    for gi, geometry in enumerate(model.get('geometries', [])):
        positions=geometry.get('positions') or []
        indices=geometry.get('indices') or []
        vc=len(positions)//3; tc=len(indices)//3
        if not vc or not tc: continue
        meshes.append({'sourceIndex':gi,'name':geometry.get('name') or 'mesh',
            'positions':[round(float(v),6) for v in positions],
            'normals':[round(float(v),6) for v in (geometry.get('normals') or [])],
            'uvs':[round(float(v),6) for v in (geometry.get('uvs') or [])],
            'indices':[int(v) for v in indices],
            'materialId':int(geometry.get('materialId',-1))})
        total_v+=vc; total_t+=tc
    if not meshes: raise ValueError('模型中没有可显示的三角网格')
    materials=[]
    for m in model.get('materials') or []:
        ti=m.get('tex'); texture=model.get('textures',[])[ti] if isinstance(ti,int) and 0<=ti<len(model.get('textures',[])) else ''
        materials.append({'texture':texture,'properties':m.get('properties') or {}})
    actions=[]
    for i,a in enumerate(model.get('actions') or []):
        actions.append({'index':i,'name':a.get('name') or '动作 %d'%(i+1),
                        'start':a.get('start',0),'end':a.get('end',0)})
    return {'format':model.get('format') or 'jumpX','version':model.get('version'),
            'meshes':meshes,'materials':materials,'textures':model.get('textures') or [],
            'vertices':total_v,'triangles':total_t,'bones':len(model.get('bones') or []),
            'actions':actions,'warnings':(model.get('warnings') or [])[:20]}


def model_frame(model, action_index, seconds):
    if model.get('format')=='EG3D':
        runtime=model.get('eg3d'); clips=runtime.clips if runtime else []
        clip=max(0,min(int(action_index),len(clips)-1)) if clips else None
        duration=clips[clip]['duration_ms'] if clip is not None else 0
        ms=(float(seconds)*1000)%duration if duration else 0
        values=runtime.evaluate(clip,ms) if runtime else []
    else:
        from jumpx_backend import Animation
        runtime=model.get('_browser_animation')
        if runtime is None:
            runtime=Animation(model);model['_browser_animation']=runtime
        actions=model.get('actions') or []
        a=actions[max(0,min(int(action_index),len(actions)-1))] if actions else {'start':0,'end':runtime.frames-1}
        start=float(a.get('start',0));end=float(a.get('end',runtime.frames-1)); span=max(1,end-start+1)
        values=runtime.evaluate(start+(float(seconds)*30)%span)
    return [{'positions':[round(float(x),6) for x in pos.ravel()], 'visible':bool(visible)} for pos,_normal,visible in values]
