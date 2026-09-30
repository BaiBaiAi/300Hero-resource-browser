"""EG3D V4 decoding and playback, independent of Tk/OpenGL.

Offsets are relative to file byte 12; metadata is one JSON array at
12 + u32(header+8). Format facts cross-checked against the public
DennisHerrm/io_scene_eg3d-1.14.0-300-Hero-importer- parser and game samples.
Implementation here is tailored to Resource Studio's geometry interface.
"""
import json
import math
import struct
import numpy as np


def segments(raw):
    if len(raw) < 12 or raw[:4] != b'EG3D':
        raise ValueError('非 EG3D 容器或文件头不完整')
    version, size = struct.unpack_from('<II', raw, 4)
    if version != 4:
        raise ValueError('暂不支持 EG3D 版本 %d' % version)
    end = 12 + size
    if end >= len(raw):
        raise ValueError('EG3D 二进制段超出文件边界')
    meta = json.loads(raw[end:].rstrip(b'\0 \r\n\t').decode('utf-8'))
    if not isinstance(meta, list) or len(meta) < 7:
        raise ValueError('EG3D 元数据段不完整')
    return raw[12:end], meta


class Reader:
    def __init__(self, data):
        self.data = data

    def array(self, offset, dtype, count, components=1):
        dt = np.dtype(dtype)
        if not isinstance(offset, int) or offset < 0 or count < 0:
            raise ValueError('EG3D 无效数据地址或数量')
        if offset + dt.itemsize * count * components > len(self.data):
            raise ValueError('EG3D 数据越界: offset=%d count=%d' % (offset, count))
        a = np.frombuffer(self.data, dt, count * components, offset)
        return a.reshape(count, components).copy()

    def attr(self, a, count):
        dtype = {0: 'u1', 1: 'u1', 2: '<u2', 3: 'i1', 4: '<f2', 5: '<f2'}.get(a[1])
        if dtype is None:
            raise ValueError('未知 EG3D 属性类型 %s' % a[1])
        comps = a[3]
        stride = max(4, comps) if a[1] in (0, 3) else comps
        raw = self.array(a[5], dtype, count, stride)
        if a[0] == 1 and a[1] == 3:
            packed = raw.astype('i1').view('<i2').reshape(count, 2).astype(float)
            h = np.clip(packed[:, 0] / 16384, -1, 1)
            angle = packed[:, 1] * (math.pi / 32768)
            radius = np.sqrt(1 - h*h)
            return np.column_stack((-radius*np.sin(angle), -h, -radius*np.cos(angle)))
        value = raw[:, :comps].astype(float)
        if a[4] == 1 and np.dtype(dtype).kind in 'iu':
            value /= np.iinfo(dtype).max
        if len(a) >= 8 and a[6] is not None and a[7] is not None:
            value = value * self.array(a[6], '<f4', 1, comps) + self.array(a[7], '<f4', 1, comps)
        if not np.isfinite(value).all():
            raise ValueError('EG3D 属性包含非有限数值')
        return value


def matrices(p, q, scale):
    q = q / np.maximum(np.linalg.norm(q, axis=1, keepdims=True), 1e-15)
    x, y, z, w = q.T
    m = np.zeros((len(p), 4, 4), dtype=float)
    m[:, 0, 0] = 1 - 2*(y*y+z*z)
    m[:, 0, 1] = 2*(x*y-z*w)
    m[:, 0, 2] = 2*(x*z+y*w)
    m[:, 1, 0] = 2*(x*y+z*w)
    m[:, 1, 1] = 1 - 2*(x*x+z*z)
    m[:, 1, 2] = 2*(y*z-x*w)
    m[:, 2, 0] = 2*(x*z-y*w)
    m[:, 2, 1] = 2*(y*z+x*w)
    m[:, 2, 2] = 1 - 2*(x*x+y*y)
    m[:, :3, :3] *= scale[:, None, :]
    m[:, :3, 3] = p
    m[:, 3, 3] = 1
    return m


def sample(times, values, t, rotation=False, step=False):
    hi = int(np.searchsorted(times, t, side='right'))
    if hi == 0:
        return values[0]
    if hi == len(times) or step:
        return values[hi-1]
    a, b = values[hi-1], values[hi]
    f = (t-times[hi-1]) / max(times[hi]-times[hi-1], 1e-15)
    if rotation:
        a = a / max(np.linalg.norm(a), 1e-15)
        b = b / max(np.linalg.norm(b), 1e-15)
        dot = np.dot(a, b)
        if dot < 0:
            b, dot = -b, -dot
        if dot < 0.9995:
            angle = math.acos(np.clip(dot, -1, 1))
            return (a*math.sin((1-f)*angle) + b*math.sin(f*angle))/math.sin(angle)
        v = a*(1-f) + b*f
        return v / max(np.linalg.norm(v), 1e-15)
    return a*(1-f) + b*f


class Animation:
    # Engine node space Y-up -> authored mesh space Z-up.
    align = np.array([[1,0,0,0], [0,0,-1,0], [0,1,0,0], [0,0,0,1]], dtype=float)

    def __init__(self, reader, meta):
        self.nodes = meta[5]
        n = len(self.nodes)
        self.p = np.zeros((n,3)); self.q = np.zeros((n,4)); self.q[:,3] = 1
        self.scale = np.ones((n,3)); self.parents = np.full(n,-1,dtype=int)
        for i, node in enumerate(self.nodes):
            for field, dst, comps in ((1,self.p,3),(2,self.q,4),(3,self.scale,3)):
                if node[field] is not None:
                    dst[i] = reader.array(node[field], '<f4', 1, comps)[0]
            for c in node[5] or []:
                if not 0 <= c < n or self.parents[c] != -1:
                    raise ValueError('EG3D 节点层级索引无效或多个父节点')
                self.parents[c] = i
        self.order = []
        pending = set(range(n))
        while pending:
            ready = sorted(i for i in pending if self.parents[i] not in pending)
            if not ready:
                raise ValueError('EG3D 节点层级存在循环')
            self.order.extend(ready); pending.difference_update(ready)
        self.palettes = []
        for ids, offset in meta[4]:
            ids = np.asarray(ids, dtype=int)
            if np.any((ids < 0) | (ids >= n)):
                raise ValueError('EG3D 骨骼组索引越界')
            inv = reader.array(offset, '<f4', len(ids), 16).reshape(-1,4,4).transpose(0,2,1)
            self.palettes.append((ids, inv))
        self.clips = []
        for clip in meta[7] if len(meta)>7 else []:
            tracks=[]; duration=0.; unsupported=set()
            for tr in clip[1]:
                node, kind, interp, count, comps, time_desc, val_desc = tr[:7]
                if not 0 <= node < n or count < 1:
                    raise ValueError('EG3D 动画轨道索引或键数量无效')
                if time_desc[1] != 5:
                    raise ValueError('未知 EG3D 动画时间编码')
                times = np.cumsum(reader.array(time_desc[0], '<u2', count).ravel(),dtype=float)
                dt = {0:'<f4', 6:'u1'}.get(val_desc[1])
                if dt is None:
                    raise ValueError('未知 EG3D 动画数值编码 %s' % val_desc[1])
                values=reader.array(val_desc[0],dt,count,comps).astype(float)
                if not np.isfinite(values).all():
                    raise ValueError('EG3D 动画包含非有限数值')
                if kind in (0,1,2) and comps != (4 if kind==1 else 3):
                    raise ValueError('EG3D TRS 轨道维数无效')
                if kind not in (0,1,2,'rendererenable','active'):
                    unsupported.add(str(kind))
                tracks.append((node,kind,interp,times,values))
                duration=max(duration,float(times[-1]))
            self.clips.append({'name':clip[0], 'duration_ms':duration, 'tracks':tracks,
                               'unsupported': sorted(unsupported)})
        self.geometries=[]

    def world(self, clip=None, ms=0):
        p,q,sc=self.p.copy(),self.q.copy(),self.scale.copy()
        visible=np.ones(len(p),dtype=bool)
        active=np.ones(len(p),dtype=bool)
        if clip is not None:
            for node,kind,interp,times,values in self.clips[clip]['tracks']:
                v=sample(times,values,ms,kind==1,interp==1 or isinstance(kind,str))
                if kind in (0,1,2):
                    (p,q,sc)[kind][node]=v
                elif kind=='rendererenable': visible[node]=bool(v[0])
                elif kind=='active': active[node]=bool(v[0])
        w=matrices(p,q,sc)
        for i in self.order:
            parent=self.parents[i]
            if parent>=0:
                w[i]=w[parent]@w[i]
                active[i] &= active[parent]
        return w, visible & active

    def evaluate(self, clip=None, ms=0, in_place=False):
        world, visible=self.world(clip,ms)
        palettes=[self.align@world[ids]@inv for ids,inv in self.palettes]
        shift=np.zeros(3)
        if in_place and clip is not None:
            root=next((i for i,n in enumerate(self.nodes) if n[0].lower() in ('bip01','bip001')),None)
            if root is not None:
                start,_=self.world(clip,0)
                shift=(self.align@(world[root,:,3]-start[root,:,3]))[:3]
        result=[]
        for g in self.geometries:
            node=g['node']; v=g['local']; normals=g['local_normals']
            palette=g['palette']
            if palette is not None:
                m=np.zeros((len(v),4,4))
                for k in range(4):
                    m+=palettes[palette][g['bone_ids'][:,k]]*g['weights'][:,k,None,None]
                out=np.einsum('nij,nj->ni',m[:,:3,:3],v)+m[:,:3,3]
                linear=m[:,:3,:3]
            else:
                m=self.align@world[node] if node>=0 else np.eye(4)
                out=v@m[:3,:3].T+m[:3,3]
                linear=np.broadcast_to(m[:3,:3],(len(v),3,3))
            # Cofactor transform handles nonuniform scales and collapsed effect bones.
            cof=np.stack((np.cross(linear[:,:,1],linear[:,:,2]),
                          np.cross(linear[:,:,2],linear[:,:,0]),
                          np.cross(linear[:,:,0],linear[:,:,1])),axis=2)
            nr=np.einsum('nij,nj->ni',cof,normals)
            determinant=np.einsum('ni,ni->n',linear[:,:,0],cof[:,:,0])
            nr*=np.where(determinant<0,-1.,1.)[:,None]
            nr/=np.maximum(np.linalg.norm(nr,axis=1,keepdims=True),1e-12)
            result.append((out-shift,nr,True if node<0 else bool(visible[node])))
        return result


def parse_model(raw):
    data,meta=segments(raw); reader=Reader(data); animation=Animation(reader,meta)
    out={'version':4,'format':'EG3D','geometries':[],'materials':[],
         'textures':[t.get('uri','') for t in meta[6]], 'bones':[],
         'actions':[], 'particles':[], 'animNames':[], 'eg3d':animation}
    for material in meta[2]:
        props=material[2] if len(material)>2 and isinstance(material[2],dict) else {}
        out['materials'].append({'name':material[0], 'tag':material[1],
            'tex':props.get('textures',{}).get('_MainTex'), 'properties':props})
    instances={}
    for i,node in enumerate(meta[5]):
        if len(node)>6 and node[6] is not None:
            instances.setdefault(node[6],[]).append((i,node[7]))
    for group_id,group in enumerate(meta[3]):
        # Meshes referenced only by ParticleSystem.renderer are emitter resources,
        # not standalone scene objects. Their node anchors are listed separately.
        if meta[5] and group_id not in instances:
            continue
        if group and isinstance(group[0],int): group=[group]
        for node,palette in instances.get(group_id,[(-1,None)]):
            for sub_id,desc in enumerate(group):
                mi,count,attrs,index_desc=desc[:4]
                if node>=0 and len(meta[5][node])>9 and isinstance(meta[5][node][9],dict):
                    mi=meta[5][node][9].get('material',mi)
                if not 0 <= mi < len(out['materials']) or count<=0:
                    raise ValueError('EG3D 网格材质或顶点数量无效')
                decoded={a[0]:reader.attr(a,count) for a in attrs}
                v=decoded[0]
                idx=reader.array(index_desc[0],'<u2',index_desc[1]).ravel().astype(int)
                if len(idx)%3 or np.any(idx>=count):
                    raise ValueError('EG3D 三角形索引无效')
                triangles=idx.reshape(-1,3)
                normals=decoded.get(1)
                if normals is None:
                    normals=np.zeros_like(v)
                    face=np.cross(v[triangles[:,1]]-v[triangles[:,0]],v[triangles[:,2]]-v[triangles[:,0]])
                    for k in range(3): np.add.at(normals,triangles[:,k],face)
                    normals/=np.maximum(np.linalg.norm(normals,axis=1,keepdims=True),1e-12)
                entry={'node':node,'palette':None,'local':v,'local_normals':normals}
                if palette is not None and 4 in decoded and 5 in decoded:
                    if not 0<=palette<len(animation.palettes):
                        raise ValueError('EG3D 网格骨骼组无效')
                    ids=decoded[4].astype(int); weights=decoded[5]
                    valid=(ids>=0)&(ids<len(animation.palettes[palette][0]))
                    if np.any((~valid)&(weights>1e-6)):
                        raise ValueError('EG3D 有效权重引用了不存在的骨骼')
                    ids=np.where(valid,ids,0)
                    if np.any(weights<0) or np.any(weights.sum(axis=1)<=1e-12):
                        raise ValueError('EG3D 蒙皮权重无效')
                    weights=weights/weights.sum(axis=1,keepdims=True)
                    entry.update(palette=palette,bone_ids=ids,weights=weights)
                animation.geometries.append(entry)
                uv=decoded.get(6)
                geo={'name':(meta[5][node][0] if node>=0 else 'mesh%d'%group_id)+'/%d'%sub_id,
                     'materialId':mi,'vertexCount':count,'faceCount':len(triangles),
                     'indices':tuple(idx.tolist()),'positions':tuple(v.ravel()),
                     'normals':tuple(normals.ravel()),'uvs':tuple(uv.ravel()) if uv is not None else None,
                     'joints':None,'box':tuple(np.r_[v.min(axis=0),v.max(axis=0)]),
                     'colors':tuple(decoded[3].ravel()) if 3 in decoded else None,
                     'eg3d_index':len(animation.geometries)-1}
                out['geometries'].append(geo)
    # Bind mesh coordinates are already Z-up. Rigid node instances need their transform.
    rest=animation.evaluate()
    for g,e,(v,n,visible) in zip(out['geometries'],animation.geometries,rest):
        if e['palette'] is None:
            g['positions']=tuple(v.ravel());g['normals']=tuple(n.ravel())
            g['box']=tuple(np.r_[v.min(axis=0),v.max(axis=0)])
    if animation.clips:
        _,initial_visibility=animation.world(0,0)
        for g,e in zip(out['geometries'],animation.geometries):
            g['visible']=bool(initial_visibility[e['node']]) if e['node']>=0 else True
    # Particle components have their own simulation; expose actual node anchors.
    world,_=animation.world()
    for i,node in enumerate(meta[5]):
        for component in (node[8] or []) if len(node)>8 else []:
            if isinstance(component,dict) and component.get('type')=='ParticleSystem':
                center=(animation.align@world[i])[:3,3]
                out['particles'].append({'name':node[0],'bone':-1,'flags':0,
                                         'off':tuple(center),'node':i})
    for i,clip in enumerate(animation.clips):
        out['animNames'].append(clip['name'])
        out['actions'].append({'name':clip['name'],'start':0,
            'end':max(0,math.ceil(clip['duration_ms']*30/1000)), 'clip':i,
            'duration_ms':clip['duration_ms']})
    return out
