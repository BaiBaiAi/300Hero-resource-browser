"""JUMPX track decoding and vectorised skinning for Resource Studio.

Format facts cross-checked with game descriptors and the public 300 Heroes
importer eg3d_x.py; no Blender runtime dependency.
"""
import math
import numpy as np
from eg3d_backend import matrices


def unpack_position(words, low, high):
    values=np.column_stack([(words>>shift)&1023 for shift in (0,10,20)]).astype(float)
    return np.asarray(low)+values*(np.asarray(high)-low)/1023


def unpack_rotation(words):
    # The three signed 12-bit axis components and the 8-bit degree angle.
    axis=np.column_stack([(words>>np.uint64(shift))&np.uint64(4095)
                          for shift in (16,32,48)]).astype(float)
    axis=np.where(axis>=2048,axis-4096,axis)
    axis/=np.maximum(np.linalg.norm(axis,axis=1,keepdims=True),1e-12)
    half=(words&np.uint64(255)).astype(float)*math.pi/360
    return np.column_stack((axis*np.sin(half)[:,None],np.cos(half)))


def clean_tracks(bones):
    """Repair isolated invalid source keys in memory and report exactly how many."""
    warnings=[]
    for bone in bones:
        for field,comps in (('posT',3),('rotT',4),('scaleT',3)):
            data=bone.get(field)
            if data:
                a=np.frombuffer(data,'<f4').reshape(-1,comps).copy()
                valid=np.isfinite(a).all(axis=1)
                if field=='rotT': valid &= np.linalg.norm(a,axis=1)>1e-9
                bad=np.flatnonzero(~valid)
                if len(bad):
                    good=np.flatnonzero(valid)
                    if not len(good):
                        raise ValueError('骨骼 %s 的 %s 全部关键帧无效' % (bone['name'],field))
                    right=np.minimum(np.searchsorted(good,bad),len(good)-1)
                    left=np.maximum(right-1,0)
                    nearest=np.where(abs(good[left]-bad)<=abs(good[right]-bad),good[left],good[right])
                    a[bad]=a[nearest]
                    warnings.append('%s/%s: %d 个无效关键帧，使用最近有效值' % (bone['name'],field,len(bad)))
                if field=='rotT': a/=np.maximum(np.linalg.norm(a,axis=1,keepdims=True),1e-9)
                bone[field]=a.astype('<f4').tobytes()
    return warnings


class Animation:
    def __init__(self, model):
        bones=model['bones'];self.model=model
        self.frames=max((b['frameCount'] for b in bones),default=1)
        self.p=[];self.q=[];self.s=[];self.visibility=[]
        for b in bones:
            for key,comps,default,dst in [('posT',3,[0,0,0],self.p),('rotT',4,[0,0,0,1],self.q),('scaleT',3,[1,1,1],self.s)]:
                data=b.get(key)
                dst.append(np.frombuffer(data,'<f4').reshape(-1,comps) if data else np.asarray([default],dtype=float))
            self.visibility.append(np.frombuffer(b['visibleT'],'<u4') if b.get('visibleT') else np.ones(1))
        self.root=next((i for i,b in enumerate(bones) if b['name'].lower() in ('bip01','bip001')),0)
        w=self.world(0)
        # Zero scale at bind time is an authored hidden effect. Pseudoinverse
        # avoids aborting unrelated geometry; visible body bones remain exact.
        self.inverse=np.linalg.pinv(w)
        self.skin=[]
        for g in model['geometries']:
            v=np.asarray(g['positions'] or [],dtype=float).reshape(-1,3)
            n=np.asarray(g['normals'],dtype=float).reshape(-1,3) if g.get('normals') else None
            joints=g.get('joints')
            if joints:
                jj=np.asarray(joints);counts=jj[:,0].astype(int)
                ids=jj[:,1:5].astype(int);weights=jj[:,5:9].astype(float)
                valid=(np.arange(4)[None,:]<counts[:,None])&(ids>=0)&(ids<len(bones))&np.isfinite(weights)&(weights>0)
                weights=np.where(valid,weights,0);ids=np.where(valid,ids,0)
                sums=weights.sum(1,keepdims=True);weights/=np.maximum(sums,1e-12)
                self.skin.append((v,n,ids,weights,sums[:,0]>1e-12))
            else:self.skin.append((v,n,None,None,None))

    def world(self, frame):
        f=max(0.,min(float(frame),self.frames-1));lo=int(f);t=f-lo
        def interpolate(track):
            return track[min(lo,len(track)-1)]*(1-t)+track[min(lo+1,len(track)-1)]*t
        p=np.array([interpolate(a) for a in self.p])
        sc=np.array([interpolate(a) for a in self.s])
        qa=np.array([a[min(lo,len(a)-1)] for a in self.q],dtype=float)
        qb=np.array([a[min(lo+1,len(a)-1)] for a in self.q],dtype=float)
        dot=np.sum(qa*qb,axis=1);qb=np.where((dot<0)[:,None],-qb,qb);dot=abs(dot)
        angle=np.arccos(np.clip(dot,-1,1));den=np.maximum(np.sin(angle),1e-12)
        a=np.where(dot>.9995,1-t,np.sin((1-t)*angle)/den)
        b=np.where(dot>.9995,t,np.sin(t*angle)/den)
        q=qa*a[:,None]+qb*b[:,None]
        # JUMPX stores row-vector R(q), translation in the last row.
        # Scale multiplies the rotation's rows in this convention.
        out=matrices(p,q,np.ones_like(sc))
        out[:,:3,:3]*=sc[:,:,None]
        out[:,3,:3]=p;out[:,:3,3]=0
        return out

    def evaluate(self, frame, in_place=False):
        frame=max(0.,min(float(frame),self.frames-1))
        w=self.world(frame);delta=self.inverse@w
        normal_delta=np.linalg.pinv(delta[:,:3,:3]).transpose(0,2,1)
        if in_place:delta[:,3,:3]-=w[self.root,3,:3]-self.world(0)[self.root,3,:3]
        result=[]
        for g,(v,n,ids,weights,weighted) in zip(self.model['geometries'],self.skin):
            visible=True
            if g.get('visibleT'):
                visibility=np.frombuffer(g['visibleT'],'<f4')
                visible=bool(visibility[min(int(frame),len(visibility)-1)]>.001)
            if ids is None:
                result.append((v,n,visible));continue
            out=np.zeros_like(v);nr=np.zeros_like(n) if n is not None else None
            # Transform each of four influences separately; no N x 4 x 4 x 4
            # temporary, and zero-weight vertices keep their authored positions.
            for k in range(4):
                m=delta[ids[:,k]];ww=weights[:,k,None]
                out+=(np.einsum('nj,nji->ni',v,m[:,:3,:3])+m[:,3,:3])*ww
                if n is not None:nr+=np.einsum('nj,nji->ni',n,normal_delta[ids[:,k]])*ww
            out[~weighted]=v[~weighted]
            if nr is not None:
                nr[~weighted]=n[~weighted];nr/=np.maximum(np.linalg.norm(nr,axis=1,keepdims=True),1e-12)
            bone_visible=np.array([a[min(int(frame),len(a)-1)] for a in self.visibility],dtype=bool)
            if weighted.any() and not np.any(bone_visible[ids] & (weights>0)):
                visible=False
            result.append((out,nr,visible))
        return result
