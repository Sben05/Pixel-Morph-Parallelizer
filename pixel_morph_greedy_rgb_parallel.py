#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
import os
import time
import multiprocessing as mp
from dataclasses import dataclass
from typing import Tuple, List

import numpy as np
from PIL import Image, ImageFilter


@dataclass
class Progress:
    total:int
    prefix:str=""
    every:int= 1
    start:float=0.0

    def __post_init__(self):
        self.start=time.perf_counter()

    def tick(self,done:int):
        if done==0:
            self._print(done)
            return
        if (done% self.every)!=0 and done!=self.total:
            return
        self._print(done)

    def _print(self,done:int):
        now=time.perf_counter()
        pct=100.0* done/max(1,self.total)
        elapsed=now- self.start
        rate=done/elapsed if elapsed>1e-9 else 0.0
        remaining=(self.total-done)/rate if rate>1e-9 else float("inf")

        if math.isfinite(remaining):
            if remaining>=3600:
                eta=f"{remaining/3600:.2f}h"
            elif remaining>=60:
                eta=f"{remaining/60:.1f}m"
            else:
                eta=f"{remaining:.0f}s"
        else:
            eta="?"

        msg=f"\r{self.prefix} {done}/{self.total} ({pct:5.1f}%) | elapsed {elapsed:6.1f}s | ETA {eta:>6}"
        print(msg,end="")
        if done==self.total:
            print()


def load_crop_resize_rgb(path:str,size:int)->np.ndarray:
    im=Image.open(path).convert("RGB")
    w,h=im.size
    s=min(w,h)
    left=(w-s)//2
    top=(h-s)//2
    im=im.crop((left,top,left+s,top+s)).resize((size,size),resample=Image.BICUBIC)
    return np.asarray(im,dtype=np.uint8)

def rgb_to_luma01(img_u8:np.ndarray)->np.ndarray:
    rgb=img_u8.astype(np.float32)/255.0
    return 0.299*rgb[...,0]+0.587*rgb[...,1]+0.114*rgb[...,2]

def sobel_mag01(gray01:np.ndarray)->np.ndarray:
    g=gray01.astype(np.float32)
    p=np.pad(g,((1,1),(1,1)),mode="edge")
    gx=(-1*p[:-2,:-2]+1*p[:-2,2:]
        -2*p[1:-1,:-2]+2*p[1:-1,2:]
        -1*p[2:,:-2]+1*p[2:,2:])
    gy=(-1*p[:-2,:-2]+-2*p[:-2,1:-1]+-1*p[:-2,2:]
        +1*p[2:,:-2]+ 2*p[2:,1:-1]+ 1*p[2:,2:])
    mag=np.sqrt(gx*gx+gy*gy)
    mag=mag/(mag.max()+1e-8)
    return mag


def stratified_sample_indices(h:int,w:int,n:int,rng:np.random.Generator)->np.ndarray:
    total=h*w
    if n<=0 or n>=total:
        return np.arange(total,dtype=np.int32)

    s=max(1,int(np.sqrt(n)))
    xs=np.linspace(0,w,s+1,dtype=np.int32)
    ys=np.linspace(0,h,s+1,dtype=np.int32)

    idxs=[]
    for j in range(s):
        for i in range(s):
            x0,x1=int(xs[i]),int(xs[i+1])
            y0,y1=int(ys[j]),int(ys[j+1])
            if x1<=x0: x1=min(w,x0+ 1)
            if y1<=y0: y1=min(h,y0+ 1)
            x=int(rng.integers(x0,x1))
            y=int(rng.integers(y0,y1))
            idxs.append(y*w+x)

    idxs=np.array(idxs,dtype=np.int32)
    if idxs.size>n:
        idxs=rng.choice(idxs,size=n,replace=False)
    elif idxs.size<n:
        remaining=n-idxs.size
        pool=np.setdiff1d(np.arange(total,dtype=np.int32),idxs,assume_unique=False)
        extra=rng.choice(pool,size=remaining,replace=False)
        idxs=np.concatenate([idxs,extra])

    return idxs

def idx_to_xy_rgb_u8(img_u8:np.ndarray,idx:np.ndarray)->Tuple[np.ndarray,np.ndarray]:
    h,w,_=img_u8.shape
    ys=(idx//w).astype(np.int32)
    xs=(idx% w).astype(np.int32)
    xy=np.stack([xs,ys],axis=1).astype(np.float32)
    rgb_u8=img_u8[ys,xs].astype(np.uint8)
    return xy,rgb_u8


def compute_importance(size:int,tgt_idx:np.ndarray,tgt_u8:np.ndarray,mode:str,w_luma:float,w_center:float,w_edge:float)->np.ndarray:
    h=w=size
    ys=(tgt_idx//w).astype(np.float32)
    xs=(tgt_idx% w).astype(np.float32)

    cx=(w-1)/2.0
    cy=(h-1)/2.0
    dx=(xs-cx)
    dy=(ys-cy)
    dist=np.sqrt(dx*dx+dy*dy)
    maxd=np.sqrt(cx*cx+cy*cy)+1e-8
    center=1.0-(dist/maxd)

    luma_full=rgb_to_luma01(tgt_u8).reshape(-1)
    luma=luma_full[tgt_idx].astype(np.float32)

    edge_full=sobel_mag01(rgb_to_luma01(tgt_u8)).reshape(-1)
    edge=edge_full[tgt_idx].astype(np.float32)

    if mode=="luma":
        score=luma
    elif mode=="center":
        score=center
    elif mode=="edge":
        score=edge
    elif mode=="combo":
        score=w_luma*luma+w_center*center+w_edge*edge
    else:
        raise ValueError("Unknown importance mode")

    return score.astype(np.float32)


def rgb_u8_to_hue01(rgb_u8:np.ndarray)->np.ndarray:
    rgb=rgb_u8.astype(np.float32)/255.0
    r=rgb[:,0]; g=rgb[:,1]; b=rgb[:,2]
    maxc=np.maximum(np.maximum(r,g),b)
    minc=np.minimum(np.minimum(r,g),b)
    delta=maxc-minc

    h=np.zeros_like(maxc,dtype=np.float32)
    mask=delta>1e-8

    mr=mask&(maxc==r)
    hg=(g-b)/(delta+1e-12)
    h[mr]=np.mod(hg[mr],6.0)

    mg=mask&(maxc==g)
    hb=(b-r)/(delta+1e-12)
    h[mg]=hb[mg]+2.0

    mb=mask&(maxc==b)
    hr=(r-g)/(delta+1e-12)
    h[mb]=hr[mb]+4.0

    h=(h/6.0)% 1.0
    return h.astype(np.float32)


def _tree_query(tree,pts:np.ndarray,k:int)->np.ndarray:
    try:
        _,nn=tree.query(pts,k=k,workers=1)
    except TypeError:
        _,nn=tree.query(pts,k=k)
    if nn.ndim==1:
        nn=nn[:,None]
    return nn.astype(np.int32)

def knn_lists_binned(src_rgb:np.ndarray,tgt_rgb:np.ndarray,k:int,bins:int=16)->np.ndarray:
    step=256//bins
    if 256%bins!=0:
        raise ValueError("bins must divide 256 cleanly in fallback mode")

    src=src_rgb.astype(np.int32)
    br=np.clip(src[:,0]//step,0,bins-1)
    bg=np.clip(src[:,1]//step,0,bins-1)
    bb=np.clip(src[:,2]//step,0,bins-1)
    key=(br*bins+bg)*bins+bb

    buckets:List[List[int]]=[[] for _ in range(bins*bins*bins)]
    for i,kk in enumerate(key.tolist()):
        buckets[kk].append(i)

    tgt=tgt_rgb.astype(np.int32)
    tbr=np.clip(tgt[:,0]//step,0,bins-1)
    tbg=np.clip(tgt[:,1]//step,0,bins- 1)
    tbb=np.clip(tgt[:,2]//step,0,bins-1)

    out=np.empty((tgt.shape[0],k),dtype=np.int32)

    for i in range(tgt.shape[0]):
        r0,g0,b0=int(tbr[i]),int(tbg[i]),int(tbb[i])
        cand:List[int]=[]
        radius=0
        while len(cand)<k and radius<bins:
            rmin,rmax=max(0,r0-radius),min(bins-1,r0+radius)
            gmin,gmax=max(0,g0-radius),min(bins-1,g0+radius)
            bmin,bmax=max(0,b0-radius),min(bins-1,b0+radius)
            for rr in range(rmin,rmax+1):
                for gg in range(gmin,gmax+1):
                    for bb_ in range(bmin,bmax+1):
                        kk=(rr*bins+gg)*bins+bb_
                        cand.extend(buckets[kk])
            radius+=1

        if not cand:
            out[i]=0
            continue

        cand_arr=np.array(cand,dtype=np.int32)
        d=src_rgb[cand_arr].astype(np.float32)-tgt_rgb[i].astype(np.float32)
        dist2=d[:,0]*d[:,0]+d[:,1]*d[:,1]+d[:,2]*d[:,2]
        kk=min(k,cand_arr.size)
        sel=np.argpartition(dist2,kth=kk-1)[:kk]
        best=cand_arr[sel]
        best=best[np.argsort(dist2[sel],kind="mergesort")]
        if best.size<k:
            best=np.pad(best,(0,k-best.size),mode="edge")
        out[i]=best[:k]

    return out

def build_knn_lists_local(src_rgb:np.ndarray,tgt_rgb:np.ndarray,k:int)->Tuple[np.ndarray,str]:
    try:
        from scipy.spatial import cKDTree
        tree=cKDTree(src_rgb.astype(np.float32),leafsize=32,compact_nodes=True,balanced_tree=True)
        nn=_tree_query(tree,tgt_rgb.astype(np.float32),k=k)
        return nn.astype(np.int32),"scipy.cKDTree"
    except Exception:
        return knn_lists_binned(src_rgb,tgt_rgb,k=k,bins=16),"binned-fallback"

def greedy_assign_targets_to_sources(neighbors:np.ndarray,importance_order:np.ndarray,n_src:int)->np.ndarray:
    m,_=neighbors.shape
    chosen=np.full(m,-1,dtype=np.int32)
    used=np.zeros(n_src,dtype=np.bool_)
    for t in importance_order:
        row=neighbors[t]
        pick=-1
        for cand in row:
            if not used[cand]:
                pick=int(cand)
                used[cand]=True
                break
        chosen[t]=pick
    return chosen

def resolve_unassigned_local(src_rgb:np.ndarray,tgt_rgb:np.ndarray,chosen:np.ndarray,k_init:int,k_max:int)->None:
    used=np.zeros(src_rgb.shape[0],dtype=np.bool_)
    used[chosen[chosen>=0]]=True
    unassigned=np.where(chosen<0)[0].astype(np.int32)
    if unassigned.size==0:
        return

    try:
        from scipy.spatial import cKDTree
        tree=cKDTree(src_rgb.astype(np.float32),leafsize=32,compact_nodes=True,balanced_tree=True)
        backend="scipy.cKDTree"
    except Exception:
        backend="none"

    if backend!="scipy.cKDTree":
        remaining=np.where(~used)[0].astype(np.int32)
        for i,t in enumerate(unassigned):
            picked=int(remaining[i%remaining.size])
            chosen[t]=picked
            used[picked]=True
        return

    for t in unassigned:
        k=min(k_max,max(k_init*2,64,1))
        picked=-1
        while True:
            nn=_tree_query(tree,tgt_rgb[int(t)].astype(np.float32)[None,:],k=k)[0]
            for cand in nn:
                if not used[cand]:
                    picked=int(cand)
                    used[picked]=True
                    break
            if picked!=-1 or k>=k_max:
                break
            k=min(k_max,k*2)

        if picked==-1:
            rem=np.where(~used)[0]
            if rem.size>0:
                picked=int(rem[0])
                used[picked]=True
        chosen[int(t)]=picked


def ease_in_out_pow(t:float,p:float)->float:
    t=max(0.0,min(1.0,t))
    a=t**p
    b=(1-t)**p
    return a/(a+b+1e-12)

def render_bilinear_splat(size:int,xy:np.ndarray,col01:np.ndarray,bg_rgb01=(0.0,0.0,0.0))->np.ndarray:
    h=w=size
    x=np.clip(xy[:,0],0.0,w-1.001)
    y=np.clip(xy[:,1],0.0,h-1.001)

    x0=np.floor(x).astype(np.int32)
    y0=np.floor(y).astype(np.int32)
    x1=np.clip(x0+1,0,w-1)
    y1=np.clip(y0+1,0,h-1)

    dx=(x-x0).astype(np.float32)
    dy=(y-y0).astype(np.float32)

    w00=(1-dx)*(1-dy)
    w10=dx*(1-dy)
    w01=(1-dx)*dy
    w11=dx*dy

    idx00=y0*w+x0
    idx10=y0*w+x1
    idx01=y1*w+x0
    idx11=y1*w+x1

    rr=np.zeros(h*w,dtype=np.float32)
    gg=np.zeros(h*w,dtype=np.float32)
    bb=np.zeros(h*w,dtype=np.float32)
    cc=np.zeros(h*w,dtype=np.float32)

    r=col01[:,0].astype(np.float32)
    g=col01[:,1].astype(np.float32)
    b=col01[:,2].astype(np.float32)

    np.add.at(rr,idx00,r*w00);np.add.at(gg,idx00,g*w00);np.add.at(bb,idx00,b*w00);np.add.at(cc,idx00,w00)
    np.add.at(rr,idx10,r*w10);np.add.at(gg,idx10,g*w10);np.add.at(bb,idx10,b*w10);np.add.at(cc,idx10,w10)
    np.add.at(rr,idx01,r*w01);np.add.at(gg,idx01,g*w01);np.add.at(bb,idx01,b*w01);np.add.at(cc,idx01,w01)
    np.add.at(rr,idx11,r*w11);np.add.at(gg,idx11,g*w11);np.add.at(bb,idx11,b*w11);np.add.at(cc,idx11,w11)

    out=np.empty((h*w,3),dtype=np.float32)
    out[:]=np.array(bg_rgb01,dtype=np.float32)

    mask=cc>1e-8
    out[mask,0]=rr[mask]/cc[mask]
    out[mask,1]=gg[mask]/cc[mask]
    out[mask,2]=bb[mask]/cc[mask]

    out=np.clip(out.reshape(h,w,3),0.0,1.0)
    return (out*255.0).astype(np.uint8)

def total_motion_cost(start_xy:np.ndarray,end_xy:np.ndarray)->float:
    d=start_xy-end_xy
    return float(np.sum(d[:,0]*d[:,0]+d[:,1]*d[:,1]))


_G_SRC_RGB_U8=None
_G_TGT_RGB_U8=None
_G_IMPORTANCE=None
_G_SRC_PERM=None
_G_TGT_PERM=None
_G_K_INIT=None
_G_K_MAX=None

def _worker_init(src_rgb_u8,tgt_rgb_u8,importance,src_perm,tgt_perm,k_init,k_max):
    global _G_SRC_RGB_U8,_G_TGT_RGB_U8,_G_IMPORTANCE,_G_SRC_PERM,_G_TGT_PERM,_G_K_INIT,_G_K_MAX
    _G_SRC_RGB_U8=src_rgb_u8
    _G_TGT_RGB_U8=tgt_rgb_u8
    _G_IMPORTANCE=importance
    _G_SRC_PERM=src_perm
    _G_TGT_PERM=tgt_perm
    _G_K_INIT=int(k_init)
    _G_K_MAX=int(k_max)

def _chunk_match_worker(task):
    _,lo,hi=task
    src_ids=_G_SRC_PERM[lo:hi]
    tgt_ids=_G_TGT_PERM[lo:hi]

    n_src=int(src_ids.size)
    m=int(tgt_ids.size)
    if n_src==0 or m==0:
        return tgt_ids,np.full(m,-1,dtype=np.int32)

    src_rgb=_G_SRC_RGB_U8[src_ids].astype(np.float32)
    tgt_rgb=_G_TGT_RGB_U8[tgt_ids].astype(np.float32)

    imp=_G_IMPORTANCE[tgt_ids].astype(np.float32)
    local_order=np.argsort(-imp,kind="mergesort").astype(np.int32)

    k0=max(1,min(int(_G_K_INIT),n_src))
    kmax=max(k0,min(int(_G_K_MAX),n_src))

    neighbors,_=build_knn_lists_local(src_rgb,tgt_rgb,k=k0)
    chosen_local=greedy_assign_targets_to_sources(neighbors,local_order,n_src=n_src)
    resolve_unassigned_local(src_rgb,tgt_rgb,chosen_local,k_init=k0,k_max=kmax)

    chosen_global=src_ids[np.clip(chosen_local,0,n_src-1)]
    return tgt_ids,chosen_global.astype(np.int32)


def enforce_bijection(chosen_src_for_tgt:np.ndarray,importance:np.ndarray)->np.ndarray:
    M=chosen_src_for_tgt.size
    s=chosen_src_for_tgt.astype(np.int64,copy=True)

    bad=(s<0)|(s>=M)
    s[bad]=-1

    order=np.argsort(-importance,kind="mergesort")
    taken=np.zeros(M,dtype=np.bool_)
    out=np.full(M,-1,dtype=np.int32)

    for t in order:
        src=int(s[t])
        if src<0:
            continue
        if not taken[src]:
            taken[src]=True
            out[t]=src

    missing=np.where(~taken)[0].astype(np.int32)
    holes=np.where(out<0)[0].astype(np.int32)
    if holes.size!=missing.size:
        missing=np.resize(missing,holes.size)

    for i,t in enumerate(holes):
        out[t]=int(missing[i])

    return out


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--src",required=True)
    ap.add_argument("--tgt",required=True)
    ap.add_argument("--size",type=int,default=256)
    ap.add_argument("--n",type=int,default=0,help="particles (0 => all pixels)")
    ap.add_argument("--frames",type=int,default=200)
    ap.add_argument("--fps",type=int,default=10)
    ap.add_argument("--out",default="parallel.gif")

    ap.add_argument("--jobs",type=int,default=max(1,(os.cpu_count() or 1)))
    ap.add_argument("--mp_start",choices=["auto","fork","spawn"],default="auto",help="auto prefers fork on macOS/Linux if available")

    ap.add_argument("--importance",choices=["luma","center","edge","combo"],default="combo")
    ap.add_argument("--w_luma",type=float,default=0.45)
    ap.add_argument("--w_center",type=float,default=0.45)
    ap.add_argument("--w_edge",type=float,default=0.10)

    ap.add_argument("--k_init",type=int,default=64)
    ap.add_argument("--k_max",type=int,default=2048)

    ap.add_argument("--color_alpha",type=float,default=0.35,help="end-color mix: 0=stay source-colored, 1=full target color (too perfect)")

    ap.add_argument("--blur",type=float,default=0.0)
    ap.add_argument("--bg",choices=["black","white"],default="black")
    ap.add_argument("--easing_pow",type=float,default=2.2)
    ap.add_argument("--progress_every",type=int,default=5)
    ap.add_argument("--seed",type=int,default=0)
    args=ap.parse_args()

    T0=time.perf_counter()
    rng=np.random.default_rng(args.seed)

    t_load0=time.perf_counter()
    print("[load] reading + crop/resize images...")
    src_u8=load_crop_resize_rgb(args.src,args.size)
    tgt_u8=load_crop_resize_rgb(args.tgt,args.size)
    t_load1=time.perf_counter()

    h=w=args.size
    total=h*w
    n_particles=total if args.n<=0 else int(args.n)

    t_samp0=time.perf_counter()
    print("[sample] selecting pixels/particles...")
    idx_src=stratified_sample_indices(h,w,n_particles,rng)
    idx_tgt=(stratified_sample_indices(h,w,idx_src.size,rng) if args.n>0 else np.arange(total,dtype=np.int32))
    if idx_tgt.size!=idx_src.size:
        idx_tgt=stratified_sample_indices(h,w,idx_src.size,rng)

    src_xy,src_rgb_u8=idx_to_xy_rgb_u8(src_u8,idx_src)
    tgt_xy,tgt_rgb_u8=idx_to_xy_rgb_u8(tgt_u8,idx_tgt)
    t_samp1=time.perf_counter()

    M=int(src_xy.shape[0])

    t_imp0=time.perf_counter()
    print(f"[importance] computing target importance ({args.importance})...")
    importance=compute_importance(size=args.size,tgt_idx=idx_tgt,tgt_u8=tgt_u8,mode=args.importance,w_luma=args.w_luma,w_center=args.w_center,w_edge=args.w_edge)
    t_imp1=time.perf_counter()

    t_hue0=time.perf_counter()
    print("[segment] converting to hue + sorting into balanced segments...")
    src_hue=rgb_u8_to_hue01(src_rgb_u8)
    tgt_hue=rgb_u8_to_hue01(tgt_rgb_u8)

    src_perm=np.argsort(src_hue,kind="mergesort").astype(np.int32)
    tgt_perm=np.argsort(tgt_hue,kind="mergesort").astype(np.int32)
    t_hue1=time.perf_counter()

    jobs=max(1,min(int(args.jobs),M))

    if args.mp_start=="fork":
        ctx=mp.get_context("fork")
    elif args.mp_start=="spawn":
        ctx=mp.get_context("spawn")
    else:
        if os.name=="posix" and "fork" in mp.get_all_start_methods():
            ctx=mp.get_context("fork")
        else:
            ctx=mp.get_context()

    t_match0=time.perf_counter()
    print(f"[greedy parallel] hue-groups/jobs = {jobs} (mp={ctx.get_start_method()})")

    chosen_src_for_tgt=np.full(M,-1,dtype=np.int32)

    tasks=[]
    for i in range(jobs):
        lo=(i*M)//jobs
        hi=((i+1)*M)//jobs
        tasks.append((i,lo,hi))

    if jobs==1:
        _worker_init(src_rgb_u8,tgt_rgb_u8,importance,src_perm,tgt_perm,args.k_init,args.k_max)
        tgt_ids,src_ids=_chunk_match_worker(tasks[0])
        chosen_src_for_tgt[tgt_ids]=src_ids
    else:
        with ctx.Pool(processes=jobs,initializer=_worker_init,initargs=(src_rgb_u8,tgt_rgb_u8,importance,src_perm,tgt_perm,args.k_init,args.k_max)) as pool:
            prog=Progress(total=len(tasks),prefix="[greedy parallel]",every=1)
            done=0
            for tgt_ids,src_ids in pool.imap_unordered(_chunk_match_worker,tasks,chunksize=1):
                chosen_src_for_tgt[tgt_ids]=src_ids
                done+=1
                prog.tick(done)

    chosen_src_for_tgt=enforce_bijection(chosen_src_for_tgt,importance)
    t_match1=time.perf_counter()

    t_map0=time.perf_counter()
    print("[map] building start/end trajectories + end colors (Option A)...")

    start_xy=src_xy.astype(np.float32)
    fixed_col01=(src_rgb_u8.astype(np.float32)/255.0).astype(np.float32)
    tgt_col01=(tgt_rgb_u8.astype(np.float32)/255.0).astype(np.float32)

    end_xy=start_xy.copy()
    end_col01=fixed_col01.copy()

    t_idx=np.arange(M,dtype=np.int32)
    s_idx=chosen_src_for_tgt.astype(np.int32)

    end_xy[s_idx]=tgt_xy[t_idx].astype(np.float32)

    alpha=float(np.clip(args.color_alpha,0.0,1.0))
    end_col01[s_idx]=(1.0-alpha)*fixed_col01[s_idx]+alpha*tgt_col01[t_idx]

    t_map1=time.perf_counter()

    cost=total_motion_cost(start_xy,end_xy)
    print(f"[metric] total motion cost: {cost:.3e}")

    t_rend0=time.perf_counter()
    bg_rgb01=(0.0,0.0,0.0) if args.bg=="black" else (1.0,1.0,1.0)
    duration_ms=int(round(1000/max(1,args.fps)))

    print("[render] generating frames (position + color crossfade)...")
    frames:List[Image.Image]=[]
    prog=Progress(total=args.frames,prefix="[render]",every=max(1,args.progress_every))

    for k in range(args.frames):
        t=0.0 if args.frames==1 else (k/(args.frames-1))
        e=ease_in_out_pow(t,args.easing_pow)

        xy=(1.0-e)*start_xy+e*end_xy
        col01=(1.0-e)*fixed_col01+e*end_col01

        img_u8=render_bilinear_splat(args.size,xy,col01,bg_rgb01=bg_rgb01)
        im=Image.fromarray(img_u8)
        if args.blur and args.blur>0.0:
            im=im.filter(ImageFilter.GaussianBlur(radius=args.blur))
        frames.append(im)

        prog.tick(k+1)

    t_rend1=time.perf_counter()

    t_save0=time.perf_counter()
    print("[save] writing GIF...")
    frames[0].save(args.out,save_all=True,append_images=frames[1:],duration=duration_ms,loop=0,optimize=True)
    t_save1=time.perf_counter()

    T1=time.perf_counter()

    print("\n[timing]")
    print(f"  load:        {(t_load1-t_load0):8.3f}s")
    print(f"  sample:      {(t_samp1-t_samp0):8.3f}s")
    print(f"  importance:  {(t_imp1 -t_imp0):8.3f}s")
    print(f"  segment:     {(t_hue1 -t_hue0):8.3f}s")
    print(f"  greedy(par): {(t_match1-t_match0):8.3f}s")
    print(f"  map:         {(t_map1 -t_map0):8.3f}s")
    print(f"  render:      {(t_rend1-t_rend0):8.3f}s")
    print(f"  save:        {(t_save1-t_save0):8.3f}s")
    print(f"  TOTAL:       {(T1     -T0):8.3f}s")
    print(f"[done] wrote {args.out}")


if __name__=="__main__":
    main()
