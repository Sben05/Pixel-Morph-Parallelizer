import io,os,sys,time,random,shutil,tempfile,subprocess
from dataclasses import dataclass
import numpy as np
from PIL import Image
import streamlit as st
import matplotlib.pyplot as plt

def _css():
  '''
  st.markdown("""<style>:root{--bg0:#070A12;--bg1:#0B1020;--card:rgba(255,255,255,.06);--stroke:rgba(255,255,255,.12)}
.stApp{background:radial-gradient(1200px 700px at 20% 10%,rgba(96,165,250,.18),transparent 60%),radial-gradient(1100px 650px at 85% 0%,rgba(94,234,212,.14),transparent 62%),radial-gradient(900px 600px at 65% 95%,rgba(167,139,250,.12),transparent 60%),linear-gradient(180deg,var(--bg0),var(--bg1))}
.block-container{padding-top:1.1rem;padding-bottom:1.8rem}
div[data-testid="stExpander"],div[data-testid="stMetric"]{background:var(--card);border:1px solid var(--stroke);border-radius:16px}
.stButton button{border-radius:14px;border:1px solid rgba(255,255,255,.18);background:linear-gradient(135deg,rgba(94,234,212,.18),rgba(96,165,250,.18));color:rgba(255,255,255,.92)}
.stButton button:hover{border:1px solid rgba(94,234,212,.55);filter:brightness(1.06)}
hr{border-top:1px solid rgba(255,255,255,.12)}</style>""",unsafe_allow_html=True)
'''

def _imp():
  s=p=None;se=pe=None
  try: import pixel_morph_greedy_rgb as s
  except Exception as e: se=e
  try: import pixel_morph_greedy_rgb_parallel as p
  except Exception as e: pe=e
  return s,p,se,pe

SEQ,PAR,SEQ_ERR,PAR_ERR=_imp()
BASE_DIR=os.path.dirname(os.path.abspath(__file__))
plt.rcParams.update({"figure.dpi":120})

def _im(b): return Image.open(io.BytesIO(b)).convert("RGB")
def _lcr(b,size):
  im=_im(b);w,h= im.size;s=min(w,h)
  L=(w-s)//2;T=(h-s)//2
  im=im.crop((L,T,L+s,T+s)).resize((size,size),resample=Image.BICUBIC)
  return np.asarray(im,dtype=np.uint8)

def _luma(u8):
  rgb=u8.astype(np.float32)/255.0
  return 0.299*rgb[...,0]+0.587*rgb[...,1]+0.114*rgb[...,2]

def _sob(gray01):
  g=gray01.astype(np.float32);p=np.pad(g,((1,1),(1,1)),mode="edge")
  gx=(-1*p[:-2,:-2]+1*p[:-2,2:]-2*p[1:-1,:-2]+2*p[1:-1,2:]-1*p[2:,:-2]+1*p[2:,2:])
  gy=(-1*p[:-2,:-2]-2*p[:-2,1:-1]-1*p[:-2,2:]+1*p[2:,:-2]+2*p[2:,1:-1]+1*p[2:,2:])
  m=np.sqrt(gx*gx+gy*gy);return m/(m.max()+1e-8)

@st.cache_data(show_spinner=False)
def feature_maps(src_b,tgt_b,size):
  su8=_lcr(src_b,size);tu8=_lcr(tgt_b,size)
  sl=_luma(su8);tl=_luma(tu8)
  se=_sob(sl);te=_sob(tl)
  if SEQ is not None:
    sh=SEQ.rgb_u8_to_hue01(su8.reshape(-1,3)).reshape(size,size)
    th=SEQ.rgb_u8_to_hue01(tu8.reshape(-1,3)).reshape(size,size)
  else:
    sh=np.zeros((size,size),np.float32);th=sh.copy()
  return {"src_u8":su8,"tgt_u8":tu8,"src_luma":sl,"tgt_luma":tl,"src_edge":se,"tgt_edge":te,"src_hue":sh,"tgt_hue":th}

@st.cache_data(show_spinner=False)
def importance_map(tgt_b,size,mode,wL,wC,wE):
  tu8=_lcr(tgt_b,size);idx=np.arange(size*size,dtype=np.int32)
  if SEQ is None:
    ys=(idx//size).astype(np.float32);xs=(idx%size).astype(np.float32)
    cx=(size-1)/2.0;cy=cx
    dist=np.sqrt((xs-cx)**2+(ys-cy)**2);maxd=np.sqrt(cx*cx+cy*cy)+1e-8
    center=1.0-(dist/maxd)
    l=_luma(tu8).reshape(-1).astype(np.float32)
    e=_sob(_luma(tu8)).reshape(-1).astype(np.float32)
    if mode=="luma": sc=l
    elif mode=="center": sc=center
    elif mode=="edge": sc=e
    else: sc=wL*l+wC*center+wE*e
    return sc.reshape(size,size).astype(np.float32)
  imp=SEQ.compute_importance(size,idx,tu8,mode,wL,wC,wE)
  return imp.reshape(size,size).astype(np.float32)

@dataclass
class Prepared:
  size:int;M:int
  src_u8:np.ndarray;tgt_u8:np.ndarray
  idx_src:np.ndarray;idx_tgt:np.ndarray
  src_xy:np.ndarray;tgt_xy:np.ndarray
  src_rgb_u8:np.ndarray;tgt_rgb_u8:np.ndarray
  importance:np.ndarray;src_perm:np.ndarray;tgt_perm:np.ndarray

@st.cache_data(show_spinner=False)
def prepare(src_b,tgt_b,size,M,seed,imode,wL,wC,wE):
  if SEQ is None: raise RuntimeError("pixel_morph_greedy_rgb import failed")
  rng=np.random.default_rng(seed)
  su8=_lcr(src_b,size);tu8=_lcr(tgt_b,size)
  total=size*size;M=int(np.clip(int(M),500,total))
  idx_s=SEQ.stratified_sample_indices(size,size,M,rng)
  idx_t=SEQ.stratified_sample_indices(size,size,idx_s.size,rng)
  sxy,srgb=SEQ.idx_to_xy_rgb_u8(su8,idx_s)
  txy,trgb=SEQ.idx_to_xy_rgb_u8(tu8,idx_t)
  imp=SEQ.compute_importance(size,idx_t,tu8,imode,wL,wC,wE).astype(np.float32)
  sh=SEQ.rgb_u8_to_hue01(srgb);th=SEQ.rgb_u8_to_hue01(trgb)
  sp=np.argsort(sh,kind="mergesort").astype(np.int32)
  tp=np.argsort(th,kind="mergesort").astype(np.int32)
  return Prepared(size=size,M=int(idx_s.size),src_u8=su8,tgt_u8=tu8,idx_src=idx_s,idx_tgt=idx_t,
                  src_xy=sxy,tgt_xy=txy,src_rgb_u8=srgb,tgt_rgb_u8=trgb,importance=imp,src_perm=sp,tgt_perm=tp)

def seq_match_only(prep,jobs,k_init,k_max):
  if SEQ is None: raise RuntimeError("SEQ import required")
  M=prep.M;jobs=max(1,min(int(jobs),M))
  chosen=np.full(M,-1,dtype=np.int32)
  t0=time.perf_counter()
  for i in range(jobs):
    lo=(i*M)//jobs;hi=((i+1)*M)//jobs
    tgt_ids,src_ids=SEQ.chunk_match(prep.src_rgb_u8,prep.tgt_rgb_u8,prep.importance,prep.src_perm,prep.tgt_perm,
                                    lo,hi,k_init,k_max)
    chosen[tgt_ids]=src_ids
  t1=time.perf_counter()
  tf0=time.perf_counter()
  if np.any(chosen<0):
    used=np.zeros(M,dtype=np.bool_);used[chosen[chosen>=0]]=True
    rem=np.where(~used)[0].astype(np.int32)
    holes=np.where(chosen<0)[0].astype(np.int32)
    for j,t in enumerate(holes): chosen[t]=rem[j%rem.size]
  tf1=time.perf_counter()
  return chosen,{"match":t1-t0,"fill":tf1-tf0,"total":tf1-t0}

def par_match_only(prep,jobs,k_init,k_max,mp_start):
  if PAR is None: raise RuntimeError("PAR import required")
  import multiprocessing as mp
  M=prep.M;jobs=max(1,min(int(jobs),M))
  if mp_start=="fork": ctx=mp.get_context("fork")
  elif mp_start=="spawn": ctx=mp.get_context("spawn")
  else:
    if os.name=="posix" and "fork" in mp.get_all_start_methods(): ctx=mp.get_context("fork")
    else: ctx=mp.get_context()
  chosen=np.full(M,-1,dtype=np.int32)
  tasks=[(i,(i*M)//jobs,((i+1)*M)//jobs) for i in range(jobs)]
  t0=time.perf_counter()
  if jobs==1:
    PAR._worker_init(prep.src_rgb_u8,prep.tgt_rgb_u8,prep.importance,prep.src_perm,prep.tgt_perm,k_init,k_max)
    tgt_ids,src_ids=PAR._chunk_match_worker(tasks[0]);chosen[tgt_ids]=src_ids
  else:
    with ctx.Pool(processes=jobs,initializer=PAR._worker_init,
                  initargs=(prep.src_rgb_u8,prep.tgt_rgb_u8,prep.importance,prep.src_perm,prep.tgt_perm,k_init,k_max)) as pool:
      for tgt_ids,src_ids in pool.imap_unordered(PAR._chunk_match_worker,tasks,chunksize=1): chosen[tgt_ids]=src_ids
  t1=time.perf_counter()
  tr0=time.perf_counter();repaired=PAR.enforce_bijection(chosen,prep.importance);tr1=time.perf_counter()
  return repaired,{"match_parallel":t1-t0,"repair":tr1-tr0,"total":tr1-t0}

def fig_img(t,img,cmap=None,vmin=None,vmax=None):
  fig=plt.figure();plt.title(t)
  plt.imshow(img,cmap=cmap,vmin=vmin,vmax=vmax) if img.ndim==2 else plt.imshow(img)
  plt.axis("off");return fig

def fig_scatter(u8,xy,title,max_pts=7000):
  fig=plt.figure();plt.title(title);plt.imshow(u8);plt.axis("off")
  pts=xy
  if pts.shape[0]>max_pts:
    sel=np.random.choice(pts.shape[0],size=max_pts,replace=False);pts=pts[sel]
  plt.scatter(pts[:,0],pts[:,1],s=4,alpha=0.65);return fig

def fig_segbar(M,jobs,title):
  fig=plt.figure(figsize=(7.8,2.0));ax=plt.gca()
  ax.set_title(title);ax.set_xlim(0,M);ax.set_ylim(0,1);ax.set_yticks([])
  for i in range(jobs):
    lo=(i*M)//jobs;hi=((i+1)*M)//jobs
    ax.fill_between([lo,hi],0,1,alpha=0.85)
    ax.text((lo+hi)/2,0.5,f"t{i}",ha="center",va="center",fontsize=9)
  ax.set_xlabel("hue-sorted index");return fig

def fig_speed(Ps,Tseq,Tpar,title):
  sp=[Tseq[i]/Tpar[i] for i in range(len(Ps))]
  fig=plt.figure();plt.title(title)
  plt.plot(Ps,sp,marker="o",label="obs")
  plt.plot(Ps,Ps,linestyle="--",alpha=0.6,label="ideal")
  plt.xlabel("P");plt.ylabel("speedup");plt.legend();return fig

def fig_eff(Ps,Tseq,Tpar,title):
  sp=[Tseq[i]/Tpar[i] for i in range(len(Ps))]
  ef=[sp[i]/Ps[i] for i in range(len(Ps))]
  fig=plt.figure();plt.title(title)
  plt.plot(Ps,ef,marker="o");plt.xlabel("P");plt.ylabel("eff");return fig

def _wd():
  if "workdir" not in st.session_state: st.session_state.workdir=tempfile.mkdtemp(prefix="pixelmorph_lab_")
  return st.session_state.workdir

def _wb(path,b):
  with open(path,"wb") as f: f.write(b)

def run_original(mode,src_path,tgt_path,out_path,args,box):
  script="pixel_morph_greedy_rgb.py" if mode=="sequential" else "pixel_morph_greedy_rgb_parallel.py"
  py=sys.executable
  cmd=[py,script,"--src",src_path,"--tgt",tgt_path,"--size",str(args["size"]),
       "--n",str(args["n"]),"--frames",str(args["frames"]),"--fps",str(args["fps"]),
       "--out",out_path,"--jobs",str(args["jobs"]), "--importance",args["importance"],
       "--w_luma",str(args["w_luma"]),"--w_center",str(args["w_center"]), "--w_edge",str(args["w_edge"]),
       "--k_init",str(args["k_init"]),"--k_max",str(args["k_max"]), "--blur",str(args["blur"]),
       "--bg",args["bg"],"--easing_pow",str(args["easing_pow"]), "--progress_every",str(args["progress_every"]),
       "--seed",str(args["seed"])]
  if mode=="parallel": cmd+=["--mp_start",args["mp_start"],"--color_alpha",str(args["color_alpha"])]
  proc=subprocess.Popen(cmd,cwd=BASE_DIR,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,bufsize=1,universal_newlines=True)
  lines=[];t0=time.perf_counter()
  assert proc.stdout is not None
  for line in proc.stdout:
    lines.append(line.rstrip("\n").replace("\r",""))
    if len(lines)%20==0: box.code("\n".join(lines[-220:]))
  rc=proc.wait();t1=time.perf_counter();box.code("\n".join(lines[-220:]))
  return rc,t1-t0

st.set_page_config(page_title="Pixel Morph Lab",layout="wide")
_css()
st.title("Pixel Morph Lab")

if SEQ is None or PAR is None:
  with st.expander("Import status",expanded=True):
    st.write("seq:", "OK" if SEQ is not None else f"FAIL: {SEQ_ERR}")
    st.write("par:", "OK" if PAR is not None else f"FAIL: {PAR_ERR}")
    st.write("Put this file next to both scripts, run Streamlit there.")
  st.stop()

L,R=st.columns([0.33,0.67],gap="large")

with L:
  st.subheader("Inputs")
  up_src=st.file_uploader("Source",type=["jpg","jpeg","png"])
  up_tgt=st.file_uploader("Target",type=["jpg","jpeg","png"])
  st.subheader("Explorer")
  size=st.slider("size",64,768,256,step=64)
  vizM=st.slider("M(viz)",1000,60000,12000,step=1000)
  seed=st.number_input("seed",min_value=0,max_value=10**9,value=0)
  st.subheader("Importance")
  imode=st.selectbox("mode",["combo","luma","center","edge"],index=0)
  wL=st.slider("w_luma",0.0,1.0,0.45,0.01)
  wC=st.slider("w_center",0.0,1.0,0.45,0.01)
  wE=st.slider("w_edge",0.0,1.0,0.10,0.01)
  st.subheader("Matching")
  jobs=st.slider("jobs",1,32,min(8,os.cpu_count() or 8))
  k_init=st.slider("k_init",1,256,64)
  k_max=st.slider("k_max",64,4096,2048,step=64)
  mp_start=st.selectbox("mp_start",["auto","fork","spawn"],index=0)
  st.subheader("Bench")
  benchM=st.slider("M(bench)",2000,150000,25000,step=1000)
  maxP=st.slider("maxP",1,32,min(8,os.cpu_count() or 8))
  sweep=st.selectbox("sweep",["pow2","1..P"],index=0)
  reps=st.slider("repeats",1,5,1)
  st.subheader("Full run")
  frames=st.slider("frames",1,250,120,5)
  fps=st.slider("fps",1,60,10)
  n=st.number_input("n(0=all)",min_value=0,value=0,step=1000)
  bg=st.selectbox("bg",["black","white"],index=0)
  blur=st.slider("blur",0.0,6.0,0.0,0.1)
  ca=st.slider("color_alpha(par)",0.0,1.0,0.35,0.01)
  run_mode=st.radio("mode",["sequential","parallel"],horizontal=True)
  do_full=st.button("Run full",type="primary",disabled=(up_src is None or up_tgt is None))

with R:
  if up_src is None or up_tgt is None:
    st.info("Upload both images.");st.stop()

  sb=up_src.getvalue();tb=up_tgt.getvalue()
  step=st.radio("Step",["1 Resize","2 Sample","3 Importance","4 Hue split","5 Greedy demo","6 Repair view","7 Benchmark"],horizontal=False)

  maps=feature_maps(sb,tb,size)
  prep=prepare(sb,tb,size,vizM,seed,imode,wL,wC,wE)

  if step=="1 Resize":
    c1,c2=st.columns(2)
    with c1: st.image(maps["src_u8"],caption="src",use_container_width=True)
    with c2: st.image(maps["tgt_u8"],caption="tgt",use_container_width=True)

  elif step=="2 Sample":
    c1,c2=st.columns(2)
    with c1: st.pyplot(fig_scatter(maps["src_u8"],prep.src_xy,f"src M={prep.M}"),clear_figure=True)
    with c2: st.pyplot(fig_scatter(maps["tgt_u8"],prep.tgt_xy,f"tgt M={prep.M}"),clear_figure=True)

  elif step=="3 Importance":
    imp=importance_map(tb,size,imode,wL,wC,wE)
    c1,c2=st.columns(2)
    with c1:
      st.pyplot(fig_img("tgt luma",maps["tgt_luma"],cmap="gray"),clear_figure=True)
      st.pyplot(fig_img("tgt edge",maps["tgt_edge"],cmap="magma"),clear_figure=True)
    with c2: st.pyplot(fig_img("importance",imp,cmap="inferno"),clear_figure=True)
    K=st.slider("topK",50,3000,600,50)
    order=np.argsort(-prep.importance,kind="mergesort")[:min(K,prep.importance.size)]
    top_xy=prep.tgt_xy[order]
    fig=plt.figure();plt.title("top targets");plt.imshow(maps["tgt_u8"]);plt.axis("off")
    plt.scatter(top_xy[:,0],top_xy[:,1],s=10,alpha=0.75);st.pyplot(fig,clear_figure=True)

  elif step=="4 Hue split":
    c1,c2=st.columns(2)
    with c1: st.pyplot(fig_img("src hue",maps["src_hue"],cmap="turbo",vmin=0,vmax=1),clear_figure=True)
    with c2: st.pyplot(fig_img("tgt hue",maps["tgt_hue"],cmap="turbo",vmin=0,vmax=1),clear_figure=True)
    st.pyplot(fig_segbar(prep.M,jobs,"segments"),clear_figure=True)
    seg=st.slider("seg",0,jobs-1,0)
    lo=(seg*prep.M)//jobs;hi=((seg+1)*prep.M)//jobs
    sids=prep.src_perm[lo:hi];tids=prep.tgt_perm[lo:hi]
    c1,c2=st.columns(2)
    with c1:
      fig=plt.figure();plt.title(f"src seg{seg}");plt.imshow(maps["src_u8"]);plt.axis("off")
      pts=prep.src_xy[sids];plt.scatter(pts[:,0],pts[:,1],s=8,alpha=0.75);st.pyplot(fig,clear_figure=True)
    with c2:
      fig=plt.figure();plt.title(f"tgt seg{seg}");plt.imshow(maps["tgt_u8"]);plt.axis("off")
      pts=prep.tgt_xy[tids];plt.scatter(pts[:,0],pts[:,1],s=8,alpha=0.75);st.pyplot(fig,clear_figure=True)

  elif step=="5 Greedy demo":
    seg=st.slider("seg",0,jobs-1,0)
    lo=(seg*prep.M)//jobs;hi=((seg+1)*prep.M)//jobs
    sids=prep.src_perm[lo:hi];tids=prep.tgt_perm[lo:hi]
    sr=prep.src_rgb_u8[sids].astype(np.float32);tr=prep.tgt_rgb_u8[tids].astype(np.float32)
    dk=st.slider("demo k",4,128,min(32,k_init),4)
    topN=st.slider("analyze N",10,250,60,5)
    neigh=SEQ.build_knn_lists_local(sr,tr,k=int(dk))
    impL=prep.importance[tids];ordL=np.argsort(-impL,kind="mergesort")[:min(topN,impL.size)]
    used=np.zeros(sids.size,dtype=np.bool_);conf=0;picks=[]
    for tloc in ordL:
      row=neigh[int(tloc)]
      if used[int(row[0])]: conf+=1
      pick=-1
      for c in row:
        c=int(c)
        if not used[c]: pick=c;used[c]=True;break
      picks.append((int(tloc),int(pick)))
    st.metric("first-choice conflicts",f"{conf}")
    pi=st.slider("viz pick",0,len(picks)-1,0)
    tloc,sloc=picks[pi]
    cand=neigh[tloc];candg=sids[cand]
    chosen=sids[sloc] if sloc>=0 else None
    c1,c2=st.columns(2)
    with c1:
      fig=plt.figure();plt.title("src candidates");plt.imshow(maps["src_u8"]);plt.axis("off")
      pts=prep.src_xy[candg];plt.scatter(pts[:,0],pts[:,1],s=60,alpha=0.9)
      if chosen is not None:
        ch=prep.src_xy[chosen];plt.scatter([ch[0]],[ch[1]],s=200,marker="x")
      st.pyplot(fig,clear_figure=True)
    with c2:
      fig=plt.figure();plt.title("tgt");plt.imshow(maps["tgt_u8"]);plt.axis("off")
      tg=prep.tgt_xy[tids[tloc]];plt.scatter([tg[0]],[tg[1]],s=240,marker="x")
      st.pyplot(fig,clear_figure=True)

  elif step=="6 Repair view":
    chosen,ts=seq_match_only(prep,jobs,k_init,k_max)
    uniq,cnts=np.unique(chosen,return_counts=True)
    dups=int(np.sum(cnts>1));miss=int(prep.M-uniq.size)
    c1,c2,c3=st.columns(3)
    c1.metric("jobs",f"{jobs}")
    c2.metric("dups(pre)",f"{dups}")
    c3.metric("missing(pre)",f"{miss}")
    repaired=PAR.enforce_bijection(chosen,prep.importance)
    st.metric("unique(after)",f"{np.unique(repaired).size}/{prep.M}")

  else:
    Ps=[]
    if sweep=="pow2":
      p=1
      while p<=maxP: Ps.append(p);p*=2
      if Ps[-1]!=maxP: Ps.append(maxP)
    else: Ps=list(range(1,maxP+1))
    st.write("P:",Ps)
    bench=prepare(sb,tb,size,benchM,seed,imode,wL,wC,wE)
    if st.button("Run benchmark",type="primary"):
      rows=[]
      for P in Ps:
        tse=[];tpa=[]
        for _ in range(reps):
          _,ts=seq_match_only(bench,P,k_init,k_max);_,tp=par_match_only(bench,P,k_init,k_max,mp_start)
          tse.append(ts["total"]);tpa.append(tp["total"])
        rows.append((P,float(np.mean(tse)),float(np.mean(tpa))))
      P_list=[a for a,_,_ in rows];Tseq=[b for _,b,_ in rows];Tpar=[c for *_,c in rows]
      c1,c2,c3=st.columns(3)
      c1.metric("M",f"{bench.M}")
      c2.metric("best speedup",f"{max([Tseq[i]/Tpar[i] for i in range(len(P_list))]):.2f}x")
      c3.metric("Pmax",f"{max(P_list)}")
      fig=plt.figure();plt.title("match-only time")
      plt.plot(P_list,Tseq,marker="o",label="seq(jobs=P)")
      plt.plot(P_list,Tpar,marker="o",label="par(jobs=P)")
      plt.xlabel("P");plt.ylabel("sec");plt.legend();st.pyplot(fig,clear_figure=True)
      st.pyplot(fig_speed(P_list,Tseq,Tpar,"speedup"),clear_figure=True)
      st.pyplot(fig_eff(P_list,Tseq,Tpar,"efficiency"),clear_figure=True)

  st.divider()
  st.subheader("Full run")
  if do_full:
    wd=_wd();stamp=f"{int(time.time())}_{random.randint(1000,9999)}"
    sp=os.path.join(wd,f"src_{stamp}.jpg");tp=os.path.join(wd,f"tgt_{stamp}.jpg")
    op=os.path.join(wd,f"out_{run_mode}_{stamp}.gif")
    _wb(sp,sb);_wb(tp,tb)
    args={"size":int(size),"n":int(n),"frames":int(frames),"fps":int(fps),"jobs":int(jobs),
          "importance":str(imode),"w_luma":float(wL),"w_center":float(wC),"w_edge":float(wE),
          "k_init":int(k_init),"k_max":int(k_max),"blur":float(blur),"bg":str(bg),
          "easing_pow":2.2,"progress_every":10**9,"seed":int(seed),
          "mp_start":str(mp_start),"color_alpha":float(ca)}
    box=st.empty()
    with st.spinner("Running..."):
      rc,wall=run_original(run_mode,sp,tp,op,args,box)
    if rc!=0 or (not os.path.exists(op)): st.error(f"failed rc={rc}")
    else:
      gb=open(op,"rb").read()
      st.success(f"done {wall:.3f}s")
      st.image(gb,caption="gif",use_container_width=True)
      st.download_button("Download",data=gb,file_name=os.path.basename(op),mime="image/gif")

  with st.expander("Cleanup"):
    if st.button("Delete workdir"):
      wd=_wd();shutil.rmtree(wd,ignore_errors=True);st.session_state.pop("workdir",None)
      st.success("deleted")
#[theme]
#primaryColor="#d6b1b1"
#backgroundColor="#ffbbfe"
#secondaryBackgroundColor="#c3ffff"
#textColor="#a752d4"
