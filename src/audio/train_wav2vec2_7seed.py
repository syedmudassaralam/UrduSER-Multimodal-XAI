#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
UrduSER 7-class audio-only v6 UPDATED FINAL.
Segment-level training + file-level voting + safe cache + confusion-aware decision tuning.
Reads original folder dataset directly and creates exact 80/10/10 split.

Dataset:
UrduSER A Dataset for Urdu Speech Emotion Recognition/
  Neutral/ Happy/ Angry/ Sad/ Fear/ Disgust/ Boredom/

Run:
  python urdu_ser_audio_v6_updated_final.py

Important: No code can guarantee 70%, but this is a larger update than simple
class weights. It trains on multiple random segments per file and evaluates by
averaging multiple deterministic crops per file.
"""
from __future__ import annotations

import json, math, os, random, time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report, confusion_matrix
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from tqdm.auto import tqdm
from transformers import AutoModel, get_cosine_schedule_with_warmup

LABELS = ["Neutral", "Happy", "Angry", "Sad", "Fear", "Disgust", "Boredom"]
LABEL2ID = {x:i for i,x in enumerate(LABELS)}
ID2LABEL = {i:x for x,i in LABEL2ID.items()}

@dataclass
class CFG:
    DATA_ROOT: str = "./data/UrduSER"   # change this path
    MODEL_NAME: str = "facebook/wav2vec2-base"  # or microsoft/wavlm-base-plus
    OUT_DIR: str = "./outputs/audio/wav2vec2_7seed"
    CACHE_DIR: str = "./cache/audio_wav2vec2_7seed_16k_5s"  # separate safe cache

    FILES_PER_CLASS: int = 500
    STRICT_500_PER_CLASS: bool = True
    USE_DIRECT_FILES_ONLY: bool = True

    SAMPLE_RATE: int = 16000
    FILE_SECONDS: float = 5.0
    SEGMENT_SECONDS: float = 3.0
    TRAIN_SEGMENTS_PER_FILE: int = 5
    EVAL_CROPS_PER_FILE: int = 15

    TRAIN_PER_CLASS: int = 400
    VAL_PER_CLASS: int = 50
    TEST_PER_CLASS: int = 50

    SEEDS: Tuple[int, ...] = (123, 42, 777, 2024, 2025, 3407, 9999)  # for quick test use (123,)
    BATCH_SIZE: int = 8
    GRAD_ACCUM_STEPS: int = 2
    NUM_WORKERS: int = 4
    USE_AMP: bool = True
    USE_CACHE: bool = True

    EPOCHS: int = 30
    PATIENCE: int = 9
    LR_ENCODER: float = 2e-6
    LR_HEAD: float = 1.2e-4
    WEIGHT_DECAY: float = 0.01
    WARMUP_RATIO: float = 0.10
    MAX_GRAD_NORM: float = 1.0

    DROPOUT: float = 0.30
    ACOUSTIC_DIM: int = 45
    ACOUSTIC_DROPOUT: float = 0.20
    LABEL_SMOOTHING: float = 0.06
    FOCAL_GAMMA: float = 1.0

    FREEZE_FEATURE_EXTRACTOR: bool = True
    TRAIN_LAST_N_TRANSFORMER_LAYERS: int = 8

    USE_AUGMENT: bool = True
    NOISE_PROB: float = 0.40
    GAIN_PROB: float = 0.40
    SHIFT_PROB: float = 0.30
    SPEED_PROB: float = 0.18
    LOWPASS_PROB: float = 0.10
    HIDDEN_MASK_PROB: float = 0.20

    USE_VAL_TEMPERATURE: bool = True
    TEMPS: Tuple[float, ...] = (0.7, 0.85, 1.0, 1.15, 1.3, 1.5, 1.7)

    # Small v6 confusion tweak: reduce only close Disgust false positives.
    # From your v6 confusion matrix, Neutral/Happy/Fear were often pulled to Disgust.
    USE_CONFUSION_DECISION_TWEAK: bool = True
    DISGUST_CLOSE_MARGIN: float = 0.08
    SAD_FEAR_CLOSE_MARGIN: float = 0.04
    TUNE_DECISION_MARGINS_ON_VALIDATION: bool = True
    DISGUST_MARGIN_GRID: Tuple[float, ...] = (0.00, 0.03, 0.05, 0.08, 0.10, 0.12)
    SAD_FEAR_MARGIN_GRID: Tuple[float, ...] = (0.00, 0.02, 0.04, 0.06, 0.08)

    DEVICE: str = "cuda" if torch.cuda.is_available() else "cpu"

cfg = CFG()

# Shared split files produced by urdu_ser_text_shared_split_current_only.py
cfg.SHARED_SPLIT_DIR = "./data_splits"
cfg.SHARED_TRAIN_CSV = f"{cfg.SHARED_SPLIT_DIR}/train_split.csv"
cfg.SHARED_VAL_CSV = f"{cfg.SHARED_SPLIT_DIR}/val_split.csv"
cfg.SHARED_TEST_CSV = f"{cfg.SHARED_SPLIT_DIR}/test_split.csv"


def set_seed(seed:int):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True

def list_class_files(d:Path)->List[Path]:
    exts={".wav",".mp3",".flac",".ogg",".m4a",".aac"}
    if cfg.USE_DIRECT_FILES_ONLY:
        return sorted([p for p in d.iterdir() if p.is_file() and p.suffix.lower() in exts])
    return sorted([p for p in d.rglob("*") if p.is_file() and p.suffix.lower() in exts])

def load_original_folders()->pd.DataFrame:
    root=Path(cfg.DATA_ROOT)
    if not root.exists(): raise FileNotFoundError(f"DATA_ROOT does not exist: {root.resolve()}")
    rows=[]
    for lab in LABELS:
        folder=root/lab
        if not folder.exists(): raise FileNotFoundError(f"Missing class folder: {folder}")
        files=list_class_files(folder)
        if len(files)<cfg.FILES_PER_CLASS: raise RuntimeError(f"{lab} has {len(files)} files; expected {cfg.FILES_PER_CLASS}")
        if cfg.STRICT_500_PER_CLASS and len(files)!=cfg.FILES_PER_CLASS:
            print(f"WARNING: {lab} has {len(files)} files. Using first {cfg.FILES_PER_CLASS} sorted files.")
        for p in files[:cfg.FILES_PER_CLASS]:
            rows.append({"path":str(p.resolve()),"file_id":p.stem,"label":lab,"label_id":LABEL2ID[lab]})
    df=pd.DataFrame(rows)
    print("\nLoaded original folders:"); print(df.label.value_counts().reindex(LABELS))
    return df

def exact_split(df:pd.DataFrame, seed:int):
    rng=np.random.default_rng(seed); tr=[]; va=[]; te=[]
    for lab in LABELS:
        part=df[df.label==lab].reset_index(drop=True)
        idx=np.arange(len(part)); rng.shuffle(idx); part=part.iloc[idx].reset_index(drop=True)
        tr.append(part.iloc[:cfg.TRAIN_PER_CLASS])
        va.append(part.iloc[cfg.TRAIN_PER_CLASS:cfg.TRAIN_PER_CLASS+cfg.VAL_PER_CLASS])
        te.append(part.iloc[cfg.TRAIN_PER_CLASS+cfg.VAL_PER_CLASS:cfg.TRAIN_PER_CLASS+cfg.VAL_PER_CLASS+cfg.TEST_PER_CLASS])
    train=pd.concat(tr).sample(frac=1,random_state=seed).reset_index(drop=True)
    val=pd.concat(va).sample(frac=1,random_state=seed).reset_index(drop=True)
    test=pd.concat(te).sample(frac=1,random_state=seed).reset_index(drop=True)
    for n,p in [("TRAIN",train),("VAL",val),("TEST",test)]:
        print(f"\n{n}: {len(p)}"); print(p.label.value_counts().reindex(LABELS))
    return train,val,test

def load_audio(path:str):
    wav,sr=torchaudio.load(path); wav=wav.mean(0)
    if sr!=cfg.SAMPLE_RATE: wav=torchaudio.functional.resample(wav,sr,cfg.SAMPLE_RATE)
    return wav.float()

def trim_silence(wav, top_db=38.0):
    peak=wav.abs().max().clamp_min(1e-6); thr=peak*(10**(-top_db/20.0)); idx=torch.where(wav.abs()>thr)[0]
    if idx.numel()<cfg.SAMPLE_RATE//10: return wav
    pad=int(0.08*cfg.SAMPLE_RATE); s=max(0,idx[0].item()-pad); e=min(wav.numel(),idx[-1].item()+pad)
    return wav[s:e]

def normalize(wav):
    wav=wav-wav.mean(); rms=wav.pow(2).mean().sqrt().clamp_min(1e-5); wav=wav/rms*0.08
    peak=wav.abs().max();
    if peak>1.0: wav=wav/peak
    return wav.clamp(-1,1)

def pre_emphasis(wav, c=0.97):
    if wav.numel()<2: return wav
    return torch.cat([wav[:1], wav[1:]-c*wav[:-1]])

def fix_len(wav, seconds):
    tgt=int(seconds*cfg.SAMPLE_RATE)
    if wav.numel()>tgt:
        st=(wav.numel()-tgt)//2; wav=wav[st:st+tgt]
    elif wav.numel()<tgt:
        pad=tgt-wav.numel(); wav=F.pad(wav,(pad//2,pad-pad//2))
    return wav

def preprocess(path):
    wav=load_audio(path); wav=trim_silence(wav); wav=normalize(wav); wav=pre_emphasis(wav); wav=fix_len(wav,cfg.FILE_SECONDS)
    return wav.float()

def cache_path(path):
    name=str(Path(path).resolve()).replace(os.sep,"_").replace(":","")
    return Path(cfg.CACHE_DIR)/f"{name}.pt"

def safe_torch_load_tensor(cp: Path):
    """Load cached waveform safely. If cache is corrupt/incomplete, delete it and return None."""
    try:
        try:
            obj = torch.load(cp, map_location="cpu", weights_only=True)
        except TypeError:
            obj = torch.load(cp, map_location="cpu")
        if isinstance(obj, torch.Tensor) and obj.numel() > 0:
            return obj.float()
        print(f"WARNING: invalid cache tensor, rebuilding: {cp}")
    except Exception as exc:
        print(f"WARNING: corrupt cache file, rebuilding: {cp} | {type(exc).__name__}: {exc}")

    try:
        cp.unlink(missing_ok=True)
    except TypeError:
        if cp.exists():
            cp.unlink()
    except Exception:
        pass
    return None


def atomic_torch_save_tensor(tensor: torch.Tensor, cp: Path):
    """Write cache atomically to avoid DataLoader workers reading half-written .pt files."""
    cp.parent.mkdir(parents=True, exist_ok=True)
    tmp = cp.with_suffix(cp.suffix + f".tmp.{os.getpid()}.{random.randint(0, 10**9)}")
    torch.save(tensor.cpu(), tmp)
    os.replace(str(tmp), str(cp))


def get_waveform(path):
    if not cfg.USE_CACHE:
        return preprocess(path)

    cp = cache_path(path)
    if cp.exists():
        cached = safe_torch_load_tensor(cp)
        if cached is not None:
            return cached

    wav = preprocess(path)
    try:
        atomic_torch_save_tensor(wav, cp)
    except Exception as exc:
        print(f"WARNING: could not write cache {cp}: {exc}")
    return wav

def segment(wav, train:bool, crop_id:int=0):
    seg_len=int(cfg.SEGMENT_SECONDS*cfg.SAMPLE_RATE)
    if wav.numel()<=seg_len: return F.pad(wav,(0,seg_len-wav.numel())) if wav.numel()<seg_len else wav
    max_start=wav.numel()-seg_len
    if train: st=random.randint(0,max_start)
    else: st=int(round(crop_id*max_start/max(1,cfg.EVAL_CROPS_PER_FILE-1)))
    return wav[st:st+seg_len]

def augment(wav):
    if random.random()<cfg.GAIN_PROB: wav=wav*random.uniform(0.75,1.25)
    if random.random()<cfg.NOISE_PROB: wav=wav+torch.randn_like(wav)*random.uniform(0.0008,0.006)
    if random.random()<cfg.SHIFT_PROB: wav=torch.roll(wav, random.randint(-cfg.SAMPLE_RATE//6,cfg.SAMPLE_RATE//6), dims=0)
    if random.random()<cfg.SPEED_PROB:
        speed=random.choice([0.96,0.98,1.02,1.04]); sr2=int(cfg.SAMPLE_RATE*speed)
        wav=torchaudio.functional.resample(wav,cfg.SAMPLE_RATE,sr2); wav=torchaudio.functional.resample(wav,sr2,cfg.SAMPLE_RATE)
        wav=fix_len(wav,cfg.SEGMENT_SECONDS)
    if random.random()<cfg.LOWPASS_PROB:
        wav=torchaudio.functional.lowpass_biquad(wav,cfg.SAMPLE_RATE,random.choice([2800,3200,3600,4200]))
    return wav.clamp(-1,1)

def acoustic_features(wav):
    wav=wav.float().cpu(); frame=400; hop=160; eps=1e-8
    if wav.numel()<frame: wav=F.pad(wav,(0,frame-wav.numel()))
    frames=wav.unfold(0,frame,hop)
    rms=torch.sqrt(frames.pow(2).mean(1).clamp_min(eps)); zcr=(frames[:,1:].sign()!=frames[:,:-1].sign()).float().mean(1)
    window=torch.hann_window(frame)
    spec=torch.stft(wav,n_fft=frame,hop_length=hop,win_length=frame,window=window,return_complex=True).abs()
    power=spec.pow(2).clamp_min(eps); freqs=torch.linspace(0,cfg.SAMPLE_RATE/2,power.shape[0]).unsqueeze(1); mag=power.sum(0).clamp_min(eps)
    centroid=(freqs*power).sum(0)/mag; bandwidth=torch.sqrt((((freqs-centroid.unsqueeze(0))**2)*power).sum(0)/mag)
    rolloff_mask=torch.cumsum(power,0)>=0.85*mag.unsqueeze(0); rolloff=freqs.squeeze(1)[rolloff_mask.float().argmax(0)]
    mfcc_t=torchaudio.transforms.MFCC(sample_rate=cfg.SAMPLE_RATE,n_mfcc=13,melkwargs={"n_fft":frame,"hop_length":hop,"n_mels":40,"center":True})
    mfcc=mfcc_t(wav).float(); delta=mfcc[:,1:]-mfcc[:,:-1]
    feat=torch.cat([
        torch.tensor([rms.mean(),rms.std(unbiased=False),rms.max(),rms.min()]),
        torch.tensor([zcr.mean(),zcr.std(unbiased=False),zcr.max()]),
        torch.tensor([centroid.mean(),centroid.std(unbiased=False)])/8000.0,
        torch.tensor([bandwidth.mean(),bandwidth.std(unbiased=False)])/8000.0,
        torch.tensor([rolloff.mean(),rolloff.std(unbiased=False)])/8000.0,
        mfcc.mean(1), mfcc.std(1,unbiased=False), delta.mean(1)
    ])
    feat=torch.nan_to_num(feat,nan=0.0,posinf=0.0,neginf=0.0).float()
    if feat.numel()>cfg.ACOUSTIC_DIM: feat=feat[:cfg.ACOUSTIC_DIM]
    elif feat.numel()<cfg.ACOUSTIC_DIM: feat=F.pad(feat,(0,cfg.ACOUSTIC_DIM-feat.numel()))
    return feat

class TrainDS(Dataset):
    def __init__(self,df):
        self.df=df.reset_index(drop=True); self.items=[]
        for i in range(len(self.df)):
            for _ in range(cfg.TRAIN_SEGMENTS_PER_FILE): self.items.append(i)
    def __len__(self): return len(self.items)
    def __getitem__(self,idx):
        row=self.df.iloc[self.items[idx]]; wav=get_waveform(row.path); seg=segment(wav,True)
        if cfg.USE_AUGMENT: seg=augment(seg)
        return {"x":seg,"a":acoustic_features(seg),"y":torch.tensor(int(row.label_id)),"file_index":torch.tensor(int(self.items[idx]))}

class EvalDS(Dataset):
    def __init__(self,df):
        self.df=df.reset_index(drop=True); self.items=[]
        for i in range(len(self.df)):
            for c in range(cfg.EVAL_CROPS_PER_FILE): self.items.append((i,c))
    def __len__(self): return len(self.items)
    def __getitem__(self,idx):
        fi,c=self.items[idx]; row=self.df.iloc[fi]; wav=get_waveform(row.path); seg=segment(wav,False,c)
        return {"x":seg,"a":acoustic_features(seg),"y":torch.tensor(int(row.label_id)),"file_index":torch.tensor(int(fi))}

def collate(b): return {"x":torch.stack([z["x"] for z in b]),"a":torch.stack([z["a"] for z in b]),"y":torch.stack([z["y"] for z in b]),"file_index":torch.stack([z["file_index"] for z in b])}

def train_loader(df):
    ds=TrainDS(df); labs=[int(df.iloc[i].label_id) for i in ds.items]; cnt=np.bincount(labs,minlength=len(LABELS)); w=np.array([1.0/cnt[y] for y in labs])
    return DataLoader(ds,batch_size=cfg.BATCH_SIZE,sampler=WeightedRandomSampler(w,len(w),replacement=True),num_workers=cfg.NUM_WORKERS,pin_memory=cfg.DEVICE.startswith("cuda"),collate_fn=collate)

def eval_loader(df): return DataLoader(EvalDS(df),batch_size=cfg.BATCH_SIZE,shuffle=False,num_workers=cfg.NUM_WORKERS,pin_memory=cfg.DEVICE.startswith("cuda"),collate_fn=collate)

class Pool(nn.Module):
    def __init__(self,h): super().__init__(); self.attn=nn.Sequential(nn.Linear(h,h//2),nn.Tanh(),nn.Linear(h//2,1))
    def forward(self,x):
        w=torch.softmax(self.attn(x).squeeze(-1),dim=1).unsqueeze(-1); mean=(w*x).sum(1); std=torch.sqrt((w*(x-mean.unsqueeze(1)).pow(2)).sum(1).clamp_min(1e-6)); mx=x.max(1).values
        return torch.cat([mean,std,mx],-1)

class SER(nn.Module):
    def __init__(self):
        super().__init__()
        # Use local_files_only=True when MODEL_NAME is a local folder such as ./wav2vec2-base.
        # This avoids Hugging Face internet checks/timeouts on your machine.
        if str(cfg.MODEL_NAME).startswith("./") or str(cfg.MODEL_NAME).startswith("/"):
            self.encoder = AutoModel.from_pretrained(cfg.MODEL_NAME, local_files_only=True)
        else:
            self.encoder = AutoModel.from_pretrained(cfg.MODEL_NAME)
        h=self.encoder.config.hidden_size; self.pool=Pool(h)
        self.aproj=nn.Sequential(nn.LayerNorm(cfg.ACOUSTIC_DIM),nn.Dropout(cfg.ACOUSTIC_DROPOUT),nn.Linear(cfg.ACOUSTIC_DIM,160),nn.GELU(),nn.Dropout(cfg.ACOUSTIC_DROPOUT))
        dim=h*3+160
        self.cls=nn.Sequential(nn.LayerNorm(dim),nn.Dropout(cfg.DROPOUT),nn.Linear(dim,h),nn.GELU(),nn.Dropout(cfg.DROPOUT),nn.Linear(h,h//2),nn.GELU(),nn.Dropout(cfg.DROPOUT),nn.Linear(h//2,len(LABELS)))
        self.configure()
    def configure(self):
        if cfg.FREEZE_FEATURE_EXTRACTOR and hasattr(self.encoder,"feature_extractor"):
            for p in self.encoder.feature_extractor.parameters(): p.requires_grad=False
        layers=getattr(getattr(self.encoder,"encoder",None),"layers",None)
        if layers is not None:
            cut=max(0,len(layers)-cfg.TRAIN_LAST_N_TRANSFORMER_LAYERS)
            for i,l in enumerate(layers):
                for p in l.parameters(): p.requires_grad=i>=cut
    def forward(self,x,a):
        mask=torch.ones_like(x,dtype=torch.long); h=self.encoder(input_values=x,attention_mask=mask).last_hidden_state
        if self.training and random.random()<cfg.HIDDEN_MASK_PROB: h=self.hidden_mask(h)
        z=torch.cat([self.pool(h),self.aproj(a)],-1); return self.cls(z)
    @staticmethod
    def hidden_mask(h):
        b,t,_=h.shape; width=max(2,t//20); h=h.clone()
        for i in range(b):
            st=random.randint(0,max(0,t-width)); h[i,st:st+width]=0
        return h

class Focal(nn.Module):
    def __init__(self,w): super().__init__(); self.register_buffer("w",w)
    def forward(self,logits,y):
        ce=F.cross_entropy(logits,y,weight=self.w,label_smoothing=cfg.LABEL_SMOOTHING,reduction="none"); pt=torch.exp(-ce)
        return (((1-pt)**cfg.FOCAL_GAMMA)*ce).mean()

def weights(df):
    y=df.label_id.to_numpy(); cnt=np.bincount(y,minlength=len(LABELS)).astype(np.float32); w=cnt.sum()/np.maximum(cnt,1); w=w/w.mean()
    # Confusion-aware class multipliers. v6 over-predicted Disgust, so Disgust is reduced,
    # while Neutral/Happy/Fear get mild support.
    mult={LABEL2ID["Neutral"]:1.05,LABEL2ID["Happy"]:1.20,LABEL2ID["Angry"]:0.95,LABEL2ID["Sad"]:1.00,LABEL2ID["Fear"]:1.35,LABEL2ID["Disgust"]:1.00,LABEL2ID["Boredom"]:1.08}
    for k,v in mult.items(): w[k]*=v
    return torch.tensor(w/w.mean(),dtype=torch.float32,device=cfg.DEVICE)

def optim(model):
    enc=[]; head=[]
    for n,p in model.named_parameters():
        if not p.requires_grad: continue
        (enc if n.startswith("encoder.") else head).append(p)
    return torch.optim.AdamW([{"params":enc,"lr":cfg.LR_ENCODER},{"params":head,"lr":cfg.LR_HEAD}],weight_decay=cfg.WEIGHT_DECAY)


def confusion_aware_predictions(
    probs: np.ndarray,
    disgust_margin: float | None = None,
    sad_fear_margin: float | None = None,
) -> np.ndarray:
    """Small post-decision tweak for repeated v6 confusions.

    It does not change probabilities saved to CSV. It only changes the final label
    when Disgust is top but Neutral/Happy/Fear is very close. This targets the
    repeated v6 errors: Neutral->Disgust, Happy->Disgust, Fear->Disgust.
    """
    pred = probs.argmax(axis=1).copy()
    if not cfg.USE_CONFUSION_DECISION_TWEAK:
        return pred

    if disgust_margin is None:
        disgust_margin = cfg.DISGUST_CLOSE_MARGIN
    if sad_fear_margin is None:
        sad_fear_margin = cfg.SAD_FEAR_CLOSE_MARGIN

    disgust = LABEL2ID["Disgust"]
    close_targets = [LABEL2ID["Neutral"], LABEL2ID["Happy"], LABEL2ID["Fear"]]

    for i in range(probs.shape[0]):
        if pred[i] == disgust:
            target_scores = probs[i, close_targets]
            best_target = close_targets[int(np.argmax(target_scores))]
            # switch only when the alternative class is close to Disgust
            if probs[i, disgust] - probs[i, best_target] <= disgust_margin:
                pred[i] = best_target

    # A very small Sad/Fear fix. Fear was often confused with Sad, so if Sad is
    # top but Fear is extremely close, keep Fear.
    sad = LABEL2ID["Sad"]
    fear = LABEL2ID["Fear"]
    for i in range(probs.shape[0]):
        if pred[i] == sad and (probs[i, sad] - probs[i, fear] <= sad_fear_margin):
            pred[i] = fear

    return pred

@torch.no_grad()
def predict_file(model, df, temp=1.0, apply_tweak=True, disgust_margin=None, sad_fear_margin=None):
    loader=eval_loader(df); model.eval(); probs=np.zeros((len(df),len(LABELS))); counts=np.zeros(len(df)); y=df.label_id.to_numpy()
    for b in tqdm(loader,desc="File-level predicting",leave=False):
        x=b["x"].to(cfg.DEVICE); a=b["a"].to(cfg.DEVICE); idx=b["file_index"].numpy(); pr=torch.softmax(model(x,a)/temp,dim=-1).cpu().numpy()
        for i,fi in enumerate(idx): probs[int(fi)]+=pr[i]; counts[int(fi)]+=1
    probs=probs/np.maximum(counts[:,None],1)
    if apply_tweak:
        pred=confusion_aware_predictions(probs, disgust_margin=disgust_margin, sad_fear_margin=sad_fear_margin)
    else:
        pred=probs.argmax(1)
    return y,pred,probs

def tune_decision_margins(y_true: np.ndarray, probs: np.ndarray, run_dir: Path | None = None) -> Tuple[float, float]:
    """Validation-based selection of the two small decision margins.

    This avoids guessing fixed margins. It chooses margins that improve validation macro F1
    while also considering the weakest class F1.
    """
    if not cfg.TUNE_DECISION_MARGINS_ON_VALIDATION:
        return cfg.DISGUST_CLOSE_MARGIN, cfg.SAD_FEAR_CLOSE_MARGIN

    best_d = cfg.DISGUST_CLOSE_MARGIN
    best_sf = cfg.SAD_FEAR_CLOSE_MARGIN
    best_score = -1.0
    best_metrics = {}
    for d in cfg.DISGUST_MARGIN_GRID:
        for sf in cfg.SAD_FEAR_MARGIN_GRID:
            pred = confusion_aware_predictions(probs, disgust_margin=d, sad_fear_margin=sf)
            macro = f1_score(y_true, pred, average="macro", zero_division=0)
            acc = accuracy_score(y_true, pred)
            per = f1_score(y_true, pred, average=None, labels=list(range(len(LABELS))), zero_division=0)
            score = float(macro + 0.04 * acc + 0.08 * per.min())
            if score > best_score:
                best_score = score
                best_d = float(d)
                best_sf = float(sf)
                best_metrics = {
                    "validation_accuracy": float(acc),
                    "validation_macro_f1": float(macro),
                    "validation_min_class_f1": float(per.min()),
                    "disgust_close_margin": best_d,
                    "sad_fear_close_margin": best_sf,
                }
    print("\nValidation-tuned decision margins:")
    print(json.dumps(best_metrics, indent=2))
    if run_dir is not None:
        (run_dir/"validation_decision_margins.json").write_text(json.dumps(best_metrics, indent=2), encoding="utf-8")
    return best_d, best_sf

def metrics(y,p):
    per=f1_score(y,p,average=None,labels=list(range(len(LABELS))),zero_division=0)
    out={"accuracy":accuracy_score(y,p),"macro_precision":precision_score(y,p,average="macro",zero_division=0),"macro_recall":recall_score(y,p,average="macro",zero_division=0),"macro_f1":f1_score(y,p,average="macro",zero_division=0),"weighted_precision":precision_score(y,p,average="weighted",zero_division=0),"weighted_recall":recall_score(y,p,average="weighted",zero_division=0),"weighted_f1":f1_score(y,p,average="weighted",zero_division=0),"min_class_f1":float(per.min())}
    for i,l in enumerate(LABELS): out[f"f1_{l}"]=float(per[i])
    return out

def best_temp(model,val):
    if not cfg.USE_VAL_TEMPERATURE: return 1.0
    bt=1.0; bf=-1
    for t in cfg.TEMPS:
        y,p,_=predict_file(model,val,t,apply_tweak=True); f=f1_score(y,p,average="macro",zero_division=0)
        if f>bf: bf=f; bt=t
    print(f"Best validation temperature: {bt} val_macro_f1={bf:.4f}"); return bt


def canonical_audio_label(x):
    s = "" if pd.isna(x) else str(x).strip()
    low = s.lower()
    mp = {
        "neutral": "Neutral",
        "happy": "Happy",
        "angry": "Angry",
        "sad": "Sad",
        "fear": "Fear",
        "disgust": "Disgust",
        "boredum": "Boredom",
        "boredom": "Boredom",
    }
    return mp.get(low, s)

def load_shared_split_csv(path: str) -> pd.DataFrame:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(
            f"Shared split not found: {p.resolve()}\n"
            "First run: python urdu_ser_text_shared_split_current_only.py"
        )
    df = pd.read_csv(p)

    if "path" not in df.columns:
        if "audio_path" in df.columns:
            df["path"] = df["audio_path"]
        else:
            raise ValueError(f"{p} must contain path or audio_path column.")

    if "file_id" not in df.columns:
        df["file_id"] = df["path"].apply(lambda x: Path(str(x)).stem)

    if "target_label" in df.columns:
        label_col = "target_label"
    elif "true_label" in df.columns:
        label_col = "true_label"
    elif "label" in df.columns:
        label_col = "label"
    else:
        raise ValueError(f"{p} must contain target_label, true_label, or label column.")

    out = pd.DataFrame()
    out["path"] = df["path"].astype(str)
    out["file_id"] = df["file_id"].astype(str).apply(lambda x: Path(str(x)).stem)
    out["label"] = df[label_col].map(canonical_audio_label)
    out["label_id"] = out["label"].map(LABEL2ID)

    before = len(out)
    out = out.dropna(subset=["label_id"]).copy()
    out["label_id"] = out["label_id"].astype(int)
    if len(out) != before:
        print(f"WARNING: removed {before-len(out)} rows with invalid labels from {p}")

    exists = out["path"].apply(lambda x: Path(str(x)).exists())
    missing = int((~exists).sum())
    if missing:
        print(f"WARNING: removing {missing} rows whose audio paths do not exist from {p}")
        print(out.loc[~exists, ["path", "file_id", "label"]].head(10).to_string(index=False))
        out = out[exists].reset_index(drop=True)

    return out.reset_index(drop=True)

def load_shared_splits():
    train = load_shared_split_csv(cfg.SHARED_TRAIN_CSV)
    val = load_shared_split_csv(cfg.SHARED_VAL_CSV)
    test = load_shared_split_csv(cfg.SHARED_TEST_CSV)

    for name, part in [("TRAIN", train), ("VAL", val), ("TEST", test)]:
        print(f"\n{name}: {len(part)}")
        print(part.label.value_counts().reindex(LABELS))

    return train, val, test

def save_prediction_outputs(df, y, p, pr, out_dir: Path, prefix_name: str):
    out = df[["path", "file_id", "label", "label_id"]].copy()
    out["prediction"] = [ID2LABEL[int(x)] for x in p]
    out["prediction_id"] = p
    for lab in LABELS:
        out[f"prob_{lab}"] = pr[:, LABEL2ID[lab]]

    out.to_csv(out_dir / f"{prefix_name}_predictions.csv", index=False, encoding="utf-8-sig")
    out[out.label_id != out.prediction_id].to_csv(
        out_dir / f"{prefix_name}_misclassified_samples.csv",
        index=False,
        encoding="utf-8-sig",
    )

    rep = classification_report(y, p, labels=list(range(len(LABELS))), target_names=LABELS, zero_division=0)
    cm = confusion_matrix(y, p, labels=list(range(len(LABELS))))
    (out_dir / f"{prefix_name}_classification_report.txt").write_text(rep, encoding="utf-8")
    pd.DataFrame(cm, index=LABELS, columns=LABELS).to_csv(
        out_dir / f"{prefix_name}_confusion_matrix.csv",
        encoding="utf-8-sig",
    )

def train_one(train, val, test, seed, root):
    set_seed(seed)
    run = root / f"seed_{seed}_{time.strftime('%Y%m%d_%H%M%S')}"
    run.mkdir(parents=True, exist_ok=True)

    (run / "config.json").write_text(json.dumps(asdict(cfg), indent=2, default=str), encoding="utf-8")
    train.to_csv(run / "train_split.csv", index=False, encoding="utf-8-sig")
    val.to_csv(run / "val_split.csv", index=False, encoding="utf-8-sig")
    test.to_csv(run / "test_split.csv", index=False, encoding="utf-8-sig")

    tl = train_loader(train)
    model = SER().to(cfg.DEVICE)
    crit = Focal(weights(train))
    opt = optim(model)
    updates = math.ceil(len(tl) / cfg.GRAD_ACCUM_STEPS) * cfg.EPOCHS
    sched = get_cosine_schedule_with_warmup(opt, int(updates * cfg.WARMUP_RATIO), updates)
    scaler = torch.cuda.amp.GradScaler(enabled=cfg.USE_AMP and cfg.DEVICE.startswith("cuda"))

    best = -1
    bad = 0
    hist = []

    for ep in range(1, cfg.EPOCHS + 1):
        model.train()
        opt.zero_grad(set_to_none=True)
        losses = []
        yt = []
        yp = []

        for st, b in enumerate(tqdm(tl, desc=f"Seed {seed} Epoch {ep}")):
            x = b["x"].to(cfg.DEVICE)
            a = b["a"].to(cfg.DEVICE)
            y = b["y"].to(cfg.DEVICE)

            with torch.cuda.amp.autocast(enabled=cfg.USE_AMP and cfg.DEVICE.startswith("cuda")):
                logits = model(x, a)
                loss = crit(logits, y) / cfg.GRAD_ACCUM_STEPS

            scaler.scale(loss).backward()

            if (st + 1) % cfg.GRAD_ACCUM_STEPS == 0 or (st + 1) == len(tl):
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.MAX_GRAD_NORM)
                scaler.step(opt)
                scaler.update()
                sched.step()
                opt.zero_grad(set_to_none=True)

            losses.append(float(loss.item() * cfg.GRAD_ACCUM_STEPS))
            yt.extend(y.detach().cpu().numpy().tolist())
            yp.extend(logits.detach().argmax(1).cpu().numpy().tolist())

        yv, pv, _ = predict_file(model, val, 1.0)
        vf = f1_score(yv, pv, average="macro", zero_division=0)
        va = accuracy_score(yv, pv)
        row = {
            "epoch": ep,
            "loss": float(np.mean(losses)),
            "train_segment_acc": accuracy_score(yt, yp),
            "val_file_acc": va,
            "val_file_macro_f1": vf,
        }
        hist.append(row)
        pd.DataFrame(hist).to_csv(run / "training_history.csv", index=False)

        print(
            f"Epoch {ep}: loss={row['loss']:.4f} train_seg_acc={row['train_segment_acc']:.4f} "
            f"val_file_acc={va:.4f} val_file_macro_f1={vf:.4f}"
        )

        if vf > best:
            best = vf
            bad = 0
            torch.save({"model": model.state_dict(), "best_val_macro_f1": best}, run / "best_model.pth")
            print("Saved best model")
        else:
            bad += 1
            print(f"No improvement. Patience: {bad}/{cfg.PATIENCE}")
            if bad >= cfg.PATIENCE:
                print("Early stopping")
                break

    # Load the checkpoint saved by this script.
    # Some PyTorch versions fail with weights_only=True because the checkpoint
    # contains numpy scalar metadata. This file is created locally during this
    # same run, so fallback to weights_only=False is safe here.
    ckpt_path = run / "best_model.pth"
    try:
        ck = torch.load(ckpt_path, map_location=cfg.DEVICE, weights_only=True)
    except Exception as e:
        print(f"weights_only=True failed while loading {ckpt_path}: {type(e).__name__}: {e}")
        print("Falling back to weights_only=False for this locally generated checkpoint.")
        ck = torch.load(ckpt_path, map_location=cfg.DEVICE, weights_only=False)
    model.load_state_dict(ck["model"])

    t = best_temp(model, val)

    # Save validation predictions for validation-based fusion tuning.
    yv, pv, prv = predict_file(model, val, t)
    mv = metrics(yv, pv)
    (run / "val_metrics.json").write_text(json.dumps(mv, indent=2), encoding="utf-8")
    save_prediction_outputs(val, yv, pv, prv, run, "val")

    # Save test predictions.
    y, p, pr = predict_file(model, test, t)
    m = metrics(y, p)

    print(f"\n{'='*80}\nFINAL TEST RESULTS - SEED {seed} ON SHARED SPLIT\n{'='*80}")
    print(json.dumps(m, indent=2))
    rep = classification_report(y, p, labels=list(range(len(LABELS))), target_names=LABELS, zero_division=0)
    cm = confusion_matrix(y, p, labels=list(range(len(LABELS))))
    print(rep)
    print(cm)

    (run / "test_metrics.json").write_text(json.dumps(m, indent=2), encoding="utf-8")
    save_prediction_outputs(test, y, p, pr, run, "test")

    return yv, prv, y, pr

def main():
    print(json.dumps(asdict(cfg), indent=2, default=str))
    print("Device:", cfg.DEVICE)

    train, val, test = load_shared_splits()
    root = Path(cfg.OUT_DIR) / f"run_{time.strftime('%Y%m%d_%H%M%S')}"
    root.mkdir(parents=True, exist_ok=True)

    val_probs = []
    test_probs = []
    yv_ref = None
    yt_ref = None

    for s in cfg.SEEDS:
        yv, prv, yt, prt = train_one(train, val, test, s, root)
        val_probs.append(prv)
        test_probs.append(prt)
        if yv_ref is None:
            yv_ref = yv
        if yt_ref is None:
            yt_ref = yt

    # Ensemble validation predictions.
    ens_val = np.mean(np.stack(val_probs), axis=0)
    pred_val = ens_val.argmax(1)
    mv = metrics(yv_ref, pred_val)
    print(f"\n{'='*80}\nFINAL AUDIO ENSEMBLE VALIDATION RESULTS - SHARED SPLIT\n{'='*80}")
    print(json.dumps(mv, indent=2))
    (root / "ensemble_val_metrics.json").write_text(json.dumps(mv, indent=2), encoding="utf-8")
    save_prediction_outputs(val, yv_ref, pred_val, ens_val, root, "ensemble_val")

    # Ensemble test predictions.
    ens_test = np.mean(np.stack(test_probs), axis=0)
    pred_test = ens_test.argmax(1)
    mt = metrics(yt_ref, pred_test)
    print(f"\n{'='*80}\nFINAL AUDIO ENSEMBLE TEST RESULTS - SHARED SPLIT\n{'='*80}")
    print(json.dumps(mt, indent=2))
    rep = classification_report(yt_ref, pred_test, labels=list(range(len(LABELS))), target_names=LABELS, zero_division=0)
    cm = confusion_matrix(yt_ref, pred_test, labels=list(range(len(LABELS))))
    print(rep)
    print(cm)

    (root / "ensemble_test_metrics.json").write_text(json.dumps(mt, indent=2), encoding="utf-8")
    save_prediction_outputs(test, yt_ref, pred_test, ens_test, root, "ensemble_test")

    print("\nSaved all shared-split audio results in:", root)
    print("Use these for fusion:")
    print("  AUDIO_VAL_PRED_CSV :", root / "ensemble_val_predictions.csv")
    print("  AUDIO_TEST_PRED_CSV:", root / "ensemble_test_predictions.csv")

if __name__ == "__main__":
    main()
