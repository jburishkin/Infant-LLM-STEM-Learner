#!/usr/bin/env python3
"""
Infant LLM v6 — STEM Learner (local mode, no SSO)
- No login required
- Same UI, header/footer, dynamic favicon, memory, temperature, personality, math eval
"""
import os, re, ast, operator, pickle, numpy as np
from flask import Flask, request, jsonify, render_template_string
import torch, torch.nn as nn, torch.optim as optim

try:
    import faiss
    HAVE_FAISS = True
except Exception:
    HAVE_FAISS = False

# ---------------- Hardware ----------------
def get_device():
    try:
        if torch.cuda.is_available():
            print("🟢 NVIDIA GPU detected")
            return torch.device("cuda")
    except Exception: pass
    try:
        if hasattr(torch, "xpu") and torch.xpu.is_available():
            print("🔵 Intel XPU detected")
            return torch.device("xpu")
    except Exception: pass
    try:
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            print("🟣 Apple MPS detected")
            return torch.device("mps")
    except Exception: pass
    print("⚪ Using CPU")
    return torch.device("cpu")
DEVICE = get_device()

# ---------------- Tiny LLM ----------------
def simple_tokenize(text):
    text = text.lower()
    text = re.sub(r"([.,!?;:()\"'])", r" \1 ", text)
    return [t for t in text.split() if t]

class TinyGenerator(nn.Module):
    def __init__(self, vocab, emb=64, hid=128):
        super().__init__()
        self.embed = nn.Embedding(vocab, emb, padding_idx=0)
        self.rnn = nn.GRU(emb, hid, batch_first=True)
        self.fc = nn.Linear(hid, vocab)
    def forward(self, x, h=None):
        x = self.embed(x)
        o, h = self.rnn(x, h)
        return self.fc(o), h

# ---------------- Memory ----------------
class MemoryStore:
    def __init__(self, dim, path="infant_data_v6_local"):
        self.dim = dim; self.path = path
        os.makedirs(path, exist_ok=True)
        self.texts=[]; self.types=[]; self.vecs=None; self._index=None
        self._load()
    def _p(self): return os.path.join(self.path,"memory.pkl")
    def _load(self):
        if os.path.exists(self._p()):
            with open(self._p(),"rb") as f:d=pickle.load(f)
            self.texts=d.get("texts",[]); self.types=d.get("types",["general"]*len(self.texts))
            v=d.get("vecs"); self.vecs=np.asarray(v,dtype=np.float32) if v is not None else None
            self._build()
    def _build(self):
        if self.vecs is None or not len(self.vecs): self._index=None; return
        if HAVE_FAISS:
            v=self.vecs.copy()
            n=np.linalg.norm(v,axis=1,keepdims=True); n[n==0]=1; v=v/n
            idx=faiss.IndexFlatIP(self.dim); idx.add(v); self._index=idx
        else: self._index=None
    def _save(self):
        with open(self._p(),"wb") as f:
            pickle.dump({"texts":self.texts,"types":self.types,"vecs":self.vecs},f)
    def add(self,t,v,ty="general"):
        v=np.asarray(v,dtype=np.float32).reshape(1,-1)
        self.vecs=v if self.vecs is None else np.vstack([self.vecs,v])
        self.texts.append(t); self.types.append(ty); self._build(); self._save()
    def query(self,q,k=6,min_s=0.2,typ=None):
        if self.vecs is None or not len(self.texts):return[]
        q=np.asarray(q,dtype=np.float32).reshape(1,-1)
        q=q/(np.linalg.norm(q)+1e-12)
        if HAVE_FAISS and self._index is not None:
            D,I=self._index.search(q,min(k*3,len(self.texts)))
            res=[(self.texts[i],self.types[i],float(s)) for i,s in zip(I[0],D[0]) if i>=0 and s>=min_s]
        else:
            s=(self.vecs@q.T).reshape(-1); n=np.linalg.norm(self.vecs,axis=1)
            s=s/(n*(np.linalg.norm(q)+1e-12)); idx=np.argsort(-s)[:k*3]
            res=[(self.texts[i],self.types[i],float(s[i])) for i in idx if s[i]>=min_s]
        if typ not in (None,"any"):
            res=[r for r in res if r[1]==typ] or res
        return res[:k]

# ---------------- Infant Core ----------------
STEM_TYPES={"math","science","coding","engineering","technology","general"}
class Infant:
    def __init__(self):
        self.vocab={"<pad>":0,"<unk>":1,"<eos>":2}
        self.idx={0:"<pad>",1:"<unk>",2:"<eos>"}
        self.next=3
        self.model=None; self.opt=None
        self.loss=nn.CrossEntropyLoss(ignore_index=0)
        self.mem=MemoryStore(64)
        self.device=DEVICE
    def _tok(self,t): 
        t=[x.lower() for x in simple_tokenize(t)]
        ids=[]
        for w in t:
            if w not in self.vocab:
                self.vocab[w]=self.next; self.idx[self.next]=w; self.next+=1; self._expand()
            ids.append(self.vocab[w])
        return ids
    def _init(self):
        if self.model is None:
            v=len(self.vocab)
            self.model=TinyGenerator(v,64,128).to(self.device)
            self.opt=optim.Adam(self.model.parameters(),lr=1e-3)
    def _expand(self):
        if self.model is None:self._init();return
        v_new=len(self.vocab);v_old=self.model.embed.num_embeddings
        if v_new<=v_old:return
        m2=TinyGenerator(v_new,64,128).to(self.device)
        sd=m2.state_dict();old=self.model.state_dict()
        for k,v in old.items():
            if k in sd: sd[k][:v.shape[0]]=v
        m2.load_state_dict(sd)
        self.model=m2;self.opt=optim.Adam(self.model.parameters(),lr=1e-3)
    def emb(self,t):
        if self.model is None:self._init()
        ids=[self.vocab.get(x,1) for x in simple_tokenize(t)]
        if not ids:return np.zeros(64,np.float32)
        with torch.no_grad():
            e=self.model.embed.weight.detach().cpu().numpy()
            return np.mean(e[np.array(ids)],0).astype(np.float32)
    def learn(self,t):
        m=re.match(r"^learn\s+([a-z]+)\s*:\s*(.+)$",t,re.I)
        if m:ty=m.group(1).lower();t2=m.group(2)
        else:ty="general";t2=t.replace("learn ","",1)
        self._tok(t2);self._init()
        x=torch.tensor([self._tok(t2)],dtype=torch.long).to(self.device)
        y=x.clone()
        self.model.train()
        o,_=self.model(x);B,L,V=o.shape
        l=self.loss(o.view(B*L,V),y.view(B*L))
        self.opt.zero_grad();l.backward();self.opt.step()
        self.mem.add(t2,self.emb(t2),ty)
        return float(l.item()),ty
    def query(self,t,typ=None):return self.mem.query(self.emb(t),typ=typ)
    def gen(self,t,temperature=0.9,max_len=40):
        self._init();ids=[self.vocab.get(x,1) for x in simple_tokenize(t)]
        if not ids:return""
        x=torch.tensor([ids],dtype=torch.long).to(self.device)
        with torch.no_grad():_,h=self.model(x)
        cur=x[0,-1].view(1,1);out=[]
        for _ in range(max_len):
            o,h=self.model(cur,h)
            logits=o[0,0].cpu().numpy()/max(1e-6,temperature)
            p=np.exp(logits-logits.max());p/=p.sum()
            n=int(np.random.choice(len(p),p=p))
            if n==2:break
            out.append(self.idx.get(n,"<unk>"))
            cur=torch.tensor([[n]],dtype=torch.long).to(self.device)
        return" ".join(out)

infant=Infant()

# ---------------- Flask UI ----------------
app=Flask(__name__)
HTML=open("infant_ui_v6.html").read() if os.path.exists("infant_ui_v6.html") else "UI missing"

@app.route("/")
def home():return render_template_string(HTML)

@app.route("/memlist")
def memlist():
    it=[{"text":t,"type":ty}for t,ty in zip(infant.mem.texts,infant.mem.types)][-30:]
    return jsonify({"items":it})

@app.route("/chat",methods=["POST"])
def chat():
    d=request.get_json()or{}
    msg=d.get("msg","").strip()
    T=float(d.get("temperature",0.9))
    personality=d.get("personality","playful")
    pref=d.get("prefer","any")
    if msg.lower().startswith("learn "):
        loss,ty=infant.learn(msg)
        return jsonify({"reply":f"Learned ({ty}) loss≈{loss:.3f}","memory":[],"updated_memory":True})
    if msg.lower().startswith("memory "):
        q=msg[7:];hits=infant.query(q,typ=pref)
        mem=[{"text":t,"type":ty,"score":s}for t,ty,s in hits]
        return jsonify({"reply":"Memory query done.","memory":mem})
    hits=infant.query(msg,typ=pref)
    mem=[{"text":t,"type":ty,"score":s}for t,ty,s in hits]
    ctx=" ".join([t for t,_,_ in hits[:3]])
    gen=infant.gen(f"{ctx} {msg}",temperature=T)
    return jsonify({"reply":gen or '...thinking...',"memory":mem})

if __name__=="__main__":
    app.run(host="0.0.0.0",port=5000)
