#!/usr/bin/env python3
# Infant LLM v6.3 — Growth Stages Edition
import os, re, ast, operator, pickle, json, hashlib, time
from typing import List, Tuple, Optional
from pathlib import Path

import numpy as np
from flask import Flask, request, jsonify, render_template, send_from_directory

import torch
import torch.nn as nn
import torch.optim as optim

# Optional FAISS
try:
    import faiss
    HAVE_FAISS = True
except Exception:
    HAVE_FAISS = False

APP_ROOT = Path(__file__).resolve().parent
SESSIONS = APP_ROOT / "sessions"
SESSIONS.mkdir(exist_ok=True)

def get_device():
    try:
        if torch.cuda.is_available():
            print("🟢 NVIDIA GPU detected")
            return torch.device("cuda")
    except Exception:
        pass
    try:
        if hasattr(torch, "xpu") and torch.xpu.is_available():
            print("🔵 Intel XPU detected")
            return torch.device("xpu")
    except Exception:
        pass
    try:
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            print("🟣 Apple MPS detected")
            return torch.device("mps")
    except Exception:
        pass
    print("⚪ Using CPU")
    return torch.device("cpu")

DEVICE = get_device()

# ---------------- Tokenizer ----------------
def simple_tokenize(text: str):
    text = text.lower()
    text = re.sub(r"([.,!?;:()\"'])", r" \1 ", text)
    return [t for t in text.split() if t]

# ---------------- Tiny Model ----------------
class TinyGenerator(nn.Module):
    def __init__(self, vocab_size: int, emb_dim: int = 64, hid_dim: int = 128, pad_idx: int = 0):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, emb_dim, padding_idx=pad_idx)
        self.rnn   = nn.GRU(emb_dim, hid_dim, batch_first=True)
        self.fc    = nn.Linear(hid_dim, vocab_size)
    def forward(self, x, h=None):
        x = self.embed(x)
        out, h = self.rnn(x, h)
        return self.fc(out), h

# ---------------- Memory Store ----------------
class MemoryStore:
    def __init__(self, dim: int, path_dir="infant_data_v6_3"):
        self.dim = dim
        self.path_dir = APP_ROOT / path_dir
        self.path_dir.mkdir(exist_ok=True, parents=True)
        self.texts: List[str] = []
        self.types: List[str] = []
        self.vecs: Optional[np.ndarray] = None
        self._index = None
        self._load()

    def _path(self): return self.path_dir / "memory.pkl"

    def _load(self):
        p = self._path()
        if p.exists():
            with open(p, "rb") as f:
                d = pickle.load(f)
            self.texts = d.get("texts", [])
            self.types = d.get("types", ["general"] * len(self.texts))
            v = d.get("vecs", None)
            self.vecs = np.asarray(v, dtype=np.float32) if v is not None else None
            self._build()
        else:
            self.texts, self.types, self.vecs, self._index = [], [], None, None

    def _save(self):
        with open(self._path(), "wb") as f:
            pickle.dump({"texts": self.texts, "types": self.types, "vecs": self.vecs}, f)

    def _build(self):
        if self.vecs is None or len(self.vecs) == 0:
            self._index = None
            return
        if HAVE_FAISS:
            v = self.vecs.copy()
            n = np.linalg.norm(v, axis=1, keepdims=True); n[n==0]=1.0
            v = v / n
            idx = faiss.IndexFlatIP(self.dim)
            idx.add(v)
            self._index = idx
        else:
            self._index = None

    def add(self, text: str, vec: np.ndarray, mtype: str = "general"):
        vec = np.asarray(vec, dtype=np.float32).reshape(1, -1)
        self.vecs = vec if self.vecs is None else np.vstack([self.vecs, vec])
        self.texts.append(text)
        self.types.append(mtype)
        self._build()
        self._save()

    def clear(self):
        self.texts = []
        self.types = []
        self.vecs = None
        self._build()
        self._save()

    def query(self, qvec: np.ndarray, k:int=6, min_score:float=0.2, type_pref:Optional[str]=None):
        if self.vecs is None or len(self.texts) == 0:
            return []
        q = np.asarray(qvec, dtype=np.float32).reshape(1, -1)
        q = q / (np.linalg.norm(q) + 1e-12)

        def filter_type(cands):
            if type_pref in (None, "any"):
                return cands
            f = [(i, s) for i, s in cands if self.types[i] == type_pref]
            return f if f else cands

        if HAVE_FAISS and self._index is not None:
            D, I = self._index.search(q, min(k*3, len(self.texts)))
            raw = [(int(i), float(s)) for i, s in zip(I[0], D[0]) if i >= 0 and s >= min_score]
            raw.sort(key=lambda x: -x[1])
            raw = filter_type(raw)[:k]
            return [(self.texts[i], self.types[i], s) for i, s in raw]

        sims = (self.vecs @ q.T).reshape(-1)
        norms = np.linalg.norm(self.vecs, axis=1)
        sims = sims / (norms*(np.linalg.norm(q) + 1e-12))
        idxs = np.argsort(-sims)[:k*3]
        raw = [(int(i), float(sims[i])) for i in idxs if sims[i] >= min_score]
        raw = filter_type(raw)[:k]
        return [(self.texts[i], self.types[i], s) for i, s in raw]

# ---------------- Infant Core ----------------
STEM_TYPES = {"math","science","coding","engineering","technology","general"}

class Infant:
    def __init__(self):
        self.vocab = {"<pad>":0,"<unk>":1,"<eos>":2}
        self.idx   = {0:"<pad>",1:"<unk>",2:"<eos>"}
        self.next  = 3
        self.model = None
        self.opt   = None
        self.loss  = nn.CrossEntropyLoss(ignore_index=0)
        self.mem   = MemoryStore(dim=64, path_dir="infant_data_v6_3")
        self.device= DEVICE

    def _tok_and_grow(self, text: str):
        ids = []
        for w in simple_tokenize(text):
            if w not in self.vocab:
                self.vocab[w] = self.next
                self.idx[self.next] = w
                self.next += 1
                self._expand()
            ids.append(self.vocab[w])
        return ids

    def _init(self):
        if self.model is None:
            V = len(self.vocab)
            self.model = TinyGenerator(V, 64, 128).to(self.device)
            self.opt   = optim.Adam(self.model.parameters(), lr=1e-3)

    def _expand(self):
        if self.model is None:
            self._init(); return
        V_old = self.model.embed.num_embeddings
        V_new = len(self.vocab)
        if V_new <= V_old: return
        m2 = TinyGenerator(V_new, 64, 128).to(self.device)
        sd = m2.state_dict()
        old = self.model.state_dict()
        for k,v in old.items():
            if k in sd:
                try: sd[k][:v.shape[0]] = v
                except Exception: pass
        m2.load_state_dict(sd)
        self.model = m2
        self.opt   = optim.Adam(self.model.parameters(), lr=1e-3)

    def emb(self, text: str) -> np.ndarray:
        self._init()
        ids = [self.vocab.get(t,1) for t in simple_tokenize(text)]
        if not ids: return np.zeros(64, np.float32)
        with torch.no_grad():
            E = self.model.embed.weight.detach().cpu().numpy()
            return np.mean(E[np.array(ids)], axis=0).astype(np.float32)

    MATH_RE = re.compile(r"^[\s0-9+\-*/().=^%]+$")
    def classify(self, text: str) -> str:
        t = text.lower().strip()
        m = re.match(r"^learn\s+([a-z]+)\s*:\s*(.+)$", t)
        if m:
            ty = m.group(1).lower()
            return ty if ty in STEM_TYPES else "general"
        if self.MATH_RE.match(t) or t.endswith("=") or (any(ch in t for ch in "+-*/^=") and any(c.isdigit() for c in t)):
            return "math"
        code_h = ("def ","class ","print(","function ","{","};","console.log","for(","while(")
        sci_h  = ("atom","cell","molecule","h2o","gravity","boil","°c","celsius","kelvin","force","mass","energy","dna","photosynthesis")
        tech_h = ("server","http","api","database","linux","kernel","python","java","c++","network","router","switch","gpu","cpu")
        eng_h  = ("torque","beam","load","stress","strain","voltage","current","ohm","watt","newton","pascal","bridge","circuit","gear")
        if any(h in t for h in code_h): return "coding"
        if any(h in t for h in eng_h):  return "engineering"
        if any(h in t for h in tech_h): return "technology"
        if any(h in t for h in sci_h):  return "science"
        return "general"

    # Math evaluator (enabled in Teenager+ stages)
    _ops = {
        ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,
        ast.Div: operator.truediv, ast.Pow: operator.pow, ast.Mod: operator.mod,
        ast.USub: operator.neg, ast.UAdd: operator.pos
    }
    def _eval_ast(self, node):
        if isinstance(node, ast.Num): return node.n
        if isinstance(node, ast.BinOp): return self._ops[type(node.op)](self._eval_ast(node.left), self._eval_ast(node.right))
        if isinstance(node, ast.UnaryOp): return self._ops[type(node.op)](self._eval_ast(node.operand))
        raise ValueError("Unsupported expression")
    def try_eval_math(self, text: str):
        cands = re.findall(r"[0-9\.\s+\-*/()^%=]+", text)
        if not cands: return None
        expr = max((c.strip() for c in cands), key=len).rstrip("=")
        if not expr: return None
        try:
            node = ast.parse(expr, mode="eval").body
            val = self._eval_ast(node)
            return expr, val
        except Exception:
            return None

    def learn(self, line: str) -> Tuple[float,str]:
        s = line.strip()
        m = re.match(r"^learn\s+([a-z]+)\s*:\s*(.+)$", s, flags=re.I)
        if m:
            ty, text = m.group(1).lower(), m.group(2).strip()
            if ty not in STEM_TYPES: ty="general"
        elif s.lower().startswith("learn "):
            text = s[6:].strip()
            ty = self.classify(text)
        else:
            text = s; ty = self.classify(text)

        ids = self._tok_and_grow(text)
        if not ids: return 0.0, ty
        self._init()
        x = torch.tensor([ids], dtype=torch.long).to(self.device)
        y = x.clone()
        self.model.train()
        o,_ = self.model(x)
        B,L,V = o.shape
        loss = self.loss(o.view(B*L,V), y.view(B*L))
        self.opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(),1.0)
        self.opt.step()

        self.mem.add(text, self.emb(text), ty)
        return float(loss.item()), ty

    def recall(self, text: str, k:int=6, typ:Optional[str]=None):
        return self.mem.query(self.emb(text), k=k, type_pref=typ)

    def generate(self, prompt: str, temperature: float=0.9, max_len:int=40):
        self._init()
        ids = [self.vocab.get(t,1) for t in simple_tokenize(prompt)]
        if not ids: return ""
        x = torch.tensor([ids], dtype=torch.long).to(self.device)
        with torch.no_grad():
            _, h = self.model(x)
        cur = x[0,-1].view(1,1)
        out=[]
        with torch.no_grad():
            for _ in range(max_len):
                o,h = self.model(cur,h)
                logits = o[0,0].cpu().numpy() / max(1e-6, temperature)
                p = np.exp(logits - logits.max()); p /= p.sum()
                n = int(np.random.choice(len(p), p=p))
                if n == 2: break
                out.append(self.idx.get(n,"<unk>"))
                cur = torch.tensor([[n]], dtype=torch.long).to(self.device)
        return " ".join(out)

infant = Infant()

# ---------------- Flask app ----------------
app = Flask(__name__, template_folder=str(APP_ROOT / "templates"), static_folder=str(APP_ROOT / "static"))

# Load metadata for footer
INFO = json.load(open(str((APP_ROOT.parent / "docs" / "project_info.json")), "r")) if (APP_ROOT.parent / "docs" / "project_info.json").exists() else {"version":"6.3","edition":"Growth Stages Edition"}

def ip_key():
    ip = request.headers.get("X-Forwarded-For", request.remote_addr or "local")
    # simplify: take first IP if forwarded
    if ip and "," in ip: ip = ip.split(",")[0].strip()
    return hashlib.sha1(ip.encode("utf-8")).hexdigest()

def session_path():
    return SESSIONS / f"{ip_key()}.json"

def load_session():
    sp = session_path()
    if sp.exists():
        try:
            return json.load(open(sp,"r"))
        except Exception:
            return {"stage":"infant","learned":0,"maturity":0,"override":False}
    return {"stage":"infant","learned":0,"maturity":0,"override":False}

def save_session(d):
    json.dump(d, open(session_path(),"w"))

STAGE_ORDER = ["infant","child","teenager","adult"]
NEXT_STAGE = {"infant":"child","child":"teenager","teenager":"adult","adult":None}
STAGE_TARGET = {"infant":20,"child":60,"teenager":120,"adult":250}

@app.route("/docs/<path:fname>")
def serve_docs(fname):
    return send_from_directory(str(APP_ROOT.parent / "docs"), fname)

@app.route("/")
def home():
    ses = load_session()
    return render_template("infant_ui_v6_3.html", version=INFO.get("version","6.3"), edition=INFO.get("edition","Growth Stages Edition"))

@app.route("/memlist")
def memlist():
    ses = load_session()
    items = [{"text": t, "type": ty} for t, ty in list(zip(infant.mem.texts, infant.mem.types))[-30:]]
    learned = len(infant.mem.texts)
    diversity = len(set(infant.mem.types)) or 1
    target = max(STAGE_TARGET.get(ses.get("stage","infant"),20), 1)
    if learned > 0:
        maturity = int(min(100, round((learned + diversity*5) / target * 100)))
    else:
        maturity = 0
    ses.update({"learned":learned,"maturity":maturity})
    save_session(ses)
    return jsonify({"items": items, "stats": {
        "learned": learned,
        "maturity": maturity,
        "stage": ses.get("stage","infant"),
        "override": ses.get("override", False)
    }})

STAGE_BASE_MATURITY = {
    "infant": 0,
    "child": 25,
    "teenager": 50,
    "adult": 75
}

@app.route("/set_stage", methods=["POST"])
def set_stage():
    ses = load_session()
    data = request.get_json() or {}
    mode = (data.get("mode") or "auto").lower()

    if mode == "manual":
        st = (data.get("stage") or "infant").lower()
        if st not in STAGE_ORDER:
            st = "infant"
        ses["stage"] = st
        ses["override"] = True

        # Always set maturity to the baseline of the selected stage
        base = STAGE_BASE_MATURITY.get(st, 0)
        ses["maturity"] = base
    else:
        # Auto growth mode re-enables dynamic stage progression
        ses["override"] = False

        # Recalculate stage based on maturity automatically
        maturity = ses.get("maturity", 0)
        if maturity < 25:
            ses["stage"] = "infant"
        elif maturity < 50:
            ses["stage"] = "child"
        elif maturity < 75:
            ses["stage"] = "teenager"
        else:
            ses["stage"] = "adult"

    save_session(ses)
    return jsonify({
        "ok": True,
        "stage": ses["stage"],
        "maturity": ses["maturity"],
        "override": ses["override"]
    })

@app.route("/clear_memory", methods=["POST"])
def clear_memory():
    infant.mem.clear()
    ses = {"stage":"infant","learned":0,"maturity":0,"override":False}
    save_session(ses)
    return jsonify({"reply": "Memory cleared and stage reset to Infant."})

@app.route("/chat", methods=["POST"])
def chat():
    d = request.get_json() or {}
    msg  = (d.get("msg") or "").strip()
    pref = d.get("prefer","any").lower()
    mode = (d.get("mode") or "auto").lower()

    ses = load_session()
    stage = ses.get("stage","infant")
    # Learning
    if msg.lower().startswith("learn "):
        loss, ty = infant.learn(msg)
        # update maturity
        learned = len(infant.mem.texts)
        diversity = len(set(infant.mem.types)) or 1
        target = STAGE_TARGET.get(stage,20)
        maturity = int(min(100, round((learned + diversity*5) / target * 100)))
        ses.update({"learned":learned,"maturity":maturity})
        # auto-suggest growth if threshold exceeded
        suggest = None
        nxt = NEXT_STAGE.get(stage)
        if nxt and maturity >= (80 if stage=="infant" else 85 if stage=="child" else 90):
            suggest = nxt
        save_session(ses)
        return jsonify({"reply": f"Learned ({ty}) loss≈{loss:.3f}", "updated_memory": True, "stage": stage, "suggest_growth": suggest})

    # Memory command
    if msg.lower().startswith("memory"):
        q = msg[6:].strip().lower()
        cat_set = {"math","science","coding","engineering","technology","general"}
        if q == "" or q is None:
            mem = [{"text": t, "type": ty, "score": 1.0} for t, ty in list(zip(infant.mem.texts, infant.mem.types))[-30:]]
            return jsonify({"reply": "Recent memory (top 30).", "memory": mem, "stage": stage})
        if q in cat_set:
            mem = [{"text": t, "type": ty, "score": 1.0} for t, ty in zip(infant.mem.texts, infant.mem.types) if ty == q]
            return jsonify({"reply": f"Listed all {q} memories.", "memory": mem, "stage": stage})
        hits = infant.recall(q, typ=pref)
        mem = [{"text": t, "type": ty, "score": s} for t, ty, s in hits]
        return jsonify({"reply": "Memory query done.", "memory": mem, "stage": stage})

    # Stage capabilities
    allow_math = stage in ("teenager","adult")
    smart_recall = stage in ("child","teenager","adult")

    # Math eval if enabled
    if allow_math:
        m_eval = infant.try_eval_math(msg)
        if m_eval is not None:
            expr, val = m_eval
            return jsonify({"reply": f"{expr} = {val}", "memory": [], "stage": stage})

    # Retrieval
    hits = infant.recall(msg, typ=pref)
    mem  = [{"text": t, "type": ty, "score": s} for t, ty, s in hits]

    # Modes
    if mode == "recall":
        if hits: return jsonify({"reply": hits[0][0], "memory": mem, "stage": stage})
        return jsonify({"reply": "I don't have that yet — try teaching me with 'learn ...'.", "memory": mem, "stage": stage})

    if mode == "creative":
        ctx = " ".join([t for t,_,_ in hits[:3]])
        gen = infant.generate(f"Context: {ctx}\nUser: {msg}\nReply:", temperature=0.8)
        if hits and len(gen.strip()) < 3: gen = hits[0][0]
        return jsonify({"reply": gen or "…thinking…", "memory": mem, "stage": stage})

    # Auto (Smart Recall if enabled)
    if smart_recall and hits:
        top_text, top_type, top_score = hits[0]
        if top_score > 0.5 and msg.lower().startswith(("what","who","where","why","how","when")):
            return jsonify({"reply": top_text, "memory": mem, "stage": stage})

    ctx = " ".join([t for t,_,_ in hits[:3]])
    gen = infant.generate(f"Context: {ctx}\nUser: {msg}\nReply:", temperature=0.9)
    if hits and len(gen.strip()) < 3: gen = hits[0][0]
    return jsonify({"reply": gen or "…thinking…", "memory": mem, "stage": stage})

if __name__ == "__main__":
    # Let it bind on all interfaces for LAN access
    app.run(host="0.0.0.0", port=5000)
