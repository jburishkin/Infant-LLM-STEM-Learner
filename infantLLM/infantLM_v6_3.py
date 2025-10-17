# infantLM_v6_3.py
# Infant LLM v6.3.1 — Flask WebUI + slow, realistic growth + auto/manual stages
# Drop-in server compatible with infant_ui_v6_3.html / static/style.css

import os, re, json, math, time, random
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict

import numpy as np
from flask import Flask, request, jsonify, render_template, send_from_directory

# ---------------------------
# App & Paths
# ---------------------------
APP_DIR = os.path.dirname(os.path.abspath(__file__))
TEMPLATES_DIR = os.path.join(APP_DIR, "templates")
STATIC_DIR = os.path.join(APP_DIR, "static")
SESSION_FILE = os.path.join(APP_DIR, "session.json")
MEMORY_FILE = os.path.join(APP_DIR, "memory.json")

app = Flask(__name__, template_folder=TEMPLATES_DIR, static_folder=STATIC_DIR)

VERSION = "v6.3.1"
EDITION = "WebUI"

# ---------------------------
# Growth Constants
# ---------------------------
STAGE_ORDER = ["infant", "child", "teenager", "adult"]
NEXT_STAGE = {"infant": "child", "child": "teenager", "teenager": "adult"}
STAGE_BASE_MATURITY = {"infant": 0, "child": 25, "teenager": 50, "adult": 75}

# Slow, realistic cognitive development scaling
# (Approx. learned items needed to reach 100% per stage)
DIFFICULTY_SCALE = {
    "infant": 200,
    "child": 600,
    "teenager": 1500,
    "adult": 4000,
}

# ---------------------------
# Utility: Session
# ---------------------------
def session_path() -> str:
    return SESSION_FILE

def load_session() -> dict:
    sp = session_path()
    if os.path.exists(sp):
        try:
            ses = json.load(open(sp, "r"))
            # Hard guard for first-run behavior
            if ses.get("learned", 0) == 0:
                ses["maturity"] = 0
            # Fill missing keys if needed
            ses.setdefault("stage", "infant")
            ses.setdefault("override", False)
            ses.setdefault("learned", 0)
            ses.setdefault("maturity", 0)
            return ses
        except Exception:
            return {"stage": "infant", "learned": 0, "maturity": 0, "override": False}
    return {"stage": "infant", "learned": 0, "maturity": 0, "override": False}

def save_session(ses: dict):
    with open(session_path(), "w") as f:
        json.dump(ses, f)

# ---------------------------
# Utility: Tokenizer
# ---------------------------
_word_re = re.compile(r"\w+|\S", re.UNICODE)

def simple_tokenize(text: str) -> List[str]:
    text = text.lower()
    # Safe punctuation spacing (escape quotes correctly!)
    text = re.sub(r"([.,!?;:()\"'])", r" \1 ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return _word_re.findall(text)

# ---------------------------
# Memory Store
# ---------------------------
@dataclass
class MemoryStore:
    texts: List[str] = field(default_factory=list)
    types: List[str] = field(default_factory=list)
    vecs: Optional[np.ndarray] = None  # (N, D)

    def save(self):
        data = {"texts": self.texts, "types": self.types}
        with open(MEMORY_FILE, "w") as f:
            json.dump(data, f)

    def load(self):
        if os.path.exists(MEMORY_FILE):
            try:
                data = json.load(open(MEMORY_FILE, "r"))
                self.texts = data.get("texts", [])
                self.types = data.get("types", [])
                self._rebuild_vecs()
            except Exception:
                self.texts, self.types, self.vecs = [], [], None

    def _hash_embed(self, tokens: List[str], dim: int = 256) -> np.ndarray:
        # Simple hashing bag-of-words embedding
        v = np.zeros(dim, dtype=np.float32)
        for t in tokens:
            idx = (hash(t) % dim + dim) % dim
            v[idx] += 1.0
        norm = np.linalg.norm(v) + 1e-6
        return v / norm

    def _rebuild_vecs(self):
        if not self.texts:
            self.vecs = None
            return
        embs = []
        for txt in self.texts:
            embs.append(self._hash_embed(simple_tokenize(txt)))
        self.vecs = np.stack(embs, axis=0)  # (N, D)

    def add(self, text: str, mtype: str) -> float:
        # Simulated training loss: shrinks slightly with memory size
        base = 3.0
        shrink = min(2.0, len(self.texts) * 0.02)
        loss = base + random.uniform(-0.2, 0.2) - shrink
        self.texts.append(text)
        self.types.append(mtype)
        v = self._hash_embed(simple_tokenize(text))
        if self.vecs is None:
            self.vecs = v.reshape(1, -1)
        else:
            self.vecs = np.vstack([self.vecs, v.reshape(1, -1)])
        self.save()
        return max(0.15, float(loss))

    def search(self, query: str, topk: int = 3) -> List[Tuple[float, str, str]]:
        if not self.texts or self.vecs is None:
            return []
        qv = self._hash_embed(simple_tokenize(query))
        # Cosine similarity since vecs are normalized
        sims = self.vecs @ qv
        idxs = np.argsort(-sims)[:topk]
        out = []
        for i in idxs:
            out.append((float(sims[i]), self.texts[i], self.types[i]))
        return out

    def clear(self):
        self.texts, self.types, self.vecs = [], [], None
        if os.path.exists(MEMORY_FILE):
            try:
                os.remove(MEMORY_FILE)
            except Exception:
                pass

# ---------------------------
# Infant Model (very simple)
# ---------------------------
class InfantModel:
    def __init__(self):
        self.mem = MemoryStore()
        self.mem.load()

    def detect_type(self, text: str) -> str:
        t = text.lower()
        if t.startswith("learn math:"):
            return "math"
        if t.startswith("learn science:"):
            return "science"
        if t.startswith("learn engineering:"):
            return "engineering"
        if t.startswith("learn technology:"):
            return "technology"
        if t.startswith("learn coding:"):
            return "coding"
        return "general"

    def normalize_learn_text(self, raw: str) -> Tuple[str, str]:
        t = raw.strip()
        low = t.lower()
        # strip "learn ..." prefix
        if low.startswith("learn "):
            t = t[6:].strip()
            low = t.lower()
        # typed prefixes
        mtype = "general"
        typed_prefixes = [
            ("math:", "math"),
            ("science:", "science"),
            ("engineering:", "engineering"),
            ("technology:", "technology"),
            ("coding:", "coding"),
        ]
        for pref, ty in typed_prefixes:
            if low.startswith(pref):
                mtype = ty
                t = t[len(pref):].strip()
                break
        if not t:
            t = raw.strip()
        return t, mtype

    def learn(self, raw: str) -> Tuple[float, str]:
        text, ty = self.normalize_learn_text(raw)
        loss = self.mem.add(text, ty)
        return loss, ty

    def reply(self, msg: str, stage: str, prefer: str) -> Tuple[str, List[Dict[str, str]]]:
        # Retrieve memory hints
        hints_raw = self.mem.search(msg, topk=3)
        hints = [{"score": f"{s:.3f}", "text": t, "type": ty} for s, t, ty in hints_raw]

        # Math capability ramps at teenager+
        # Handle simple arithmetic if input resembles a query
        if stage in ("teenager", "adult"):
            ans = self._try_eval_math(msg)
            if ans is not None:
                return f"{ans}", hints

        # Otherwise stitch a naive answer based on memory
        if not hints_raw:
            return "I don't know yet. Please teach me.", hints

        # Create a tiny extractive-style response from top hint
        top_text = hints_raw[0][1]
        # if user asks "what is/are X", attempt to echo relevant snippet
        m = re.search(r"what (?:is|are)\s+(.*)\??", msg.lower())
        if m:
            subj = m.group(1).strip()
            # try to find a clause mentioning subj in top_text
            if subj in top_text.lower():
                return top_text, hints

        # fallback: simple synthesis of hints texts
        joined = "; ".join([t for _, t, _ in hints_raw])
        return joined, hints

    def _try_eval_math(self, msg: str) -> Optional[str]:
        # Accepts inputs like "what is 2+3" or "2+3=" or "5*(4+1)="
        q = msg.strip().lower()
        # Strip leading prompt words:
        q = re.sub(r"^(what\s+is|calc|calculate|compute|evaluate)\s+", "", q)
        q = q.replace("=", "").strip()
        # Allow digits, operators, parentheses, spaces, decimal points
        if not re.fullmatch(r"[0-9+\-*/().\s]+", q):
            return None
        try:
            # Safe evaluation using AST-like restriction via Python eval with no builtins
            # Here we trust the regexp filter to keep it arithmetic-only
            val = eval(q, {"__builtins__": {}}, {})
            if isinstance(val, (int, float)):
                # trim trailing .0
                if isinstance(val, float) and abs(val - int(val)) < 1e-9:
                    val = int(val)
                return str(val)
        except Exception:
            return None
        return None

infant = InfantModel()

# ---------------------------
# Flask Views
# ---------------------------
@app.route("/")
def home():
    return render_template("infant_ui_v6_3.html", version=VERSION, edition=EDITION)

@app.route("/health")
def health():
    return jsonify({"ok": True, "version": VERSION})

# ---------------------------
# Growth & Memory APIs
# ---------------------------
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
        # Set maturity baseline to the stage’s base
        ses["maturity"] = STAGE_BASE_MATURITY.get(st, 0)
    else:
        # Auto mode reverts stage to match current maturity
        ses["override"] = False
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
    return jsonify({"ok": True, "stage": ses["stage"], "maturity": ses["maturity"], "override": ses["override"]})

@app.route("/memlist")
def memlist():
    ses = load_session()
    items = [{"text": t, "type": ty} for t, ty in list(zip(infant.mem.texts, infant.mem.types))[-30:]]
    learned = len(infant.mem.texts)
    diversity = len(set(infant.mem.types)) or 1
    st = ses.get("stage", "infant")
    override = ses.get("override", False)

    difficulty = DIFFICULTY_SCALE.get(st, 400)
    progress = (learned + diversity * 2) / difficulty

    if not override:
        # Auto Mode — slow exponential growth
        maturity = int(min(100, round(100 * (1 - math.exp(-progress * 0.6)))))
    else:
        # Manual Mode — baseline + very slow crawl (optional)
        base = STAGE_BASE_MATURITY.get(st, 0)
        maturity = int(min(100, base + (progress * 10)))  # crawl up to +10% max from activity

    ses.update({"learned": learned, "maturity": maturity})

    # Auto stage progression (only when override=False)
    if not override:
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
        "items": items,
        "stats": {
            "learned": learned,
            "maturity": maturity,
            "stage": ses["stage"],
            "override": ses["override"],
        }
    })

@app.route("/clear_memory", methods=["POST"])
def clear_memory():
    infant.mem.clear()
    ses = {"stage": "infant", "learned": 0, "maturity": 0, "override": False}
    save_session(ses)
    return jsonify({"ok": True})

# ---------------------------
# Chat / Learn API
# ---------------------------
@app.route("/chat", methods=["POST"])
def chat():
    data = request.get_json() or {}
    msg = (data.get("msg") or "").strip()
    mode = (data.get("mode") or "auto").lower()
    prefer = (data.get("prefer") or "any").lower()

    ses = load_session()
    stage = ses.get("stage", "infant")
    override = ses.get("override", False)

    # Learning path: "learn ..."
    if msg.lower().startswith("learn "):
        loss, ty = infant.learn(msg)

        # Recompute progress/maturity
        learned = len(infant.mem.texts)
        diversity = len(set(infant.mem.types)) or 1

        # Use stage-based slow growth difficulty
        difficulty = DIFFICULTY_SCALE.get(stage, 400)
        progress = (learned + diversity * 2) / difficulty

        if not override:
            maturity = int(min(100, round(100 * (1 - math.exp(-progress * 0.6)))))
        else:
            # Manual mode baseline + tiny crawl
            base = STAGE_BASE_MATURITY.get(stage, 0)
            maturity = int(min(100, base + (progress * 10)))

        ses.update({"learned": learned, "maturity": maturity})

        message = f"Learned ({ty}) loss≈{loss:.3f}"

        # Auto-promotion only in Auto Mode
        if not override:
            prev_stage = stage
            if stage == "infant" and maturity >= 25:
                stage = "child"
                ses["stage"] = "child"
                message += "\n🌱 Your LLM has grown to CHILD!"
            elif stage == "child" and maturity >= 50:
                stage = "teenager"
                ses["stage"] = "teenager"
                message += "\n🌿 Your LLM has grown to TEENAGER!"
            elif stage == "teenager" and maturity >= 75:
                stage = "adult"
                ses["stage"] = "adult"
                message += "\n🌺 Your LLM has matured into ADULT!"
            # If stage changed, UI will auto-refresh via /memlist
        else:
            # Manual mode: no promotional hints
            pass

        save_session(ses)
        return jsonify({
            "reply": message,
            "updated_memory": True,
            "stage": stage
        })

    # Non-learning: respond
    # Provide memory hints
    hints_raw = infant.mem.search(msg, topk=3)
    hints = [{"text": t, "type": ty} for _, t, ty in hints_raw]

    # Use model reply
    reply, _ = infant.reply(msg, stage, prefer)

    return jsonify({
        "reply": reply,
        "updated_memory": False,
        "memory": hints,
        "stage": stage
    })

# ---------------------------
# Static passthrough for favicon (optional)
# ---------------------------
@app.route('/favicon.ico')
def favicon():
    path = os.path.join(STATIC_DIR, 'favicon.ico')
    if os.path.exists(path):
        return send_from_directory(STATIC_DIR, 'favicon.ico', mimetype='image/vnd.microsoft.icon')
    return ('', 204)

# ---------------------------
# Main
# ---------------------------
if __name__ == "__main__":
    host = os.environ.get("INFANT_HOST", "0.0.0.0")
    port = int(os.environ.get("INFANT_PORT", "5000"))
    print("⚪ Using CPU")
    print(f"Infant LLM {VERSION} — {EDITION} (type 'help' for commands if using CLI endpoints)")
    app.run(host=host, port=port, debug=False)
