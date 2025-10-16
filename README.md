<p align="center">
  <img src="https://github.com/jburishkin/Infant-LLM-STEM-Learner/blob/main/STEMLEARNER.png" alt="Infant LLM — STEM Learner Banner" width="100%">
</p>

# 🧠 Infant LLM v6 — STEM Learner

> An evolving local neural assistant that learns **Science, Technology, Engineering, and Math (STEM)** directly from your input.  
> Teach it facts, equations, and code — it remembers, reasons, and grows over time.

---

### 🎥 See It in Action

| Chat & Teach | Memory & STEM Context |
|:-------------:|:--------------------:|
| ![Chat Demo][(https://github.com/yourusername/infant-llm/assets/demo-chat.gif)](https://github.com/jburishkin/Infant-LLM-STEM-Learner/blob/main/InfLM6_1_use.png) | ![Memory Demo](https://github.com/yourusername/infant-llm/assets/demo-memory.gif) |

*(Replace these with your actual `.gif` or `.png` files once uploaded to your repo’s `/assets/` folder.)*


---

## 🚀 Overview

**Infant LLM v6** is a lightweight, self-growing language model that starts with no predefined knowledge.  
It learns entirely from **your interactions** and evolves as you teach it.

It’s built for **education**, **STEM experimentation**, and **AI research** — exploring how small models can grow into capable assistants through incremental learning.

---

## 🧩 Key Features

- 🧠 **Self-learning neural model** — grows only from user-taught input  
- 📚 **Persistent FAISS memory** — remembers every fact, equation, or code sample  
- 🧮 **STEM awareness** — classifies math, science, coding, and engineering knowledge  
- 💻 **Interactive web UI** — modern interface with personality and creativity sliders  
- ⚙️ **Hardware flexibility** — runs on CPU, NVIDIA, AMD, Intel, or Apple GPUs  
- 🔐 **Offline by design** — fully local, no data leaves your device  


---

## 🧰 Requirements

- **Python 3.10+**
- Recommended packages:
  ```bash
  pip install flask torch numpy faiss-cpu
  ```
- Optional (GPU acceleration):
  ```bash
  pip install faiss-gpu
  ```

---

## 📂 Project Layout

```
infantLLM/
 ├── infantLM_v6_local.py     # Main Flask + PyTorch logic
 ├── infant_ui_v6.html        # Web UI for chat + learning
 ├── infant_data_v6_local/    # Auto-created folder for memory
 └── README.md
```

---

## 🧪 Run the App

```bash
python3 infantLM_v6_local.py
```

Then open your browser to:

👉 [http://localhost:5000](http://localhost:5000)

---

## 💬 Example Interaction

| You | Infant |
|-----|--------|
| `learn the sky is blue` | 🧠 Learned: *the sky is blue* |
| `learn math: 1+1=2` | 🧮 Learned: *1+1=2* |
| `1+1=` | 🧮 *1+1 = 2* |
| `learn dolphins swim in the ocean` | 🧠 Learned: *dolphins swim in the ocean* |
| `memory sky` | Hint: *the sky is blue* |

---

## 🎛️ Interface Features

- **Personality** — switch between *Playful (🧠)* or *Serious (⚙️)*  
- **Temperature** — control creativity (0.1 → logical, 2.0 → imaginative)  
- **Knowledge Filter** — query specific categories like *math* or *coding*  
- **Blue Favicon & Footer Icons** — match mode (🧠 playful / ⚙️ serious)

---

## 🗃️ Memory System

- Uses the model’s embedding layer to vectorize each learned phrase  
- Stores them persistently with FAISS for efficient similarity search  
- Reloads automatically between runs from:
  ```
  infant_data_v6_local/memory.pkl
  ```

---

## ⚙️ Hardware Detection

Automatically detects and optimizes for your best available device:

| Device | Backend |
|---------|----------|
| 🟢 NVIDIA GPU | CUDA |
| 🟠 AMD GPU | ROCm |
| 🔵 Intel GPU | XPU |
| 🟣 Apple Silicon | MPS |
| ⚪ CPU | Fallback |

---

## 🔒 Privacy

Infant LLM v6 runs **completely offline**:
- No API keys, no cloud access, no telemetry.  
- All learning stays on your system.  
- Perfect for secure or air-gapped environments.

---

## 🧭 Roadmap

- 🧩 Modular “plugin” learning files  
- 🧠 Incremental self-correction  
- 🔐 Optional enterprise SSO (Google / Microsoft / GCC High)  
- 🌐 Shared LAN knowledge sync between instances  

---

## 🧑‍💻 Contributing

Contributions are welcome!  
Open an issue or PR to add:
- new learning features  
- improved STEM understanding  
- UI enhancements
- SSO login
- or better model persistence  

---

## 🪪 License

**MIT License**

Free for personal and commercial use.  
Please credit **Infant LLM** when redistributing or adapting this project.

---

<p align="center">
  <sub>🧠 Infant LLM v6 — STEM Learner | Built with PyTorch + Flask | Runs locally, learns from you</sub>
</p>
