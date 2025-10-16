#!/usr/bin/env bash
# Infant LLM v6.3 — Growth Stages Edition (headless-friendly)

clear
echo "=============================================================="
echo " 🌱  Infant LLM v6.3 — STEM Learner"
echo "--------------------------------------------------------------"
echo "  A living AI experiment that grows through human teaching."
echo "  Runs locally with Flask + PyTorch (CPU/GPU)."
echo "=============================================================="
sleep 1

if [ ! -d "infantenv" ]; then
  echo "🧠 Creating virtual environment..."
  python3 -m venv infantenv
fi

source infantenv/bin/activate

echo "📦 Installing dependencies..."
pip install --upgrade pip
pip install -r requirements.txt

echo ""
echo "🚀 Launching Infant LLM v6.3..."
echo "💻 Access it at: http://0.0.0.0:5000  (or http://localhost:5000)"
echo ""
python3 infantLLM/infantLM_v6_3.py
