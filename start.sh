#!/usr/bin/env bash
set -euo pipefail

echo "==> MLExplain -- Interactive ML Model Explainer"
echo "==> Setting up environment ..."

if [ ! -d "venv" ]; then
    python3 -m venv venv
fi
source venv/bin/activate

pip install --quiet -r requirements.txt

echo "==> Starting server on port 8006 ..."
export PORT=8006
python app.py
