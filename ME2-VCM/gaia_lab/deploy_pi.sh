#!/usr/bin/env bash
# Deploy Gaia's Chamber to a Raspberry Pi 5 (4 GB).
# Installs CPU-only PyTorch + dependencies, then starts the server.
set -euo pipefail

echo "=== Gaia's Chamber — Pi 5 Deployment ==="

# 1. Install system deps
sudo apt-get update
sudo apt-get install -y python3-pip python3-venv

# 2. Create venv
python3 -m venv ~/.gaia-venv
source ~/.gaia-venv/bin/activate

# 3. Install CPU-only PyTorch (smaller, faster on Pi)
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install numpy scipy soundfile flask

# 4. Copy project files (assumes you've already scp'd the gaia-lab dir)
echo "Project files should be in ~/gaia-lab/"
test -f ~/gaia-lab/src/gaia_app.py || { echo "ERROR: ~/gaia-lab/src/gaia_app.py not found"; exit 1; }
test -f ~/gaia-lab/models/crnn_hf_20_gen.pth || { echo "ERROR: model checkpoint not found"; exit 1; }

# 5. Start the server
echo "Starting Gaia's Chamber on port 8901..."
cd ~/gaia-lab/src
python3 gaia_app.py &

echo ""
echo "=== Gaia's Chamber is running ==="
echo "Open: http://$(hostname -I | awk '{print $1}'):8901"
echo ""
echo "To run as a systemd service:"
cat <<'EOF'
[Service]
WorkingDirectory=/home/pi/gaia-lab/src
ExecStart=/home/pi/.gaia-venv/bin/python3 gaia_app.py
Restart=always
EOF
