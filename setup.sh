#!/usr/bin/env bash
# Bootstrap the Shap-E demo environment.
# Auteur : BENCHERAIK Abdessamad
# Usage: bash setup.sh [--cpu]   (default: GPU/CUDA)

set -euo pipefail

USE_CPU=false
[[ "${1:-}" == "--cpu" ]] && USE_CPU=true

echo "=== Creating conda environment 'shap-e-demo' ==="
if $USE_CPU; then
    # Strip CUDA line for CPU-only machines
    sed '/pytorch-cuda/d' environment.yml > /tmp/env_cpu.yml
    conda env create -f /tmp/env_cpu.yml
else
    conda env create -f environment.yml
fi

echo ""
echo "=== Activating and verifying ==="
conda run -n shap-e-demo python - <<'EOF'
import torch
print(f"PyTorch  : {torch.__version__}")
print(f"CUDA ok  : {torch.cuda.is_available()}")
import shap_e; print("Shap-E   : ok")
import open3d as o3d; print(f"Open3D   : {o3d.__version__}")
EOF

echo ""
echo "Setup complete. Activate with:"
echo "  conda activate shap-e-demo"
echo ""
echo "Then generate your first 3D object:"
echo "  python generate_3d.py --prompt 'a ceramic teapot'"
