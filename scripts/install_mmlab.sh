#!/usr/bin/env bash
# Install the OpenMMLab stack (mmcv, mmdet, mmpose) into the uv-managed venv.
#
# mmcv requires building C++ extensions from source and has a broken
# build-system declaration (uses pkg_resources without declaring it).
# This script works around that by:
#   1. Using pip --no-build-isolation (relies on setuptools<70 in the venv)
#   2. Pre-installing cython + wheel (needed by mmcv and xtcocotools)
#   3. Building mmcv, chumpy, and xtcocotools from source
#   4. Installing mmdet and mmpose (pure Python)
#
# Usage:
#   uv sync              # install core deps first (pins setuptools<70)
#   bash scripts/install_mmlab.sh

set -euo pipefail

PIP=".venv/bin/pip"

echo "=== Installing OpenMMLab stack ==="

# Pre-requisites for building C extensions
echo "→ Installing build tools (pip, wheel, cython)..."
uv pip install pip wheel cython

# Build mmcv from source (compiles C++ ops — takes ~5 minutes)
echo "→ Building mmcv from source (this takes several minutes)..."
$PIP install --no-build-isolation "mmcv>=2.1.0,<2.2.0"

# Build chumpy and xtcocotools (needed by mmpose, also have broken builds)
echo "→ Building chumpy and xtcocotools..."
uv pip install pip  # chumpy's setup.py imports pip
$PIP install --no-build-isolation chumpy==0.70
$PIP install --no-build-isolation xtcocotools==1.14.3

# Install mmdet and mmpose (pure Python wheels)
echo "→ Installing mmdet and mmpose..."
uv pip install "mmdet>=3.2.0" "mmpose>=1.3.0"

echo ""
echo "=== OpenMMLab stack installed successfully ==="
echo ""
echo "Verify with:"
echo '  uv run --no-sync python -c "import mmcv, mmdet, mmpose; print(\"mmcv\", mmcv.__version__); print(\"mmdet\", mmdet.__version__); print(\"mmpose\", mmpose.__version__)"'
