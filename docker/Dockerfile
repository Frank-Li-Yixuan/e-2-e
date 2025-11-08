# CUDA + Python runtime for stable, reproducible runs
# Use a widely available CUDA+cudnn tag
FROM --platform=linux/amd64 nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Base tools and libs
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-pip python3-venv python3-dev \
    git ffmpeg libgl1 libglib2.0-0 \
    build-essential curl ca-certificates && \
    rm -rf /var/lib/apt/lists/*

# Python deps: enforce ABI-safe numpy/opencv, Torch CU124, and evaluation stack
RUN python3 -m pip install -U pip wheel setuptools && \
    python3 -m pip uninstall -y numpy opencv-python opencv-contrib-python opencv-python-headless || true && \
    python3 -m pip install --no-cache-dir "numpy<2.0" "opencv-python-headless==4.8.1.78" && \
    python3 -m pip install --no-cache-dir torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124 && \
    python3 -m pip install --no-cache-dir ultralytics insightface onnxruntime-gpu onnxruntime pycocotools lpips pytorch-fid easyocr matplotlib seaborn pandas tqdm pyyaml pillow "jedi>=0.18.2" && \
    # Re-pin to avoid accidental upgrades by transitive deps
    python3 -m pip install --no-cache-dir "numpy<2.0" "opencv-python-headless==4.8.1.78"

WORKDIR /workspace

# Quick sanity on container start (visible in logs)
CMD ["bash","-lc","python - <<'PY'\nimport numpy as np\nprint('NUMPY VERSION:',np.__version__)\ntry:\n    import cv2\n    print('CV2 VERSION:',cv2.__version__)\nexcept Exception as e:\n    print('[ERROR] cv2 import failed:', e)\nPY\n"]
