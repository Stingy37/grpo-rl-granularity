#!/usr/bin/env bash
# One-time environment setup: a persistent conda env in ~/.conda/envs/grpo-rl (the pod's /opt/conda
# is on an ephemeral overlay and loses packages on restart).
#
# Why these exact builds: the pod runs driver 570 (CUDA 12.8). vLLM >= 0.28 pins torch 2.13, whose
# wheels are CUDA 13 and need an R580 driver. vLLM 0.26.0 is the newest release pinning torch 2.11.0;
# its "+cu129" GitHub wheel with torch 2.11.0+cu129 runs on the 12.8 driver through CUDA minor-version
# compatibility (vLLM's documented install path for CUDA 12.9). faiss-cpu is pinned to 1.13.2 because
# 1.15.1 returns wrong results from multi-threaded batch IndexIVF search on this index.
set -euo pipefail
ENV_NAME="${ENV_NAME:-grpo-rl}"
CONDA_BIN="${CONDA_BIN:-/opt/conda/bin/conda}"

if [ ! -x "$HOME/.conda/envs/$ENV_NAME/bin/python" ]; then
    "$CONDA_BIN" create -y -q -n "$ENV_NAME" python=3.12 pip
fi
PIP="$HOME/.conda/envs/$ENV_NAME/bin/pip"
PY="$HOME/.conda/envs/$ENV_NAME/bin/python"

"$PIP" install -q "vllm @ https://github.com/vllm-project/vllm/releases/download/v0.26.0/vllm-0.26.0+cu129-cp38-abi3-manylinux_2_28_x86_64.whl" \
    --extra-index-url https://download.pytorch.org/whl/cu129
"$PIP" install -q "peft==0.21.2" "bitsandbytes==0.50.2" "faiss-cpu==1.13.2" pyarrow accelerate huggingface_hub
# vllm pulls torchcodec (video decoding) from PyPI as a CUDA-13 build whose libtorchcodec_image.so needs
# libnvrtc.so.13 and makes `import vllm` fail with OSError. No cu129 build of that version exists and
# nothing here decodes video, so it is removed; vllm treats a missing torchcodec as an optional dependency.
"$PIP" uninstall -y -q torchcodec
# The pod has no CUDA toolkit (no nvcc). vLLM's attention (FlashAttention), fp8 GEMMs (CUTLASS), LoRA (Triton) and
# torch.compile all ship prebuilt or use Triton's own compiler, but FlashInfer's top-k/top-p sampler JIT-compiles
# with nvcc at warm-up, so scripts/train_rl.sh and src/main.py set VLLM_USE_FLASHINFER_SAMPLER=0.

"$PY" - <<'EOF'
import torch, vllm, faiss, peft, bitsandbytes, transformers, pyarrow
print("torch", torch.__version__, "cuda", torch.version.cuda, "available:", torch.cuda.is_available(),
      torch.cuda.get_device_name(0) if torch.cuda.is_available() else "")
print("vllm", vllm.__version__, "| transformers", transformers.__version__, "| peft", peft.__version__,
      "| bitsandbytes", bitsandbytes.__version__, "| faiss", faiss.__version__, "| pyarrow", pyarrow.__version__)
EOF
echo "ok: run training with  bash scripts/train_rl.sh  (PY=$PY)"
