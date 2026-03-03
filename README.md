# Qwen3-0.6B Megakernel — Reproduction Guide

Custom CUDA megakernel for Qwen3-0.6B inference on RTX 5090, benchmarked against vLLM's standard PagedAttention baseline.

## Benchmark Results (RTX 5090, float16, 32 max tokens)

| Metric | Megakernel | vLLM (enforce-eager) | Speedup |
|---|---|---|---|
| Avg latency | 0.052s | 0.120s | **2.3x** |
| Median latency | 0.052s | 0.154s | **2.9x** |
| Tokens/sec | 606.7 | 266.5 | **2.3x** |
| Throughput (end-to-end) | 402.6 tok/s | 199.3 tok/s | **2.0x** |
| Req/s | 17.04 | 8.33 | **2.0x** |
| Variance (min to max) | 0.048-0.058s | 0.048-0.162s | 10x tighter |

---

## Requirements

- NVIDIA RTX 5090 (sm_120 architecture)
- CUDA 13.0 driver (nvidia-smi should show Driver 580+)
- Python 3.12
- Ubuntu 24 (tested on Vast.ai container)

**Note:** The CUDA megakernel compiles for sm_120 (Blackwell architecture) and requires a physical RTX 5090. It cannot run on older GPUs or on macOS. Mac users must use a remote GPU instance — see the macOS section below.

---

## Repository Structure

```
qwen_megakernel/
├── megakernel_5090.cu          # Main CUDA megakernel (sm_120)
├── megakernel.py               # FastAPI OpenAI-compatible server
├── qwen_ops.cpp                # PyTorch C++ extension bindings
├── setup.py                    # Build script for CUDA extension
├── serve.sh                    # vLLM baseline server launcher
├── client_benchmark.py         # Benchmark client (OpenAI SDK)
├── prompt.py                   # 100 benchmark prompts
├── rmsnorm.cuh / rmsnorm.cu    # RMSNorm CUDA kernel
├── swiglu.cuh / swiglu.cu      # SwiGLU CUDA kernel
├── swiglu_binding.cpp          # SwiGLU PyTorch binding
└── Model/
    └── Qwen06B_architecture.py # Weight loading + Decoder class
```

---

## Setup by Platform

All platforms ultimately run the same server on Linux. The difference is only in how you connect and copy files.

---

### Windows (VS Code + WSL)

**Prerequisites:**
- VS Code with the Remote - SSH extension installed
- WSL2 with Ubuntu (run `wsl --install` in PowerShell if not already set up)
- NVIDIA Nsight Systems 2025.3.2 for viewing profiles (Windows native)

**Step 1 — Generate SSH key in WSL:**
```bash
ssh-keygen -t ed25519 -C "your_email@example.com"
cat ~/.ssh/id_ed25519.pub
```
Paste the public key into your Vast.ai instance under Manage SSH Keys.

**Step 2 — Connect VS Code to the remote server:**

1. Open VS Code and press Ctrl+Shift+P
2. Type Remote-SSH: Connect to Host, then Add New SSH Host
3. Enter: `ssh -p <PORT> root@<IP>`
4. Select `C:\Users\YourName\.ssh\config` to save
5. Click Connect — VS Code opens a new window on the server

**Step 3 — Copy files from WSL to server:**
```bash
export SERVER_IP=<your_instance_ip>
export SERVER_PORT=<your_instance_port>

ssh -i ~/.ssh/id_ed25519 -p $SERVER_PORT root@$SERVER_IP "mkdir -p ~/qwen_megakernel/Model"

scp -i ~/.ssh/id_ed25519 -P $SERVER_PORT \
  megakernel_5090.cu megakernel.py qwen_ops.cpp \
  rmsnorm.cu rmsnorm.cuh swiglu.cu swiglu.cuh swiglu_binding.cpp \
  client_benchmark.py serve.sh setup.py prompt.py \
  root@$SERVER_IP:~/qwen_megakernel/

scp -i ~/.ssh/id_ed25519 -P $SERVER_PORT \
  Model/Qwen06B_architecture.py \
  root@$SERVER_IP:~/qwen_megakernel/Model/
```

**Step 4 — SSH tunnel for accessing the server locally:**
```bash
ssh -i ~/.ssh/id_ed25519 -p $SERVER_PORT root@$SERVER_IP \
    -L 8000:localhost:8000 -N -o ServerAliveInterval=30 &

curl http://localhost:8000/health
```

**Viewing Nsight profiles on Windows:**

Copy .nsys-rep files from the server to WSL:
```bash
scp -i ~/.ssh/id_ed25519 -P $SERVER_PORT \
  root@$SERVER_IP:/tmp/megakernel_profile.nsys-rep \
  ~/your_project/megakernel_profile.nsys-rep
```

Open in Nsight Systems on Windows via File > Open and navigate to:
```
\\wsl.localhost\Ubuntu\home\<your_username>\your_project\megakernel_profile.nsys-rep
```

---

### macOS

**Prerequisites:**
- VS Code with the Remote - SSH extension
- Homebrew: `brew install openssh`

macOS has no CUDA support. All GPU work runs on the remote Linux server. Your Mac is purely the client and editor.

**Step 1 — Generate SSH key:**
```bash
ssh-keygen -t ed25519 -C "your_email@example.com"
cat ~/.ssh/id_ed25519.pub
```
Paste the public key into Vast.ai under Manage SSH Keys.

**Step 2 — Connect VS Code:**

1. Install the Remote - SSH extension in VS Code
2. Press Cmd+Shift+P and type Remote-SSH: Connect to Host
3. Enter: `ssh -p <PORT> root@<IP>`
4. Select `~/.ssh/config` to save
5. Click Connect

**Step 3 — Copy files from Mac to server:**
```bash
export SERVER_IP=<your_instance_ip>
export SERVER_PORT=<your_instance_port>

ssh -i ~/.ssh/id_ed25519 -p $SERVER_PORT root@$SERVER_IP "mkdir -p ~/qwen_megakernel/Model"

scp -i ~/.ssh/id_ed25519 -P $SERVER_PORT \
  megakernel_5090.cu megakernel.py qwen_ops.cpp \
  rmsnorm.cu rmsnorm.cuh swiglu.cu swiglu.cuh swiglu_binding.cpp \
  client_benchmark.py serve.sh setup.py prompt.py \
  root@$SERVER_IP:~/qwen_megakernel/

scp -i ~/.ssh/id_ed25519 -P $SERVER_PORT \
  Model/Qwen06B_architecture.py \
  root@$SERVER_IP:~/qwen_megakernel/Model/
```

**Step 4 — SSH tunnel:**
```bash
ssh -i ~/.ssh/id_ed25519 -p $SERVER_PORT root@$SERVER_IP \
    -L 8000:localhost:8000 -N -o ServerAliveInterval=30 &

curl http://localhost:8000/health
```

**Viewing Nsight profiles on Mac:**

Nsight Systems does not have a macOS GUI. Options:
- Copy .nsys-rep to a Windows or Linux machine with Nsight Systems installed
- Use the Nsight Systems CLI on the server: `nsys stats /tmp/megakernel_profile.nsys-rep`

---

### Local RTX 5090 — Windows + WSL2

If you have a physical RTX 5090 in your Windows machine, you can run everything locally via WSL2 with CUDA passthrough. No remote server needed.

**Prerequisites:**
- Windows 11 with WSL2 enabled
- NVIDIA Driver 580+ installed on Windows (WSL2 inherits this automatically)
- Ubuntu 24.04 in WSL2

**Step 1 — Verify CUDA is accessible in WSL2:**
```bash
nvidia-smi        # should show RTX 5090 and CUDA 13.0
nvcc --version    # should show CUDA 13.0
```

If nvcc is not found, install the CUDA toolkit from developer.nvidia.com/cuda-downloads and select WSL-Ubuntu as the target.

**Step 2 — Copy files into WSL:**
```bash
mkdir -p ~/qwen_megakernel/Model

# Access Windows files via /mnt/c/
cp /mnt/c/Users/YourName/qwen_project/megakernel_5090.cu ~/qwen_megakernel/
cp /mnt/c/Users/YourName/qwen_project/Model/Qwen06B_architecture.py ~/qwen_megakernel/Model/
# repeat for all files listed in Repository Structure above
```

**Step 3 — Install dependencies:**
```bash
pip install --pre torch --index-url https://download.pytorch.org/whl/nightly/cu128
pip install transformers fastapi "uvicorn[standard]" orjson pydantic openai ninja accelerate vllm
```

**Step 4 — Build the CUDA extension:**
```bash
cd ~/qwen_megakernel

CUDA_PATH=$(dirname $(which nvcc))/..
CUDA_HOME=$CUDA_PATH TORCH_CUDA_ARCH_LIST="12.0a" python setup.py build_ext --inplace
```

On local WSL2, the CUDA version mismatch patch is usually not needed. If you see CUDA_MISMATCH_MESSAGE, apply the same sed patch from the cloud setup section below.

**Step 5 — Full ncu profiling is available on local GPU:**

Unlike cloud containers, a local GPU gives full hardware counter access:
```bash
ncu --set full \
    --export ~/qwen_megakernel/megakernel_ncu_profile \
    python -c "
from Model.Qwen06B_architecture import Decoder
d = Decoder(verbose=False)
d.generate('What is the capital of France?', max_tokens=32)
"
```

Open `megakernel_ncu_profile.ncu-rep` in NVIDIA Nsight Compute on Windows:
```
\\wsl.localhost\Ubuntu\home\<username>\qwen_megakernel\megakernel_ncu_profile.ncu-rep
```

Everything from the cloud setup Steps 4 onward applies identically. The server runs in WSL and is accessible at localhost:8000 from both WSL and Windows.

---

### Local RTX 5090 — Native Linux

**Prerequisites:**
- Ubuntu 22.04 or 24.04
- NVIDIA Driver 580+: `sudo apt install nvidia-driver-580`
- CUDA 13.0 toolkit

**Step 1 — Verify:**
```bash
nvidia-smi        # should show RTX 5090
nvcc --version    # should show CUDA 13.0
```

**Step 2 — Install dependencies:**
```bash
pip install --pre torch --index-url https://download.pytorch.org/whl/nightly/cu128
pip install transformers fastapi "uvicorn[standard]" orjson pydantic openai ninja accelerate vllm
```

**Step 3 — Build:**
```bash
cd ~/qwen_megakernel

CUDA_HOME=/usr/local/cuda-13.0 TORCH_CUDA_ARCH_LIST="12.0a" \
python setup.py build_ext --inplace
```

On native Linux the CUDA version mismatch patch is usually not needed. If you hit the error, apply the same sed patch from the cloud setup section.

Full ncu profiling works without restrictions on native Linux:
```bash
ncu --set full --export /tmp/megakernel_ncu \
    python -c "
from Model.Qwen06B_architecture import Decoder
d = Decoder(verbose=False)
d.generate('What is the capital of France?', max_tokens=32)
"
```

---

## Cloud Setup (Vast.ai) — Full Steps

### Step 1 — Provision Instance

Rent an RTX 5090 on Vast.ai (https://cloud.vast.ai):
- GPU: RTX 5090
- Image: pytorch/pytorch:latest or any Ubuntu 24 image
- Disk: 30GB minimum

### Step 2 — Install Dependencies

```bash
pip install --pre torch --index-url https://download.pytorch.org/whl/nightly/cu128
pip install transformers fastapi "uvicorn[standard]" orjson pydantic openai ninja accelerate vllm
```

### Step 3 — Build the CUDA Extension

```bash
cd ~/qwen_megakernel

# Patch PyTorch CUDA version check (driver 13.0 vs PyTorch built with 12.8)
LINE=$(grep -n "raise RuntimeError(CUDA_MISMATCH_MESSAGE" \
    /venv/main/lib/python3.12/site-packages/torch/utils/cpp_extension.py | cut -d: -f1)
sed -i "${LINE}s/raise RuntimeError/pass  # raise RuntimeError/" \
    /venv/main/lib/python3.12/site-packages/torch/utils/cpp_extension.py

CUDA_HOME=/usr/local/cuda-13.0 TORCH_CUDA_ARCH_LIST="12.0a" \
python setup.py build_ext --inplace
```

Verify:
```bash
python -c "import qwen_megakernel_C; print('ABI version:', qwen_megakernel_C.abi_version())"
# Expected: ABI version: 2
```

### Step 4 — Create .env

```bash
cat > ~/qwen_megakernel/.env << 'EOF'
HOST=http://localhost
PORT=8000
MODEL=Qwen/Qwen3-0.6B
DTYPE=float16
MAX_MODEL_LEN=2048
GPU_MEMORY_UTILIZATION=0.90
MAX_NUM_SEQS=1
BLOCK_SIZE=16
SWAP_SPACE=4
KV_CACHE_DTYPE=auto
NUM_REQUESTS=100
MAX_TOKENS=32
TEMPERATURE=0.0
LOG_FILE=latency_log_megakernel.json
CONCURRENCY=1
EOF
```

### Step 5 — Run Megakernel Server

```bash
cd ~/qwen_megakernel
python megakernel.py
```

Expected output:
```
[megakernel_server] Loading Qwen/Qwen3-0.6B with megakernel...
All weight pointers 16-byte aligned OK
[megakernel_server] Model loaded in ~8s
[megakernel_server] Serving on port 8000
```

Health check:
```bash
curl http://localhost:8000/health
# {"status":"ok"}
```

### Step 6 — Run Megakernel Benchmark

In a second terminal on the server:
```bash
cd ~/qwen_megakernel
python client_benchmark.py
```

Raw kernel benchmark with no HTTP overhead:
```bash
python -c "
from prompt import PROMPTS
from Model.Qwen06B_architecture import Decoder
import time, statistics

d = Decoder(verbose=False)
d.generate(PROMPTS[0], max_tokens=32)

latencies = []
for p in PROMPTS:
    t0 = time.perf_counter()
    d.generate(p, max_tokens=32)
    latencies.append(time.perf_counter() - t0)

print(f'Avg:    {statistics.mean(latencies):.3f}s')
print(f'Median: {statistics.median(latencies):.3f}s')
print(f'Min:    {min(latencies):.3f}s')
print(f'Max:    {max(latencies):.3f}s')
print(f'Tok/s:  {32/statistics.mean(latencies):.1f}')
"
```

### Step 7 — Run vLLM Baseline

```bash
pkill -f megakernel.py
sed -i 's/LOG_FILE=.*/LOG_FILE=latency_log_vllm_baseline.json/' .env
bash serve.sh
```

Wait for `Application startup complete`, then in a second terminal:
```bash
python client_benchmark.py
```

### Step 8 — Profile with Nsight Systems

```bash
export PATH=/opt/nvidia/nsight-compute/2025.3.1/host/target-linux-x64:$PATH

# Megakernel — 100 prompts
nsys profile --output /tmp/megakernel_profile_100 \
    python -c "
from prompt import PROMPTS
from Model.Qwen06B_architecture import Decoder
d = Decoder(verbose=False)
d.generate(PROMPTS[0], max_tokens=32)
for p in PROMPTS:
    d.generate(p, max_tokens=32)
"

# vLLM — profile the server process directly
nsys profile \
    --output /tmp/vllm_server_profile \
    --trace cuda,cudnn,cublas,osrt \
    --trace-fork-before-exec true \
    python -m vllm.entrypoints.openai.api_server \
    --model Qwen/Qwen3-0.6B \
    --enforce-eager \
    --max-model-len 2048 \
    --max-num-seqs 1 \
    --dtype float16 \
    --port 8000 &

until curl -s http://localhost:8000/health > /dev/null 2>&1; do sleep 2; done

python -c "
from prompt import PROMPTS
from openai import OpenAI
client = OpenAI(base_url='http://localhost:8000/v1', api_key='x')
for p in PROMPTS:
    client.chat.completions.create(
        model='Qwen/Qwen3-0.6B',
        messages=[{'role':'user','content':p}],
        max_tokens=32,
        extra_body={'chat_template_kwargs': {'enable_thinking': False}}
    )
"
pkill -f vllm
```

Copy profiles to local machine:
```bash
scp -i ~/.ssh/id_ed25519 -P $SERVER_PORT \
  root@$SERVER_IP:/tmp/megakernel_profile_100.nsys-rep \
  root@$SERVER_IP:/tmp/vllm_server_profile.nsys-rep \
  ./profiles/
```

Open in NVIDIA Nsight Systems 2025.3.2.

---

## Key Design Decisions

**Single kernel launch per inference phase** — `launch_ldg_generate_nosync` runs all 32 decode steps inside one launch using GPU-side grid barriers between layers. Traditional frameworks launch a separate kernel per operation per token.

**Partial KV cache reset** — only zeros positions actually written (tracked via high-water mark) instead of clearing the full 235MB cache every request. For a 16-token prompt + 32-token output this is ~42x less data zeroed.

**Single cudaStreamSynchronize per request** — all N token generation steps run on-device with no CPU sync until the final DtoH transfer of the output log.

**Pinned memory output buffer** — the output token log is backed by pinned CPU memory for fast DMA transfer (~10us for 128 int32 tokens).

---

## Troubleshooting

| Error | Fix |
|---|---|
| `CUDA_MISMATCH_MESSAGE` during build | Apply the sed patch in the build step |
| `ABI version mismatch` | Rebuild: `python setup.py build_ext --inplace --force` |
| `qwen_megakernel_C op decode unavailable` | PyTorch was upgraded — rebuild the extension |
| `Address already in use` on port 8000 | `pkill -f megakernel.py` or `pkill -f vllm` |
| `ERR_NVGPUCTRPERM` for ncu | Container restriction — use nsys instead, or use a local or bare-metal GPU for full ncu access |
| `nvidia-smi not found` in WSL2 | Install NVIDIA driver 580+ on the Windows host — WSL2 inherits it automatically |
