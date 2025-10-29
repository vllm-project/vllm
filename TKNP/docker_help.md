# Build and use docker containers to build vLLM system


```bash
dzdo docker pull nvcr.io/nvidia/ai-dynamo/vllm-runtime:0.4.1

# Run interactive shell
dzdo docker run --gpus all -it --rm   --name vllm-dev   --ipc=host   --shm-size=20g   --ulimit memlock=-1   -p 8000:8000 nvcr.io/nvidia/ai-dynamo/vllm-runtime:0.4.1 bash

```

## Restart docker if you cannot see images 

```bash
dzdo systemctl status docker
```

# To install new packages interactively 

```bash
# start the container, install packages, do not exit. Keep the container open.
pip install <pkg-name>

# from a different terminal, run the script below
dzdo docker ps

# copy the container id and commit the changes 

dzdo docker commit <container-id> susavlsh10/vllm-tknp:v1

dzdo docker commit 595878e9b149 susavlsh10/vllm-tknp:v1

```

```bash
dzdo docker run --gpus all -it --rm   --name vllm-dev   --ipc=host   --shm-size=20g   --ulimit memlock=-1    -v "$HOME:/workspace" -w /workspace -p 8000:8000 susavlsh10/vllm-tknp:v0  bash
```

  -v "$HOME/Documents/MLSystems/vllm-distributed:/workspace" \
  -w /workspace \


```bash
dzdo docker run --gpus all -it --rm \
  --name vllm-run \
  --ipc=host --shm-size=20g --ulimit memlock=-1 \
  -p 8001:8001 \
  -v "$HOME:$HOME" \
  -v "$HOME/Documents/MLSystems/vllm-distributed:/workspace" \
  -v /mnt/nvme/hf_cache:/mnt/nvme/hf_cache \
  -w /workspace \
  -e HF_HOME=/mnt/nvme/hf_cache \
  --entrypoint bash \
  susavlsh10/vllm-tknp:v1

```

if the container is already running 

```bash
dzdo docker exec -it vllm-run bash
```

# Editable copy of vLLM 

```bash
VLLM_USE_PRECOMPILED=1 pip install --editable .
```
vllm base commit : 5f0af36af555a3813b9d30983bd29c384b84b647

export VLLM_COMMIT=5f0af36af555a3813b9d30983bd29c384b84b647 # use full commit hash from the main branch
export VLLM_PRECOMPILED_WHEEL_LOCATION=https://wheels.vllm.ai/${VLLM_COMMIT}/vllm-1.0.0.dev-cp38-abi3-manylinux1_x86_64.whl
uv pip install --editable .