Information about prefill/decode disaggregate serving: https://docs.vllm.ai/en/stable/features/disagg_prefill/#development

# setup

### nvidia
Install nvidia tools (setup_gpu_1.sh)
```
# install nvidia drivers and tools
sudo apt update
sudo add-apt-repository -y restricted
sudo add-apt-repository -y universe
sudo apt update
sudo apt install -y nvidia-utils-550 nvidia-driver-550
sudo nvidia-smi -mig 0 # disable multi instance GPU
sudo reboot
```

### vllm
https://docs.vllm.ai/en/latest/getting_started/quickstart/#prerequisites

Install UV & restart shell
```
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Install nvidia toolchain (cuda-keyring for downloading nvidia software)
```
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb
sudo apt-get update
sudo apt-get install -y cuda-toolkit-12-8 ccache

echo 'export PATH=/usr/local/cuda-12.8/bin:$PATH' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH=/usr/local/cuda-12.8/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc
source ~/.bashrc
```

Install dependencies via python
```
source .venv/bin/activate
uv pip install -U pip
uv pip install "torch==2.10.0+cu128" --index-url https://download.pytorch.org/whl/cu128
uv pip install packaging setuptools wheel ninja cmake numpy nixl
uv pip install -U "triton==3.5"
```

Install VLLM
```
VLLM_USE_PRECOMPILED=1 uv pip install --editable .
```

If you get an error like:
```
[stdout]
      VLLM_PRECOMPILED_WHEEL_COMMIT not valid: nightly, trying to fetch base commit in main branch
      Upstream main branch latest commit: 4ce2d0145312809ef6122ccb7be8ae7cafa462a9
      Using precompiled wheel commit 4ce2d0145312809ef6122ccb7be8ae7cafa462a9 with variant cu128
      Trying to fetch nightly build metadata from https://wheels.vllm.ai/4ce2d0145312809ef6122ccb7be8ae7cafa462a9/cu128/vllm/metadata.json
      Trying the default variant from remote
      Trying to fetch nightly build metadata from https://wheels.vllm.ai/4ce2d0145312809ef6122ccb7be8ae7cafa462a9/vllm/metadata.json
```
and, you have cloned `vllm` (not using `hcasalet/villum`), then you should change to the latest release which will have the precompiled wheel. For example:
```
git checkout releases/v0.23.0
```
If for some reason vllm is looking for cuda13.0, try setting `VLLM_PRECOMPILED_WHEEL_VARIANT=cu128`


##### Serving
Set your token
```
export HF_TOKEN=""
```

Set the model, use unsloth's gpt-oss-20b because vllm lists this one as supported on vllm cpu.
```
export MODEL="Qwen/Qwen3-30B-A3B"
```

Set the HF_HOME variable to the following location so we can share the model cache.
```
export HF_HOME="/data/huggingface"
```

add to bashrc
```
echo 'export HF_HOME=/data/huggingface' >> ~/.bashrc
source ~/.bashrc
```

Start prefill service
```
VLLM_NIXL_SIDE_CHANNEL_PORT=5600 \
UCX_NET_DEVICES=all \
CUDA_VISIBLE_DEVICES=0 \
VLLM_NIXL_SIDE_CHANNEL_HOST=[IP_ADDRESS] \
vllm serve "$MODEL" \
--port 8100 \
--enforce-eager \
--gpu-memory-utilization 0.9 \
--kv-transfer-config '{"kv_connector":"NixlConnector","kv_role":"kv_producer","kv_load_failure_policy":"fail"}'
```

Start decode service
```
CUDA_VISIBLE_DEVICES=0 \
UCX_NET_DEVICES=all \
VLLM_NIXL_SIDE_CHANNEL_PORT=5601 \
VLLM_NIXL_SIDE_CHANNEL_HOST=[IP_ADDRESS] \
vllm serve "$MODEL" \
--port 8200 \
--enforce-eager \
--gpu-memory-utilization 0.9 \
--kv-transfer-config '{"kv_connector":"NixlConnector","kv_role":"kv_consumer","kv_load_failure_policy":"fail"}'
```

Start the proxy
```
python tests/v1/kv_connector/nixl_integration/toy_proxy_server.py \
    --port 8192 \
    --prefiller-hosts localhost \
    --prefiller-ports 8100 \
    --decoder-hosts localhost \
    --decoder-ports 8200
```

# CPU only setup
### Build wheel from source for CPU
ref: https://docs.vllm.ai/en/v0.17.1/getting_started/installation/cpu/#full-build
we need to build with compilation (only takes a couple minutes) because the Clemson c4130 does not have AVX-512 so it fails if you run without compiling.
```
sudo apt-get update -y
sudo apt-get install -y gcc-12 g++-12 libnuma-dev
sudo update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-12 10 --slave /usr/bin/g++ g++ /usr/bin/g++-12
```

```
uv venv --python 3.12 --seed --managed-python
source .venv/bin/activate
```

```
git clone villum
cd villum

uv pip install -r requirements/build/cpu.txt --torch-backend cpu --index-strategy unsafe-best-match
uv pip install -r requirements/cpu.txt --torch-backend cpu --index-strategy unsafe-best-match
uv pip install nixl==1.3.0
```

```
VLLM_TARGET_DEVICE=cpu uv pip install . --no-build-isolation
```


### setup environment for specific CPU
Before use vLLM CPU installed via wheels, make sure TCMalloc and Intel OpenMP are installed and added to LD_PRELOAD: 
```
# install TCMalloc, Intel OpenMP is installed with vLLM CPU
sudo apt-get install -y --no-install-recommends libtcmalloc-minimal4

# manually find the path
sudo find / -iname '*libtcmalloc_minimal.so.4' 2>/dev/null
sudo find / -iname '*libiomp5.so' 2>/dev/null
TC_PATH=...
IOMP_PATH=...

# add them to LD_PRELOAD
export LD_PRELOAD="$TC_PATH:$IOMP_PATH:$LD_PRELOAD"
```

for exmaple mine looks like:
```
echo $LD_PRELOAD
/usr/lib/x86_64-linux-gnu/libtcmalloc_minimal.so.4:/users/ctknab/villum/.venv/lib/libiomp5.so
```

ref: https://docs.vllm.ai/en/v0.6.0/getting_started/cpu-installation.html#related-runtime-environment-variables

run `lscpu` to see the current architecture. on Clemson c4130, there are 2 NUMA nodes. So we should setup the CPU cores dedicated to the OpenMP threads. This is the NUMA node setup on this cpu:
```
NUMA:
  NUMA node(s):              2
  NUMA node0 CPU(s):         0,2,4,6,8,10,12,14,16,18,20,22,24,26,28,30,32,34,36,38,40,42,44,46
  NUMA node1 CPU(s):         1,3,5,7,9,11,13,15,17,19,21,23,25,27,29,31,33,35,37,39,41,43,45,47
```

so, set
```
# Map worker threads exclusively to physical cores on Node 0 and Node 1
export VLLM_CPU_OMP_THREADS_BIND="0,2,4,6,8,10,12,14,16,18,20,22|1,3,5,7,9,11,13,15,17,19,21,23"
```

### serving a model
```
# VLLM_CPU_KVCACHE_SPACE=12 vllm serve Qwen/Qwen3-30B-A3B     --dtype bfloat16     -tp 2     --distributed-executor-backend mp     --max-model-len 2048
export VLLM_CPU_OMP_THREADS_BIND=0-31
export VLLM_CPU_KVCACHE_SPACE=32
vllm serve Qwen/Qwen3-30B-A3B     --port 8000     --dtype auto
```

then in a separate terminal, curl the endpoint
```
curl http://localhost:8000/v1/completions   -H "Content-Type: application/json"   -d '{
    "model": "Qwen/Qwen3-30B-A3B",
    "prompt": "The primary structural difference between a operating system process and a thread is that",
    "max_tokens": 128,
    "temperature": 0.5
  }'
```

# etc
use `df -h` to see drive sizes
```
Filesystem                                       Size  Used Avail Use% Mounted on
tmpfs                                             38G  2.2M   38G   1% /run
/dev/sda3                                         63G  8.2G   52G  14% /
tmpfs                                            189G     0  189G   0% /dev/shm
tmpfs                                            5.0M     0  5.0M   0% /run/lock
/dev/sda1                                        256M   11M  246M   5% /boot/efi
ops.clemson.cloudlab.us:/proj/microfastpath-PG0  100G   70G   31G  70% /proj/microfastpath-PG0
ops.clemson.cloudlab.us:/share                   500G  5.9G  495G   2% /share
tmpfs                                             38G   16K   38G   1% /run/user/20004
```

and `lsblk` to see mounted disks
```
lsblk
NAME    MAJ:MIN RM   SIZE RO TYPE MOUNTPOINTS
sda       8:0    0 931.5G  0 disk
├─sda1    8:1    0   256M  0 part /boot/efi
├─sda2    8:2    0     1M  0 part
├─sda3    8:3    0    64G  0 part /
└─sda99 259:0    0     8G  0 part [SWAP]
sdb       8:16   0 931.5G  0 disk
```

check to make sure it isnt already mounted by going to `/data`

if you want to use the sbd disk, we need to format it
```
sudo mkfs.ext4 /dev/sdb
```

then mount it to a folder
```
# create the folder
sudo mkdir -p /data

# mount the drive to it
sudo mount /dev/sdb /data

# give your user ownership
sudo chown -R $USER:$(id -gn) /data
```

add it to the filesystem directory so it doesnt unmount
```
echo '/dev/sdb /data ext4 defaults,noatime 0 2' | sudo tee -a /etc/fstab
```


# docker
### installing
install with `./install-docker.sh`

verify its running
```
sudo systemctl status docker
```

if not then start it
```
sudo systemctl start docker
```

add yourself to the docker group
```
sudo usermod -a -G docker $USER
newgrp docker
```

### running
```
docker compose up -d
```

port forward (run on your laptop)
```
ssh -L 3000:127.0.0.1:3000 -i /path/to/your/cloudlab_key ctknab@your-cloudlab-hostname.edu
```

for example:
```
ssh -L 3000:127.0.0.1:3000 -i ~/.ssh/id_ed25519 ctknab@clnode246.clemson.cloudlab.us
```

then go to 
```
http://localhost:3000
```

and login with username: admin, password: admin. skip the prompt to change the default password.

### benchmarking
download the dataset
```
wget https://huggingface.co/datasets/anon8231489123/ShareGPT_Vicuna_unfiltered/resolve/main/ShareGPT_V3_unfiltered_cleaned_split.json
```

run the benchmark
```
vllm bench serve     --backend vllm     --base-url http://localhost:8000     --model Qwen/Qwen3-30B-A3B     --dataset-name sharegpt --dataset-path ./ShareGPT_V3_unfiltered_cleaned_split.json     --num-prompts 20     --request-rate 2.0
```

running with different batch sizes
```
 vllm bench serve     --backend vllm     --base-url http://localhost:8000     --model Qwen/Qwen3-30B-A3B     --dataset-name sharegpt --dataset-path ./ShareGPT_V3_unfiltered_cleaned_split.json     --num-prompts 20     --request-rate inf --max-concurrency 1
```
change the `--max-concurrency`
