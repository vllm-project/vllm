# AutoDL RTX 4090 实例 vLLM 开发环境搭建记录

> 日期：2026-08-27（v2，含完整踩坑修正）
> 机器：AutoDL 容器实例 / RTX 4090 24G / 系统盘 30G + 数据盘 50G (`/root/autodl-tmp`)
> vLLM 基准 commit：`fd57c4b7afebc0b43d25ed7f5848fc35786463d0`

## 一、目录规划

| 路径 | 用途 |
|------|------|
| `/root/autodl-tmp/repos/vllm` | vLLM 源码（可编辑安装） |
| `/root/autodl-tmp/models/hub` | HF 模型缓存（`HF_HOME`） |
| `/root/autodl-tmp/wheels` | 手动下载的预编译 wheel |
| `/root/autodl-tmp/datasets` | 测试数据集 |
| `/root/autodl-tmp/outputs` | 安装日志、benchmark 结果 |
| 系统盘 conda env `vllm-dev` | Python 3.12 环境（随镜像保存） |

## 二、搭建步骤（经实测修正的顺序）

### 1. Git 与 SSH 配置

```bash
git config --global user.name "emprorsky"
git config --global user.email "704813907@qq.com"
ssh-keygen -t ed25519 -C "seetacloud-vllm"    # 一路回车
cat ~/.ssh/id_ed25519.pub                      # 公钥添加到 GitHub Settings → SSH keys
ssh -T git@github.com                          # 验证，出现 Hi emprorsky! 即成功
```

### 2. 克隆仓库（fetch 走 HTTPS + 加速，push 走 SSH）

```bash
source /etc/network_turbo
mkdir -p /root/autodl-tmp/repos
cd /root/autodl-tmp/repos
git clone https://github.com/emprorsky/vllm_xxy.git vllm
cd vllm
git remote set-url --push origin git@github.com:emprorsky/vllm_xxy.git
git remote add upstream https://github.com/vllm-project/vllm.git
git switch -c project/kv-aware-scheduling
unset http_proxy https_proxy    # 关学术加速
```

### 3. conda 环境与 .bashrc

```bash
conda create -n vllm-dev python=3.12 -y
```

`~/.bashrc` 末尾追加（**完整版，含 Xet 禁用，见卡点 5**）：

```bash
# ==== vLLM 开发环境配置 ====
export HF_ENDPOINT=https://hf-mirror.com
export HF_HOME=/root/autodl-tmp/models/hub
export HF_HUB_DISABLE_XET=1    # Xet CDN 国内 401，禁用走普通 HTTP
alias turbo='source /etc/network_turbo'
source /root/miniconda3/etc/profile.d/conda.sh
```

### 4. 安装构建依赖（含 torch 2.13.0+cu130，约 3GB）

```bash
source /root/miniconda3/etc/profile.d/conda.sh && conda activate vllm-dev
cd /root/autodl-tmp/repos/vllm
pip install -r requirements/build/cuda.txt
```

### 5. 安装 vLLM（预编译模式）——推荐直接用"本地 wheel"方式

v2 修正：原文档写的 `pip install -e .` 直接跑会卡（见卡点 2），**推荐直接走下面三步**：

```bash
# (1) 获取当前 commit 对应的预编译 wheel 文件名
cd /root/autodl-tmp/repos/vllm
COMMIT=$(git rev-parse HEAD)
curl -s "https://wheels.vllm.ai/${COMMIT}/vllm/metadata.json" | python3 -c "
import json,sys
for w in json.load(sys.stdin):
    if 'x86_64' in w['filename']:
        print(w['filename'])
"

# (2) 手动下载（wget 有进度条+断点续传，直连 S3 约 13MB/s）
mkdir -p /root/autodl-tmp/wheels
cd /root/autodl-tmp/wheels
wget -c "https://wheels.vllm.ai/${COMMIT}/<第1步输出的文件名>"

# (3) 用本地 wheel 安装（跳过 pip 内部下载）
cd /root/autodl-tmp/repos/vllm
export VLLM_USE_PRECOMPILED=1
export VLLM_PRECOMPILED_WHEEL_LOCATION="/root/autodl-tmp/wheels/<文件名>"
pip install -e . --no-build-isolation
```

> 实际下载的 wheel：`vllm-0.28.1rc1.dev1+gfd57c4b7a-cp38-abi3-manylinux_2_28_x86_64.whl`（316MB）
> URL 中 `+` 字符需编码为 `%2B`。

## 三、卡点与解决方法（按遇到顺序）

### 卡点 1：`pip install -e .` 报 `ModuleNotFoundError: No module named 'torch'`

**原因**：`setup.py` 在生成元数据阶段就要 import torch，但 `--no-build-isolation` 模式下依赖当前环境。

**解决**：先装构建依赖再装 vllm（Step 4 → Step 5 的顺序，不能颠倒）。

### 卡点 2：`Building editable for vllm (pyproject.toml) ... -` 长时间卡住 ⭐核心卡点

**现象**：PyPI 依赖全部下完（阿里云源速度正常），卡在 `Building editable for vllm (pyproject.toml) ...` 后只有一个转动的光标，几十分钟不动。

**原因**：预编译模式会从 `wheels.vllm.ai`（S3/CloudFront）下载约 316MB 的预编译 wheel。**pip 内部用 `urlopen` 下载，无进度条、无超时重试**，网络稍慢就像挂死。

**解决**：见上面 Step 5 的三步法——`wget -c` 手动下载 + `VLLM_PRECOMPILED_WHEEL_LOCATION` 环境变量喂给 pip，彻底绕开 pip 内部下载。

### 卡点 3：Agent/后台跑 pip 下载反而变慢

**现象**：通过 Agent 后台（`nohup`/异步）执行 `pip install` 时，部分包（opencv 等）下载只有 0.2-0.7MB/s，比前台手动跑慢几十倍，且容易中途无声退出（日志停在半截下载）。

**原因**：后台 shell 环境变量与网络路由和交互终端不完全一致，pip 源配置可能没生效。

**解决**：**大文件下载一律在前台交互终端手动跑**（有进度条）。若必须后台，用 `nohup pip install ... > /root/autodl-tmp/outputs/xxx.log 2>&1 &` + `tail -f` 盯日志，且要检查最终是否出现 `Successfully installed`——**日志停在半截 ≠ 装完**。

### 卡点 4：conda activate 报错

**现象**：`CondaError: Run 'conda init' before 'conda activate'`

**解决**：先 `source /root/miniconda3/etc/profile.d/conda.sh` 再 activate（已写入 `.bashrc`，新终端无此问题）。

### 卡点 5：模型下载 401 Unauthorized（Xet 协议）⭐高频坑

**现象**：vllm 启动时小文件（config.json 等）下载正常，但权重文件失败，报：

```
RuntimeError: Task error: File reconstruction error: CAS Client Error:
Request error: HTTP status client error (401 Unauthorized),
domain: https://cas-server.xethub.hf.co/...
```

**原因**：HuggingFace 新版 `huggingface_hub` 默认用 **Xet 协议**下载大文件，但 Xet 的 CDN（`cas-server.xethub.hf.co`）不走 hf-mirror 镜像、需要认证，国内直连必挂 401。小文件走普通 HTTP 走镜像成功，大文件走 Xet 失败。

**解决**：

```bash
export HF_HUB_DISABLE_XET=1    # 已写入 .bashrc；当前终端需手动 export
```

禁用后走普通 HTTP，hf-mirror 即可正常代理下载。

### 卡点 6：模型下到了系统盘 ~/.cache/huggingface

**现象**：模型缓存出现在 `~/.cache/huggingface`（系统盘）而非数据盘。

**原因**：启动 vllm 的终端没加载 `.bashrc`（比如通过 IDE/脚本启动），`HF_HOME` 未生效，回退到默认 `~/.cache`。

**解决**：启动前确认 `echo $HF_HOME` 输出 `/root/autodl-tmp/models/hub`；不对就手动 `export HF_HOME=...`。下错了就 `rm -rf ~/.cache/huggingface/hub/models--*` 清理。

### 卡点 7：安装中途 vllm 启动失败（EngineCore failed）

**现象**：`RuntimeError: Engine core initialization failed` + Traceback 里 `ModuleNotFoundError: No module named 'accelerate'`。

**原因**：pip 安装未完成就尝试启动 vllm（后台 pip 中途死了），缺 `accelerate` 等运行时依赖。

**解决**：确认 pip 安装完整结束（日志出现 `Successfully installed`）后再启动。补漏：`pip install accelerate`。

### 卡点 8：下载权重时终端长时间无输出（假死）

**现象**：日志停在 `Using FlashAttention version 2` 后不动。

**原因**：不是卡死——并发下载 4 个 safetensors 文件时终端不刷新，进度条要等当前批次完成才显示。

**确认方法**：

```bash
du -sh /root/autodl-tmp/models/hub/hub/models--Qwen--Qwen2.5-7B-Instruct   # 看数字是否增长
watch -n 5 'du -sh /root/autodl-tmp/models/hub/hub/models--Qwen--Qwen2.5-7B-Instruct'
```

数字在涨就是正常下载。Qwen2.5-7B 全部权重约 15GB。

## 四、验证

```bash
conda activate vllm-dev
python -c "import vllm; print(vllm.__version__)"
# 输出: 0.1.dev20460+gfd57c4b7a
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
# 输出: 2.13.0+cu130 True
python -c "import accelerate; print(accelerate.__version__)"
# 输出: 1.14.0
```

## 五、日常工作流

- **改 Python 代码**（调度器、frontend 等绝大多数场景）：直接改 `/root/autodl-tmp/repos/vllm/vllm/` 下源码，即时生效
- **改 C++/CUDA 代码**：需要重新构建，`pip install -e . --no-build-isolation` 或针对性 rebuild
- **同步上游**：`git fetch upstream && git rebase upstream/main`（先 `turbo` 开加速）
- **下载模型**：直接用模型名（如 `Qwen/Qwen2.5-7B-Instruct`），HF_ENDPOINT 镜像 + Xet 禁用自动生效，缓存在数据盘
- **备选下载**（hf-mirror 慢时）：modscope 国内 CDN：
  ```bash
  pip install modelscope
  python -c "
  from modelscope import snapshot_download
  snapshot_download('Qwen/Qwen2.5-7B-Instruct', cache_dir='/root/autodl-tmp/models/ms')"
  # 启动时 --model /root/autodl-tmp/models/ms/Qwen/Qwen2.5-7B-Instruct
  ```
- **冒烟测试**：
  ```bash
  conda activate vllm-dev
  # 新终端已自动配好环境变量；旧终端需手动：
  export HF_ENDPOINT=https://hf-mirror.com HF_HOME=/root/autodl-tmp/models/hub HF_HUB_DISABLE_XET=1

  python -m vllm.entrypoints.openai.api_server \
      --model Qwen/Qwen2.5-7B-Instruct \
      --gpu-memory-utilization 0.9 --max-model-len 4096
  ```
  成功标志：`Uvicorn running on http://0.0.0.0:8000`
  测试：`curl http://localhost:8000/v1/models`

## 六、冒烟测试结果（2026-08-27 实测）

启动成功标志：日志出现 `Application startup complete`（路由列表之后）。

```bash
curl http://localhost:8000/v1/chat/completions -H "Content-Type: application/json" -d '{
  "model": "Qwen/Qwen2.5-7B-Instruct",
  "messages": [{"role": "user", "content": "你好，介绍一下你自己"}]
}'
```

### 快速评测基线（Qwen2.5-7B / 4090 / bf16 / max_tokens=256）

脚本：`/root/autodl-tmp/outputs/quick_bench.py`，结果存 `quick_bench.json`

| 并发 | 请求数 | TTFT 均值 | TTFT P99 | ITL 均值 | 输出吞吐 (tok/s) |
|------|-------|----------|----------|---------|----------------|
| 1 | 4 | 6.11s | 8.10s | 15.8ms | 62.8 |
| 4 | 8 | 2.12s | 4.19s | 16.1ms | 246.7 |
| 8 | 16 | 2.12s | 4.20s | 16.1ms | 491.9 |
| 16 | 16 | 0.07s | 0.07s | 16.5ms | 947.8 |

要点：
- 吞吐随并发近线性扩展（62.8 → 947.8 tok/s），continuous batching 工作正常
- 并发 16 时 TTFT 骤降到 0.07s——同 prompt 命中了 **prefix cache（KV 复用）**，正是 `project/kv-aware-scheduling` 分支要研究的方向
- 并发 1 的 6.1s TTFT 含 CUDA graph 预热，正常现象
- 改造调度代码后重跑该脚本即可对比基线

## 七、经验教训总结

1. **大下载不要走 Agent 后台**——环境不一致导致变慢甚至无声失败，一律前台手动跑
2. **pip 卡住先怀疑隐形大文件下载**——`Building editable` 卡住 = 在拉预编译 wheel，用 `VLLM_PRECOMPILED_WHEEL_LOCATION` + `wget -c` 绕开
3. **HF 下载 401 = Xet 协议坑**——`HF_HUB_DISABLE_XET=1` 一行解决，已固化到 `.bashrc`
4. **终端环境变量要主动验证**——IDE/脚本启动的 shell 不一定加载 `.bashrc`，关键变量（HF_HOME）先 `echo` 确认
5. **"日志不动"≠"卡死"**——并发下载时终端不刷新，用 `du -sh` 看磁盘增长判断
