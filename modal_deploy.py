"""
Modal 部署脚本: Qwen3-ASR-0.6B 语音识别服务 (预编译镜像方案)
基于 huangazazaz/vllm_fix_asr 分支

策略: 先安装官方 vLLM pip 包 (预编译 C++/CUDA 扩展, 秒装),
     再把 fork 的 ASR Python 代码覆盖上去, 跳过源码编译。

部署前: python download_model.py  (模型下载到 ./asr-0.6/)
部署:   modal deploy modal_deploy.py
"""

import os
import subprocess
import modal

# =============================================================================
# 配置
# =============================================================================

MODEL_ID = "Qwen/Qwen3-ASR-0.6B-hf"
MODEL_LOCAL_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "asr-0.6",
)
MODEL_REMOTE_PATH = "/models/Qwen3-ASR-0.6B-hf"

GPU_TYPE = "T4"
GPU_COUNT = 1
SCALEDOWN_WINDOW = 15 * 60

VLLM_FORK_REPO = "https://github.com/huangazazaz/vllm_fix_asr.git"

# =============================================================================
# 持久化卷
# =============================================================================

vllm_cache_vol = modal.Volume.from_name(
    "vllm-cache", create_if_missing=True
)

# =============================================================================
# 构建镜像: 官方 vLLM (预编译) + fork ASR 文件覆盖
# =============================================================================

# 仅在本地部署时校验模型目录存在 (容器内 MODAL_ENVIRONMENT 会有值)
if not os.environ.get("MODAL_ENVIRONMENT"):
    if not os.path.isdir(MODEL_LOCAL_DIR):
        raise FileNotFoundError(
            f"模型目录不存在: {MODEL_LOCAL_DIR}\n"
            f"请先运行: python download_model.py"
        )

vllm_image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.9.0-devel-ubuntu22.04",
        add_python="3.12",
    )
    .entrypoint([])
    .apt_install("git", "libsndfile1", "ffmpeg", "libsoxr-dev")
    # === 安装官方最新 vLLM (已有 qwen3_asr_realtime.py!) ===
    .run_commands(
        "pip install --upgrade pip",
        "pip install vllm[audio]",
    )
    # === 用 fork 覆盖官方 vLLM（包含 input_stream 修复） ===
    .run_commands(
        f"git clone --depth 1 --single-branch {VLLM_FORK_REPO} /opt/vllm-fork",
        # 清理缓存
        "find /usr/local/lib/python3.12/site-packages/vllm -name '*.pyc' -delete",
        "find /usr/local/lib/python3.12/site-packages/vllm -name '__pycache__' -type d -exec rm -rf {} + 2>/dev/null || true",
        # 覆盖 fork 所有文件
        "cp -rf /opt/vllm-fork/vllm/* /usr/local/lib/python3.12/site-packages/vllm/",
        # 额外音频依赖
        "pip install soundfile librosa soxr pyav 2>/dev/null || true",
    )
    # === 本地模型打包 ===
    .add_local_dir(
        local_path=MODEL_LOCAL_DIR,
        remote_path=MODEL_REMOTE_PATH,
    )
)

# =============================================================================
# 服务
# =============================================================================

app = modal.App("vllm-qwen3-asr")


@app.server(
    image=vllm_image,
    gpu=f"{GPU_TYPE}:{GPU_COUNT}",
    scaledown_window=SCALEDOWN_WINDOW,
    startup_timeout=600,
    port=8000,
    volumes={
        "/root/.cache/vllm": vllm_cache_vol,
    },
    unauthenticated=True,
)
class VllmAsrServer:
    """vLLM ASR 服务"""

    @modal.enter()
    def start(self):
        import json
        import os as _os

        # 验证模型文件
        assert _os.path.isdir(MODEL_REMOTE_PATH), (
            f"Model not found: {MODEL_REMOTE_PATH}"
        )

        # 修改 config.json: 用 Qwen3ASRRealtimeGeneration 替换基类架构
        config_path = f"{MODEL_REMOTE_PATH}/config.json"
        with open(config_path) as f:
            config = json.load(f)
        old_arch = config.get("architectures", [])
        config["architectures"] = ["Qwen3ASRRealtimeGeneration"]
        with open(config_path, "w") as f:
            json.dump(config, f, indent=2)
        print(f" Patched architectures: {old_arch} → Qwen3ASRRealtimeGeneration")

        print(f" Model: {MODEL_REMOTE_PATH}")
        print(f" GPU:   {GPU_TYPE} x {GPU_COUNT}")

        cmd = [
            "vllm", "serve",
            MODEL_REMOTE_PATH,
            "--served-model-name", MODEL_ID,
            "--host", "0.0.0.0",
            "--port", "8000",
            "--enforce-eager",
            "--gpu-memory-utilization", "0.90",
            "--max-model-len", "4096",
            "--max-num-seqs", "16",
            "--uvicorn-log-level", "info",
        ]

        print(f" CMD: {' '.join(cmd)}")
        self.process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        # 启动线程读取 vLLM 输出
        import threading
        self.output_lines = []
        def read_output():
            for line in self.process.stdout:
                self.output_lines.append(line)
                print(f" [vLLM] {line.rstrip()}")
        self.output_thread = threading.Thread(target=read_output, daemon=True)
        self.output_thread.start()

        print(" Waiting...")
        self._wait_ready(timeout=600)

    def _wait_ready(self, timeout: int = 600):
        import time
        import urllib.request
        import urllib.error

        start = time.time()
        while time.time() - start < timeout:
            if self.process.poll() is not None:
                print(" vLLM crashed! Last output:")
                for line in self.output_lines[-20:]:
                    print(f"   {line.rstrip()}")
                raise RuntimeError(
                    f"vLLM exited with code {self.process.returncode}"
                )
            try:
                req = urllib.request.Request("http://localhost:8000/health")
                with urllib.request.urlopen(req, timeout=5) as resp:
                    if resp.status == 200:
                        print(f" Ready in {time.time() - start:.0f}s")
                        return
            except (urllib.error.URLError, ConnectionRefusedError, TimeoutError):
                pass
            time.sleep(5)
        raise TimeoutError(f"Not ready within {timeout}s")

    @modal.exit()
    def stop(self):
        if hasattr(self, "process") and self.process:
            print(" Shutting down...")
            self.process.terminate()
            try:
                self.process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                self.process.kill()
            print(" Stopped.")


@app.local_entrypoint()
def main():
    print("=" * 60)
    print(" Qwen3-ASR-0.6B Modal 部署 (预编译方案)")
    print("=" * 60)
    print()
    print(" 策略: 官方 vLLM (预编译) + fork ASR 文件覆盖")
    print(f" 模型: {MODEL_LOCAL_DIR} → {MODEL_REMOTE_PATH}")
    print()
    print(" 部署: modal deploy modal_deploy.py")
    print("=" * 60)
