"""Modal 部署 Qwen3-ASR realtime（含无限重复修复）

使用方法:
    1. modal setup          # 绑定账号
    2. modal deploy deploy_modal.py   # 部署
    3. 拿到公网 URL，就能测试了

测试:
    pip install websockets pybase64
    python examples/speech_to_text/realtime/openai_realtime_client.py \
        --model Qwen3ASRRealtimeGeneration --host <url> --port 443
"""

import modal

MODEL_ID = "Qwen/Qwen3-ASR-1.5B"

# 镜像：安装 Python 包 + 下载模型
image = (
    modal.Image.debian_slim(python_version="3.12")
    # 从你的 fork 源码安装修复版 vllm（含补丁）
    .pip_install(
        "git+https://github.com/huangazazaz/vllm_fix_asr.git@fix/qwen3-asr-realtime-reset-context"
    )
    # 下载模型到镜像缓存，启动时直接加载
    .pip_install("huggingface_hub[hf_transfer]")
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1"})
    .run_commands(
        f"python -c \"from huggingface_hub import snapshot_download; "
        f"snapshot_download('{MODEL_ID}')\""
        " && echo 'Model cached'"
    )
)

app = modal.App("qwen3-asr", image=image)


@app.function(gpu="T4", timeout=3600, container_idle_timeout=300)
@modal.asgi_app()
def serve():
    from vllm.entrypoints.openai.api_server import build_app

    return build_app(
        model=MODEL_ID,
        task="realtime",
        enforce_eager=True,
        max_model_len=4096,
    )
