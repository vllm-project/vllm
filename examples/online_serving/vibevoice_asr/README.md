# VibeVoice-ASR on vLLM (OpenAI server)

这一套是把微软 **VibeVoice-ASR** 通过 `vllm.general_plugins` 插件接入 vLLM，并封装成
“一键准备模型 + 一键启动 OpenAI server”的最小工作流（本目录内脚本即可）。

## 0. 前置条件

- 已有可用的 vLLM 环境（示例默认 conda env: `vllm-src`）
- 安装 VibeVoice 插件（提供 `vllm_plugin` 模块）：

```bash
python -m pip install -U --no-deps "vibevoice[vllm] @ https://github.com/microsoft/VibeVoice/archive/refs/heads/main.zip"
python -m pip install -U diffusers
```

## 1. 一键启动（自动准备模型）

默认下载/准备模型到 `/data2/mayufeng/models/VibeVoice-ASR`，并在 GPU6 端口 8006 启动：

```bash
bash examples/online_serving/vibevoice_asr/start_vibevoice_asr_server.sh \
  --gpu 6 \
  --port 8006 \
  --conda-env vllm-src \
  --model-dir /data2/mayufeng/models/VibeVoice-ASR \
  --allowed-local-media-path /home/mayufeng/projects
```

说明：

- 脚本默认 `--served-model-name vibevoice`（与 VibeVoice 官方 `vllm_plugin/tests/test_api.py` 保持一致）。
- 默认参数基本对齐 VibeVoice 官方 `vllm_plugin/scripts/start_server.py`（长音频/高吞吐场景更友好）：
    - `--dtype bfloat16`
    - `--max-model-len 65536`
    - `--max-num-seqs 64`
    - `--max-num-batched-tokens 32768`
    - `--enforce-eager`
    - `--no-enable-prefix-caching`
    - `--enable-chunked-prefill`
    - `--trust-remote-code`

另外，脚本仍然默认强制 `--attention-backend TRITON_ATTN`（更稳），如果你确认 `flashinfer` 后端在本机可用，
可以用 `--attention-backend` 切回更快的后端。

脚本会自动做这些事情：

- `snapshot_download` 拉取 `microsoft/VibeVoice-ASR` 到 `--model-dir`
- 运行 `python -m vllm_plugin.tools.generate_tokenizer_files`
- （如有需要）自动修补 `tokenizer_config.json`：删除 `tokenizer_class`（保留 `.bak` 备份）
- 启动 `vllm serve`（OpenAI server），默认强制使用本目录的 `chat_template.jinja`（兼容 OpenAI multi-part `content=[{...}, {...}]`）
- 额外设置环境变量：
    - `VIBEVOICE_FFMPEG_MAX_CONCURRENCY`（默认 64）
    - `PYTORCH_ALLOC_CONF`（默认 `expandable_segments:True`）

## 2. OpenAI 请求示例

### 2.1 使用官方 streaming 测试脚本（推荐）

VibeVoice 插件自带一个 streaming client（支持 wav/mp3/flac/视频等，自动转 data: URL）：

```bash
conda run -n vllm-src python -m vllm_plugin.tests.test_api \
  /home/mayufeng/projects/sample.wav \
  --url http://localhost:8006
```

#### 2.1.1 推荐解码参数（抑制 n-gram 复读）

我们在长音频（~3min）上实测发现，**仅设置 `repetition_penalty` 往往不足以抑制尾部循环复读**，
配合 `frequency_penalty` 效果更明显。推荐先用这组（ASR 场景友好、能显著降低重复）：

```json
{
  "temperature": 0,
  "repetition_penalty": 1.1,
  "frequency_penalty": 0.5,
  "presence_penalty": 0.0
}
```

说明：

- 这是 **OpenAI 请求参数**（`/v1/chat/completions` payload 字段），需要由 client / proxy 注入。
- 如果你希望“上游请求传进来的这些字段不生效”，建议在上游 proxy（例如 LiteLLM）里做参数白名单/覆盖（见下文 2.4）。

说明：首次启动时 vLLM 可能需要较长时间做 torch.compile / CUDA graph warmup；该脚本默认会
先轮询 `/health` 最多 120 秒（可用 `--wait-ready-secs` 调整/关闭）。

### 2.2 使用本地文件（file://）

启动 server 时必须设置 `--allowed-local-media-path`，且音频文件必须位于该目录下。

```bash
conda run -n vllm-src python examples/online_serving/vibevoice_asr/smoke_test_client.py \
  --base-url http://localhost:8006 \
  --audio /home/mayufeng/projects/sample.wav \
  --model vibevoice \
  --prompt "Transcribe the audio."
```

### 2.3 使用 data: URL（不依赖 allowed-local-media-path）

也可以把音频转 base64 data URL（会更大，适合小文件验证）。vLLM 的 multimodal loader
支持 `data:audio/wav;base64,...`。

### 2.4 接入 LiteLLM 并“固定”解码参数（可选）

如果你是通过 LiteLLM 转发到 vLLM，并希望 **VibeVoice-ASR 始终使用固定的解码参数**（上游请求里即便传了
`temperature/repetition_penalty/frequency_penalty/presence_penalty` 也不生效），通常有两种做法：

1) **在 LiteLLM 侧做覆盖（推荐）**：在转发前把这些字段强制写成固定值，或直接丢弃上游同名字段。
2) **单独加一层轻量 proxy**：在 vLLM 前面起一个很薄的 HTTP 转发服务，统一注入固定参数。

#### 2.4.1 LiteLLM middleware（async_pre_call_hook）示例

本目录提供了一个可直接复用的 LiteLLM Proxy 回调（等价于“middleware”）：

- `examples/online_serving/vibevoice_asr/litellm/custom_callbacks.py`

它会在请求转发前强制覆盖（忽略上游传入）：

- `temperature=0`
- `repetition_penalty=1.1`
- `frequency_penalty=0.5`
- `presence_penalty=0.0`

LiteLLM `config.yaml` 示例（你需要确保 LiteLLM 启动时能 import 到该 module；最简单是把
`custom_callbacks.py` 复制/挂载到 LiteLLM 的工作目录并按实际 module 名修改）：

```yaml
model_list:
  - model_name: vibevoice-asr
    litellm_params:
      model: hosted_vllm/vibevoice
      api_base: http://127.0.0.1:8006/v1
      api_key: ""

litellm_settings:
  callbacks:
    - examples.online_serving.vibevoice_asr.litellm.custom_callbacks.proxy_handler_instance
```

说明：如果你是用 docker 跑 LiteLLM，通常需要 `-v` 把该回调文件挂进去，并设置 `PYTHONPATH`
（或直接放到 `config.yaml` 同目录，改成 `custom_callbacks.proxy_handler_instance`）。

## 3. 备注：为什么默认强制 `--chat-template`

VibeVoice 的 `VibeVoiceASRTextTokenizerFast` 内置 `chat_template` 只支持 string 形式的 message
content；当你用 OpenAI 多模态请求（`content=[{...}, {...}]` 的 list 结构）时，会触发：

`TypeError: can only concatenate str (not "list") to str`

因此本目录的启动脚本默认强制 `--chat-template examples/online_serving/vibevoice_asr/chat_template.jinja`
来保证兼容 OpenAI multi-part `content` 格式。

如果你有自定义模板需求，可以用 `--chat-template` 指向你自己的 jinja 文件覆盖。
