# Serve Qwen3-Omni on four Ascend 910B NPUs

Serve the BF16 weights of [Qwen3-Omni-30B-A3B-Instruct](https://huggingface.co/Qwen/Qwen3-Omni-30B-A3B-Instruct) with `vllm serve` and receive text answers to text, image, audio and video inputs. The example uses tensor parallelism across four NPUs on one host and enables expert parallelism for the MoE layers.

## Prepare the runtime and checkpoint

The reference deployment used devices 0 through 3 on an eight-device Ascend 910B4-1 host (64 GiB HBM per device). It ran the Linux AArch64 image `quay.io/ascend/vllm-omni:v0.30.0` with the following software:

| Component | Tested version |
| --- | --- |
| vLLM | `0.30.0+empty` |
| vLLM Ascend | `0.19.1rc2.dev2331+g16074608c` |
| CANN | `9.1.0` |
| PyTorch / torch-npu | `2.10.0+cpu` / `2.10.0.post4` |
| Transformers | `5.14.1` |

## Pull the image and start the container

Run these commands on a Linux AArch64 host with the Ascend driver installed and the `ascend` runtime registered with Docker. Check the devices with `npu-smi info`. The [upstream NPU installation guide](https://docs.vllm.ai/projects/vllm-omni/en/latest/getting_started/installation/npu/) lists the images by hardware family.

```bash
docker pull --platform linux/arm64 quay.io/ascend/vllm-omni:v0.30.0
docker image inspect --format '{{.Id}}' quay.io/ascend/vllm-omni:v0.30.0
```

The tested arm64 image ID is `sha256:96c8901953dbe758199d97795279762131b58faadb02fa593289ea01e3e68dbd`. Set `CHECKPOINT` to the absolute host path containing the full Instruct checkpoint. The container mounts it read-only. This checkpoint declares BF16 and loads with `quantization=None`.

```bash
export CHECKPOINT=/absolute/path/to/Qwen3-Omni-30B-A3B-Instruct
docker run --rm -it --name qwen3-omni-910b \
  --runtime=ascend --privileged --network=host --shm-size=16g \
  -e ASCEND_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
  -e ASCEND_RT_VISIBLE_DEVICES=0,1,2,3 \
  --mount "type=bind,src=${CHECKPOINT},dst=/models/Qwen3-Omni-30B-A3B-Instruct,readonly" \
  --entrypoint /bin/bash quay.io/ascend/vllm-omni:v0.30.0
```

The reference host granted the container privileged access to all eight devices and selected four for the engine. Adapt both device lists to your allocation. Host networking lets a client on the same host reach the loopback address used below.

## Start the server

Run the following in the serving container. The environment block records the values used in the tested deployment. The cache path must be writable by the server process.

```bash
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3
export CPU_AFFINITY_CONF=1
export HCCL_CONNECT_TIMEOUT=7200
export HCCL_EXEC_TIMEOUT=0
export HCCL_OP_EXPANSION_MODE=AIV
export HCCL_RDMA_PCIE_DIRECT_POST_NOSTRICT=TRUE
export OMP_NUM_THREADS=1
export PYTHONFAULTHANDLER=1
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export RUNAI_STREAMER_LOG_LEVEL=INFO
export RUNAI_STREAMER_LOG_TO_STDERR=1
export RUNAI_STREAMER_MEMORY_LIMIT=0
export SAFETENSORS_FAST_GPU=1
export TASK_QUEUE_ENABLE=1
export VLLM_CACHE_ROOT=/tmp/vllm-cache
export VLLM_PLUGINS=ascend

vllm serve /models/Qwen3-Omni-30B-A3B-Instruct \
  --disable-access-log-for-endpoints /metrics \
  --enable-prompt-tokens-details \
  --trust-remote-code \
  --max-num-seqs=100 \
  --max-model-len=40960 \
  --max-num-batched-tokens=16384 \
  --tensor-parallel-size=4 \
  --enable-expert-parallel \
  --distributed_executor_backend mp \
  --no-enable-prefix-caching \
  --async-scheduling \
  --compilation-config '{"cudagraph_mode": "FULL_DECODE_ONLY"}' \
  --gpu-memory-utilization=0.95 \
  --additional-config '{"enable_flashcomm1": false, "weight_nz_mode": 2}' \
  --host 127.0.0.1 --port 40000 \
  --served-model-name qwen3-omni-30b
```

The tested plugin reported that `enable_flashcomm1` was deprecated. The value above records the setting used with this build. The engine log should show `Qwen3OmniMoeForConditionalGeneration`, `dtype=torch.bfloat16`, `quantization=None` and a world size of four.

## Verify text and audio inputs

Run the client on the Docker host or open a second container shell with `docker exec -it qwen3-omni-910b bash`. Once `/health` succeeds, send a short chat request.

```bash
curl -fsS http://127.0.0.1:40000/health
curl -fsS http://127.0.0.1:40000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"qwen3-omni-30b","messages":[{"role":"user","content":"Reply with the word ready."}],"max_tokens":16}'
```

To verify that audio reaches the model, use a short WAV containing a known sentence. This example downloads a French speech sample from Qwen. You can also supply your own recording at `asr_fr.wav`. The request embeds the file so the inference server needs no outbound media access.

```bash
curl -fL -o asr_fr.wav https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen3-Omni/cookbook/asr_fr.wav
python3 - <<'PY'
import base64
import json
from pathlib import Path
from urllib.request import Request, urlopen

audio = base64.b64encode(Path("asr_fr.wav").read_bytes()).decode("ascii")
content = [
    {"type": "audio_url", "audio_url": {"url": "data:audio/wav;base64," + audio}},
    {"type": "text", "text": "Transcribe the recording in its original language."},
]
body = {"model": "qwen3-omni-30b", "messages": [{"role": "user", "content": content}],
        "temperature": 0, "max_tokens": 256}
request = Request("http://127.0.0.1:40000/v1/chat/completions",
                  data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
with urlopen(request, timeout=90) as response:
    result = json.load(response)
print(result["choices"][0]["message"]["content"])
print(result["usage"]["prompt_tokens_details"])
PY
```

In the reference run, the transcript began with “Bien, on va passer à des questions un peu plus générales” and the direct response reported 115 audio tokens. For video, a six-second clip with successive blue, yellow and green frames produced the three colors in order. See [multimodal inputs](../features/multimodal_inputs.md) for the image and video request formats. These checks exercise text output.

## Troubleshooting

- **Architecture inspection aborts.** In the tested image, loading the default set of plugins led to a registry subprocess abort with `corrupted size vs. prev_size`. Setting `VLLM_PLUGINS=ascend` allowed architecture inspection to complete. This comparison narrowed the failure to the plugin configuration without identifying an individual faulty plugin.
- **Unknown quantization method: ascend.** With that plugin selection, passing `--quantization=ascend` failed validation. Use the unquantized BF16 checkpoint and the command above.
- **An API gateway returns HTTP 200 and asks for the recording.** Send the same request directly to vLLM. Compare the transcript and audio token count, then check whether the gateway forwards the complete `audio_url` content part. Apply the same comparison to `video_url` when a video answer misses known frames.
