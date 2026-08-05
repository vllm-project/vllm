# AMD NPU Vision Validation (Qwen2.5-VL)

This guide describes how to validate the experimental **NPU vision tower** support for
Qwen2.5-VL on AMD Ryzen AI NPUs (STX / KRK). The LLM still runs on GPU (ROCm); only the
vision encoder is offloaded to the NPU via FlexMLRT.

## Overview

| Component | Runtime |
| --------- | ------- |
| Qwen2.5-VL language model | GPU (ROCm) |
| Vision tower (when enabled) | AMD NPU (FlexMLRT) |
| CPU preprocessing | Host CPU (PyTorch, from stitched ONNX) |

Enable NPU vision by setting `VLLM_VISION_NPU_CACHE` to the path of a compiled
`.rai` cache file. When unset, vLLM uses the default PyTorch GPU vision path.

## Prerequisites

### Hardware and OS

Follow the [Ryzen AI Linux installation guide](https://ryzenai.docs.amd.com/en/latest/linux.html):

- Ubuntu 24.04 LTS
- Kernel >= 6.10
- Python 3.12.x
- Supported platform: **STX** or **KRK** (Strix / Krackan)

### NPU drivers and XRT

1. Download and install the NPU driver packages from the
   [Ryzen AI Software Drivers](https://account.amd.com/en/forms/downloads/xef.html?filename=RAI%5F1.7.1%5FLinux%5FNPU%5FXRT.zip)
   page (see the Linux guide for package names).
2. Set up the environment:

```bash
export LD_LIBRARY_PATH=/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH
source /opt/xilinx/xrt/setup.sh
```

#### Bypass `amdxdna` execution timeout (recommended)

To bypass the default `amdxdna` driver execution timeout (which can interfere with
longer NPU workloads under XRT), reload the kernel module with `timeout_in_sec=0`:

```bash
sudo rmmod amdxdna
sudo modprobe amdxdna timeout_in_sec=0
```

#### Verify the NPU device

```bash
xrt-smi examine
```

You should see an NPU device (e.g. `NPU Strix`).

### Ryzen AI software (FlexMLRT)

Install Ryzen AI 1.7.1 from the
[Linux installation guide](https://ryzenai.docs.amd.com/en/latest/linux.html):

```bash
# After downloading ryzen_ai-1.7.1.tgz
./install_ryzen_ai.sh -a yes -p <TARGET-PATH>/venv
source <TARGET-PATH>/venv/bin/activate
echo $RYZEN_AI_INSTALLATION_PATH
```

Optional sanity check using the bundled quicktest:

```bash
cd <TARGET-PATH>/venv/quicktest
python quicktest.py
```

Locate `libflexmlrt.so` under your Ryzen AI installation (or a local FlexMLRT build).
You will need this library on `LD_LIBRARY_PATH` when running vLLM.

## Model cache (Hugging Face)

Download the precompiled vision cache:

**Repo:** [lichang55245/Qwen2.5-Vision-Tower-AMD-NPU-model-cache](https://huggingface.co/lichang55245/Qwen2.5-Vision-Tower-AMD-NPU-model-cache)

The repo contains (same directory):

| File | Purpose |
| ---- | ------- |
| `qwen2_5_vl_vision_stitched_7b.rai` | Compiled NPU cache (FlexMLRT input) |
| `qwen2_5_vl_vision_stitched_7b.onnx` | ONNX model |
| `qwen2_5_vl_vision_stitched_7b.onnx.data` | ONNX weights |

```bash
pip install huggingface_hub

huggingface-cli download lichang55245/Qwen2.5-Vision-Tower-AMD-NPU-model-cache \
  --local-dir ./qwen2_5_vl_npu_cache

export VLLM_VISION_NPU_CACHE="$(pwd)/qwen2_5_vl_npu_cache/qwen2_5_vl_vision_stitched_7b.rai"
```

Keep the `.rai` and `.onnx` files in the **same directory** (as published on Hugging Face).

## Build the vLLM NPU bridge extension

The pybind module `_vision_flexmlrt_npu` is not built by default. After checking out the
`npu-vision-support` branch:

```bash
cd vllm/vision_npu/bridge
rm -rf build && mkdir build && cd build

cmake .. \
  -DFLEXMLRT_INCLUDE_DIR=<path-to-flexmlRT>/include \
  -DFLEXMLRT_LIB_DIR=<path-to-flexmlRT>/lib \
  -Dpybind11_DIR=$(python -m pybind11 --cmakedir) \
  -DPython_EXECUTABLE=$(which python)

make -j$(nproc)
make install
```

This installs `_vision_flexmlrt_npu.cpython-*-linux-gnu.so` into `vllm/vision_npu/`.

Install vLLM in your environment (editable or from source) so Python picks up the
extension and updated `vision_npu` package.

## XRT configuration (`xrt.ini`)

NPU initialization requires `XRT_INI_PATH` pointing at an INI file. Create
`xrt.ini` (any path) with the following content used for Qwen2.5-VL NPU vision
validation on STX:

```ini
[Debug]
num_heap_pages=8
```

Save the file and export its path:

```bash
# Example: save next to your model cache
cat > ./xrt.ini <<'EOF'
[Debug]
num_heap_pages=8
EOF

export XRT_INI_PATH="$(pwd)/xrt.ini"
```

`num_heap_pages` controls NPU heap allocation for FlexMLRT; `8` is the value
used in our STX validation runs.

## Environment variables

| Variable | Required | Description |
| -------- | -------- | ----------- |
| `VLLM_VISION_NPU_CACHE` | Yes (to enable NPU vision) | Absolute path to `*.rai` file |
| `VLLM_VISION_NPU_DEVICE` | No | NPU device name (default: `stx`) |
| `XRT_INI_PATH` | Yes | Path to `xrt.ini` (see above) |
| `LD_LIBRARY_PATH` | Yes | Must include directory containing `libflexmlrt.so` |

Example:

```bash
export VLLM_VISION_NPU_CACHE=/path/to/qwen2_5_vl_npu_cache/qwen2_5_vl_vision_stitched_7b.rai
export VLLM_VISION_NPU_DEVICE=stx
export XRT_INI_PATH=/path/to/xrt.ini
export LD_LIBRARY_PATH=<flexmlRT-lib-dir>:$LD_LIBRARY_PATH
```

To disable NPU vision and fall back to GPU PyTorch vision, unset `VLLM_VISION_NPU_CACHE`.

Optional debug:

```bash
export VLLM_NPU_TIMING=1          # log NPU vision timing
export VLLM_LOGGING_LEVEL=DEBUG   # FlexMLRT bridge debug on stderr
```

## Validation steps

### 1. Driver and library check

```bash
xrt-smi examine
python -c "from vllm.vision_npu._vision_flexmlrt_npu import VisionFlexMLRTModel; print('bridge OK')"
```

If the import fails with `libflexmlrt.so: cannot open shared object file`, fix
`LD_LIBRARY_PATH`.

### 2. Start vLLM with NPU vision

Use any Qwen2.5-VL checkpoint compatible with your GPU setup. Example:

```bash
python -m vllm.entrypoints.openai.api_server \
  --model <path-to-Qwen2.5-VL-7B-Instruct> \
  --dtype bfloat16 \
  --trust-remote-code \
  --limit-mm-per-prompt '{"image": 1}' \
  --skip-mm-profiling \
  --port 8000
```

Confirm startup logs mention NPU vision, for example:

- `[Qwen2.5VL] Using NPU vision backend`
- `[FlexMLRT Backend] Loaded RAI cache ...`

### 3. Image request smoke test

```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "<your-model-name>",
    "messages": [{
      "role": "user",
      "content": [
        {"type": "image_url", "image_url": {"url": "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-VL/assets/demo.jpeg"}},
        {"type": "text", "text": "Describe this image."}
      ]
    }],
    "max_tokens": 64
  }'
```

A successful response with coherent image description indicates end-to-end NPU vision +
GPU LLM inference.

### 4. Compare with GPU-only baseline (optional)

Restart the server **without** `VLLM_VISION_NPU_CACHE` and run the same request. Outputs
should be qualitatively similar (small numeric differences in vision embeddings are
expected between NPU and GPU paths).

## Troubleshooting

| Symptom | Likely fix |
| ------- | ---------- |
| `libflexmlrt.so: cannot open shared object file` | Add FlexMLRT `lib` dir to `LD_LIBRARY_PATH` |
| `DRM_IOCTL_AMDXDNA_CREATE_BO ... Invalid argument` | Set `XRT_INI_PATH` to an `xrt.ini` with `[Debug] num_heap_pages=8`; verify NPU drivers with `xrt-smi` |
| `VLLM_VISION_NPU_CACHE must point to a .rai file` | Point env var at the `.rai` file, not `vaiml_par_0/` |
| `Cannot find ONNX model ... near RAI bundle` | Keep `.onnx` / `.onnx.data` next to the `.rai` (HF layout) |
| `FlexMLRT vision model creation failed` | Check subgraph name is `vaiml_par_0` (built into bridge); verify `.rai` matches Qwen2.5-VL 7B stitched vision |
| ImportError for `_vision_flexmlrt_npu` | Rebuild the bridge extension (see above) |

## Scope and limitations

- **PR scope:** NPU vision encoder only; no async NPU/GPU pipelining in this branch.
- **Model:** Validated with Qwen2.5-VL 7B stitched vision cache; other models are unsupported.
- **Video:** NPU path applies to image encoding; video still uses the GPU vision path.
- **Subgraph name:** Hardcoded to `vaiml_par_0` for the published cache.

## References

- [Ryzen AI Linux installation](https://ryzenai.docs.amd.com/en/latest/linux.html)
- [Qwen2.5-VL NPU vision cache (Hugging Face)](https://huggingface.co/lichang55245/Qwen2.5-Vision-Tower-AMD-NPU-model-cache)
- FlexMLRT RAI loading pattern: `xmc/src/voe/flexmlRT/test/generic/test.cpp` (`-r` / `-S vaiml_par_0`)
