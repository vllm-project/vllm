# DoSA sparse indexer

DoSA accelerates long-context DSA prefill by screening keys with certified score
bounds and computing exact scores for the remaining candidates. It is disabled
by default. Enable it with `AttentionConfig.indexer_dosa`:

```bash
vllm serve MODEL_DIR \
    --tensor-parallel-size 8 --enable-expert-parallel \
    --kv-cache-dtype fp8 --max-num-batched-tokens 32768 \
    --attention-config '{"indexer_dosa": true}'
```

The optimized path supports SM100 GPUs with CUDA 13, FP8 index keys, 32 or 64
index heads, head dimension 128 and top-k 2048. It handles contiguous gathered
contexts from 131072 through 1048576 keys. Short contexts, decode, MXFP4,
context parallelism, speculative decoding and candidate-block selection use
the existing indexer backend. Model context limits are configured separately.
Query spans shorter than one physical DoSA slab also use the existing backend.

DoSA selects the exact top-k under the native SM100 MX/BF16 score. Queries and
keys, BF16-rounded weights, and UE8M0 key scales are the same as in the existing
backend. Scaled FP8 head dot products accumulate in FP32; per-head ReLU values
then round to BF16 before the native BF16 weighted accumulation and reduction.

The seed supplies a native-score threshold. Screening bounds account for native
rounding, and surviving keys receive the same native score before selection.
Capacity overflow and failed numerical admission use exhaustive recovery under
the same native score arithmetic. Equal-score ties may choose different valid
key IDs; matching scores does not guarantee identical generated text.

Scratch and private graphs are owned by the existing workspace manager and
included in model memory profiling.
Bootstrap and carry state belong to one invocation; unrelated requests do not
share selection hints.

## Benchmarks

End-to-end TTFT on eight SM100 GPUs, FP8 KV, TP8/EP8. Each value is stock / DoSA,
using the median of three formal requests after one warmup.

| Context | GLM-5.2 Q16K | GLM-5.2 Q32K | DeepSeek-V3.2 Q16K | DeepSeek-V3.2 Q32K |
| --- | ---: | ---: | ---: | ---: |
| 128K | 0.99× | 1.01× | 0.93× | 0.88× |
| 256K | 1.23× | 1.08× | — | — |
| 384K | 1.17× | 1.16× | 0.97× | 0.97× |
| 512K | 1.29× | 1.36× | 1.02× | 1.02× |
| 768K | 1.66× | 1.69× | 1.33× | 1.35× |
| ~1M | **2.18×** | **2.36×** | **1.80×** | **1.83×** |

TTFT uses SQLite 3.53.4 source prefixes; task accuracy uses 100 fixed
LongBench-v2 requests for GLM-5.2. DeepSeek results cover latency only;
requests above its 163840-token context are extrapolation tests.

The GLM-5.2 task benchmark uses 100 fixed LongBench-v2 requests with zero-shot,
greedy nonthinking answers, a 128-token output budget and Q16K. Stock scores
54/100 and DoSA 58/100; a separate stock A/A rerun scores 55/100 and does not
replace the original baseline. DeepSeek results in this submission cover TTFT
only, with no quality claim for extrapolated contexts.

## TTFT reproduction

<details>
<summary>TTFT reproduction: complete scripts and commands</summary>

Use this PR checkout installed in an existing virtual environment on eight SM100
GPUs (two four-GPU nodes), with CUDA 13. The reported runs use vLLM base
`fdc5fde9d786832ab638bc0c2eb2328b637ecd0a`, PyTorch `2.14.0+cu130`, and
FlashInfer `0.7.2`. Activate the installed environment on each host; the commands
below use it without installing or resolving another environment:

```bash
source /path/to/vllm/.venv/bin/activate
```

Save the two scripts below as `prepare_inputs.py` and `ttft_client.py` in the same
working directory. Run all client-side commands from that directory. The input
script reconstructs the original raw SQLite 3.53.4 source prefixes; it does not
apply a chat template or tokenize, shorten, or resample the benchmark requests.
Both arms use the same generated files.

**Save as `prepare_inputs.py`:**

```python
"""Reconstruct SQLite inputs and the reported model-specific TTFT plans."""

import argparse
import io
import json
from pathlib import Path
from urllib.request import urlopen
from zipfile import ZipFile

URL = "https://www.sqlite.org/2026/sqlite-amalgamation-3530400.zip"
PREFIXES = {
    "ds": (
        ("128k", 131072, 493014),
        ("256k", 262144, 981111),
        ("384k", 393216, 1401009),
        ("512k", 524288, 1802647),
        ("768k", 786432, 2675019),
        ("1m", 1048569, 3511994),
    ),
    "glm": (
        ("128k", 131072, 537893),
        ("256k", 262144, 1060634),
        ("384k", 393216, 1507846),
        ("512k", 524288, 1955848),
        ("768k", 786432, 2896538),
        ("1m", 1048448, 3765787),
    ),
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, help="Reuse SQLite 3.53.4 sqlite3.c")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.source is None:
        with (
            urlopen(URL) as response,
            ZipFile(io.BytesIO(response.read())) as archive,
            archive.open("sqlite-amalgamation-3530400/sqlite3.c") as source,
        ):
            text = io.TextIOWrapper(source, encoding="utf-8", newline=None).read()
    else:
        with args.source.open(encoding="utf-8", newline=None) as source:
            text = source.read()
    for model, prefixes in PREFIXES.items():
        directory = args.output_dir.resolve() / model
        directory.mkdir(parents=True, exist_ok=True)
        requests = []
        for name, tokens, end in prefixes:
            path = directory / f"{name}.txt"
            path.write_bytes(text[:end].encode("utf-8"))
            requests.append({"name": name, "input": str(path), "tokens": tokens})
            print(f"{model} {name}: {end} characters, {path}")
        reported = [
            item for item in requests
            if model != "ds" or item["name"] != "256k"
        ]
        (directory / "requests.json").write_text(json.dumps(reported, indent=2) + "\n")
        middle = [item for item in requests if item["name"] in ("384k", "768k")]
        (directory / "requests-mid.json").write_text(
            json.dumps(middle, indent=2) + "\n"
        )
        only256 = [item for item in requests if item["name"] == "256k"]
        (directory / "requests-256k.json").write_text(
            json.dumps(only256, indent=2) + "\n"
        )


if __name__ == "__main__":
    main()
```

Generate the twelve inputs once:

```bash
uv run --active --no-sync python prepare_inputs.py --output-dir inputs
```

The reported request plans are `inputs/ds/requests.json` (five lengths,
excluding 256K) and `inputs/glm/requests.json` (all six lengths). DeepSeek 256K
was cancelled and has no reported paired result. Its input and
`requests-256k.json` are retained as optional, unmeasured data. To run only 384K
and 768K, use `requests-mid.json`. Generated-input token counts are:

| Model | 128K | 256K | 384K | 512K | 768K | Near-1M |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| DeepSeek-V3.2 | 131072 | 262144 | 393216 | 524288 | 786432 | 1048569 |
| GLM-5.2-FP8 | 131072 | 262144 | 393216 | 524288 | 786432 | 1048448 |

**Start the server on both nodes.** Set `MODEL_DIR` to the corresponding local
checkpoint, `MASTER_ADDR` to node 0's address, and `NODE_RANK` to `0` or `1`.
Use `MODEL_KIND=ds` for DeepSeek-V3.2 or `MODEL_KIND=glm` for GLM-5.2-FP8. Each
node exposes its four GPUs. Run stock and DoSA serially on the same node pair.
The GLM compatibility settings below are identical in its two arms; DeepSeek
uses the native communication defaults. CUDA graph policy, native producer,
TP/EP execution, and the dense logits budget retain their defaults.

```bash
export MODEL_KIND=ds
export MODEL_DIR=/path/to/DeepSeek-V3.2-hf
export MASTER_ADDR=NODE_0_IP
export NODE_RANK=0
export CHUNK=16384
export DOSA=false
export CUDA_VISIBLE_DEVICES=0,1,2,3
export VLLM_SERVER_DEV_MODE=1
export VLLM_ALLOW_LONG_MAX_MODEL_LEN=1
unset NCCL_NVLS_ENABLE VLLM_ALLREDUCE_USE_FLASHINFER
unset VLLM_ALLREDUCE_USE_SYMM_MEM VLLM_USE_NCCL_SYMM_MEM

MODEL_ARGS=()
if [ "$MODEL_KIND" = glm ]; then
  export NCCL_NVLS_ENABLE=0
  export VLLM_ALLREDUCE_USE_FLASHINFER=0
  export VLLM_ALLREDUCE_USE_SYMM_MEM=0
  MODEL_ARGS=(--compilation-config '{"pass_config":{"fi_allreduce_fusion_max_size_mb":0}}')
fi
NODE_ARGS=()
if [ "$NODE_RANK" = 1 ]; then
  NODE_ARGS=(--headless)
fi

uv run --active --no-sync vllm serve "$MODEL_DIR" \
  --served-model-name dosa-benchmark \
  --tensor-parallel-size 8 --enable-expert-parallel \
  --nnodes 2 --node-rank "$NODE_RANK" \
  --master-addr "$MASTER_ADDR" --master-port 29810 \
  --distributed-executor-backend mp \
  --max-model-len 1048576 --max-num-batched-tokens "$CHUNK" \
  --max-num-seqs 1 --kv-cache-dtype fp8 \
  --gpu-memory-utilization 0.90 --seed 233767787 \
  --host 0.0.0.0 --port 30210 \
  --attention-config "{\"indexer_dosa\":$DOSA}" \
  "${MODEL_ARGS[@]}" "${NODE_ARGS[@]}"
```

The PR includes the common manual FlashInfer fusion-budget and zero compile-range
fixes used for GLM. `VLLM_SERVER_DEV_MODE=1` exposes vLLM's native prefix-cache
reset endpoint, also used by its benchmark sweep runner. Do not install capture
or profiling hooks in the latency server.

**Save as `ttft_client.py`:** This is the actual measurement client, unchanged.
It calls vLLM's `async_request_openai_completions`; its TTFT boundary is the first
streamed completion choice, including an empty-text choice. File reads and the
prefix-cache reset occur outside that timer, while server tokenization remains
inside TTFT.

```python
"""Serial FP8 model TTFT using the frozen upstream vLLM benchmark client."""

import argparse
import asyncio
import dataclasses
import json
import statistics
import time
import uuid
from pathlib import Path

import aiohttp

from vllm.benchmarks.lib.endpoint_request_func import (
    AIOHTTP_TIMEOUT,
    RequestFuncInput,
    async_request_openai_completions,
)


def save(path, value):
    temporary = path.with_suffix(".pending")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


async def run(args):
    plan = json.loads(args.requests.read_bytes())
    args.output.mkdir(parents=True, exist_ok=False)
    state = {
        "status": "RUNNING",
        "started_at": time.time(),
        "arm": args.arm,
        "source": args.source,
        "request_plan": str(args.requests),
        "client": (
            "vllm.benchmarks.lib.endpoint_request_func."
            "async_request_openai_completions"
        ),
        "protocol": (
            "original UTF8 text; fresh RID and prefix reset; "
            "1 warm + 3 formal; first streamed completion choice"
        ),
        "generation": {"max_tokens": 2, "temperature": 0, "ignore_eos": False},
        "requests": [],
    }
    save(args.output / "STATE.json", state)
    try:
        async with aiohttp.ClientSession(timeout=AIOHTTP_TIMEOUT) as session:
            for item in plan:
                prompt_path = Path(item["input"])
                original = prompt_path.read_bytes()
                prompt = original.decode("utf-8")
                cell = {
                    "name": item["name"],
                    "input": str(prompt_path),
                    "bytes": len(original),
                    "status": "MEASURING",
                    "receipts": [],
                }
                state["requests"].append(cell)
                save(args.output / "STATE.json", state)
                for repeat in range(4):
                    role = "warm" if repeat == 0 else f"formal{repeat}"
                    request_id = f"dosa-{args.arm}-{item['name']}-{uuid.uuid4().hex}"
                    async with session.post(
                        args.base_url.rstrip("/") + "/reset_prefix_cache"
                    ) as response:
                        response.raise_for_status()
                        reset = await response.json()
                    if reset != {"success": True}:
                        raise RuntimeError(f"Prefix reset failed: {reset}")
                    request = RequestFuncInput(
                        prompt=prompt,
                        api_url=args.base_url.rstrip("/") + "/v1/completions",
                        prompt_len=item.get("tokens", 0),
                        output_len=2,
                        model=args.model,
                        extra_body={"temperature": 0},
                        ignore_eos=False,
                        request_id=request_id,
                    )
                    result = dataclasses.asdict(
                        await async_request_openai_completions(request, session)
                    )
                    receipt = {
                        "request_id": request_id,
                        "role": role,
                        "cache_reset": reset,
                        **result,
                    }
                    receipt_name = f"{item['name']}-{role}.json"
                    save(args.output / receipt_name, receipt)
                    cell["receipts"].append(receipt_name)
                    save(args.output / "STATE.json", state)
                    if not result["success"] or result["ttft"] <= 0:
                        raise RuntimeError(f"{item['name']} {role}: {result['error']}")
                    if "tokens" in item and result["prompt_len"] != item["tokens"]:
                        raise AssertionError(
                            (item["name"], result["prompt_len"], item["tokens"])
                        )
                    print(
                        json.dumps(
                            {
                                "name": item["name"],
                                "role": role,
                                "tokens": result["prompt_len"],
                                "ttft_s": result["ttft"],
                            }
                        ),
                        flush=True,
                    )
                measured = [
                    json.loads((args.output / name).read_bytes())
                    for name in cell["receipts"][1:]
                ]
                cell.update(
                    status="COMPLETE",
                    prompt_tokens=measured[0]["prompt_len"],
                    median_ttft_s=statistics.median(x["ttft"] for x in measured),
                )
                if len({x["prompt_len"] for x in measured}) != 1:
                    raise AssertionError("Prompt length changed between repetitions")
                save(args.output / "STATE.json", state)
        state["status"] = "COMPLETE"
    except Exception as error:
        state.update(status="FAILED", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        state["finished_at"] = time.time()
        save(args.output / "STATE.json", state)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--requests", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--arm", choices=("stock", "dosa"), required=True)
    parser.add_argument("--source", required=True)
    asyncio.run(run(parser.parse_args()))
```

Once the server is healthy, run the client on a separate CPU host with the same
installed vLLM environment:

```bash
uv run --active --no-sync python ttft_client.py \
  --requests "inputs/$MODEL_KIND/requests.json" \
  --base-url "http://$MASTER_ADDR:30210" --model dosa-benchmark \
  --arm stock --source dosa-branch \
  --output "results/$MODEL_KIND-stock-16k"
```

For every input the client sends one excluded warmup followed by three measured
requests, resetting the native prefix cache and assigning a fresh request ID
before each request. It sends the full raw UTF-8 text with `max_tokens=2`,
`temperature=0`, and default EOS behavior. `STATE.json` records the median of the
three measured TTFTs; every warmup and formal receipt is retained separately.

Stop the owned server normally, restart both nodes with `DOSA=true`, and repeat
with `--arm dosa` and a new output directory. Repeat both arms with
`CHUNK=32768`, using new `*-32k` output directories. For GLM set `MODEL_KIND=glm`
and `MODEL_DIR=/path/to/GLM-5.2-FP8` on both server nodes and the client; retain all
four common GLM compatibility settings in both arms. Keep the same hardware pair
within each stock/DoSA comparison. DeepSeek timing requests above 163840 tokens
test context extrapolation beyond its checkpoint's configured 163840-token
range; they do not establish model quality at those lengths.

</details>
