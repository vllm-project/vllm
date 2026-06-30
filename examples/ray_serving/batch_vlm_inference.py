# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Batch VLM inference with Ray Data and vLLM.

Supports two datasets:
  --dataset coco       Real-world captioning (lmms-lab/COCO-Caption2017, ~5k val)
  --dataset random-mm  Synthetic multimodal data with configurable ISL/OSL

The script is hardware-agnostic: runtime environment variables are selected
automatically based on the detected GPU backend (ROCm or CUDA).

Verified hardware:
  NVIDIA: B200
  AMD:    MI300X, MI325X, MI355X

Example:
    python batch_vlm_inference.py \\
        --model Qwen/Qwen3-VL-235B-A22B-Instruct-FP8 \\
        --dataset random-mm --isl 1024 --osl 512 --n-samples 1024 \\
        --enable-expert-parallel

Learn more about Ray Data's LLM integration:
https://docs.ray.io/en/latest/data/working-with-llms.html

Docker / Ray setup and compatibility

The container may not include this file or Ray (ROCm OpenAI image or similar).
Copy the script in (``docker cp .../batch_vlm_inference.py <container>:/tmp/``)
or bind-mount your vLLM tree (e.g. ``-v /path/to/vllm:/workspace``) and run from
there. Install once: ``pip3 install "ray[data]>=2.44.1"``. Sanity check:
``python3 -c "import ray; from ray.data.llm import build_llm_processor; \
print(ray.__version__)"``.

Ray Data LLM requires ``vllm.inputs.data``; use a vLLM revision that includes
https://github.com/vllm-project/vllm/pull/46013 (or ``main`` after it merges).
"""

import argparse
import functools
from io import BytesIO
from typing import Any

import datasets
import pybase64 as base64
import ray
import torch
from packaging.version import Version
from PIL import Image
from ray.data.llm import build_llm_processor, vLLMEngineProcessorConfig

assert Version(ray.__version__) >= Version("2.44.1"), (
    "Ray version must be at least 2.44.1"
)


def get_runtime_env_vars() -> dict[str, str]:
    """Return platform-specific vLLM environment variables.

    ROCm: enables AITER kernels and ROCm-specific memory optimizations.
    CUDA: sets the multiprocessing backend required for async scheduling.
    """
    env: dict[str, str] = {"VLLM_WORKER_MULTIPROC_METHOD": "spawn"}
    if torch.version.hip:
        env.update(
            {
                "HIP_FORCE_DEV_KERNARG": "1",
                "VLLM_ROCM_USE_AITER": "1",
                "VLLM_ROCM_USE_AITER_MHA": "1",
                "VLLM_ROCM_SHUFFLE_KV_CACHE_LAYOUT": "1",
                "SAFETENSORS_FAST_GPU": "1",
            }
        )
    return env


def build_engine_config(args: argparse.Namespace) -> vLLMEngineProcessorConfig:
    """Build a vLLMEngineProcessorConfig from parsed arguments."""
    engine_kwargs: dict[str, Any] = dict(
        tensor_parallel_size=args.tensor_parallel_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_model_len,
        max_num_seqs=args.max_num_seqs,
        max_num_batched_tokens=args.max_num_batched_tokens,
        kv_cache_dtype=args.kv_cache_dtype,
        enable_expert_parallel=args.enable_expert_parallel,
        mm_encoder_tp_mode="data",
        async_scheduling=True,
        distributed_executor_backend="mp",
        enable_chunked_prefill=True,
        compilation_config={
            "mode": 3,
            "cudagraph_mode": "FULL_AND_PIECEWISE",
            # Enable AITER-fused rms_norm and fp8 quantization kernels on ROCm.
            **(
                {"custom_ops": ["+rms_norm", "+quant_fp8"]} if torch.version.hip else {}
            ),
        },
        limit_mm_per_prompt={"image": 1},
    )

    return vLLMEngineProcessorConfig(
        model_source=args.model,
        engine_kwargs=engine_kwargs,
        runtime_env={"env_vars": get_runtime_env_vars()},
        batch_size=args.batch_size,
        max_concurrent_batches=args.max_concurrent_batches,
        has_image=True,
        experimental={
            # Prefetch enough tasks to keep the engine saturated between batches.
            "max_tasks_in_flight_per_actor": 2 * args.max_concurrent_batches,
        },
    )


def preprocess_coco(row: dict[str, Any], max_tokens: int = 256) -> dict[str, Any]:
    """Prepare a COCO-Caption2017 row for VLM inference.

    The dataset provides an image and a question field. A system prompt
    instructs the model to reason step-by-step before answering.
    """
    image = Image.open(BytesIO(row["image"]["bytes"]))
    return {
        "messages": [
            {
                "role": "system",
                "content": (
                    "Analyze the image and question carefully, using step-by-step "
                    "reasoning. First, describe any image provided in detail. Then, "
                    "present your reasoning. Finally, state your answer clearly."
                ),
            },
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": row["question"]},
                ],
            },
        ],
        "sampling_params": {"max_tokens": max_tokens, "temperature": 0.3},
    }


def preprocess_random_mm(row: dict[str, Any], max_tokens: int = 512) -> dict[str, Any]:
    """Prepare a synthetic random-mm row for VLM inference.

    Decodes base64-encoded JPEG images from the RandomMultiModalDataset
    format into PIL Images that Ray Data can serialize efficiently.
    """
    images = [
        Image.open(BytesIO(base64.b64decode(item["image_url"]["url"].split(",", 1)[1])))
        for item in (row.get("multi_modal_data") or [])
        if item.get("type") == "image_url"
    ]
    content: list[dict[str, Any]] = [{"type": "image", "image": img} for img in images]
    content.append({"type": "text", "text": row["prompt"]})
    return {
        "messages": [{"role": "user", "content": content}],
        "sampling_params": {"max_tokens": max_tokens, "temperature": 0.0},
    }


def load_coco_dataset(args: argparse.Namespace) -> ray.data.Dataset:
    """Load the COCO-Caption2017 validation split as a Ray Dataset."""
    hf_ds = datasets.load_dataset("lmms-lab/COCO-Caption2017", "default")
    ds = ray.data.from_huggingface(hf_ds["val"])
    ds = ds.limit(args.n_samples)
    return ds


def load_random_mm_dataset(args: argparse.Namespace) -> ray.data.Dataset:
    """Generate synthetic multimodal samples using vLLM's RandomMultiModalDataset.

    Samples are generated eagerly in the driver process and ingested via
    ray.data.from_items(). This avoids per-worker tokenizer loading with no
    throughput benefit at typical benchmark scales.
    """
    from transformers import AutoTokenizer

    from vllm.benchmarks.datasets import RandomMultiModalDataset

    print(
        f"Generating {args.n_samples} synthetic samples "
        f"(isl={args.isl}, osl={args.osl}, "
        f"image={args.image_size}px, seed={args.seed}) ..."
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    samples = RandomMultiModalDataset(random_seed=args.seed).sample(
        tokenizer=tokenizer,
        num_requests=args.n_samples,
        input_len=args.isl,
        output_len=args.osl,
        range_ratio=0.0,  # fixed lengths for controlled benchmarking
        limit_mm_per_prompt={"image": 1},
        bucket_config={(args.image_size, args.image_size, 1): 1.0},
    )
    rows = [
        {"prompt": s.prompt, "multi_modal_data": s.multi_modal_data or []}
        for s in samples
    ]
    print(f"Generated {len(rows)} samples.")
    return ray.data.from_items(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Batch VLM inference with Ray Data and vLLM",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    # Model
    parser.add_argument("--model", default="Qwen/Qwen3-VL-235B-A22B-Instruct-FP8")
    # Dataset
    parser.add_argument("--dataset", choices=["coco", "random-mm"], default="coco")
    parser.add_argument(
        "--n-samples",
        type=int,
        default=None,
        help="samples to process (default: 10 × batch-size)",
    )
    parser.add_argument(
        "--isl",
        type=int,
        default=1024,
        help="[random-mm] input sequence length in text tokens",
    )
    parser.add_argument("--osl", type=int, default=512, help="max output tokens")
    parser.add_argument(
        "--image-size",
        type=int,
        default=512,
        help="[random-mm] synthetic image height and width in pixels",
    )
    parser.add_argument("--seed", type=int, default=42, help="[random-mm] random seed")
    # Engine
    parser.add_argument("--tensor-parallel-size", type=int, default=8)
    parser.add_argument(
        "--enable-expert-parallel",
        action="store_true",
        help="enable for MoE models (e.g. Qwen3-VL-235B; not needed for 32B)",
    )
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.94)
    parser.add_argument("--max-model-len", type=int, default=32768)
    parser.add_argument("--max-num-seqs", type=int, default=10240)
    parser.add_argument("--max-num-batched-tokens", type=int, default=32768)
    parser.add_argument("--kv-cache-dtype", default="fp8")
    # Ray processor
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--max-concurrent-batches", type=int, default=64)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # Pass GPU count explicitly to ray.init — some ROCm environments (e.g. TheRock)
    # ship libamd_smi instead of librocm_smi64, causing Ray's auto-detection to
    # return 0 GPUs. torch.accelerator works correctly on ROCm via HIP.
    num_gpus = torch.accelerator.device_count()
    ray.init(num_gpus=num_gpus if num_gpus > 0 else None)

    if args.n_samples is None:
        args.n_samples = 10 * args.batch_size

    if args.dataset == "random-mm":
        ds = load_random_mm_dataset(args)
        preprocess = functools.partial(preprocess_random_mm, max_tokens=args.osl)
    else:
        ds = load_coco_dataset(args)
        preprocess = functools.partial(preprocess_coco, max_tokens=args.osl)

    config = build_engine_config(args)
    processor = build_llm_processor(config, preprocess=preprocess)

    print(f"Processing {args.n_samples} samples with {args.model} ...")
    results = processor(ds).take_all()
    print(f"Done — {len(results)} results.")
    if results:
        print("Sample output:", results[0].get("generated_text", "")[: args.osl])


if __name__ == "__main__":
    main()
