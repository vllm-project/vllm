# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Benchmark packed-encoder FlexAttention and optionally compare embeddings."""

import argparse
import gc
import hashlib
import json
import statistics
import subprocess
import time
from pathlib import Path

import numpy as np
import torch

import vllm
from vllm import LLM

PHRASE = (
    "A detailed discussion about efficient search, document matching, "
    "language models, and retrieval quality. "
)
PROFILES = {
    "pack4x2048": [2048] * 4,
    "irregular8192": [512, 768, 1024, 1536, 2048, 2304],
}


def exact_token_prompts(tokenizer, lengths: list[int]) -> list[list[int]]:
    base = tokenizer.encode(PHRASE, add_special_tokens=False)
    if tokenizer.cls_token_id is None or tokenizer.sep_token_id is None:
        raise ValueError("The selected tokenizer must provide CLS and SEP token IDs")

    prompts = []
    for length in lengths:
        content_length = length - 2
        content = (base * ((content_length + len(base) - 1) // len(base)))[
            :content_length
        ]
        prompts.append([tokenizer.cls_token_id, *content, tokenizer.sep_token_id])
    return prompts


def load_reference(path: Path, embeddings: np.ndarray) -> dict[str, float]:
    reference = np.load(path)["embeddings"]
    if reference.shape != embeddings.shape:
        raise ValueError(
            f"Reference shape {reference.shape} does not match {embeddings.shape}"
        )
    if not np.isfinite(reference).all():
        raise ValueError("Reference embeddings contain non-finite values")
    if not np.isfinite(embeddings).all():
        raise ValueError("Candidate embeddings contain non-finite values")

    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        reference64 = reference.astype(np.float64)
        embeddings64 = embeddings.astype(np.float64)
        dot = np.sum(reference64 * embeddings64, axis=1)
        norms = np.linalg.norm(reference64, axis=1) * np.linalg.norm(
            embeddings64, axis=1
        )
        cosine = dot / np.maximum(norms, np.finfo(np.float64).tiny)
        abs_delta = np.abs(reference64 - embeddings64)

    if not np.isfinite(cosine).all() or not np.isfinite(abs_delta).all():
        raise ValueError("Embedding comparison produced non-finite metrics")

    metrics = {
        "min_cosine_similarity": float(np.min(cosine)),
        "max_abs_delta": float(np.max(abs_delta)),
    }
    if not np.isfinite(list(metrics.values())).all():
        raise ValueError("Embedding comparison produced non-finite metrics")
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="Alibaba-NLP/gte-modernbert-base")
    parser.add_argument("--profile", choices=PROFILES, required=True)
    parser.add_argument("--q-block-size", type=int, default=64)
    parser.add_argument("--kv-block-size", type=int, default=64)
    parser.add_argument("--max-model-len", type=int, default=8192)
    parser.add_argument("--warmups", type=int, default=4)
    parser.add_argument("--repeats", type=int, default=15)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--reference", type=Path)
    parser.add_argument("--min-cosine", type=float, default=0.999)
    args = parser.parse_args()
    if not np.isfinite(args.min_cosine):
        raise ValueError("--min-cosine must be finite")

    lengths = PROFILES[args.profile]
    llm = LLM(
        model=args.model,
        runner="pooling",
        convert="embed",
        dtype="half",
        max_model_len=args.max_model_len,
        max_num_seqs=len(lengths),
        max_num_batched_tokens=max(4096, sum(lengths)),
        enforce_eager=True,
        gpu_memory_utilization=0.65,
        attention_config={
            "backend": "FLEX_ATTENTION",
            "flex_attn_q_block_size": args.q_block_size,
            "flex_attn_kv_block_size": args.kv_block_size,
        },
    )
    prompts = exact_token_prompts(llm.get_tokenizer(), lengths)
    iteration_embeddings = []
    latencies_ms = []
    first_request_ms = None
    for iteration in range(args.warmups + args.repeats):
        start = time.perf_counter()
        result = llm.embed(prompts, use_tqdm=False)
        elapsed_ms = (time.perf_counter() - start) * 1000
        if first_request_ms is None:
            first_request_ms = elapsed_ms
        iteration_embeddings.append(
            np.asarray([item.outputs.embedding for item in result], dtype=np.float32)
        )
        if iteration >= args.warmups:
            latencies_ms.append(elapsed_ms)

    torch.cuda.synchronize()
    del llm
    gc.collect()
    all_embeddings = np.stack(iteration_embeddings)
    embeddings = all_embeddings[0]
    reference_metrics = None
    if args.reference is not None:
        reference_metrics = load_reference(args.reference, embeddings)
        if reference_metrics["min_cosine_similarity"] < args.min_cosine:
            raise ValueError(
                "Embedding comparison failed: "
                f"min cosine {reference_metrics['min_cosine_similarity']:.7f} "
                f"< {args.min_cosine:.7f}"
            )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    embedding_path = args.output.with_suffix(".npz")
    np.savez_compressed(
        embedding_path,
        embeddings=embeddings,
        iteration_embeddings=all_embeddings,
    )
    try:
        revision = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        revision = "unknown"
    record = {
        "model": args.model,
        "profile": args.profile,
        "token_lengths": lengths,
        "total_tokens": sum(lengths),
        "max_model_len": args.max_model_len,
        "attention_backend": "FLEX_ATTENTION",
        "dtype": "float16",
        "q_block_size": args.q_block_size,
        "kv_block_size": args.kv_block_size,
        "warmups": args.warmups,
        "repeats": args.repeats,
        "first_request_ms": first_request_ms,
        "p50_ms": statistics.median(latencies_ms),
        "p95_ms": float(np.percentile(latencies_ms, 95)),
        "latencies_ms": latencies_ms,
        "gpu": torch.cuda.get_device_name(0),
        "torch_version": torch.__version__,
        "vllm_version": vllm.__version__,
        "git_revision": revision,
        "embedding_sha256": hashlib.sha256(embeddings.tobytes()).hexdigest(),
        "reference_comparison": reference_metrics,
        "embedding_file": str(embedding_path),
    }
    args.output.write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps(record, separators=(",", ":")))


if __name__ == "__main__":
    main()
