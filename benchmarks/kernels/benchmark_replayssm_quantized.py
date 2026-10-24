# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Time Bolt ReplaySSM verification with CUDA graphs on one GB10.

Run from the checkout with its configured CUDA environment:
  .venv/bin/python benchmarks/kernels/benchmark_replayssm_quantized.py \
      --mode all --output /work/results/replayssm-components.json

This measures SSM verification and GPU cursor updates, not model throughput.
The ten-step accepted-count cycle has mean 4.3, including the mandatory token.
Shared checkpoints use round-to-nearest. Separate FP16 recurrence uses SR7.
Run the focused correctness tests before interpreting these timings.
"""

import argparse
import hashlib
import importlib.metadata
import json
import shutil
import subprocess
from datetime import UTC, datetime
from pathlib import Path

import torch

from vllm.config.mamba import MambaBackendEnum
from vllm.model_executor.layers.mamba.mamba_utils import (
    MambaStateDtypeCalculator,
    MambaStateShapeCalculator,
)

MODES = ("shared-bf16", "shared-fp8", "shared-int8", "separate-fp8", "separate-nvfp4")
ACCEPTED = (1, 2, 3, 4, 4, 5, 5, 6, 6, 7)


def prepare(mode):
    checkpoint_mode, state_format = mode.split("-")
    separate = checkpoint_mode == "separate"
    heads, groups, tokens = 128, 8, 8
    torch.manual_seed(42)
    source = torch.randn(1, heads, 64, 128, device="cuda") / 16
    slots = torch.tensor([1], device="cuda", dtype=torch.int32)
    # Model projection splits retain the full feature-row stride.
    widths = (heads * 64, groups * 128, groups * 128)
    convolved = torch.randn(tokens, sum(widths), device="cuda", dtype=torch.bfloat16)
    x, b, c = convolved.split(widths, dim=-1)
    projected = torch.empty(
        tokens,
        2 * heads * 64 + 2 * groups * 128 + heads,
        device="cuda",
        dtype=torch.bfloat16,
    )
    dt = projected[:, -heads:]
    x = x.view(tokens, heads, 64)
    x.div_(32)
    b, c = b.view(tokens, groups, 128), c.view(tokens, groups, 128)
    b.div_(16)
    dt.fill_(32)
    dt = dt[..., None].expand(-1, -1, 64)
    a = torch.full((heads,), -1 / 32, device="cuda")
    d, bias = (
        torch.zeros(heads, device="cuda", dtype=torch.bfloat16) for _ in range(2)
    )
    out = torch.empty_like(x)
    rings = MambaStateShapeCalculator.replayssm_ring_shapes(
        heads,
        64,
        128,
        groups,
        1,
        16,
        MambaBackendEnum.FLASHINFER,
        num_speculative_tokens=7,
        checkpoint_mode=checkpoint_mode,
        pending_state=separate,
    )
    read_format = "auto" if state_format == "bf16" else state_format
    extras = MambaStateShapeCalculator.replayssm_read_shapes(
        (heads, 64, 128),
        read_format,
        checkpoint_mode,
        n_groups=groups,
        pending_state=separate,
    )
    state_dtype = {
        "bf16": torch.bfloat16,
        "fp8": torch.float8_e4m3fn,
        "int8": torch.int8,
    }.get(state_format, torch.float16)
    if separate:
        state_dtype = torch.float16
    shapes = ((1,), (heads, 64, 128), *rings, *extras)
    dtypes = (
        torch.bfloat16,
        state_dtype,
        torch.bfloat16,
        torch.float32,
        torch.bfloat16,
        *MambaStateDtypeCalculator.replayssm_read_dtypes(read_format, checkpoint_mode),
    )
    cache = tuple(
        torch.zeros(2, *shape, dtype=dtype, device="cuda")
        for shape, dtype in zip(shapes, dtypes, strict=True)
    )
    if separate:
        from vllm.model_executor.layers.mamba.ops.replayssm_cutedsl import (
            initialize_replayssm_state,
            selective_state_update_replayssm,
        )

        initialize_replayssm_state(cache, source, slots, state_dtype=state_format)
        accepted = torch.zeros(1, device="cuda", dtype=torch.int32)
        cu_seqlens = torch.tensor([0, tokens], device="cuda", dtype=torch.int32)

        def step(count):
            accepted.fill_(count)
            selective_state_update_replayssm(
                cache,
                x,
                dt,
                a,
                b,
                c,
                d,
                bias,
                slots,
                out,
                state_dtype=state_format,
                num_accepted_tokens=accepted,
                cu_seqlens=cu_seqlens,
                max_seqlen=tokens,
                seed=42,
            )
    else:
        from flashinfer.mamba.checkpointing_ssu import checkpointing_ssu

        from vllm.model_executor.layers.mamba.ops.replayssm_quantization import (
            quantize_replayssm_state,
        )

        scale = None
        if state_format == "bf16":
            cache[1][1].copy_(source[0])
        else:
            scale = cache[5]
            quantize_replayssm_state(source, cache[1], scale, slots)
        starts = torch.zeros(2, device="cuda", dtype=torch.int32)
        history = torch.zeros_like(starts)
        arguments = dict(
            state=cache[1],
            state_scale=scale,
            x_cache=cache[2],
            dt_cache=cache[3],
            B_cache=cache[4],
            ring_start=starts,
            prev_num_accepted_tokens=history,
            x=x[None],
            dt=dt[None],
            A=a[:, None, None].expand(-1, 64, 128),
            B=b[None],
            C=c[None],
            D=d[:, None].expand(-1, 64),
            dt_bias=bias[:, None].expand(-1, 64),
            dt_softplus=True,
            out=out[None],
            state_batch_indices=slots,
            algorithm="monolith",
            d_split=1,
        )

        def step(count):
            checkpointing_ssu(**arguments)
            flush = history + tokens > 16
            starts.copy_(torch.where(flush, (starts + history) % 24, starts))
            history.copy_(torch.where(flush, 0, history) + count)

    layout = {
        name: {
            "shape": list(value.shape),
            "stride": list(value.stride()),
            "dtype": str(value.dtype),
        }
        for name, value in {
            "x": x,
            "dt": dt,
            "B": b,
            "C": c,
            "A": a,
            "D": d,
            "dt_bias": bias,
        }.items()
    }
    return step, out, layout


def measure(mode, iterations):
    step, out, layout = prepare(mode)
    for _ in range(2):
        for count in ACCEPTED:
            step(count)
    torch.accelerator.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        for count in ACCEPTED:
            step(count)
    for _ in range(10):
        graph.replay()
    torch.accelerator.synchronize()
    start, end = (torch.cuda.Event(enable_timing=True) for _ in range(2))
    start.record()
    for _ in range(iterations):
        graph.replay()
    end.record()
    end.synchronize()
    if not torch.isfinite(out).all():
        raise RuntimeError(f"{mode} produced a nonfinite output")
    result = {
        "mode": mode,
        "input_layout": layout,
        "microseconds_per_verification": start.elapsed_time(end)
        * 1000
        / (iterations * len(ACCEPTED)),
        "graph_replays": iterations,
        "verification_steps_per_graph": len(ACCEPTED),
        "checkpoint_rounding": "SR7 recurrence; nearest read"
        if mode.startswith("separate")
        else "nearest; SR disabled",
        "algorithm": "native mma.sync pending"
        if mode.startswith("separate")
        else "FlashInfer monolith d_split=1",
    }
    print(json.dumps(result), flush=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=(*MODES, "all"), default="all")
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.iterations < 1:
        parser.error("--iterations must be positive")
    if torch.cuda.get_device_capability() != (12, 1):
        parser.error("This comparison requires a GB10 SM121 GPU")
    root = Path(__file__).resolve().parents[2]
    source_commit = None
    if shutil.which("git"):
        source = subprocess.run(
            ["git", "-C", str(root), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
        )
        if source.returncode == 0:
            source_commit = source.stdout.strip()
    overlay = root.parent / "validation-source.tar.gz"
    source_record = root.parent / "source-revision.txt"
    report = {
        "status": "started",
        "started_at": datetime.now(UTC).isoformat(),
        "source_commit": source_commit,
        "source_revision_record": source_record.read_text().strip()
        if source_record.exists()
        else None,
        "source_overlay_sha256": hashlib.sha256(overlay.read_bytes()).hexdigest()
        if overlay.exists()
        else None,
        "environment": {
            "gpu": torch.cuda.get_device_name(),
            "capability": [12, 1],
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "flashinfer": importlib.metadata.version("flashinfer-python"),
            "cutlass_dsl": importlib.metadata.version("nvidia-cutlass-dsl"),
        },
        "geometry": {
            "batch": 1,
            "heads": 128,
            "head_dim": 64,
            "state_size": 128,
            "groups": 8,
            "verification_tokens": 8,
            "history": 16,
        },
        "accepted_count_cycle": ACCEPTED,
        "mean_accepted_count": sum(ACCEPTED) / len(ACCEPTED),
        "scope": (
            "CUDA graph SSM verification plus GPU cursor updates; excludes model, "
            "sampling, and serving. This is not an accuracy check or model "
            "throughput result."
        ),
        "selection_limit": (
            "Shared modes use monolith d_split=1. This fixes the component baseline; "
            "it does not establish the fastest FlashInfer choice or serving default."
        ),
        "flush_policy": {
            "shared": "Flush old history when history + 8 > 16",
            "separate": "Flush only at accepted 16-token boundaries",
            "comparison_limit": (
                "Flush frequency and replay work differ. Timing differences cannot "
                "be attributed to quantization alone."
            ),
            "reference_160_verifications_from_empty": {
                "accepted_tokens": 688,
                "shared_flushes": 63,
                "separate_flushes": 43,
                "note": "Schedule illustration; excludes one pending-window delay.",
            },
        },
        "results": [],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    try:
        for mode in MODES if args.mode == "all" else (args.mode,):
            report["results"].append(measure(mode, args.iterations))
        report["status"] = "completed"
    except Exception as error:
        report["status"] = "failed"
        report["error"] = repr(error)
        raise
    finally:
        report["finished_at"] = datetime.now(UTC).isoformat()
        args.output.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
