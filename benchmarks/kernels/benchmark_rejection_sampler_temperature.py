# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from functools import partial

import torch

from vllm.triton_utils import triton
from vllm.utils.argparse_utils import FlexibleArgumentParser
from vllm.v1.worker.gpu.sample.gumbel import apply_temperature
from vllm.v1.worker.gpu.spec_decode.rejection_sampler_utils import (
    rejection_sample,
)


def make_inputs(
    batch_size: int,
    num_speculative_steps: int,
    vocab_size: int,
    temperature: float,
    dtype: torch.dtype,
    with_draft_logits: bool,
) -> dict[str, torch.Tensor]:
    device = "cuda"
    num_logits = batch_size * (num_speculative_steps + 1)
    target_logits = torch.randn(num_logits, vocab_size, dtype=dtype, device=device)
    draft_sampled = torch.randint(
        vocab_size,
        (batch_size, num_speculative_steps + 1),
        dtype=torch.int64,
        device=device,
    )
    draft_sampled[:, 0] = 0
    idx_mapping = torch.arange(batch_size, dtype=torch.int32, device=device)
    expanded_idx_mapping = idx_mapping.repeat_interleave(num_speculative_steps + 1)
    expanded_local_pos = torch.arange(
        num_speculative_steps + 1, dtype=torch.int32, device=device
    ).repeat(batch_size)

    inputs = {
        "target_logits": target_logits,
        "draft_sampled": draft_sampled.flatten(),
        "cu_num_logits": torch.arange(batch_size + 1, dtype=torch.int32, device=device)
        * (num_speculative_steps + 1),
        "pos": torch.arange(num_logits, dtype=torch.int32, device=device),
        "idx_mapping": idx_mapping,
        "expanded_idx_mapping": expanded_idx_mapping,
        "expanded_local_pos": expanded_local_pos,
        "temperature": torch.full(
            (batch_size,), temperature, dtype=torch.float32, device=device
        ),
        "seed": torch.arange(batch_size, dtype=torch.int64, device=device),
    }
    if with_draft_logits:
        inputs["draft_logits"] = torch.randn(
            batch_size,
            num_speculative_steps,
            vocab_size,
            dtype=dtype,
            device=device,
        )
    return inputs


def run_baseline(
    inputs: dict[str, torch.Tensor], num_speculative_steps: int
) -> tuple[torch.Tensor, torch.Tensor]:
    processed_logits = torch.empty_like(
        inputs["target_logits"], dtype=torch.float32
    ).copy_(inputs["target_logits"])
    apply_temperature(
        processed_logits,
        inputs["expanded_idx_mapping"],
        inputs["temperature"],
    )
    return rejection_sample(
        target_logits=processed_logits,
        draft_logits=inputs.get("draft_logits"),
        num_speculative_steps=num_speculative_steps,
        **{
            key: value
            for key, value in inputs.items()
            if key not in ("target_logits", "draft_logits")
        },
    )


def run_fused(
    inputs: dict[str, torch.Tensor], num_speculative_steps: int
) -> tuple[torch.Tensor, torch.Tensor]:
    return rejection_sample(
        target_logits=inputs["target_logits"],
        draft_logits=inputs.get("draft_logits"),
        num_speculative_steps=num_speculative_steps,
        apply_target_temperature=True,
        **{
            key: value
            for key, value in inputs.items()
            if key not in ("target_logits", "draft_logits")
        },
    )


def assert_outputs_equal(
    baseline: tuple[torch.Tensor, torch.Tensor],
    fused: tuple[torch.Tensor, torch.Tensor],
    num_speculative_steps: int,
) -> None:
    baseline_sampled, baseline_num_sampled = baseline
    fused_sampled, fused_num_sampled = fused
    torch.testing.assert_close(fused_num_sampled, baseline_num_sampled, rtol=0, atol=0)
    steps = torch.arange(
        num_speculative_steps + 1, device=baseline_sampled.device
    ).unsqueeze(0)
    valid = steps < baseline_num_sampled.unsqueeze(1)
    torch.testing.assert_close(
        fused_sampled[valid], baseline_sampled[valid], rtol=0, atol=0
    )


def measure_peak_memory(callable_) -> int:
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    allocated = torch.cuda.memory_allocated()
    output = callable_()
    torch.cuda.synchronize()
    peak = torch.cuda.max_memory_allocated() - allocated
    del output
    return peak


def main(args) -> None:
    dtype = getattr(torch, args.dtype)
    print(
        "batch  rows  baseline_ms  fused_ms  speedup  "
        "baseline_peak_MiB  fused_peak_MiB  saved_MiB"
    )
    for batch_size in args.batch_sizes:
        inputs = make_inputs(
            batch_size,
            args.num_speculative_steps,
            args.vocab_size,
            args.temperature,
            dtype,
            args.with_draft_logits,
        )
        baseline_call = partial(run_baseline, inputs, args.num_speculative_steps)
        fused_call = partial(run_fused, inputs, args.num_speculative_steps)

        baseline_output = baseline_call()
        fused_output = fused_call()
        torch.cuda.synchronize()
        assert_outputs_equal(baseline_output, fused_output, args.num_speculative_steps)
        del baseline_output, fused_output

        baseline_ms = triton.testing.do_bench(
            baseline_call, warmup=args.warmup_ms, rep=args.rep_ms
        )
        fused_ms = triton.testing.do_bench(
            fused_call, warmup=args.warmup_ms, rep=args.rep_ms
        )
        baseline_peak = measure_peak_memory(baseline_call)
        fused_peak = measure_peak_memory(fused_call)
        mib = 1024**2
        print(
            f"{batch_size:5d}  {batch_size * (args.num_speculative_steps + 1):4d}  "
            f"{baseline_ms:11.3f}  {fused_ms:8.3f}  "
            f"{baseline_ms / fused_ms:7.3f}x  "
            f"{baseline_peak / mib:17.1f}  {fused_peak / mib:14.1f}  "
            f"{(baseline_peak - fused_peak) / mib:9.1f}"
        )


if __name__ == "__main__":
    parser = FlexibleArgumentParser(
        description="Benchmark in-kernel target temperature for rejection sampling."
    )
    parser.add_argument("--batch-sizes", type=int, nargs="+", default=[1, 8, 32, 128])
    parser.add_argument("--num-speculative-steps", type=int, default=4)
    parser.add_argument("--vocab-size", type=int, default=151936)
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument(
        "--dtype", choices=["float32", "float16", "bfloat16"], default="bfloat16"
    )
    parser.add_argument("--with-draft-logits", action="store_true")
    parser.add_argument("--warmup-ms", type=int, default=100)
    parser.add_argument("--rep-ms", type=int, default=500)
    main(parser.parse_args())
