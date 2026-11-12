# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""PrefixLM prefill attention: Triton kernel vs its FlashAttention path vs FlashInfer.

A PrefixLM decision request is one shared state followed by one block per
question, each attending bidirectionally inside its block and to everything
before it. Two batch shapes cover how such requests are prefilled:

- ``state``: every request prefills its state and first question from scratch.
- ``questions``: every question resumes after a prefix-cache hit on its state.

Backends timed on the same batch and inputs (outputs are checked against the
first one):

- ``triton``: ``TRITON_ATTN`` with its own kernel (FA2 path disabled).
- ``triton_fa``: ``TRITON_ATTN`` as shipped; it hands these batches to
  FlashAttention-2 on SM 8.x and FlashAttention-3 on SM 9.x.
- ``flashinfer``: ``FLASHINFER`` with native mm-prefix ranges (#46558).

Example (a Qwen3.5-9B full-attention layer):

    cd benchmarks/attention_benchmarks
    python benchmark_prefix_lm.py --state-lens 2048 8192 14336 \
        --num-requests 1 4 --questions 11
"""

import argparse
import statistics

import torch
from common import BenchmarkConfig, run_do_bench
from runner import (
    _build_common_attn_metadata,
    _create_backend_impl,
    _create_input_tensors,
    _create_kv_cache,
    _create_metadata_builder,
    _create_vllm_config,
    _get_backend_config,
    _kv_cache_spec,
    log_warnings_and_errors_only,
)

from vllm.config import set_current_vllm_config
from vllm.v1.attention.backends.utils import (
    get_supported_kv_cache_layouts,
    resolve_kv_cache_layout,
)

BACKENDS = {
    "triton": "TRITON_ATTN",
    "triton_fa": "TRITON_ATTN",
    "flashinfer": "FLASHINFER",
}


def prefix_lm_batch(
    shape: str, num_requests: int, state_len: int, question_len: int, questions: int
) -> tuple[list[int], list[int], dict[int, list[tuple[int, int]]]]:
    """Query lengths, sequence lengths and bidirectional ranges of a batch."""
    state = (0, state_len - 1)
    question = (state_len, state_len + question_len - 1)
    seq_len = state_len + question_len
    if shape == "state":
        num_seqs, q_len = num_requests, seq_len
    else:
        num_seqs, q_len = num_requests * questions, question_len
    q_lens = [q_len] * num_seqs
    kv_lens = [seq_len] * num_seqs
    ranges = {i: [state, question] for i in range(num_seqs)}
    return q_lens, kv_lens, ranges


def run(
    name: str,
    config: BenchmarkConfig,
    q_lens: list[int],
    kv_lens: list[int],
    ranges: dict[int, list[tuple[int, int]]],
    reference: torch.Tensor | None,
) -> tuple[float, torch.Tensor]:
    device = torch.device(config.device)
    block_size = config.block_size
    max_blocks = len(q_lens) * -(-max(kv_lens) // block_size)
    vllm_config = _create_vllm_config(config, max_blocks, max(kv_lens))
    # Builders size their staging buffers by the step budget.
    scheduler = vllm_config.scheduler_config
    budget = max(scheduler.max_num_batched_tokens, sum(q_lens))
    scheduler.max_num_batched_tokens = budget
    # FlashInfer reads the ranges only for models flagged as mm-prefix LMs.
    vllm_config.model_config.__dict__["is_mm_prefix_lm"] = True
    dtype = vllm_config.model_config.dtype
    with set_current_vllm_config(vllm_config):
        backend_class, impl, layer = _create_backend_impl(
            _get_backend_config(config.backend), config, device, dtype
        )
        supported = get_supported_kv_cache_layouts([backend_class])
        layout = resolve_kv_cache_layout(vllm_config, [[m.name for m in supported]])
        common = _build_common_attn_metadata(q_lens, kv_lens, block_size, device)
        common.seq_lens_cpu_upper_bound = common.seq_lens.cpu()
        common.mm_req_doc_ranges = ranges
        spec = backend_class.customize_spec(_kv_cache_spec(config, dtype))
        builder = _create_metadata_builder(
            backend_class, spec, vllm_config, device, config.backend
        )
        metadata = builder.build(common_prefix_len=0, common_attn_metadata=common)
        if name == "triton":
            metadata.prefix_lm_fa = None
        elif name == "triton_fa" and metadata.prefix_lm_fa is None:
            raise SystemExit("triton_fa: no FlashAttention PrefixLM path on this GPU")
        torch.manual_seed(0)
        q, k, v = _create_input_tensors(config, sum(q_lens), device, dtype)
        caches = _create_kv_cache(
            config, max_blocks, backend_class, device, dtype, layout
        )
        # Identical cache contents for every backend, whatever its layout.
        num_slots = max_blocks * block_size
        shape = (num_slots, config.num_kv_heads, config.head_dim)
        k_all = torch.randn(shape, device=device, dtype=dtype)
        v_all = torch.randn(shape, device=device, dtype=dtype)
        slots = torch.arange(num_slots, device=device, dtype=torch.int64)
        impl.do_kv_cache_update(layer, k_all, v_all, caches[0], slots)
        out = torch.empty_like(q[0])

        def step():
            impl.forward(layer, q[0], k[0], v[0], caches[0], metadata, output=out)

        times = run_do_bench(step, use_cuda_graphs=False, warmup_ms=config.warmup_ms)
    if reference is not None:
        diff = (out.float() - reference.float()).abs().max().item()
        if diff > 2e-2:
            raise SystemExit(f"{name}: max |diff| {diff:.3g} against the first backend")
    return statistics.median(times), out.clone()


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--backends", nargs="+", default=list(BACKENDS), choices=BACKENDS)
    p.add_argument("--shapes", nargs="+", default=["state", "questions"])
    p.add_argument("--state-lens", type=int, nargs="+", default=[2048, 8192, 14336])
    p.add_argument("--num-requests", type=int, nargs="+", default=[1, 4])
    p.add_argument("--questions", type=int, default=11)
    p.add_argument("--question-len", type=int, default=64)
    p.add_argument("--head-dim", type=int, default=256)
    p.add_argument("--num-q-heads", type=int, default=16)
    p.add_argument("--num-kv-heads", type=int, default=4)
    p.add_argument("--block-size", type=int, default=16)
    p.add_argument("--model", default="Qwen/Qwen3-0.6B")
    args = p.parse_args()

    print(
        "shape,num_requests,state_len,tokens,"
        + ",".join(f"{b}_ms" for b in args.backends)
    )
    for shape in args.shapes:
        for num_requests in args.num_requests:
            for state_len in args.state_lens:
                q_lens, kv_lens, ranges = prefix_lm_batch(
                    shape, num_requests, state_len, args.question_len, args.questions
                )
                row, reference = [], None
                for name in args.backends:
                    config = BenchmarkConfig(
                        backend=BACKENDS[name],
                        batch_spec=f"prefix_lm_{shape}",
                        num_layers=1,
                        head_dim=args.head_dim,
                        num_q_heads=args.num_q_heads,
                        num_kv_heads=args.num_kv_heads,
                        block_size=args.block_size,
                        device="cuda:0",
                        dtype=torch.bfloat16,
                        model=args.model,
                    )
                    with log_warnings_and_errors_only():
                        ms, out = run(name, config, q_lens, kv_lens, ranges, reference)
                    reference = out if reference is None else reference
                    row.append(f"{ms:.3f}")
                print(
                    f"{shape},{num_requests},{state_len},{sum(q_lens)},"
                    + ",".join(row),
                    flush=True,
                )


if __name__ == "__main__":
    main()
