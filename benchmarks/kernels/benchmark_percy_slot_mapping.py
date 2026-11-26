# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Time `_compute_slot_mappings_kernel` with and without the Percy W2
generation check, on a decode-heavy or prefill-heavy batch.

Example:
    python benchmarks/kernels/benchmark_percy_slot_mapping.py \
        --num-reqs 256 --tokens-per-req 1 --num-groups 2

"""

from dataclasses import dataclass

import torch

from vllm.utils.argparse_utils import FlexibleArgumentParser
from vllm.utils.math_utils import cdiv
from vllm.v1.attention.backends.utils import PAD_SLOT_ID
from vllm.v1.worker.gpu.block_table import PercyGenCheck


@dataclass
class SlotMappingInputs:
    tables: list[torch.Tensor]
    ptrs: torch.Tensor
    strides: torch.Tensor
    block_sizes: torch.Tensor
    enabled: torch.Tensor
    dcp_sharded: torch.Tensor
    idx_mapping: torch.Tensor
    query_start_loc: torch.Tensor
    positions: torch.Tensor
    out: torch.Tensor
    gen_check: PercyGenCheck


def build_inputs(
    num_reqs: int,
    tokens_per_req: int,
    context_len: int,
    num_groups: int,
    block_size: int,
    device: torch.device,
) -> SlotMappingInputs:
    """A batch where every request writes `tokens_per_req` tokens after
    `context_len` cached ones, with write windows that all pass."""
    blocks_per_req = cdiv(context_len + tokens_per_req, block_size)
    num_blocks = num_reqs * blocks_per_req + 1
    block_ids = torch.arange(1, num_blocks, dtype=torch.int32, device=device)
    table = block_ids.view(num_reqs, blocks_per_req).contiguous()
    tables = [table.clone() for _ in range(num_groups)]
    positions = (
        torch.arange(tokens_per_req, device=device).repeat(num_reqs) + context_len
    )
    num_tokens = num_reqs * tokens_per_req
    gen_check = PercyGenCheck(
        num_blocks, [block_size] * num_groups, num_reqs, num_tokens, device
    )
    gen_check.begin_windows()
    first_col = context_len // block_size
    for req in range(num_reqs):
        cols = range(first_col, blocks_per_req)
        window = (first_col, *(1 for _ in cols))
        gen_check.set_windows(req, (window,) * num_groups)
    gen_check.commit_windows()
    return SlotMappingInputs(
        tables=tables,
        ptrs=torch.tensor(
            [t.data_ptr() for t in tables], dtype=torch.uint64, device=device
        ),
        strides=torch.tensor(
            [t.stride(0) for t in tables], dtype=torch.int64, device=device
        ),
        block_sizes=torch.full(
            (num_groups,), block_size, dtype=torch.int32, device=device
        ),
        enabled=torch.ones(num_groups, dtype=torch.bool, device=device),
        dcp_sharded=torch.ones(num_groups, dtype=torch.bool, device=device),
        idx_mapping=torch.arange(num_reqs, dtype=torch.int32, device=device),
        query_start_loc=torch.arange(
            0, num_tokens + 1, tokens_per_req, dtype=torch.int32, device=device
        ),
        positions=positions.to(torch.int64),
        out=torch.empty(num_groups, num_tokens, dtype=torch.int64, device=device),
        gen_check=gen_check,
    )


def launch(inputs: SlotMappingInputs, gen_check: bool) -> None:
    from vllm.v1.worker.gpu.block_table import _compute_slot_mappings_kernel

    gc = inputs.gen_check
    if gen_check:
        gen_args = (gc.table, gc.meta, gc.meta.stride(0), gc.gens)
    else:
        gen_args = (gc.table, gc.table, 0, gc.table)
    num_groups = len(inputs.tables)
    _compute_slot_mappings_kernel[(num_groups, inputs.idx_mapping.shape[0] + 1)](
        inputs.out.shape[1],
        inputs.idx_mapping,
        inputs.query_start_loc,
        inputs.positions,
        inputs.ptrs,
        inputs.strides,
        inputs.block_sizes,
        inputs.enabled,
        inputs.dcp_sharded,
        inputs.out,
        inputs.out.stride(0),
        0,
        *gen_args,
        CP_SIZE=1,
        CP_INTERLEAVE=1,
        PAD_ID=PAD_SLOT_ID,
        TRITON_BLOCK_SIZE=1024,
        GEN_CHECK=gen_check,
    )


def time_us(inputs: SlotMappingInputs, gen_check: bool, iters: int) -> float:
    for _ in range(10):
        launch(inputs, gen_check)
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    torch.cuda.synchronize()
    start.record()
    for _ in range(iters):
        launch(inputs, gen_check)
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) * 1000 / iters


def main() -> None:
    parser = FlexibleArgumentParser(description=__doc__)
    parser.add_argument("--num-reqs", type=int, default=256)
    parser.add_argument("--tokens-per-req", type=int, default=1)
    parser.add_argument("--context-len", type=int, default=4096)
    parser.add_argument("--num-groups", type=int, default=1)
    parser.add_argument("--block-size", type=int, default=16)
    parser.add_argument("--iters", type=int, default=1000)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("This benchmark requires a CUDA device.")
    inputs = build_inputs(
        args.num_reqs,
        args.tokens_per_req,
        args.context_len,
        args.num_groups,
        args.block_size,
        torch.device("cuda"),
    )
    launch(inputs, gen_check=False)
    baseline = inputs.out.clone()
    launch(inputs, gen_check=True)
    assert torch.equal(inputs.out, baseline), "passing windows must not PAD"
    off = time_us(inputs, False, args.iters)
    on = time_us(inputs, True, args.iters)
    print(f"slot_mapping without check: {off:8.2f} us")
    print(f"slot_mapping with check:    {on:8.2f} us  (+{on - off:.2f} us)")


if __name__ == "__main__":
    main()
