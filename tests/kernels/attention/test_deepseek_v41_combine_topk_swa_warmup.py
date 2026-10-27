# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CombineTopkSwaIndicesKernel warmup must cover the pointer alignments that the
chunked sparse prefill passes at runtime."""

import itertools
from contextlib import contextmanager
from types import SimpleNamespace

import pytest
import torch

from vllm.platforms import current_platform

if not current_platform.is_cuda_alike():
    pytest.skip("Triton kernels require a GPU", allow_module_level=True)

triton = pytest.importorskip("triton")

from vllm.models.deepseek_v41.common.ops.cache_utils import (  # noqa: E402
    _COMBINE_TOPK_SWA_INDICES_KERNEL,
    combine_topk_swa_indices,
)

TOPK = 512
WINDOW = 128
NUM_REQS = 3
QUERY_LEN = 4


def _vllm_config():
    hf_config = SimpleNamespace(
        compress_ratios=[0, 0, 2, 2, 1, 1, 0],
        index_topk=TOPK,
        sliding_window=WINDOW,
    )
    return SimpleNamespace(
        model_config=SimpleNamespace(hf_config=hf_config),
        scheduler_config=SimpleNamespace(max_num_batched_tokens=16384),
    )


@contextmanager
def _record_compiles():
    from triton import knobs

    previous = knobs.runtime.jit_post_compile_hook
    compiled: list[str] = []

    def hook(**kwargs):
        compiled.append(getattr(kwargs.get("fn"), "name", "<unknown>"))
        if previous is not None:
            return previous(**kwargs)
        return None

    knobs.runtime.jit_post_compile_hook = hook
    try:
        yield compiled
    finally:
        knobs.runtime.jit_post_compile_hook = previous


def _int32(values: int, fill: int, offset: int) -> torch.Tensor:
    buffer = torch.full((offset + values,), fill, dtype=torch.int32, device="cuda")
    return buffer[offset:]


def test_warmup_covers_chunked_prefill_pointer_alignments():
    _COMBINE_TOPK_SWA_INDICES_KERNEL.warmup(_vllm_config())

    topk_indices = torch.zeros(
        (NUM_REQS * QUERY_LEN, TOPK), dtype=torch.int32, device="cuda"
    )
    with _record_compiles() as compiled:
        # The chunked prefill passes query_start_loc[num_decodes + chunk_start:]
        # and gather_lens[chunk_start:]. seq_lens is sliced at num_decodes too
        # on ROCm, or comes from a fresh buffer, so the int32 pointers land on
        # every 4-byte offset.
        for num_decodes, chunk_start, seq_lens_at_num_decodes in itertools.product(
            range(4), range(4), (True, False)
        ):
            batch_offset = num_decodes + chunk_start
            query_start_loc = torch.arange(
                0,
                (batch_offset + NUM_REQS + 1) * QUERY_LEN,
                QUERY_LEN,
                dtype=torch.int32,
                device="cuda",
            )[batch_offset : batch_offset + NUM_REQS + 1]
            seq_offset = batch_offset if seq_lens_at_num_decodes else chunk_start
            seq_lens = _int32(NUM_REQS, 4096, seq_offset)
            gather_lens = _int32(NUM_REQS, 1024, chunk_start)
            for ratio in (0, 1, 2):
                combine_topk_swa_indices(
                    topk_indices,
                    query_start_loc,
                    seq_lens,
                    gather_lens,
                    WINDOW,
                    ratio,
                    TOPK if ratio else 0,
                    4096,
                    4096,
                )
        torch.cuda.synchronize()

    assert not compiled, f"JIT compiled after warmup: {compiled}"
