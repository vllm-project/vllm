# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The fused Mamba align-mode state-index kernel (VLLM_MAMBA_FUSED_STATE_INDEX)
must return exactly what the torch path of mamba_get_block_table_tensor does."""

import pytest
import torch

from vllm.platforms import current_platform
from vllm.v1.attention.backends.utils import mamba_get_block_table_tensor
from vllm.v1.kv_cache_interface import MambaSpec

pytestmark = pytest.mark.skipif(
    not current_platform.is_cuda(), reason="Triton kernel needs a CUDA GPU"
)


@pytest.mark.parametrize("num_speculative_blocks", [0, 7])
@pytest.mark.parametrize("num_reqs", [1, 3, 64, 512])
def test_fused_state_indices_match_torch(monkeypatch, num_speculative_blocks, num_reqs):
    block_size, num_cols = 1024, 300
    spec = MambaSpec(
        block_size=block_size,
        shapes=((1,),),
        dtypes=(torch.float16,),
        mamba_cache_mode="align",
        num_speculative_blocks=num_speculative_blocks,
    )
    gen = torch.Generator(device="cuda").manual_seed(num_reqs)
    # A row slice of a larger table, as the model runner passes it.
    table = torch.randint(
        0,
        1 << 20,
        (num_reqs + 5, num_cols),
        dtype=torch.int32,
        device="cuda",
        generator=gen,
    )[:num_reqs]
    max_len = (num_cols - 1 - num_speculative_blocks) * block_size
    seq_lens = torch.randint(
        0, max_len, (num_reqs,), dtype=torch.int32, device="cuda", generator=gen
    )
    # Block boundaries and zero-length (CUDA-graph padding) rows.
    edge = torch.tensor([0, 1, block_size, block_size + 1], dtype=torch.int32)
    seq_lens[: min(4, num_reqs)] = edge[: min(4, num_reqs)].cuda()

    monkeypatch.setenv("VLLM_MAMBA_FUSED_STATE_INDEX", "0")
    ref = mamba_get_block_table_tensor(table, seq_lens, spec, "align")
    monkeypatch.setenv("VLLM_MAMBA_FUSED_STATE_INDEX", "1")
    out = mamba_get_block_table_tensor(table, seq_lens, spec, "align")

    assert out.dtype == ref.dtype and out.shape == ref.shape
    torch.testing.assert_close(out, ref, rtol=0, atol=0)


def test_non_unit_column_stride_uses_torch_path(monkeypatch):
    spec = MambaSpec(
        block_size=64,
        shapes=((1,),),
        dtypes=(torch.float16,),
        mamba_cache_mode="align",
        num_speculative_blocks=3,
    )
    table = torch.randint(0, 5000, (8, 128), dtype=torch.int32, device="cuda")[:, ::2]
    seq_lens = torch.randint(0, 60 * 64, (8,), dtype=torch.int32, device="cuda")
    monkeypatch.setenv("VLLM_MAMBA_FUSED_STATE_INDEX", "0")
    ref = mamba_get_block_table_tensor(table, seq_lens, spec, "align")
    monkeypatch.setenv("VLLM_MAMBA_FUSED_STATE_INDEX", "1")
    out = mamba_get_block_table_tensor(table, seq_lens, spec, "align")
    torch.testing.assert_close(out, ref, rtol=0, atol=0)
