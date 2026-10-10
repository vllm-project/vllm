# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for BaseMambaAttentionMetadataBuilder batch classification."""

from dataclasses import fields
from types import SimpleNamespace

import pytest
import torch

from tests.v1.attention.utils import (
    BatchSpec,
    MockMambaBuilder,
    create_common_attn_metadata,
)
from vllm.config.compilation import CUDAGraphMode
from vllm.v1.attention.backends.mamba1_attn import Mamba1AttentionMetadataBuilder
from vllm.v1.attention.backends.mamba2_attn import Mamba2AttentionMetadataBuilder
from vllm.v1.attention.backends.utils import mamba_get_block_table_tensor
from vllm.v1.kv_cache_interface import MambaSpec


def _make_vllm_config(
    max_model_len: int,
    max_num_seqs: int,
    num_speculative_tokens: int = 0,
    block_size: int | None = None,
):
    """Create a minimal mock VllmConfig with only the fields the builder
    accesses, avoiding any model download / HF config inspection."""
    speculative_config = (
        SimpleNamespace(
            num_speculative_tokens=num_speculative_tokens,
            parallel_drafting=False,
        )
        if num_speculative_tokens > 0
        else None
    )
    return SimpleNamespace(
        cache_config=SimpleNamespace(
            block_size=block_size,
            mamba_cache_mode="none",
            use_replayssm=False,
            replayssm_buffer_len=16,
        ),
        compilation_config=SimpleNamespace(
            cudagraph_mode=CUDAGraphMode.FULL,
            max_cudagraph_capture_size=None,
        ),
        speculative_config=speculative_config,
        num_speculative_tokens=num_speculative_tokens,
        parallel_config=SimpleNamespace(decode_context_parallel_size=1),
        scheduler_config=SimpleNamespace(max_num_seqs=max_num_seqs),
        model_config=SimpleNamespace(max_model_len=max_model_len),
    )


def test_mamba_single_token_prompt_runs_as_prefill():
    seq_lens = [8, 9, 1]
    config = _make_vllm_config(256, len(seq_lens), block_size=16)
    metadata = MockMambaBuilder.build_mamba_metadata(
        config,
        seq_lens=seq_lens,
        query_lens=[1] * len(seq_lens),
        is_prefilling=[False, False, True],
    )

    assert metadata.num_decodes == 2
    assert metadata.num_prefills == 1
    assert metadata.has_initial_states_p.tolist() == [False]


@pytest.mark.parametrize(
    ("builder_cls", "num_spec"),
    [
        (Mamba1AttentionMetadataBuilder, 0),
        (Mamba2AttentionMetadataBuilder, 0),
        (Mamba2AttentionMetadataBuilder, 2),
    ],
)
@pytest.mark.parametrize("num_prefills", [0, 1])
@pytest.mark.parametrize("mamba_cache_mode", ["none", "align"])
def test_cross_group_build(builder_cls, num_spec, num_prefills, mamba_cache_mode):
    """Share batch metadata while keeping state indices and FULL buffers local."""
    config = _make_vllm_config(48, 4, num_speculative_tokens=num_spec, block_size=16)
    config.use_v2_model_runner = True
    config.cache_config.mamba_cache_mode = mamba_cache_mode
    config.model_config.get_mamba_chunk_size = lambda: 16
    spec = MambaSpec(block_size=16, shapes=((1,), (1,)), dtypes=(torch.float32,))
    first, second, ref = (
        builder_cls(spec, ["layer0"], config, torch.device("cpu")) for _ in range(3)
    )
    batch = BatchSpec(
        seq_lens=[40, 30, 20],
        query_lens=[1 + num_spec, 1 + num_spec, 4 if num_prefills else 1 + num_spec],
    )
    first_common = create_common_attn_metadata(
        batch, 16, torch.device("cpu"), arange_block_indices=True
    ).replace(
        is_prefilling=torch.tensor([False, False, bool(num_prefills)]),
        _cross_group_cache={},
    )
    second_common = first_common.replace(
        block_table_tensor=first_common.block_table_tensor + 100
    )
    kwargs = (
        {"num_accepted_tokens": torch.ones(3, dtype=torch.int32)} if num_spec else {}
    )
    for builder, common in ((first, first_common), (second, second_common)):
        if mamba_cache_mode == "align":
            builder.mamba_aligned_state_indices = mamba_get_block_table_tensor(
                common.block_table_tensor, common.seq_lens, spec, "align"
            )
    first_meta = first.build(0, first_common, **kwargs)
    meta = second.build(0, second_common, **kwargs)
    expected = ref.build(0, second_common.replace(_cross_group_cache=None), **kwargs)

    for field in fields(meta):
        actual = getattr(meta, field.name)
        torch.testing.assert_close(actual, getattr(expected, field.name))
        if field.name.startswith("state_indices_tensor"):
            if num_prefills == 0 and field.name == "state_indices_tensor_d":
                assert actual.data_ptr() == second.state_indices_tensor_d.data_ptr()
        elif num_prefills == 0 and field.name == "num_accepted_tokens" and num_spec:
            assert actual.data_ptr() == second.decode_num_accepted_tokens.data_ptr()
        elif isinstance(actual, torch.Tensor):
            assert actual.data_ptr() == getattr(first_meta, field.name).data_ptr()
        else:
            assert actual is getattr(first_meta, field.name)
