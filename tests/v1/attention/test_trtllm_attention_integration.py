# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Integration tests for TRTLLM gen-full attention through FlashInfer."""

import unittest.mock
from functools import partial

import pytest
import torch
from torch.nn.attention.flex_attention import create_block_mask, flex_attention

from tests.v1.attention.utils import (
    BatchSpec,
    create_common_attn_metadata,
    create_vllm_config,
)
from vllm.config import set_current_vllm_config
from vllm.platforms import current_platform
from vllm.utils.math_utils import cdiv
from vllm.utils.torch_utils import set_random_seed
from vllm.v1.attention.backends.utils import (
    PerLayerParameters,
    get_kv_cache_layout,
    set_kv_cache_layout,
)
from vllm.v1.kv_cache_interface import FullAttentionSpec

if not current_platform.is_device_capability_family(100):
    pytest.skip(
        "TRTLLM integration tests require NVIDIA Blackwell (SM100).",
        allow_module_level=True,
    )

from vllm.v1.attention.backends.flashinfer import (  # noqa: E402
    FlashInferImpl,
    FlashInferMetadataBuilder,
    TRTLLMDecode,
    TRTLLMPrefill,
)


class MockAttentionLayer:
    """Minimal mock of an attention layer for testing."""

    def __init__(self, device: torch.device):
        self._q_scale = torch.tensor(1.0, device=device)
        self._k_scale = torch.tensor(1.0, device=device)
        self._v_scale = torch.tensor(1.0, device=device)
        self._q_scale_float = 1.0
        self._k_scale_float = 1.0
        self._v_scale_float = 1.0
        self._o_scale_float = None


MODEL = "Qwen/Qwen2.5-0.5B"
BLOCK_SIZE = 16
NUM_GPU_BLOCKS = 8192

BATCH_SPECS = {
    "decode_only": BatchSpec(
        seq_lens=[128, 256, 512],
        query_lens=[1, 1, 1],
    ),
    "prefill_only": BatchSpec(
        seq_lens=[64, 128, 256],
        query_lens=[16, 32, 16],
    ),
    "mixed": BatchSpec(
        seq_lens=[128, 256, 512, 128],
        query_lens=[1, 1, 8, 16],
    ),
}


def _mock_get_per_layer_parameters(vllm_config, layer_names, impl_cls):
    head_size = vllm_config.model_config.get_head_size()
    return {
        name: PerLayerParameters(
            window_left=-1,
            logits_soft_cap=0.0,
            sm_scale=1.0 / (head_size**0.5),
        )
        for name in layer_names
    }


def _create_hnd_kv_cache(
    k_contexts,
    v_contexts,
    block_size,
    num_kv_heads,
    head_size,
    dtype,
    device,
    num_blocks,
    common_attn_metadata,
):
    """Create and populate a KV cache with HND-compatible strides.

    The returned tensor has logical shape
    (num_blocks, 2, block_size, num_kv_heads, head_size) but is physically
    laid out as (num_blocks, 2, num_kv_heads, block_size, head_size) so that
    ``kv_cache.permute(0, 1, 3, 2, 4)`` yields a contiguous HND view.
    """
    seq_lens = common_attn_metadata.seq_lens.cpu()
    query_lens = (
        common_attn_metadata.query_start_loc_cpu[1:]
        - common_attn_metadata.query_start_loc_cpu[:-1]
    )
    block_table = common_attn_metadata.block_table_tensor
    slot_mapping = common_attn_metadata.slot_mapping
    batch_size = len(k_contexts)

    # Build cache in (2, num_blocks, block_size, num_kv_heads, head_size)
    # then convert to HND format (same approach as test_attention_backends.py).
    kv_cache_raw = torch.zeros(
        2,
        num_blocks,
        block_size,
        num_kv_heads,
        head_size,
        dtype=dtype,
        device=device,
    )
    kv_cache_flat = kv_cache_raw.view(2, -1, num_kv_heads, head_size)

    start_block_idx = 1
    for i in range(batch_size):
        k_ctx, v_ctx = k_contexts[i], v_contexts[i]
        start = start_block_idx * block_size
        end = start + k_ctx.shape[0]
        kv_cache_flat[0, start:end] = k_ctx
        kv_cache_flat[1, start:end] = v_ctx
        start_block_idx += cdiv(int(seq_lens[i]), block_size)

    blocks_end = start_block_idx

    # Randomly permute blocks (starting from block 1; block 0 is null).
    perm = torch.randperm(blocks_end - 1) + 1
    inv_perm = torch.zeros(blocks_end, dtype=torch.long, device=device)
    inv_perm[1:] = torch.argsort(perm) + 1
    kv_cache_raw[:, 1:blocks_end] = kv_cache_raw[:, perm]

    # Build block table.
    start_block_idx = 1
    for i in range(batch_size):
        n_blocks = cdiv(int(seq_lens[i]), block_size)
        block_table[i, :n_blocks] = inv_perm[
            start_block_idx : start_block_idx + n_blocks
        ]
        start_block_idx += n_blocks

    # Build slot mapping that is consistent with the block table.
    for i in range(batch_size):
        ctx_len = int(seq_lens[i]) - int(query_lens[i])
        token_offsets = torch.arange(int(query_lens[i])) + ctx_len
        block_indices = token_offsets // block_size
        intra_block_offsets = token_offsets % block_size
        start = common_attn_metadata.query_start_loc_cpu[i]
        end = common_attn_metadata.query_start_loc_cpu[i + 1]
        slot_mapping[start:end] = block_table[
            i, block_indices
        ] * block_size + intra_block_offsets.to(device)

    # Transpose to FlashInfer logical shape then make HND-strided.
    kv_cache = kv_cache_raw.transpose(0, 1)
    kv_cache = kv_cache.transpose(2, 3).contiguous().transpose(2, 3)
    return kv_cache


def _run_trtllm_integration(batch_spec):
    """Run TRTLLM attention through the full FlashInfer pipeline
    and compare against an SDPA reference."""
    set_random_seed(42)
    device = torch.device("cuda:0")

    vllm_config = create_vllm_config(
        model_name=MODEL,
        max_model_len=max(batch_spec.seq_lens),
        block_size=BLOCK_SIZE,
        num_gpu_blocks=NUM_GPU_BLOCKS,
    )
    vllm_config.attention_config.use_trtllm_attention = True

    num_q_heads = vllm_config.model_config.get_num_attention_heads(
        vllm_config.parallel_config
    )
    num_kv_heads = vllm_config.model_config.get_num_kv_heads(
        vllm_config.parallel_config
    )
    head_size = vllm_config.model_config.get_head_size()
    dtype = vllm_config.model_config.dtype
    scale = 1.0 / (head_size**0.5)

    # 1. Generate data and compute SDPA reference
    all_q, all_k, all_v = [], [], []
    all_sdpa_out = []
    k_contexts, v_contexts = [], []

    for i in range(batch_spec.batch_size):
        s_len = batch_spec.seq_lens[i]
        q_len = batch_spec.query_lens[i]
        ctx_len = s_len - q_len

        q = torch.randn(q_len, num_q_heads, head_size, dtype=dtype, device=device)
        k_full = torch.randn(s_len, num_kv_heads, head_size, dtype=dtype, device=device)
        v_full = torch.randn(s_len, num_kv_heads, head_size, dtype=dtype, device=device)

        # SDPA reference (N=1, H, L, D)
        q_sdpa = q.unsqueeze(0).transpose(1, 2)
        k_sdpa = k_full.unsqueeze(0).transpose(1, 2)
        v_sdpa = v_full.unsqueeze(0).transpose(1, 2)

        if num_q_heads != num_kv_heads:
            repeats = num_q_heads // num_kv_heads
            k_sdpa = k_sdpa.repeat_interleave(repeats, dim=1)
            v_sdpa = v_sdpa.repeat_interleave(repeats, dim=1)

        def causal_mask_mod(b, h, q_idx, kv_idx, *, context_len):
            return (q_idx + context_len) >= kv_idx

        mask_fn = partial(causal_mask_mod, context_len=ctx_len)
        block_mask = create_block_mask(
            mask_fn, B=None, H=None, Q_LEN=q_len, KV_LEN=s_len, device=device
        )
        sdpa_out = flex_attention(
            q_sdpa,
            k_sdpa,
            v_sdpa,
            block_mask=block_mask,
            scale=scale,
            enable_gqa=True,
        )
        all_sdpa_out.append(sdpa_out.transpose(1, 2).squeeze(0))

        all_q.append(q)
        all_k.append(k_full[ctx_len:])
        all_v.append(v_full[ctx_len:])
        k_contexts.append(k_full[:ctx_len])
        v_contexts.append(v_full[:ctx_len])

    query_vllm = torch.cat(all_q, dim=0)
    key_vllm = torch.cat(all_k, dim=0)
    value_vllm = torch.cat(all_v, dim=0)
    sdpa_output = torch.cat(all_sdpa_out, dim=0)

    common_attn_metadata = create_common_attn_metadata(batch_spec, BLOCK_SIZE, device)

    # 2. Create HND KV cache
    kv_cache = _create_hnd_kv_cache(
        k_contexts,
        v_contexts,
        BLOCK_SIZE,
        num_kv_heads,
        head_size,
        dtype,
        device,
        NUM_GPU_BLOCKS,
        common_attn_metadata,
    )

    # 3. Run through FlashInfer with TRTLLM enabled
    set_kv_cache_layout("HND")
    get_kv_cache_layout.cache_clear()

    try:
        kv_cache_spec = FullAttentionSpec(
            block_size=BLOCK_SIZE,
            num_kv_heads=num_kv_heads,
            head_size=head_size,
            dtype=dtype,
        )
        layer_names = ["test_layer_0"]

        with (
            set_current_vllm_config(vllm_config),
            unittest.mock.patch(
                "vllm.utils.flashinfer.supports_trtllm_attention",
                return_value=True,
            ),
            unittest.mock.patch(
                "vllm.v1.attention.backends.flashinfer.get_per_layer_parameters",
                _mock_get_per_layer_parameters,
            ),
        ):
            builder = FlashInferMetadataBuilder(
                kv_cache_spec, layer_names, vllm_config, device
            )
            attn_metadata = builder.build(
                common_prefix_len=0,
                common_attn_metadata=common_attn_metadata,
            )

            # Verify the correct TRTLLM metadata types were produced.
            has_prefills = any(ql > 1 for ql in batch_spec.query_lens)
            has_decodes = any(ql == 1 for ql in batch_spec.query_lens)

            if has_prefills:
                assert isinstance(attn_metadata.prefill, TRTLLMPrefill), (
                    f"Expected TRTLLMPrefill, got {type(attn_metadata.prefill)}"
                )
            if has_decodes:
                assert isinstance(attn_metadata.decode, TRTLLMDecode), (
                    f"Expected TRTLLMDecode, got {type(attn_metadata.decode)}"
                )

            impl = FlashInferImpl(
                num_heads=num_q_heads,
                head_size=head_size,
                scale=scale,
                num_kv_heads=num_kv_heads,
                alibi_slopes=None,
                sliding_window=None,
                kv_cache_dtype="auto",
            )

            mock_layer = MockAttentionLayer(device)
            output = torch.empty_like(query_vllm)

            output = impl.forward(
                mock_layer,
                query_vllm,
                key_vllm,
                value_vllm,
                kv_cache,
                attn_metadata,
                output=output,
            )

        # 4. Compare against SDPA reference
        torch.testing.assert_close(
            output,
            sdpa_output,
            atol=1e-2,
            rtol=1e-2,
        )

    finally:
        set_kv_cache_layout(None)
        get_kv_cache_layout.cache_clear()


@pytest.mark.parametrize(
    "batch_spec_name",
    list(BATCH_SPECS.keys()),
)
@torch.inference_mode()
def test_trtllm_gen_full_attention_integration(batch_spec_name: str):
    """Test TRTLLM gen-full attention through the full FlashInfer
    MetadataBuilder.build() -> FlashInferImpl.forward() pipeline,
    with real TRTLLM kernels on Blackwell."""
    _run_trtllm_integration(BATCH_SPECS[batch_spec_name])
