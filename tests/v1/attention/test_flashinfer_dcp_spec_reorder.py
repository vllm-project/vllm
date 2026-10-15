# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""FlashInfer GQA builder: reorder threshold under DCP with spec decode."""

import pytest

from vllm.platforms import current_platform

if not current_platform.is_cuda():
    pytest.skip("FlashInfer backend requires a CUDA platform.", allow_module_level=True)

import torch

from tests.v1.attention.utils import (
    BatchSpec,
    create_common_attn_metadata,
    create_vllm_config,
)
from vllm.config import SpeculativeConfig, set_current_vllm_config
from vllm.v1.attention.backend import AttentionCGSupport
from vllm.v1.attention.backends import flashinfer as flashinfer_backend
from vllm.v1.attention.backends.flashinfer import (
    FlashInferBackend,
    FlashInferDecodeKernel,
    FlashInferMetadataBuilder,
)
from vllm.v1.attention.backends.utils import (
    PerLayerParameters,
    get_dcp_local_seq_lens,
)
from vllm.v1.kv_cache_interface import FullAttentionSpec
from vllm.v1.worker.utils import AttentionGroup


@pytest.fixture
def flashinfer_dcp_vllm_config(tmp_path):
    (tmp_path / "config.json").write_text(
        '{"model_type":"llama","architectures":["LlamaForCausalLM"],'
        '"hidden_size":2048,"num_attention_heads":16,"num_key_value_heads":1,'
        '"num_hidden_layers":1,"max_position_embeddings":128}'
    )
    return create_vllm_config(
        model_name=str(tmp_path),
        tensor_parallel_size=2,
        max_model_len=128,
        max_num_seqs=8,
        max_num_batched_tokens=128,
        dtype="bfloat16",
    )


@pytest.mark.parametrize("comm_backend", ["ag_rs", "fi_a2a_fused"])
def test_flashinfer_gqa_dcp_spec_decode_clamps_reorder_threshold(
    monkeypatch, comm_backend, flashinfer_dcp_vllm_config
):
    """trtllm-gen decode receives no cp_rank/global-seq-len information, so its
    end-aligned causal mask is wrong for q_len > 1 over the DCP-interleaved
    local KV shard. The builder must keep reorder_batch_threshold at 1 under
    DCP so spec queries take the (DCP-aware) prefill path instead.
    """
    vllm_config = flashinfer_dcp_vllm_config
    vllm_config.parallel_config.decode_context_parallel_size = 2
    vllm_config.parallel_config.dcp_comm_backend = comm_backend
    monkeypatch.setattr(
        flashinfer_backend, "get_dcp_world_size_and_rank", lambda _: (2, 1)
    )
    vllm_config.speculative_config = SpeculativeConfig(
        method="ngram", num_speculative_tokens=3
    )

    monkeypatch.setattr(
        flashinfer_backend, "can_use_trtllm_attention", lambda *args, **kwargs: True
    )
    monkeypatch.setattr(
        FlashInferMetadataBuilder,
        "_get_flashinfer_trtllm_api_decode_kernel",
        staticmethod(lambda: FlashInferDecodeKernel.TRTLLM_GEN),
    )
    monkeypatch.setattr(
        flashinfer_backend,
        "get_per_layer_parameters",
        lambda *args, **kwargs: {
            "layer.0": PerLayerParameters(
                window_left=-1, logits_soft_cap=None, sm_scale=0.1, has_sinks=False
            )
        },
    )

    kv_cache_spec = FullAttentionSpec(
        block_size=16,
        num_kv_heads=vllm_config.model_config.get_num_kv_heads(
            vllm_config.parallel_config
        ),
        head_size=vllm_config.model_config.get_head_size(),
        dtype=vllm_config.model_config.dtype,
    )
    group = AttentionGroup(FlashInferBackend, ["layer.0"], kv_cache_spec, 0)
    with set_current_vllm_config(vllm_config):
        group.create_metadata_builders(vllm_config, torch.device("cpu"))
    builder = group.get_metadata_builder()

    assert builder.use_dcp
    if comm_backend == "fi_a2a_fused":
        assert builder.dcp_valid_token_mask.buffer is not None
        assert builder.dcp_valid_token_mask.buffer.shape == (128,)
    else:
        assert builder.dcp_valid_token_mask.buffer is None
    # Guard against passing vacuously with the kernel disabled.
    assert (
        builder.flashinfer_trtllm_api_decode_kernel == FlashInferDecodeKernel.TRTLLM_GEN
    )
    assert builder.reorder_batch_threshold == 1
    assert (
        FlashInferMetadataBuilder.get_cudagraph_support(vllm_config, kv_cache_spec)
        == AttentionCGSupport.UNIFORM_SINGLE_TOKEN_DECODE
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires CUDA")
def test_flashinfer_fused_dcp_builders_refresh_isolated_masks(
    monkeypatch, flashinfer_dcp_vllm_config
):
    """Local-empty rows and the old padded tail are refreshed before graph reuse."""
    vllm_config = flashinfer_dcp_vllm_config
    vllm_config.parallel_config.decode_context_parallel_size = 2
    vllm_config.parallel_config.dcp_comm_backend = "fi_a2a_fused"
    monkeypatch.setattr(
        flashinfer_backend, "get_dcp_world_size_and_rank", lambda _: (2, 1)
    )
    # Planning this metadata path does not invoke an attention kernel.
    monkeypatch.setattr(
        flashinfer_backend, "can_use_trtllm_attention", lambda *args, **kwargs: True
    )
    monkeypatch.setattr(
        FlashInferMetadataBuilder,
        "_get_flashinfer_trtllm_api_decode_kernel",
        staticmethod(lambda: FlashInferDecodeKernel.TRTLLM_GEN),
    )
    monkeypatch.setattr(
        flashinfer_backend,
        "get_per_layer_parameters",
        lambda *args, **kwargs: {
            "layer.0": PerLayerParameters(
                window_left=-1, logits_soft_cap=None, sm_scale=0.1, has_sinks=False
            )
        },
    )
    kv_cache_spec = FullAttentionSpec(
        block_size=16,
        num_kv_heads=1,
        head_size=128,
        dtype=torch.bfloat16,
    )
    device = torch.device("cuda")
    group = AttentionGroup(FlashInferBackend, ["layer.0"], kv_cache_spec, 0)
    with set_current_vllm_config(vllm_config):
        group.create_metadata_builders(vllm_config, device, num_metadata_builders=2)
        assert all(builder.use_dcp for builder in group.metadata_builders)
        masks = []
        for builder in group.metadata_builders:
            mask = builder.dcp_valid_token_mask.buffer
            assert mask is not None and mask.shape == (128,)
            mask.fill_(True)
            masks.append(mask)
        mask_ptrs = [mask.data_ptr() for mask in masks]
        assert mask_ptrs[0] != mask_ptrs[1]
        for slot, seq_lens, query_lens, num_tokens, valid_index in (
            (0, [3, 1, 0, 0], [1, 1, 0, 0], 4, 0),
            (1, [1, 5], [1, 1], 2, 1),
            (0, [1, 5], [1, 1], 2, 1),
        ):
            common = create_common_attn_metadata(
                BatchSpec(seq_lens=seq_lens, query_lens=query_lens),
                block_size=16,
                device=device,
            )
            slot_mapping = torch.full(
                (num_tokens,), -1, dtype=torch.int64, device=device
            )
            slot_mapping[: common.slot_mapping.numel()].copy_(common.slot_mapping)
            common.slot_mapping = slot_mapping
            common.num_actual_tokens = num_tokens
            common.dcp_local_seq_lens = get_dcp_local_seq_lens(common.seq_lens, 2, 1, 1)
            other_mask = masks[1 - slot].clone()
            metadata = group.build_metadata(common, ubatch_idx=slot)
            assert metadata.num_prefills == 0
            assert metadata.num_decodes == len(query_lens)
            assert metadata.num_decode_tokens == num_tokens
            assert metadata.dcp_valid_token_mask is not None
            assert metadata.dcp_valid_token_mask.data_ptr() == mask_ptrs[slot]
            expected = torch.zeros_like(masks[slot])
            expected[valid_index] = True
            torch.testing.assert_close(metadata.dcp_valid_token_mask, expected)
            torch.testing.assert_close(masks[1 - slot], other_mask)
