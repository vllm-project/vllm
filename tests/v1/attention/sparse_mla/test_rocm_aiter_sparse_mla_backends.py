# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Numerical coverage of the no-sink ROCm sparse MLA builder and forward path."""

from types import SimpleNamespace

import pytest
import torch
from transformers import DeepseekV2Config

from tests.v1.attention._mla_backends import create_and_prepopulate_kv_cache
from tests.v1.attention.utils import (
    BatchSpec,
    create_common_attn_metadata,
    create_vllm_config,
)
from vllm import _custom_ops as ops
from vllm.config import set_current_vllm_config
from vllm.model_executor.layers.attention.mla_attention import _DecodeConcatQuantFP8
from vllm.model_executor.layers.quantization.utils.quant_utils import GroupShape
from vllm.platforms import current_platform
from vllm.utils.math_utils import cdiv
from vllm.utils.torch_utils import set_random_seed
from vllm.v1.kv_cache_interface import MLAAttentionSpec

pytestmark = pytest.mark.skipif(
    not current_platform.is_rocm(), reason="ROCm-specific tests"
)

KV_LORA_RANK = 512
ROPE_DIM = 64
HEAD_SIZE = KV_LORA_RANK + ROPE_DIM


@pytest.fixture(autouse=True)
def require_aiter():
    from vllm._aiter_ops import is_aiter_found_and_supported
    from vllm.platforms.rocm import get_cdna_version

    if get_cdna_version() < 3:
        pytest.skip("AITER sparse MLA requires CDNA 3 or newer")
    if not is_aiter_found_and_supported():
        pytest.skip("AITER is required for sparse MLA")


def _batch_spec(batch_name: str, topk: int) -> BatchSpec:
    return {
        "decode": BatchSpec([1, 31, topk - 1, topk + 37], [1, 1, 1, 1]),
        "prefill": BatchSpec([9, topk + 17], [9, 13]),
        "mixed": BatchSpec([17, topk + 31, 7, topk + 9], [1, 1, 7, 11]),
        "multi_token": BatchSpec([topk + 13, topk - 3], [3, 5]),
    }[batch_name]


def _assert_fp8_output_close(actual, expected, value_scale, score_error):
    # Same precision budget as the AITER sink tests, with one output rounding
    # because there is no sink rescaling. Cancellation makes output-relative
    # error unsuitable: rounding P costs u(FP8) * sum(P * abs(V)).
    u_probability = torch.finfo(current_platform.fp8_dtype()).eps / 2
    u_output = torch.finfo(actual.dtype).eps / 2
    relative_error = (1 + u_probability) * torch.exp(2 * score_error) - 1
    allowance = (
        relative_error.unsqueeze(-1) * value_scale * (1 + u_output)
        + u_output * expected.abs()
        + torch.finfo(actual.dtype).tiny * u_output
    )
    error = (actual.double() - expected).abs()
    max_ratio = (error / allowance.clamp_min(1e-300)).max().item()
    assert max_ratio <= 1, f"exceeded FP8 rounding budget by {max_ratio:.3f}x"


def _make_backend(tmp_path, num_heads, topk, block_size, kv_cache_dtype):
    from vllm.v1.attention.backends.mla.rocm_aiter_mla_sparse import (
        ROCMAiterMLASparseBackend,
    )

    DeepseekV2Config(
        architectures=["DeepseekV2ForCausalLM"],
        hidden_size=num_heads * 128,
        num_attention_heads=num_heads,
        num_key_value_heads=1,
        num_hidden_layers=1,
        kv_lora_rank=KV_LORA_RANK,
        qk_nope_head_dim=128,
        qk_rope_head_dim=ROPE_DIM,
        v_head_dim=128,
        index_topk=topk,
        max_position_embeddings=topk + 128,
    ).save_pretrained(tmp_path)
    config = create_vllm_config(
        model_name=str(tmp_path),
        dtype=torch.bfloat16,
        max_model_len=topk + 128,
        max_num_batched_tokens=64,
        max_num_seqs=8,
        block_size=block_size,
    )
    config.cache_config.cache_dtype = kv_cache_dtype
    device = torch.device("cuda")
    scale = 0.7 * HEAD_SIZE**-0.5
    with set_current_vllm_config(config):
        impl = ROCMAiterMLASparseBackend.get_impl_cls()(
            num_heads=num_heads,
            head_size=HEAD_SIZE,
            scale=scale,
            num_kv_heads=1,
            alibi_slopes=None,
            sliding_window=None,
            kv_cache_dtype=kv_cache_dtype,
            logits_soft_cap=None,
            attn_type="decoder",
            kv_sharing_target_layer_name=None,
            kv_lora_rank=KV_LORA_RANK,
        )
        layer = SimpleNamespace(
            impl=impl,
            _q_scale=torch.tensor(0.5, device=device),
            _k_scale=torch.tensor(0.25, device=device),
            _decode_concat_quant_fp8_op=_DecodeConcatQuantFP8(
                static=True, group_shape=GroupShape.PER_TENSOR, compile_native=True
            ),
        )
        config.compilation_config.static_forward_context["test.layer"] = layer
        builder = ROCMAiterMLASparseBackend.get_builder_cls()(
            MLAAttentionSpec(
                block_size=block_size,
                num_kv_heads=1,
                head_size=HEAD_SIZE,
                dtype=torch.bfloat16,
                cache_dtype_str=kv_cache_dtype,
            ),
            ["test.layer"],
            config,
            device,
        )
    return impl, builder, layer


def _run_sparse_batch(impl, builder, layer, batch_spec, randomize_blocks, split_q):
    """Compare physical cache execution with request-local, causal sparse rows."""
    device = torch.device("cuda")
    dtype = torch.bfloat16
    block_size = builder.kv_cache_spec.block_size
    topk = builder.topk_tokens
    num_tokens = batch_spec.compute_num_tokens()
    common = create_common_attn_metadata(batch_spec, block_size, device)
    q = torch.randn(num_tokens, impl.num_heads, HEAD_SIZE, device=device, dtype=dtype)
    keys = [
        torch.randn(seq_len, HEAD_SIZE, device=device, dtype=dtype)
        for seq_len in batch_spec.seq_lens
    ]
    contexts = [
        key[: seq_len - query_len]
        for key, seq_len, query_len in zip(
            keys, batch_spec.seq_lens, batch_spec.query_lens
        )
    ]
    new_keys = torch.cat(
        [key[-query_len:] for key, query_len in zip(keys, batch_spec.query_lens)]
    )
    cache = create_and_prepopulate_kv_cache(
        kv_c_contexts=[key[:, :KV_LORA_RANK] for key in contexts],
        k_pe_contexts=[key[:, None, KV_LORA_RANK:] for key in contexts],
        block_size=block_size,
        head_size=HEAD_SIZE,
        dtype=dtype,
        device=device,
        num_blocks=1
        + sum(cdiv(seq_len, block_size) for seq_len in batch_spec.seq_lens),
        common_attn_metadata=common,
        randomize_blocks=randomize_blocks,
        kv_cache_dtype=impl.kv_cache_dtype,
        scale=layer._k_scale,
    ).squeeze(1)
    ops.concat_and_cache_mla(
        new_keys[:, :KV_LORA_RANK],
        new_keys[:, KV_LORA_RANK:],
        cache,
        common.slot_mapping,
        kv_cache_dtype=impl.kv_cache_dtype,
        scale=layer._k_scale,
    )

    # Quantize the independent logical operands, not values gathered through the
    # backend's physical-index conversion, to keep addressing errors observable.
    if impl.kv_cache_dtype.startswith("fp8"):
        fp8_dtype = current_platform.fp8_dtype()
        q_ref = (q.float() / layer._q_scale).to(fp8_dtype).double() * layer._q_scale
        keys_ref = [
            (key.float() / layer._k_scale).to(fp8_dtype).double() * layer._k_scale
            for key in keys
        ]
    else:
        q_ref = q.double()
        keys_ref = [key.double() for key in keys]

    # Poison unused rows as well as padded columns: only the causal live prefix
    # belongs in the AITER ragged index list.
    indices = torch.full((64, topk), -1, device=device, dtype=torch.int32)
    expected_outputs, expected_indices, expected_lengths = [], [], []
    value_scales, score_errors = [], []
    u_accumulator = torch.finfo(torch.float32).eps / 2
    # FP32 QK accumulation plus conversion/multiplication of the model scale.
    gamma = (HEAD_SIZE + 2) * u_accumulator / (1 - (HEAD_SIZE + 2) * u_accumulator)
    token_idx = 0
    for req_idx, (seq_len, query_len) in enumerate(
        zip(batch_spec.seq_lens, batch_spec.query_lens)
    ):
        for query_idx in range(query_len):
            visible_len = seq_len - query_len + query_idx + 1
            selected = torch.randperm(visible_len, device=device)[:topk]
            num_selected = selected.numel()
            indices[token_idx, :num_selected] = selected.to(torch.int32)
            selected_keys = keys_ref[req_idx][selected]
            scores = q_ref[token_idx] @ selected_keys.T * impl.scale
            probabilities = scores.softmax(-1)
            expected_outputs.append(probabilities @ selected_keys[:, :KV_LORA_RANK])
            value_scales.append(probabilities @ selected_keys[:, :KV_LORA_RANK].abs())
            score_errors.append(
                gamma
                * (q_ref[token_idx].abs() @ selected_keys.abs().T).amax(-1)
                * abs(impl.scale)
            )
            physical = (
                common.block_table_tensor[req_idx, selected // block_size] * block_size
                + selected % block_size
            )
            expected_indices.append(physical)
            expected_lengths.append(num_selected)
            token_idx += 1
    impl.topk_indices_buffer = indices
    metadata = builder.build(0, common)
    assert metadata.work_meta_data is not None
    assert impl.sinks is None
    assert metadata.num_decodes == sum(length == 1 for length in batch_spec.query_lens)
    assert metadata.num_prefills == sum(length > 1 for length in batch_spec.query_lens)
    torch.testing.assert_close(
        metadata.paged_kv_indptr[1:] - metadata.paged_kv_indptr[:-1],
        torch.tensor(expected_lengths, dtype=torch.int32, device=device),
        rtol=0,
        atol=0,
    )

    query = tuple(part.contiguous() for part in q.split([KV_LORA_RANK, ROPE_DIM], -1))
    output, lse = impl.forward_mqa(query if split_q else q, cache, metadata, layer)
    expected = torch.stack(expected_outputs)
    assert lse is None
    assert output.dtype == dtype
    assert output.shape == expected.shape
    assert torch.isfinite(output).all()
    if impl.kv_cache_dtype.startswith("fp8"):
        _assert_fp8_output_close(
            output, expected, torch.stack(value_scales), torch.stack(score_errors)
        )
        single_key = torch.tensor(expected_lengths, device=device) == 1
        torch.testing.assert_close(
            output[single_key], expected[single_key].to(dtype), atol=0, rtol=0
        )
    else:
        torch.testing.assert_close(output, expected.to(dtype), atol=0.01, rtol=0.01)
    physical_indices = torch.cat(expected_indices).to(torch.int32)
    torch.testing.assert_close(
        metadata.paged_kv_indices[: physical_indices.numel()],
        physical_indices,
        atol=0,
        rtol=0,
    )


@pytest.mark.parametrize("batch_name", ["decode", "prefill", "mixed", "multi_token"])
@pytest.mark.parametrize(
    ("num_heads", "topk", "block_size"), [(4, 128, 1), (16, 256, 16), (64, 2048, 64)]
)
@pytest.mark.parametrize("kv_cache_dtype", ["auto", "fp8"])
@pytest.mark.parametrize("randomize_blocks", [False, True], ids=["ordered", "shuffled"])
@torch.inference_mode()
def test_sparse_mla_builder_forward_correctness(
    batch_name,
    num_heads,
    topk,
    block_size,
    kv_cache_dtype,
    randomize_blocks,
    tmp_path,
    workspace_init,
):
    """No-sink AITER numerics include top-k conversion, head padding, and scales."""
    set_random_seed(42)
    impl, builder, layer = _make_backend(
        tmp_path, num_heads, topk, block_size, kv_cache_dtype
    )
    _run_sparse_batch(
        impl,
        builder,
        layer,
        _batch_spec(batch_name, topk),
        randomize_blocks,
        split_q=randomize_blocks,
    )


@pytest.mark.parametrize("kv_cache_dtype", ["auto", "fp8"])
@torch.inference_mode()
def test_sparse_mla_reused_builder_forward_correctness(
    kv_cache_dtype, tmp_path, workspace_init
):
    """Persistent metadata survives repeated, shrinking, and growing batches."""
    set_random_seed(42)
    impl, builder, layer = _make_backend(tmp_path, 16, 256, 16, kv_cache_dtype)
    for batch_name in ["mixed", "mixed", "decode", "prefill", "multi_token"]:
        _run_sparse_batch(
            impl, builder, layer, _batch_spec(batch_name, 256), True, split_q=True
        )
