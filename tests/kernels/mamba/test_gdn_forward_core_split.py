# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Integration test for the non-spec decode split in
``GatedDeltaNet._forward_core``.

On a pure non-spec batch that mixes prefills with 1-token decodes, the layer
peels the decodes (the contiguous decode-first front slice) off to
``fused_sigmoid_gating_delta_rule_update`` -- the same recurrent update kernel
the spec-decode path uses -- and runs only the prefill tail through
``chunk_gated_delta_rule``. This must produce the same core-attention output and
the same ssm-state pool update as running *everything* through
``chunk_gated_delta_rule`` (the previous behavior).

Both paths are exercised through the REAL ``_forward_core``:

* ``meta_split`` is built by the real ``GDNAttentionMetadataBuilder`` for a
  mixed batch, so ``num_decodes > 0`` triggers the peel (and the builder rebases
  ``chunk_indices``/``chunk_offsets`` to the prefill-only tail).
* ``meta_unified`` is the same metadata with the decodes reclassified as
  prefills and full-batch chunk metadata, which forces ``_forward_core`` through
  the existing chunk-only path on identical inputs (the conv is unified over all
  non-spec tokens in both paths, so it cancels out and only the recurrent split
  is compared).

The Triton/FLA chunk backend is forced so the prefill-only ``chunk_indices``
must stay consistent with the rebased ``cu_seqlens`` (a stringent, backend
portable check of the split wiring).
"""

from __future__ import annotations

import dataclasses
import types
from unittest.mock import patch

import pytest
import torch

from vllm.platforms import current_platform

if not (
    current_platform.is_cuda() and current_platform.is_device_capability_family(100)
):
    pytest.skip(
        reason="GDN _forward_core split test uses the CuteDSL prefill backend "
        "(requires CUDA SM10x).",
        allow_module_level=True,
    )

from tests.v1.attention.utils import (  # noqa: E402
    BatchSpec,
    create_common_attn_metadata,
    create_vllm_config,
)
from vllm.config import SpeculativeConfig, set_current_vllm_config  # noqa: E402
from vllm.model_executor.layers.mamba.gdn import qwen_gdn_linear_attn  # noqa: E402
from vllm.model_executor.layers.mamba.gdn.qwen_gdn_linear_attn import (  # noqa: E402
    ChunkGatedDeltaRule,
    QwenGatedDeltaNetAttention,
)
from vllm.model_executor.layers.mamba.mamba_utils import (  # noqa: E402
    MambaStateShapeCalculator,
)
from vllm.third_party.flash_linear_attention.ops.index import (  # noqa: E402
    prepare_chunk_indices,
    prepare_chunk_offsets,
)
from vllm.third_party.flash_linear_attention.ops.layernorm_guard import (  # noqa: E402
    rmsnorm_fn,
)
from vllm.third_party.flash_linear_attention.ops.utils import (  # noqa: E402
    FLA_CHUNK_SIZE,
)
from vllm.utils.torch_utils import _encode_layer_name  # noqa: E402
from vllm.v1.attention.backends.gdn_attn import (  # noqa: E402
    GDNAttentionMetadataBuilder,
)
from vllm.v1.kv_cache_interface import MambaSpec  # noqa: E402

# Small GDN dims; head_k_dim/head_v_dim=128 keeps the chunk/update kernels happy.
H = 4  # num key heads
HV = 8  # num value heads
K = 128  # head_k_dim
V = 128  # head_v_dim
CONV_KERNEL = 4
KEY_DIM = H * K
VALUE_DIM = HV * V
CONV_DIM = 2 * KEY_DIM + VALUE_DIM
BLOCK_SIZE = 16
PREFIX = "model.layers.0.linear_attn"


def _make_vllm_config():
    # A small, ungated GDN model whose config is cached locally; only the config
    # (scheduler/cache/compilation/hf) is used here, never the weights. Inject
    # linear_key_head_dim=128 and request the CuteDSL prefill backend -- the
    # supported GDN chunk kernel on Blackwell (the Triton/FLA chunk kernel is
    # unsupported on SM10x). CuteDSL consumes chunk_indices/chunk_offsets, so
    # this also exercises the prefill-only chunk-metadata wiring.
    cfg = create_vllm_config(
        model_name="Qwen/Qwen3.5-0.8B",
        block_size=BLOCK_SIZE,
        hf_config_override={"linear_key_head_dim": K},
    )
    cfg.additional_config = {"gdn_prefill_backend": "cutedsl"}
    return cfg


def _build_layer(
    vllm_config, conv_state, ssm_state, A_log, dt_bias, conv_weight, conv_bias
):
    """A minimal object that runs the real ``_forward_core`` bound to it."""
    layer = types.SimpleNamespace()
    layer.prefix = PREFIX
    layer.enable_packed_recurrent_decode = False
    layer.tp_size = 1
    layer.num_k_heads = H
    layer.num_v_heads = HV
    layer.head_k_dim = K
    layer.head_v_dim = V
    layer.key_dim = KEY_DIM
    layer.value_dim = VALUE_DIM
    layer.activation = "silu"
    layer.A_log = A_log
    layer.dt_bias = dt_bias
    layer.conv1d = types.SimpleNamespace(weight=conv_weight, bias=conv_bias)
    layer.kv_cache = (conv_state, ssm_state)
    with set_current_vllm_config(vllm_config):
        layer.chunk_gated_delta_rule = ChunkGatedDeltaRule()
    for name in (
        "rearrange_mixed_qkv",
        "_forward_core",
    ):
        setattr(
            layer,
            name,
            types.MethodType(getattr(QwenGatedDeltaNetAttention, name), layer),
        )
    return layer


def _run_forward_core(layer, meta, mixed_qkv, b, a, num_tokens):
    core_attn_out = torch.zeros(
        num_tokens, HV, V, dtype=mixed_qkv.dtype, device=mixed_qkv.device
    )
    ctx = types.SimpleNamespace(attn_metadata={PREFIX: meta})
    with patch.object(qwen_gdn_linear_attn, "get_forward_context", return_value=ctx):
        layer._forward_core(
            mixed_qkv=mixed_qkv.clone(),
            b=b.clone(),
            a=a.clone(),
            core_attn_out=core_attn_out,
        )
    return core_attn_out


@pytest.mark.parametrize("state_dtype", [torch.bfloat16, torch.float32])
@pytest.mark.parametrize("num_decodes,prefill_lens", [(3, [512, 300]), (4, [64, 5])])
@pytest.mark.parametrize("fresh_prefill", [False, True])
def test_forward_core_split_matches_unified(
    state_dtype: torch.dtype,
    num_decodes: int,
    prefill_lens: list[int],
    fresh_prefill: bool,
) -> None:
    torch.manual_seed(0)
    device = torch.device("cuda")
    vllm_config = _make_vllm_config()

    # Decode-first batch: D 1-token decodes (with context), then the prefills.
    decode_seq_lens = [64] * num_decodes
    prefill_seq_lens = [
        pl if (fresh_prefill and i == 0) else pl + 37
        for i, pl in enumerate(prefill_lens)
    ]
    seq_lens = decode_seq_lens + prefill_seq_lens
    query_lens = [1] * num_decodes + list(prefill_lens)
    batch = BatchSpec(seq_lens=seq_lens, query_lens=query_lens)

    builder = GDNAttentionMetadataBuilder(
        # GDNAttentionMetadataBuilder declares kv_cache_spec as AttentionSpec but
        # asserts isinstance(kv_cache_spec, MambaSpec). MambaSpec is correct here.
        kv_cache_spec=MambaSpec(  # type: ignore[arg-type]
            block_size=BLOCK_SIZE, shapes=((16, 64),), dtypes=(torch.float16,)
        ),
        layer_names=[PREFIX],
        vllm_config=vllm_config,
        device=device,
    )
    common = create_common_attn_metadata(
        batch, BLOCK_SIZE, device, arange_block_indices=True
    )
    with set_current_vllm_config(vllm_config):
        meta_split = builder.build(common_prefix_len=0, common_attn_metadata=common)

    assert meta_split.spec_sequence_masks is None
    assert meta_split.num_decodes == num_decodes
    assert meta_split.num_prefills == len(prefill_lens)
    assert meta_split.num_decode_tokens == num_decodes
    assert builder.gdn_prefill_backend == "cutedsl"

    num_tokens = sum(query_lens)

    # Full-batch chunk metadata for the unified reference path, built the same
    # way the builder would for a non-split batch (backend-matched).
    cu_full = meta_split.non_spec_query_start_loc
    assert cu_full is not None
    if builder.gdn_prefill_backend == "cutedsl":
        from vllm.model_executor.layers.mamba.ops.gdn_chunk_cutedsl import (
            prepare_metadata_cutedsl,
        )

        full_ci, full_co = prepare_metadata_cutedsl(
            cu_full, int(cu_full[-1].item()), FLA_CHUNK_SIZE
        )
    else:
        cu_full_cpu = cu_full.cpu()
        full_ci = prepare_chunk_indices(cu_full_cpu, FLA_CHUNK_SIZE).to(device)
        full_co = prepare_chunk_offsets(cu_full_cpu, FLA_CHUNK_SIZE).to(device)
    meta_unified = dataclasses.replace(
        meta_split,
        num_decodes=0,
        num_decode_tokens=0,
        num_prefills=meta_split.num_decodes + meta_split.num_prefills,
        num_prefill_tokens=(
            meta_split.num_decode_tokens + meta_split.num_prefill_tokens
        ),
        chunk_indices=full_ci,
        chunk_offsets=full_co,
        # Unified path: the chunk kernel processes the full non-spec batch.
        prefill_query_start_loc=meta_split.non_spec_query_start_loc,
        prefill_state_indices=meta_split.non_spec_state_indices_tensor,
        prefill_has_initial_state=meta_split.has_initial_state,
    )

    # Size the state pools from the indices the builder actually produced.
    assert meta_split.non_spec_state_indices_tensor is not None
    pool_size = int(meta_split.non_spec_state_indices_tensor.max().item()) + 1
    conv_state_shape, temporal_state_shape = (
        MambaStateShapeCalculator.gated_delta_net_state_shape(
            1, H, HV, K, V, CONV_KERNEL, num_spec=0
        )
    )
    conv_state0 = (
        torch.randn(pool_size, *conv_state_shape, dtype=torch.bfloat16, device=device)
        * 0.05
    )
    ssm_state0 = (
        torch.randn(pool_size, *temporal_state_shape, dtype=state_dtype, device=device)
        * 0.05
    )

    A_log = torch.randn(HV, dtype=torch.float32, device=device) * 0.1
    dt_bias = torch.randn(HV, dtype=torch.float32, device=device) * 0.1
    conv_weight = (
        torch.randn(CONV_DIM, 1, CONV_KERNEL, dtype=torch.bfloat16, device=device) * 0.1
    )
    conv_bias = torch.randn(CONV_DIM, dtype=torch.bfloat16, device=device) * 0.1

    mixed_qkv = (
        torch.randn(num_tokens, CONV_DIM, dtype=torch.bfloat16, device=device) * 0.1
    )
    a = torch.randn(num_tokens, HV, dtype=torch.bfloat16, device=device) * 0.1
    b = torch.randn(num_tokens, HV, dtype=torch.bfloat16, device=device) * 0.1

    # ---- Split path (real _forward_core, meta_split) ----
    conv_state_split = conv_state0.clone()
    ssm_state_split = ssm_state0.clone()
    layer_split = _build_layer(
        vllm_config,
        conv_state_split,
        ssm_state_split,
        A_log,
        dt_bias,
        conv_weight,
        conv_bias,
    )
    out_split = _run_forward_core(layer_split, meta_split, mixed_qkv, b, a, num_tokens)

    # ---- Unified path (real _forward_core, meta_unified) ----
    conv_state_unified = conv_state0.clone()
    ssm_state_unified = ssm_state0.clone()
    layer_unified = _build_layer(
        vllm_config,
        conv_state_unified,
        ssm_state_unified,
        A_log,
        dt_bias,
        conv_weight,
        conv_bias,
    )
    out_unified = _run_forward_core(
        layer_unified, meta_unified, mixed_qkv, b, a, num_tokens
    )

    # Conv is unified in both paths, so the conv-state update must be identical.
    torch.testing.assert_close(conv_state_split, conv_state_unified, atol=0, rtol=0)

    # Chunk vs. recurrent update accumulate in different orders; mirror the
    # tolerances used by the kernel-level parity test.
    if state_dtype == torch.float32:
        atol = rtol = 2e-2
    else:
        atol = rtol = 6e-2
    torch.testing.assert_close(out_split, out_unified, atol=atol, rtol=rtol)
    torch.testing.assert_close(ssm_state_split, ssm_state_unified, atol=atol, rtol=rtol)


class _TestGatedNorm:
    def __init__(self, weight: torch.Tensor, eps: float) -> None:
        self.weight = weight
        self.bias = None
        self.eps = eps
        self.group_size = None
        self.norm_before_gate = True
        self.activation = "silu"

    def __call__(self, x: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        return rmsnorm_fn(
            x,
            self.weight,
            None,
            z=z,
            eps=self.eps,
            norm_before_gate=True,
            activation="silu",
        )


@torch.inference_mode()
def test_rms_norm_gated_cuda_supports_inplace_output() -> None:
    """The generic prefill/decode path can normalize its output in place."""
    torch.manual_seed(2)
    device = torch.device("cuda")
    x = torch.randn(37, HV, V, dtype=torch.bfloat16, device=device)
    output_gate = torch.randn_like(x)
    norm = _TestGatedNorm(torch.randn(V, dtype=torch.float32, device=device), eps=1e-6)
    expected = norm(x.clone(), output_gate)
    layer = types.SimpleNamespace(norm=norm)
    layer._rms_norm_gated_cuda = types.MethodType(
        QwenGatedDeltaNetAttention._rms_norm_gated_cuda, layer
    )

    layer._rms_norm_gated_cuda(x, output_gate, x)

    torch.testing.assert_close(x, expected, atol=3e-2, rtol=3e-2)


def _build_mixed_mtp_layer(
    vllm_config,
    conv_state: torch.Tensor,
    ssm_state: torch.Tensor,
    a_log: torch.Tensor,
    dt_bias: torch.Tensor,
    conv_weight: torch.Tensor,
    norm_weight: torch.Tensor,
    eps: float,
):
    """Build the smallest layer object that runs the mixed fused path."""
    layer = types.SimpleNamespace(
        prefix=PREFIX,
        enable_packed_recurrent_decode=False,
        disable_tp_for_ba_proj=False,
        tp_size=1,
        num_k_heads=1,
        num_v_heads=8,
        head_k_dim=K,
        head_v_dim=V,
        key_dim=K,
        value_dim=8 * V,
        activation="silu",
        A_log=a_log,
        dt_bias=dt_bias,
        conv1d=types.SimpleNamespace(weight=conv_weight, bias=None),
        kv_cache=(conv_state, ssm_state),
        norm=_TestGatedNorm(norm_weight, eps),
        layer_norm_epsilon=eps,
        gdn_decode_kernel="phase_a",
    )
    with set_current_vllm_config(vllm_config):
        layer.chunk_gated_delta_rule = ChunkGatedDeltaRule()
    for name in (
        "rearrange_mixed_qkv",
        "_forward_core",
        "_forward_core_decode_spec_post_conv_fused_norm",
        "_forward_core_decode_spec_fused_norm",
        "_can_use_fused_gdn_mtp_decode",
        "_rms_norm_gated_cuda",
        "_forward_core_fused_norm",
        "_forward_core_fused_norm_packed",
        "split_ba",
    ):
        setattr(
            layer,
            name,
            types.MethodType(getattr(QwenGatedDeltaNetAttention, name), layer),
        )
    return layer


@torch.inference_mode()
def test_phase_a_forward_uses_packed_decode_entrypoint() -> None:
    """Phase A keeps projected QKVZ and BA packed through the model op."""
    device = torch.device("cuda")
    num_tokens = 3
    hidden_states = torch.empty(num_tokens, 1, dtype=torch.bfloat16, device=device)
    mixed_qkvz = torch.randn(
        num_tokens, CONV_DIM + VALUE_DIM, dtype=torch.bfloat16, device=device
    )
    ba = torch.randn(num_tokens, 2 * HV, dtype=torch.bfloat16, device=device)
    layer = types.SimpleNamespace(
        prefix=PREFIX,
        enable_fused_gdn_decode=True,
        norm=types.SimpleNamespace(
            weight=torch.empty(V, dtype=torch.bfloat16, device=device)
        ),
        num_v_heads=HV,
        tp_size=1,
        head_v_dim=V,
        in_proj_qkvz=lambda _: (mixed_qkvz, None),
        in_proj_ba=lambda _: (ba, None),
        _output_projection_normalized=lambda output: output,
    )
    layer.forward_cuda = types.MethodType(
        QwenGatedDeltaNetAttention.forward_cuda, layer
    )

    def packed_op(
        actual_qkvz: torch.Tensor,
        actual_ba: torch.Tensor,
        output: torch.Tensor,
        *,
        layer_name: str,
    ) -> None:
        assert actual_qkvz is mixed_qkvz
        assert actual_ba is ba
        assert layer_name == _encode_layer_name(PREFIX)
        output.fill_(1)

    with (
        patch.object(
            torch.ops.vllm,
            "qwen_gdn_attention_core_fused_norm_packed",
            side_effect=packed_op,
        ) as packed_mock,
        patch.object(torch.ops.vllm, "qwen_gdn_attention_core") as triton_mock,
    ):
        output = layer.forward_cuda(hidden_states)

    packed_mock.assert_called_once()
    triton_mock.assert_not_called()
    torch.testing.assert_close(output, torch.ones_like(output))


@torch.inference_mode()
def test_mixed_prefill_mtp_uses_fused_decode() -> None:
    """Continuous batching keeps MTP decode fused when prefill is present."""
    torch.manual_seed(1)
    device = torch.device("cuda")
    num_spec = 3
    spec_tokens = num_spec + 1
    prefill_tokens = 64
    local_h = 1
    local_hv = 8
    conv_dim = 2 * local_h * K + local_hv * V
    eps = 1e-6

    vllm_config = _make_vllm_config()
    vllm_config.cache_config.mamba_cache_mode = "none"
    vllm_config.speculative_config = SpeculativeConfig(
        method="ngram", num_speculative_tokens=num_spec
    )
    builder = GDNAttentionMetadataBuilder(
        kv_cache_spec=MambaSpec(
            block_size=BLOCK_SIZE,
            shapes=((16, 64),),
            dtypes=(torch.float16,),
            num_speculative_blocks=num_spec,
        ),
        layer_names=[PREFIX],
        vllm_config=vllm_config,
        device=device,
    )
    batch = BatchSpec(
        seq_lens=[128, 96],
        query_lens=[spec_tokens, prefill_tokens],
    )
    common = create_common_attn_metadata(
        batch, BLOCK_SIZE, device, arange_block_indices=True
    )
    common.block_table_tensor.add_(1)
    with set_current_vllm_config(vllm_config):
        metadata = builder.build(
            common_prefix_len=0,
            common_attn_metadata=common,
            num_accepted_tokens=torch.ones(
                batch.batch_size, dtype=torch.int32, device=device
            ),
            num_decode_draft_tokens_cpu=torch.tensor([num_spec, -1], dtype=torch.int32),
        )

    assert metadata.num_spec_decodes == 1
    assert metadata.num_prefills == 1
    assert metadata.spec_token_indx is not None
    assert metadata.non_spec_token_indx is not None
    assert metadata.spec_token_prefix_len == spec_tokens

    state_indices = (
        metadata.spec_state_indices_tensor,
        metadata.non_spec_state_indices_tensor,
    )
    pool_size = max(int(indices.max().item()) for indices in state_indices) + 1
    conv_state_shape, temporal_state_shape = (
        MambaStateShapeCalculator.gated_delta_net_state_shape(
            1, local_h, local_hv, K, V, CONV_KERNEL
        )
    )
    conv_state_seed = 0.05 * torch.randn(
        pool_size, *conv_state_shape, dtype=torch.bfloat16, device=device
    )
    ssm_state_seed = 0.01 * torch.randn(
        pool_size, *temporal_state_shape, dtype=torch.float32, device=device
    )
    a_log = 0.1 * torch.randn(local_hv, dtype=torch.float32, device=device)
    dt_bias = 0.1 * torch.randn(local_hv, dtype=torch.float32, device=device)
    conv_weight = 0.1 * torch.randn(
        conv_dim, 1, CONV_KERNEL, dtype=torch.bfloat16, device=device
    )
    norm_weight = torch.randn(V, dtype=torch.float32, device=device)
    num_tokens = batch.compute_num_tokens()
    mixed_qkv = 0.1 * torch.randn(
        num_tokens, conv_dim, dtype=torch.bfloat16, device=device
    )
    a = 0.1 * torch.randn(num_tokens, local_hv, dtype=torch.bfloat16, device=device)
    b = 0.1 * torch.randn_like(a)
    output_gate = 0.1 * torch.randn(
        num_tokens, local_hv, V, dtype=torch.bfloat16, device=device
    )
    mixed_qkvz = torch.cat((mixed_qkv, output_gate.flatten(1)), dim=-1)
    ba = torch.cat((b, a), dim=-1)
    context = types.SimpleNamespace(attn_metadata={PREFIX: metadata})

    reference_layer = _build_mixed_mtp_layer(
        vllm_config,
        conv_state_seed.clone(),
        ssm_state_seed.clone(),
        a_log,
        dt_bias,
        conv_weight,
        norm_weight,
        eps,
    )
    reference_out = torch.zeros_like(output_gate)
    with patch.object(
        qwen_gdn_linear_attn, "get_forward_context", return_value=context
    ):
        reference_layer._forward_core(
            mixed_qkv=mixed_qkv.clone(),
            b=b,
            a=a,
            core_attn_out=reference_out,
        )
    reference_out = reference_layer.norm(reference_out, output_gate)

    fused_layer = _build_mixed_mtp_layer(
        vllm_config,
        conv_state_seed.clone(),
        ssm_state_seed.clone(),
        a_log,
        dt_bias,
        conv_weight,
        norm_weight,
        eps,
    )
    context.no_compile_layers = {PREFIX: fused_layer}
    fused_out = torch.zeros_like(output_gate)
    fused_op = qwen_gdn_linear_attn.ops.fused_gdn_decode_post_conv_mtp
    with (
        patch.object(qwen_gdn_linear_attn, "get_forward_context", return_value=context),
        patch.object(
            qwen_gdn_linear_attn.ops,
            "fused_gdn_decode_post_conv_mtp",
            wraps=fused_op,
        ) as fused_mock,
    ):
        torch.ops.vllm.qwen_gdn_attention_core_fused_norm_packed(
            mixed_qkvz.clone(),
            ba,
            fused_out,
            layer_name=_encode_layer_name(PREFIX),
        )

    assert fused_mock.call_count == 1

    indexed_metadata = dataclasses.replace(metadata, spec_token_prefix_len=None)
    indexed_layer = _build_mixed_mtp_layer(
        vllm_config,
        conv_state_seed.clone(),
        ssm_state_seed.clone(),
        a_log,
        dt_bias,
        conv_weight,
        norm_weight,
        eps,
    )
    indexed_context = types.SimpleNamespace(
        attn_metadata={PREFIX: indexed_metadata},
        no_compile_layers={PREFIX: indexed_layer},
    )
    indexed_out = torch.zeros_like(output_gate)
    with patch.object(
        qwen_gdn_linear_attn,
        "get_forward_context",
        return_value=indexed_context,
    ):
        torch.ops.vllm.qwen_gdn_attention_core_fused_norm_packed(
            mixed_qkvz.clone(),
            ba,
            indexed_out,
            layer_name=_encode_layer_name(PREFIX),
        )

    torch.testing.assert_close(fused_out, indexed_out, atol=3e-2, rtol=3e-2)
    torch.testing.assert_close(
        fused_layer.kv_cache[0], indexed_layer.kv_cache[0], atol=0, rtol=0
    )
    torch.testing.assert_close(
        fused_layer.kv_cache[1],
        indexed_layer.kv_cache[1],
        atol=3e-2,
        rtol=3e-2,
    )
    torch.testing.assert_close(
        fused_layer.kv_cache[0], reference_layer.kv_cache[0], atol=0, rtol=0
    )
    torch.testing.assert_close(fused_out, reference_out, atol=3e-2, rtol=3e-2)
    torch.testing.assert_close(
        fused_layer.kv_cache[1],
        reference_layer.kv_cache[1],
        atol=3e-2,
        rtol=3e-2,
    )
