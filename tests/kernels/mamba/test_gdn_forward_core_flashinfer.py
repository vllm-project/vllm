# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The FlashInfer prefill routes of ``QwenGatedDeltaNetAttention._forward_core``.

Each batch runs through the real ``_forward_core`` on metadata from the real
``GDNAttentionMetadataBuilder``, along every route it can take:

* state pool (``prefill_initial_state_mode`` "none"/"all"): FlashInfer reads
  and writes the SSM cache slots in place;
* gather (forced "mixed"): the caller gathers, masks and scatters states;
* the CuteDSL chunk kernel, an independent reference for the whole layer
  (conv, output placement behind peeled decodes, state writes);
* with internal prefill checkpoints (align mode), pool vs gather.

``core_attn_out`` is allocated uninitialized; it is NaN-poisoned here so any
actual row the layer fails to write shows up.
"""

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
        reason="Compares the FlashInfer and CuteDSL GDN prefill (CUDA SM10x).",
        allow_module_level=True,
    )

from tests.v1.attention.utils import (  # noqa: E402
    BatchSpec,
    create_common_attn_metadata,
    create_vllm_config,
)
from vllm.config import set_current_vllm_config  # noqa: E402
from vllm.model_executor.layers.mamba.checkpoint import (  # noqa: E402
    MambaPrefillCheckpointExporter,
    MambaPrefillCheckpointMetadata,
)
from vllm.model_executor.layers.mamba.gdn import qwen_gdn_linear_attn  # noqa: E402
from vllm.model_executor.layers.mamba.gdn.qwen_gdn_linear_attn import (  # noqa: E402
    ChunkGatedDeltaRule,
    QwenGatedDeltaNetAttention,
)
from vllm.model_executor.layers.mamba.mamba_utils import (  # noqa: E402
    MambaStateShapeCalculator,
)
from vllm.v1.attention.backends.gdn_attn import (  # noqa: E402
    GDNAttentionMetadataBuilder,
    GDNFlashInferCheckpointPlan,
    plan_flashinfer_checkpoints,
)
from vllm.v1.attention.backends.utils import NULL_BLOCK_ID  # noqa: E402
from vllm.v1.kv_cache_interface import MambaSpec  # noqa: E402

H, HV, K, V = 4, 8, 128, 128
CONV_KERNEL = 4
CONV_DIM = 2 * H * K + HV * V
BLOCK_SIZE = 16
PREFIX = "model.layers.0.linear_attn"
PAD = 16  # CUDA graph padding rows behind num_actual_tokens


def _builder(backend: str):
    cfg = create_vllm_config(
        model_name="Qwen/Qwen3.5-0.8B",
        block_size=BLOCK_SIZE,
        hf_config_override={"linear_key_head_dim": K},
    )
    cfg.additional_config = {"gdn_prefill_backend": backend}
    builder = GDNAttentionMetadataBuilder(
        kv_cache_spec=MambaSpec(
            block_size=BLOCK_SIZE, shapes=((16, 64),), dtypes=(torch.float16,)
        ),
        layer_names=[PREFIX],
        vllm_config=cfg,
        device=torch.device("cuda"),
    )
    assert builder.gdn_prefill_backend == backend
    return cfg, builder


def _layer(cfg, weights, conv_state, ssm_state, packed_decode=False):
    layer = types.SimpleNamespace(
        prefix=PREFIX,
        enable_packed_recurrent_decode=packed_decode,
        tp_size=1,
        num_k_heads=H,
        num_v_heads=HV,
        head_k_dim=K,
        head_v_dim=V,
        key_dim=H * K,
        value_dim=HV * V,
        activation="silu",
        A_log=weights["A_log"],
        dt_bias=weights["dt_bias"],
        conv1d=types.SimpleNamespace(
            weight=weights["conv_weight"], bias=weights["conv_bias"]
        ),
        kv_cache=(conv_state, ssm_state),
        _checkpoint_buffer_spec=((64, HV, V, K), torch.float32),
        _checkpoint_exporter=MambaPrefillCheckpointExporter(state_len=CONV_KERNEL - 1),
    )
    with set_current_vllm_config(cfg):
        layer.chunk_gated_delta_rule = ChunkGatedDeltaRule()
    layer.gdn_prefill_backend = layer.chunk_gated_delta_rule.gdn_prefill_backend
    for name in (
        "rearrange_mixed_qkv",
        "_forward_core",
        "_forward_core_decode_non_spec",
        "_forward_flashinfer_prefill",
    ):
        method = getattr(QwenGatedDeltaNetAttention, name)
        setattr(layer, name, types.MethodType(method, layer))
    return layer


def _run(cfg, meta, weights, inputs, conv_state0, ssm_state0, packed_decode=False):
    """Returns (core_attn_out with padding rows, conv_state, ssm_state)."""
    conv_state, ssm_state = conv_state0.clone(), ssm_state0.clone()
    layer = _layer(cfg, weights, conv_state, ssm_state, packed_decode)
    mixed_qkv, b, a = inputs
    num_tokens = mixed_qkv.shape[0] + PAD
    # Uninitialized in serving; NaN here so any row left unwritten shows.
    core_attn_out = torch.full(
        (num_tokens, HV, V), float("nan"), dtype=mixed_qkv.dtype, device="cuda"
    )

    def pad(x):
        return torch.cat([x, torch.zeros_like(x[:PAD])])

    ctx = types.SimpleNamespace(attn_metadata={PREFIX: meta})
    with patch.object(qwen_gdn_linear_attn, "get_forward_context", return_value=ctx):
        layer._forward_core(
            mixed_qkv=pad(mixed_qkv), b=pad(b), a=pad(a), core_attn_out=core_attn_out
        )
    return core_attn_out, conv_state, ssm_state


def _setup(seq_lens, query_lens, state_dtype, gen):
    batch = BatchSpec(seq_lens=seq_lens, query_lens=query_lens)
    common = create_common_attn_metadata(
        batch, BLOCK_SIZE, torch.device("cuda"), arange_block_indices=True
    )
    pool_size = int(common.block_table_tensor.max().item()) + 2
    conv_shape, ssm_shape = MambaStateShapeCalculator.gated_delta_net_state_shape(
        1, H, HV, K, V, CONV_KERNEL, num_spec=0
    )

    def randn(*shape, dtype=torch.bfloat16, scale=0.1):
        return (
            torch.randn(*shape, generator=gen, device="cuda", dtype=torch.float32)
            * scale
        ).to(dtype)

    conv_state0 = randn(pool_size, *conv_shape, scale=0.05)
    ssm_state0 = randn(pool_size, *ssm_shape, dtype=state_dtype, scale=0.05)
    weights = dict(
        A_log=randn(HV, dtype=torch.float32),
        dt_bias=randn(HV, dtype=torch.float32),
        conv_weight=randn(CONV_DIM, 1, CONV_KERNEL),
        conv_bias=randn(CONV_DIM),
    )
    num_tokens = sum(query_lens)
    inputs = (randn(num_tokens, CONV_DIM), randn(num_tokens, HV), randn(num_tokens, HV))
    return common, weights, inputs, conv_state0, ssm_state0


# (decode seq lens, prefill seq lens, prefill query lens, expected mode)
CASES = {
    "one_fresh": ([], [300], [300], "none"),
    "one_cached": ([], [337], [300], "all"),
    "fresh_rows": ([], [200, 64, 5], [200, 64, 5], "none"),
    "cached_rows": ([], [237, 101, 42], [200, 64, 5], "all"),
    "mixed_rows": ([], [237, 64, 42], [200, 64, 5], "mixed"),
    "decodes_fresh": ([64, 64, 64], [300, 64], [300, 64], "none"),
    "decodes_cached": ([64, 64, 64], [337], [300], "all"),
    # Packed decode (its own early-return path) vs the peeled-decode kernel.
    "decodes_only": ([64, 64, 64], [], [], "mixed"),
}


@pytest.mark.parametrize("state_dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("case", CASES.values(), ids=CASES.keys())
def test_flashinfer_prefill_routes_match(case, state_dtype):
    decode_seq_lens, prefill_seq_lens, prefill_query_lens, mode = case
    gen = torch.Generator(device="cuda").manual_seed(0)
    seq_lens = decode_seq_lens + prefill_seq_lens
    query_lens = [1] * len(decode_seq_lens) + prefill_query_lens
    common, weights, inputs, conv_state0, ssm_state0 = _setup(
        seq_lens, query_lens, state_dtype, gen
    )
    num_actual = sum(query_lens)

    fi_cfg, fi_builder = _builder("flashinfer")
    with set_current_vllm_config(fi_cfg):
        meta = fi_builder.build(common_prefix_len=0, common_attn_metadata=common)
    assert meta.prefill_initial_state_mode == mode
    meta_gather = dataclasses.replace(meta, prefill_initial_state_mode="mixed")
    ref_cfg, ref_builder = _builder("cutedsl")
    with set_current_vllm_config(ref_cfg):
        meta_ref = ref_builder.build(common_prefix_len=0, common_attn_metadata=common)

    packed = not prefill_seq_lens
    runs = {
        "pool": _run(fi_cfg, meta, weights, inputs, conv_state0, ssm_state0, packed),
        "gather": _run(fi_cfg, meta_gather, weights, inputs, conv_state0, ssm_state0),
        "cutedsl": _run(ref_cfg, meta_ref, weights, inputs, conv_state0, ssm_state0),
    }
    for out, _, _ in runs.values():
        assert not out[:num_actual].isnan().any(), "unwritten output rows"

    out, conv_state, ssm_state = runs["pool"]
    # The pool route changes only where FlashInfer reads/writes states.
    # Outputs are compared over the actual rows only: padding rows behind
    # num_actual_tokens are uninitialized by contract.
    gather_out, gather_conv, gather_ssm = runs["gather"]
    torch.testing.assert_close(
        out[:num_actual], gather_out[:num_actual], atol=1e-3, rtol=1e-3
    )
    torch.testing.assert_close(conv_state, gather_conv, atol=0, rtol=0)
    torch.testing.assert_close(ssm_state, gather_ssm, atol=1e-3, rtol=1e-3)
    # Different chunk kernels: tolerances of the kernel-level parity tests.
    tol = 2e-2 if state_dtype == torch.float32 else 6e-2
    ref_out, ref_conv, ref_ssm = runs["cutedsl"]
    torch.testing.assert_close(
        out[:num_actual], ref_out[:num_actual], atol=tol, rtol=tol
    )
    torch.testing.assert_close(conv_state, ref_conv, atol=0, rtol=0)
    torch.testing.assert_close(ssm_state, ref_ssm, atol=tol, rtol=tol)


@pytest.mark.parametrize("cached", [False, True])
@pytest.mark.parametrize("prefill_lens", [[300], [300, 200]])
def test_flashinfer_checkpointed_prefill_pool_matches_gather(prefill_lens, cached):
    """Prefix-caching (align mode) prefills emit checkpoints; with the pool
    route the outputs, final states and exported checkpoints match the gather
    route."""
    gen = torch.Generator(device="cuda").manual_seed(0)
    num_decodes, context = 2, 37 if cached else 0
    seq_lens = [64] * num_decodes + [n + context for n in prefill_lens]
    query_lens = [1] * num_decodes + prefill_lens
    common, weights, inputs, conv_state0, ssm_state0 = _setup(
        seq_lens, query_lens, torch.float32, gen
    )
    fi_cfg, fi_builder = _builder("flashinfer")
    with set_current_vllm_config(fi_cfg):
        meta = fi_builder.build(common_prefix_len=0, common_attn_metadata=common)
    assert meta.prefill_initial_state_mode == ("all" if cached else "none")

    # One checkpoint per prefill row, each in a free cache slot.
    offsets = [0] * num_decodes + [192, 128][: len(prefill_lens)]
    free = conv_state0.shape[0] - 1
    slots = [NULL_BLOCK_ID] * num_decodes + [free, free - 1][: len(prefill_lens)]
    every_n_tokens, cu_starts, rows = plan_flashinfer_checkpoints(offsets, prefill_lens)
    meta = dataclasses.replace(
        meta,
        checkpoint=MambaPrefillCheckpointMetadata(
            checkpoint_offsets=torch.tensor(offsets, device="cuda", dtype=torch.int32),
            state_indices=torch.tensor(slots, device="cuda"),
        ),
        flashinfer_checkpoint_plan=GDNFlashInferCheckpointPlan(
            every_n_tokens=every_n_tokens,
            num_checkpoints=cu_starts[-1],
            cu_starts=torch.tensor(cu_starts, device="cuda"),
            rows=torch.tensor(rows, device="cuda"),
        ),
    )
    workspace = types.SimpleNamespace(
        get_simultaneous=lambda spec: [torch.empty(*spec[0], device="cuda")]
    )
    with patch.object(
        qwen_gdn_linear_attn, "current_workspace_manager", return_value=workspace
    ):
        out, conv, ssm = _run(fi_cfg, meta, weights, inputs, conv_state0, ssm_state0)
        meta_gather = dataclasses.replace(meta, prefill_initial_state_mode="mixed")
        ref_out, ref_conv, ref_ssm = _run(
            fi_cfg, meta_gather, weights, inputs, conv_state0, ssm_state0
        )
    # Actual rows only: padding rows behind num_actual_tokens are
    # uninitialized by contract.
    num_actual = sum(query_lens)
    torch.testing.assert_close(
        out[:num_actual], ref_out[:num_actual], atol=1e-3, rtol=1e-3
    )
    torch.testing.assert_close(conv, ref_conv, atol=0, rtol=0)
    torch.testing.assert_close(ssm, ref_ssm, atol=1e-3, rtol=1e-3)
    for slot in slots[num_decodes:]:
        assert ssm[slot].ne(ssm_state0[slot]).any(), "checkpoint not exported"
