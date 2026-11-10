# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Leading spec rows in mixed GDN batches: row slices match the index path.

When a batch mixes spec decodes with prefills, `GDNAttentionMetadataBuilder`
sets `spec_rows_lead` if the spec rows come first. `_forward_core` then
slices the spec and non-spec rows instead of gathering them by
`spec_token_indx` / `non_spec_token_indx` and scattering the outputs back
(and FlashInfer writes the prefill output in place). The two paths run the same
kernels on the same values, so the core-attention output, the conv state and
the recurrent state must be bitwise equal. A decode row without draft tokens is
a spec row that the runner can sort among the prefills; such a batch must keep
the index path.
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
        reason="The FlashInfer and CuteDSL GDN prefill backends need CUDA SM10x.",
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
from vllm.v1.attention.backends.gdn_attn import (  # noqa: E402
    GDNAttentionMetadataBuilder,
)
from vllm.v1.kv_cache_interface import MambaSpec  # noqa: E402

NUM_SPEC = 3
H = 4
HV = 8
K = 128
V = 128
CONV_KERNEL = 4
KEY_DIM = H * K
VALUE_DIM = HV * V
CONV_DIM = 2 * KEY_DIM + VALUE_DIM
BLOCK_SIZE = 16
PREFIX = "model.layers.0.linear_attn"

# (context length, query length, draft tokens) per request, in batch order.
# Draft tokens >= 0 make a spec row (0: a decode row without drafts), -1 a
# prefill chunk. The orders are the model runner's: drafted rows first, then
# the other rows by (query length != NUM_SPEC + 1, query length).
BATCHES = {
    "lead-uniform": [
        (90, 4, 3),
        (200, 4, 3),
        (37, 4, 3),
        (0, 300, -1),
        (61, 64, -1),
    ],
    "lead-ragged": [
        (90, 4, 3),
        (200, 2, 1),
        (37, 3, 2),
        (55, 1, 0),
        (0, 300, -1),
        (61, 5, -1),
    ],
    "not-lead": [
        (90, 4, 3),
        (200, 4, 3),
        (61, 4, -1),
        (55, 1, 0),
        (70, 1, -1),
        (44, 1, 0),
        (0, 200, -1),
    ],
}


def _make_vllm_config(backend: str):
    cfg = create_vllm_config(
        model_name="Qwen/Qwen3.5-0.8B",
        block_size=BLOCK_SIZE,
        hf_config_override={"linear_key_head_dim": K},
    )
    cfg.additional_config = {"gdn_prefill_backend": backend}
    cfg.cache_config.mamba_cache_mode = "none"
    cfg.speculative_config = SpeculativeConfig(
        method="ngram", num_speculative_tokens=NUM_SPEC
    )
    return cfg


def _build_layer(vllm_config, conv_state, ssm_state, a_log, dt_bias, conv_weight):
    """A minimal object that runs the real `_forward_core` bound to it."""
    layer = types.SimpleNamespace(
        prefix=PREFIX,
        enable_packed_recurrent_decode=False,
        tp_size=1,
        num_k_heads=H,
        num_v_heads=HV,
        head_k_dim=K,
        head_v_dim=V,
        key_dim=KEY_DIM,
        value_dim=VALUE_DIM,
        activation="silu",
        A_log=a_log,
        dt_bias=dt_bias,
        conv1d=types.SimpleNamespace(weight=conv_weight, bias=None),
        kv_cache=(conv_state, ssm_state),
    )
    with set_current_vllm_config(vllm_config):
        layer.chunk_gated_delta_rule = ChunkGatedDeltaRule()
    layer.gdn_prefill_backend = layer.chunk_gated_delta_rule.gdn_prefill_backend
    for name in ("rearrange_mixed_qkv", "_forward_flashinfer_prefill", "_forward_core"):
        setattr(
            layer,
            name,
            types.MethodType(getattr(QwenGatedDeltaNetAttention, name), layer),
        )
    return layer


@pytest.mark.parametrize("backend", ["flashinfer", "cutedsl"])
@pytest.mark.parametrize("batch_name", list(BATCHES))
@torch.inference_mode()
def test_spec_rows_lead_matches_index_path(backend: str, batch_name: str) -> None:
    torch.manual_seed(0)
    device = torch.device("cuda")
    rows = BATCHES[batch_name]
    vllm_config = _make_vllm_config(backend)
    builder = GDNAttentionMetadataBuilder(
        kv_cache_spec=MambaSpec(
            block_size=BLOCK_SIZE,
            shapes=((16, 64),),
            dtypes=(torch.float16,),
            num_speculative_blocks=NUM_SPEC,
        ),
        layer_names=[PREFIX],
        vllm_config=vllm_config,
        device=device,
    )
    if builder.gdn_prefill_backend != backend:
        pytest.skip(f"GDN prefill backend {backend} is unavailable here")

    batch = BatchSpec(
        seq_lens=[ctx + q for ctx, q, _ in rows], query_lens=[q for _, q, _ in rows]
    )
    common = create_common_attn_metadata(
        batch, BLOCK_SIZE, device, arange_block_indices=True
    )
    common.block_table_tensor.add_(1)
    accepted = [
        int(torch.randint(1, drafts + 2, ())) if drafts >= 0 else 1
        for _, _, drafts in rows
    ]
    with set_current_vllm_config(vllm_config):
        meta = builder.build(
            common_prefix_len=0,
            common_attn_metadata=common,
            num_accepted_tokens=torch.tensor(
                accepted, dtype=torch.int32, device=device
            ),
            num_decode_draft_tokens_cpu=torch.tensor(
                [drafts for _, _, drafts in rows], dtype=torch.int32
            ),
        )
    assert meta.num_prefills > 0 and meta.num_spec_decodes > 0
    assert meta.spec_rows_lead == batch_name.startswith("lead")
    index_meta = dataclasses.replace(meta, spec_rows_lead=False)

    pool_size = int(common.block_table_tensor.max().item()) + 1
    conv_state_shape, temporal_state_shape = (
        MambaStateShapeCalculator.gated_delta_net_state_shape(
            1, H, HV, K, V, CONV_KERNEL, NUM_SPEC
        )
    )
    conv_state0 = 0.05 * torch.randn(
        pool_size, *conv_state_shape, dtype=torch.bfloat16, device=device
    )
    ssm_state0 = 0.05 * torch.randn(
        pool_size, *temporal_state_shape, dtype=torch.float32, device=device
    )
    a_log = 0.1 * torch.randn(HV, dtype=torch.float32, device=device)
    dt_bias = 0.1 * torch.randn(HV, dtype=torch.bfloat16, device=device)
    conv_weight = 0.1 * torch.randn(
        CONV_DIM, 1, CONV_KERNEL, dtype=torch.bfloat16, device=device
    )
    num_tokens = batch.compute_num_tokens()
    # As in the model, mixed_qkv is a strided column view of the packed qkvz
    # projection, and z is the rest of it.
    qkvz = 0.1 * torch.randn(
        num_tokens, CONV_DIM + VALUE_DIM, dtype=torch.bfloat16, device=device
    )
    b = 0.1 * torch.randn(num_tokens, HV, dtype=torch.bfloat16, device=device)
    a = 0.1 * torch.randn_like(b)

    def run(metadata):
        layer = _build_layer(
            vllm_config,
            conv_state0.clone(),
            ssm_state0.clone(),
            a_log,
            dt_bias,
            conv_weight,
        )
        x = qkvz.clone()
        out = torch.zeros(num_tokens, HV, V, dtype=torch.bfloat16, device=device)
        ctx = types.SimpleNamespace(attn_metadata={PREFIX: metadata})
        with patch.object(
            qwen_gdn_linear_attn, "get_forward_context", return_value=ctx
        ):
            layer._forward_core(
                mixed_qkv=x[:, :CONV_DIM], b=b.clone(), a=a.clone(), core_attn_out=out
            )
        return out, layer.kv_cache[0], layer.kv_cache[1], x

    out, conv_state, ssm_state, x = run(meta)
    ref_out, ref_conv_state, ref_ssm_state, _ = run(index_meta)
    assert torch.equal(out, ref_out)
    assert torch.equal(conv_state, ref_conv_state)
    assert torch.equal(ssm_state, ref_ssm_state)
    # The in-place conv update writes only the spec rows of the qkv columns.
    num_spec_tokens = meta.num_spec_decode_tokens
    assert torch.equal(x[:, CONV_DIM:], qkvz[:, CONV_DIM:])
    if meta.spec_rows_lead:
        assert torch.equal(x[num_spec_tokens:], qkvz[num_spec_tokens:])
    else:
        assert torch.equal(x, qkvz)
