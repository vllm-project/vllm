# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from typing import Any

import pytest
import torch

from vllm.platforms import current_platform

if current_platform.is_rocm():
    pytest.skip(
        reason="FlashInfer GDN prefill is not supported on ROCm.",
        allow_module_level=True,
    )

import flashinfer.gdn_prefill  # noqa: E402
import torch.nn.functional as F  # noqa: E402

from tests.v1.attention.utils import (  # noqa: E402
    BatchSpec,
    create_common_attn_metadata,
)
from vllm.config import (  # noqa: E402
    CacheConfig,
    ModelConfig,
    SchedulerConfig,
    VllmConfig,
    set_current_vllm_config,
)
from vllm.forward_context import set_forward_context  # noqa: E402
from vllm.model_executor.layers.mamba.checkpoint import (  # noqa: E402
    MambaPrefillCheckpointExporter,
    MambaPrefillCheckpointMetadata,
)
from vllm.model_executor.layers.mamba.gdn.qwen_gdn_linear_attn import (
    QwenGatedDeltaNetAttention,
    fi_chunk_gated_delta_rule,
)  # noqa: E402
from vllm.model_executor.layers.mamba.mamba_utils import (  # noqa: E402
    is_conv_state_dim_first,
)
from vllm.model_executor.layers.mamba.ops.causal_conv1d import (  # noqa: E402
    causal_conv1d_fn,
)
from vllm.third_party.flash_linear_attention.ops import (  # noqa: E402
    fused_post_conv_prep,
)
from vllm.utils.torch_utils import (  # noqa: E402
    _encode_layer_name,
    set_default_torch_dtype,
)
from vllm.v1.attention.backends.gdn_attn import (  # noqa: E402
    GDNAttentionMetadataBuilder,
    plan_flashinfer_checkpoints,
)
from vllm.v1.attention.backends.utils import NULL_BLOCK_ID  # noqa: E402
from vllm.v1.kv_cache_interface import MambaSpec  # noqa: E402


def test_flashinfer_gdn_prefill_passes_cu_seqlens_through(monkeypatch):
    """FlashInfer accepts int32 cu_seqlens; the wrapper must not cast them."""
    captured_cu_seqlens = None

    def fake_chunk_gated_delta_rule(**kwargs):
        nonlocal captured_cu_seqlens
        captured_cu_seqlens = kwargs["cu_seqlens"]
        return kwargs["q"]

    monkeypatch.setattr(
        flashinfer.gdn_prefill,
        "chunk_gated_delta_rule",
        fake_chunk_gated_delta_rule,
    )
    q = torch.zeros(1, 2, 1, 2)
    cu_seqlens = torch.tensor([0, 2], dtype=torch.int32)

    output, final_state = fi_chunk_gated_delta_rule(
        q=q,
        k=q,
        v=q,
        g=torch.zeros(1, 2, 1),
        beta=torch.zeros(1, 2, 1),
        initial_state=torch.zeros(1, 1, 2, 2),
        output_final_state=False,
        cu_seqlens=cu_seqlens,
        use_qk_l2norm_in_kernel=False,
    )

    assert captured_cu_seqlens is not None
    assert captured_cu_seqlens is cu_seqlens
    assert output.shape == q.shape
    assert final_state is None


@pytest.mark.skipif(
    not (
        current_platform.is_device_capability(90)
        or current_platform.is_device_capability_family(100)
        or current_platform.is_device_capability_family(120)
    ),
    reason="FlashInfer GDN prefill checkpoints need SM90, SM100 or SM120.",
)
@pytest.mark.parametrize("use_cp", [True, False])
@pytest.mark.parametrize("ssm_dtype", [torch.float32, torch.bfloat16])
def test_flashinfer_checkpoints_export_the_state_at_each_offset(use_cp, ssm_dtype):
    """Each exported checkpoint is the state of the prefill stopped there.

    The leading decode row is outside the prefill kernel. The two checkpointed
    prefills put FlashInfer's stride (their offsets' gcd) below both offsets,
    so the exporter must pick each state out of the shared emitted buffer;
    the third prefill takes no checkpoint. Both FlashInfer routes are covered
    because ``use_cp="auto"`` may take either. The conv cache has MTP's spec
    slots after the window, which the conv kernels leave alone.
    """
    torch.manual_seed(0)
    device = "cuda"
    num_qk_heads, num_v_heads, head_dim, conv_dim, state_len = 2, 4, 128, 16, 3
    num_spec = 2
    offsets = [0, 128, 0, 192]
    query_lens = [200, 50, 300]
    state_indices = [NULL_BLOCK_ID, 3, NULL_BLOCK_ID, 5]
    num_blocks = 8

    num_tokens = sum(query_lens)
    q = F.normalize(
        torch.randn(1, num_tokens, num_qk_heads, head_dim, device=device), dim=-1
    ).bfloat16()
    k = F.normalize(
        torch.randn(1, num_tokens, num_qk_heads, head_dim, device=device), dim=-1
    ).bfloat16()
    v = torch.randn(1, num_tokens, num_v_heads, head_dim, device=device).bfloat16()
    g = F.logsigmoid(torch.randn(1, num_tokens, num_v_heads, device=device)).exp()
    beta = torch.sigmoid(torch.randn(1, num_tokens, num_v_heads, device=device))
    initial_state = 0.1 * torch.randn(
        len(query_lens), num_v_heads, head_dim, head_dim, device=device
    )
    prefill_cu = [0]
    for query_len in query_lens:
        prefill_cu.append(prefill_cu[-1] + query_len)

    every_n_tokens, cu_starts, rows = plan_flashinfer_checkpoints(offsets, query_lens)
    state_checkpoints = torch.empty(
        cu_starts[-1], num_v_heads, head_dim, head_dim, device=device
    )
    fi_chunk_gated_delta_rule(
        q=q,
        k=k,
        v=v,
        g=g,
        beta=beta,
        initial_state=initial_state,
        output_final_state=True,
        cu_seqlens=torch.tensor(prefill_cu, device=device),
        use_qk_l2norm_in_kernel=False,
        state_checkpoints=state_checkpoints,
        checkpoint_cu_starts=torch.tensor(cu_starts, device=device),
        checkpoint_every_n_tokens=every_n_tokens,
        use_cp=use_cp,
    )

    # The exporter sees the decode row's token in front of the prefills.
    raw_qkv = torch.randn(1 + num_tokens, conv_dim, device=device).bfloat16()
    cu_seqlens = torch.tensor([0] + [1 + n for n in prefill_cu], device=device)
    conv_state = torch.zeros(
        num_blocks,
        state_len + num_spec,
        conv_dim,
        device=device,
        dtype=torch.bfloat16,
    ).transpose(-1, -2)
    ssm_state = torch.zeros(
        num_blocks, num_v_heads, head_dim, head_dim, device=device, dtype=ssm_dtype
    )
    MambaPrefillCheckpointExporter(state_len=state_len).export(
        MambaPrefillCheckpointMetadata(
            checkpoint_offsets=torch.tensor(offsets, device=device, dtype=torch.int32),
            state_indices=torch.tensor(state_indices, device=device),
        ),
        raw_qkv=raw_qkv,
        conv_state=conv_state,
        recurrent_checkpoint=state_checkpoints,
        recurrent_state=ssm_state,
        cu_seqlens=cu_seqlens,
        recurrent_rows=torch.tensor(rows, device=device),
    )

    written = []
    for row, (offset, block) in enumerate(zip(offsets, state_indices)):
        if not offset:
            continue
        written.append(block)
        seq = row - 1
        start = prefill_cu[seq]
        _, expected_state = fi_chunk_gated_delta_rule(
            q=q[:, start : start + offset],
            k=k[:, start : start + offset],
            v=v[:, start : start + offset],
            g=g[:, start : start + offset],
            beta=beta[:, start : start + offset],
            initial_state=initial_state[seq : seq + 1],
            output_final_state=True,
            cu_seqlens=torch.tensor([0, offset], device=device),
            use_qk_l2norm_in_kernel=False,
        )
        torch.testing.assert_close(
            ssm_state[block].float(), expected_state[0], atol=2e-2, rtol=2e-2
        )
        end = cu_seqlens[row].item() + offset
        torch.testing.assert_close(
            conv_state[block, :, :state_len],
            raw_qkv[end - state_len : end].T,
            atol=0,
            rtol=0,
        )
        assert not conv_state[block, :, state_len:].any()

    untouched = [i for i in range(num_blocks) if i not in written]
    assert not ssm_state[untouched].any()
    assert not conv_state[untouched].any()


@pytest.mark.parametrize("num_tokens", [1, 326, 336])
@torch.inference_mode()
def test_fused_gate_matches_torch_without_changing_preparation(
    num_tokens: int,
):
    """Fused gate exponentiation preserves the old exp and Q/K/V/beta."""
    torch.manual_seed(0)
    device = "cuda"
    h, hv, k, v = 16, 48, 128, 128
    inputs: dict[str, Any] = dict(
        conv_output=torch.randn(
            num_tokens, 2 * h * k + hv * v, device=device, dtype=torch.bfloat16
        ),
        a=torch.empty(num_tokens, hv, device=device, dtype=torch.bfloat16).uniform_(
            -4, 4
        ),
        b=torch.randn(num_tokens, hv, device=device, dtype=torch.bfloat16),
        A_log=torch.linspace(-6, 2, hv, device=device, dtype=torch.float32),
        dt_bias=torch.empty(hv, device=device, dtype=torch.bfloat16).uniform_(-1, 1),
        num_k_heads=h,
        head_k_dim=k,
        head_v_dim=v,
    )
    reference = fused_post_conv_prep(**inputs, output_g_exp=False)
    precise = fused_post_conv_prep(**inputs, output_g_exp=True)
    for index in (0, 1, 2, 4):
        torch.testing.assert_close(precise[index], reference[index], rtol=0, atol=0)
    torch.testing.assert_close(precise[3], torch.exp(reference[3]), rtol=0, atol=0)


@pytest.mark.skipif(
    not (
        current_platform.is_device_capability(90)
        or current_platform.is_device_capability_family(100)
        or current_platform.is_device_capability_family(120)
    ),
    reason="FlashInfer GDN prefill needs SM90, SM100 or SM120.",
)
@pytest.mark.parametrize("num_tokens", [326, 336])
@torch.inference_mode()
def test_qwen4_prefill_preserves_gate_rounding_and_continuation(
    num_tokens: int, dist_init: None
):
    """Real prefill preserves the old Torch gate, output and cached state."""
    torch.manual_seed(0)
    device = torch.device("cuda")
    prefix = "model.layers.0.linear_attn"
    model_config = ModelConfig(
        model="Qwen/Qwen3.8-Flash-Next-FP8",
        dtype="bfloat16",
        max_model_len=1024,
        enforce_eager=True,
        hf_overrides={"quantization_config": None},
    )
    config = VllmConfig(
        model_config=model_config,
        cache_config=CacheConfig(
            block_size=512,
            mamba_block_size=512,
            mamba_cache_dtype="bfloat16",
            mamba_ssm_cache_dtype="float32",
            mamba_cache_mode="align",
            enable_prefix_caching=True,
        ),
        scheduler_config=SchedulerConfig(
            max_model_len=1024,
            max_num_seqs=1,
            max_num_batched_tokens=1024,
            is_encoder_decoder=model_config.is_encoder_decoder,
        ),
        additional_config={"gdn_prefill_backend": "flashinfer"},
    )
    with (
        set_current_vllm_config(config),
        set_default_torch_dtype(torch.bfloat16),
        torch.device(device),
    ):
        layer = QwenGatedDeltaNetAttention(
            model_config.hf_text_config, config, prefix=prefix
        )
        assert (layer.num_k_heads, layer.num_v_heads) == (16, 48)
        assert layer.head_k_dim == layer.head_v_dim == 128
        assert layer.gdn_prefill_backend == "flashinfer"
        layer.A_log.uniform_(-1, 1)
        layer.dt_bias.uniform_(-1, 1)
        layer.conv1d.weight.normal_(std=0.1)
        spec = layer.get_kv_cache_spec(config)
        assert isinstance(spec, MambaSpec)
        layer.bind_kv_cache(
            torch.zeros(2, 1, 1, spec.page_size_bytes, dtype=torch.uint8)
        )
        builder = GDNAttentionMetadataBuilder(spec, [prefix], config, device)
        position = 0
        for length in (num_tokens, 17):
            common = create_common_attn_metadata(
                BatchSpec(seq_lens=[position + length], query_lens=[length]),
                512,
                device,
                arange_block_indices=True,
            )
            common.block_table_tensor.add_(1)  # Reserve the null-state block.
            common = common.replace(is_prefilling=torch.tensor([True], device="cpu"))
            metadata = builder.build(0, common)
            assert metadata.num_prefills == 1
            assert metadata.flashinfer_checkpoint_plan is None
            assert metadata.prefill_has_initial_state is not None
            assert bool(metadata.prefill_has_initial_state[0]) == (position > 0)
            raw = torch.randn(length, layer.conv_dim) * 0.1
            a = torch.randn(length, layer.num_v_heads)
            b = torch.randn_like(a)
            ref_conv, ref_state = (state.clone() for state in layer.kv_cache)
            conv_view = (
                ref_conv if is_conv_state_dim_first() else ref_conv.transpose(-1, -2)
            )
            conv_output = causal_conv1d_fn(
                raw.clone().T,
                layer.conv1d.weight.view(layer.conv_dim, -1),
                layer.conv1d.bias,
                activation=layer.activation,
                conv_states=conv_view,
                has_initial_state=metadata.has_initial_state,
                cache_indices=metadata.non_spec_state_indices_tensor,
                query_start_loc=metadata.non_spec_query_start_loc,
                metadata=metadata,
            ).T
            q, k, v, g, beta = fused_post_conv_prep(
                conv_output,
                a,
                b,
                layer.A_log,
                layer.dt_bias,
                layer.num_k_heads,
                layer.head_k_dim,
                layer.head_v_dim,
                apply_l2norm=True,
                output_g_exp=False,
            )
            g = torch.exp(g)
            state_indices = metadata.prefill_state_indices
            assert state_indices is not None
            initial_state = ref_state[state_indices].clone()
            initial_state[~metadata.prefill_has_initial_state] = 0
            expected, final_state = fi_chunk_gated_delta_rule(
                q.unsqueeze(0),
                k.unsqueeze(0),
                v.unsqueeze(0),
                g.unsqueeze(0),
                beta.unsqueeze(0),
                initial_state,
                output_final_state=True,
                cu_seqlens=metadata.prefill_query_start_loc,
                use_qk_l2norm_in_kernel=False,
            )
            assert final_state is not None
            ref_state[state_indices] = final_state.to(ref_state.dtype)
            output = torch.full((length + 7, layer.num_v_heads, layer.head_v_dim), 37.0)
            with set_forward_context({prefix: metadata}, config, num_tokens=length):
                torch.ops.vllm.qwen_gdn_attention_core(
                    raw.clone(),
                    b,
                    a,
                    output,
                    layer_name=_encode_layer_name(prefix),
                )
            torch.testing.assert_close(
                output[:length], expected.squeeze(0), rtol=0, atol=0
            )
            torch.testing.assert_close(layer.kv_cache[0], ref_conv, rtol=0, atol=0)
            torch.testing.assert_close(layer.kv_cache[1], ref_state, rtol=0, atol=0)
            assert torch.all(output[length:] == 37.0)
            position += length
