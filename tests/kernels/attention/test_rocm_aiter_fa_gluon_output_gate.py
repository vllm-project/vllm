# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import os

import pytest
import torch

import vllm.config
from tests.compile.backend import TestBackend
from tests.v1.attention.utils import (
    BatchSpec,
    create_common_attn_metadata,
    dense_kv_cache_views,
)
from vllm._aiter_ops import is_aiter_found_and_supported, rocm_aiter_ops
from vllm.compilation.passes.fx_utils import find_op_nodes
from vllm.config import (
    CacheConfig,
    CompilationConfig,
    CompilationMode,
    ModelConfig,
    VllmConfig,
)
from vllm.forward_context import get_forward_context, set_forward_context
from vllm.model_executor.layers.attention import Attention
from vllm.v1.attention.backends.registry import AttentionBackendEnum
from vllm.v1.kv_cache_interface import AttentionSpec, KVCacheLayout

pytestmark = pytest.mark.skip_global_cleanup

if not is_aiter_found_and_supported():
    pytest.skip("ROCm with supported AITER is required", allow_module_level=True)

if not rocm_aiter_ops.pa_decode_gluon_supports_output_gate():
    pytest.skip("AITER pa_decode_gluon has no output_gate", allow_module_level=True)

ATTN_OP = torch.ops.vllm.unified_attention_with_output.default


class GatedAttentionModel(torch.nn.Module):
    """Qwen3-Next gated attention: q and the gate interleaved per head."""

    def __init__(self, vllm_config, num_heads, num_kv_heads, head_size, device):
        super().__init__()
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_size = head_size
        self.block_size = vllm_config.cache_config.block_size
        self.device = device
        self.layer_name = "model.layers.0.self_attn.attn"
        self.attn = Attention(
            num_heads=num_heads,
            head_size=head_size,
            scale=head_size**-0.5,
            num_kv_heads=num_kv_heads,
            cache_config=vllm_config.cache_config,
            prefix=self.layer_name,
            attn_backend=AttentionBackendEnum.ROCM_AITER_FA_GLUON.get_class(),
        )
        self.attn._k_scale = self.attn._k_scale.to(device)
        self.attn._v_scale = self.attn._v_scale.to(device)
        self.kv_cache_spec = self.attn.attn_backend.customize_spec(
            AttentionSpec(
                block_size=self.block_size,
                num_kv_heads=num_kv_heads,
                head_size=head_size,
                dtype=torch.bfloat16,
            )
        )
        self.builder = self.attn.attn_backend.get_builder_cls()(
            kv_cache_spec=self.kv_cache_spec,
            layer_names=[self.attn.layer_name],
            vllm_config=vllm_config,
            device=device,
        )

    def build_attn_metadata(self, num_tokens: int, seq_len: int):
        batch_spec = BatchSpec(
            seq_lens=[seq_len] * num_tokens, query_lens=[1] * num_tokens
        )
        common = create_common_attn_metadata(
            batch_spec, self.block_size, self.device, arange_block_indices=True
        )
        num_blocks = num_tokens * -(-seq_len // self.block_size)
        raw = torch.zeros(
            num_blocks * self.kv_cache_spec.page_size_bytes,
            dtype=torch.int8,
            device=self.device,
        )
        kv_cache = dense_kv_cache_views(
            raw,
            self.kv_cache_spec,
            num_blocks,
            num_layers=1,
            layout=KVCacheLayout.LBHNC,
        )[0]
        generator = torch.Generator(device=self.device).manual_seed(1)
        kv_cache.copy_(
            torch.randn(
                kv_cache.shape,
                generator=generator,
                dtype=kv_cache.dtype,
                device=self.device,
            )
        )
        self.attn.kv_cache = kv_cache
        return self.builder.build(common_prefix_len=0, common_attn_metadata=common)

    def forward(self, q_gate, k, v, gate_in_attention: bool = True, flat_gate=True):
        q_gate = q_gate.view(-1, self.num_heads, 2 * self.head_size)
        q, gate_by_head = torch.chunk(q_gate, 2, dim=-1)
        gate = gate_by_head.reshape(-1, self.num_heads * self.head_size)
        q = q.reshape(-1, self.num_heads * self.head_size)
        if gate_in_attention:
            return self.attn(q, k, v, output_gate=gate if flat_gate else gate_by_head)
        return self.attn(q, k, v) * torch.sigmoid(gate)


@pytest.mark.parametrize("num_tokens", [1, 7, 64, 128])
@pytest.mark.parametrize("seq_len", [37, 3000])
@pytest.mark.parametrize("gate_supported", [True, False], ids=["fused", "unsupported"])
@pytest.mark.parametrize("flat_gate", [True, False], ids=["flat", "by_head"])
def test_attention_output_gate(
    num_tokens, seq_len, gate_supported, flat_gate, monkeypatch
):
    device = os.environ.get("VLLM_TEST_CUDA_DEVICE", "cuda")
    torch.set_default_device(device)
    torch.set_default_dtype(torch.bfloat16)
    torch.manual_seed(0)
    num_heads, num_kv_heads, head_size = 24, 4, 256
    monkeypatch.setenv("VLLM_ROCM_USE_AITER", "1")
    monkeypatch.setenv("VLLM_ROCM_SHUFFLE_KV_CACHE_LAYOUT", "1")
    if not gate_supported:
        monkeypatch.setattr(
            rocm_aiter_ops, "pa_decode_gluon_supports_output_gate", lambda: False
        )
    rocm_aiter_ops.refresh_env_variables()

    vllm_config = VllmConfig(
        model_config=ModelConfig(dtype=torch.bfloat16),
        cache_config=CacheConfig(block_size=16),
        compilation_config=CompilationConfig(mode=CompilationMode.VLLM_COMPILE),
    )
    with vllm.config.set_current_vllm_config(vllm_config):
        model = GatedAttentionModel(
            vllm_config, num_heads, num_kv_heads, head_size, device
        )
        backend = TestBackend()
        q_gate = torch.randn(num_tokens, num_heads * 2 * head_size)
        k = torch.randn(num_tokens, num_kv_heads * head_size)
        v = torch.randn(num_tokens, num_kv_heads * head_size)

        def run(fn, *args):
            with set_forward_context(None, vllm_config):
                attn_metadata = model.build_attn_metadata(num_tokens, seq_len)
                forward_context = get_forward_context()
                forward_context.attn_metadata = attn_metadata
                forward_context.slot_mapping = {
                    model.layer_name: attn_metadata.slot_mapping
                }
                return fn(q_gate.clone(), k.clone(), v.clone(), *args)

        reference = run(model, False)
        torch._dynamo.mark_dynamic(q_gate, 0)
        torch._dynamo.mark_dynamic(k, 0)
        torch._dynamo.mark_dynamic(v, 0)
        gated = run(torch.compile(model, backend=backend), True, flat_gate)

    assert model.attn.fuse_output_gate == gate_supported
    sigmoid = torch.ops.aten.sigmoid.default
    assert backend.op_count(sigmoid) == (0 if gate_supported else 1)
    (attn,) = find_op_nodes(ATTN_OP, backend.graph_post_pass)
    assert (attn.kwargs.get("output_gate") is not None) == gate_supported
    torch.testing.assert_close(gated, reference, atol=2e-2, rtol=2e-2)
